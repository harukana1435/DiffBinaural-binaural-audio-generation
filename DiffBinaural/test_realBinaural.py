# System libs
import os
import random
import time
import json
import csv
# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
from imageio import imwrite as imsave
from mir_eval.separation import bss_eval_sources
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Our libs
from utils.arguments import ArgParser
from dataset.genaudio_realBinaural import GenAudioRealBinauralDataset
from modules import models
from diffusion_utils import diffusion_pytorch
from utils.helpers import AverageMeter, magnitude2heatmap, \
    istft_reconstruction, warpgrid, makedirs, save_mel_to_tensorboard, _nested_map, save_checkpoint,load_checkpoint,scan_checkpoint
import warnings
# UserWarningとFutureWarningを無視する
warnings.filterwarnings("ignore", category=FutureWarning)

# 改良コンポーネント
from training_stabilizer import TrainingStabilizer, create_stabilized_training_loop

h = None
device = None

# RealBinauralNetWrapperを使用（train_realBinaural.pyから）
class RealBinauralNetWrapper(torch.nn.Module):
    def __init__(self, nets, use_enhanced_loss=False):
        super(RealBinauralNetWrapper, self).__init__()
        self.net_frame, self.net = nets
        self.sampler = diffusion_pytorch.GaussianDiffusion(
            self.net,
            image_size = 80,
            timesteps = 1000,   # number of steps
            sampling_timesteps = 25, # if ddim else None
            loss_type = 'l1',    # L1 or L2
            objective = 'pred_noise', # pred_noise or pred_x0
            beta_schedule = 'cosine', # cosine schedule for better performance
            ddim_sampling_eta = 0,
            auto_normalize = False,
            min_snr_loss_weight=False
        )
        
        
        # 学習安定化（テスト時は使用しない）
        self.stabilizer = TrainingStabilizer()

        self.max=2.5
        self.min=-12

    def move_to_device(self, device):
        """モジュール全体を指定デバイスに移動（DataParallel対応）"""
        # メインのモジュールを移動
        self.to(device)
        
        # 各サブモジュールも個別に移動（確実性のため）
        if hasattr(self, 'net_frame') and hasattr(self.net_frame, "to"):
            self.net_frame.to(device)
        if hasattr(self, 'net') and hasattr(self.net, "to"):
            self.net.to(device)
        if hasattr(self, 'sampler') and hasattr(self.sampler, "to"):
            self.sampler.to(device)
        if hasattr(self, 'enhanced_loss') and hasattr(self.enhanced_loss, "to"):
            self.enhanced_loss.to(device)
        
        # デバイス確認
        if hasattr(self, 'net'):
            actual_device = next(self.net.parameters()).device
            print(f"RealBinauralNetWrapper moved to {device} (actual: {actual_device})")
        else:
            print(f"RealBinauralNetWrapper moved to {device}")

    def forward(self, batch_data, args):
        # テスト時は使用しない
        pass

    def sample(self, batch_data, args):
        """サンプリング"""
        model_device = next(self.net_frame.parameters()).device
        batch_data = _nested_map(batch_data, lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x)
        
        mono_mel = batch_data['mono_mel']
        binaural_mel = batch_data['binaural_mel']
        frames = batch_data['frames']
        pos = batch_data['pos_data']
        mask = batch_data['mask']
        pos_2d = batch_data['2d_pos_data']

        # magnitude normalization (train_realBinaural.py と同じ)
        mono_mel = torch.clamp(mono_mel, min=self.min, max=self.max)
        
        # 0-1 normalization
        mono_mel = 2.0 * (mono_mel - self.min) / (self.max - self.min) - 1.0
        
        # detach
        mono_mel = mono_mel.detach()
        binaural_mel = binaural_mel.detach()
        
        # Frame feature (conditions)
        if hasattr(self.net_frame, 'forward_multiframe'):
            feat_frames = self.net_frame.forward_multiframe(frames, pos_2d, mask)
            #feat_frames = self.net_frame.forward_multiframe(frames)
        else:
            print("no frame feature")
        
        # DDIM sampling（テスト時は25ステップ）
        preds = self.sampler.ddim_sample(
            condition=[mono_mel, feat_frames], 
            return_all_timesteps=True, 
            silence_mask_sampling=False,
            sampling_timesteps=25  # テスト時は25ステップ
        )

        pred = preds[:, -1, ...]
        
        # 逆正規化
        pred = 0.5 * (pred + 1.0) * (self.max - self.min) + self.min
        pred = torch.clamp(pred, min=self.min, max=self.max)
        
        return {'pred_mag': pred, 'gt_mag': binaural_mel}

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def generate(netWrapper, loader, args):
    torch.set_grad_enabled(False)
    
    mel = None
    overlap_count = None

    # switch to eval mode
    netWrapper.eval()
    
    # 両端を除外するフレーム数を設定（メルスペクトログラムの時間軸の10%程度）
    crop_frames = getattr(args, 'crop_frames', 8)  # デフォルトで8フレーム（両端から各8フレーム除外）

    for i, batch_data in enumerate(loader):

        if mel==None:
            # バッチサイズ1の場合の処理
            t = batch_data['total_time_frame'].item() if batch_data['total_time_frame'].numel() == 1 else batch_data['total_time_frame'][0].item()
            m=args.num_mels
            device=next(netWrapper.module.net.parameters()).device
            # リアルバイノーラルは2チャンネル（左右）
            mel = torch.zeros((2, m, t)).to(device)
            overlap_count = torch.zeros((2, m, t)).to(device)
            print(f"Initialized mel buffer: {mel.shape}")
        
        # forward pass
        outputs = netWrapper.module.sample(batch_data, args)
        
        B = outputs['pred_mag'].shape[0]

        print(f"Processing batch {i}: total_time_frame={batch_data['total_time_frame']}, mel buffer shape={mel.shape if mel is not None else 'None'}")

        for j in range(B):
            start_frame = batch_data['start_time_frame'][j]
            pred_mag = outputs['pred_mag'][j]  # (2, F, T)
            
            print(f"Batch {j}: start_frame={start_frame}, pred_mag.shape={pred_mag.shape}")
            
            # 両端をクロップ（境界部分の品質低下を除外）
            T = pred_mag.shape[-1]  # 時間軸の長さ
            if T > 2 * crop_frames:
                # 両端からcrop_framesずつ除外
                cropped_pred = pred_mag[:, :, crop_frames:T-crop_frames]
                cropped_start = start_frame + crop_frames
                cropped_end = cropped_start + cropped_pred.shape[-1]
                
                print(f"  Cropping: {T} -> {cropped_pred.shape[-1]} frames, placing at [{cropped_start}:{cropped_end}]")
                
                # 範囲チェック
                if cropped_end <= mel.shape[-1]:
                    mel[:, :, cropped_start:cropped_end] += cropped_pred
                    overlap_count[:, :, cropped_start:cropped_end] += 1
                else:
                    print(f"  Warning: cropped_end ({cropped_end}) > mel.shape[-1] ({mel.shape[-1]})")
            else:
                # セグメントが短すぎる場合は中央部分のみ使用
                center_start = T // 4
                center_end = T - T // 4
                if center_end > center_start:
                    cropped_pred = pred_mag[:, :, center_start:center_end]
                    cropped_start = start_frame + center_start
                    cropped_end = cropped_start + cropped_pred.shape[-1]
                    
                    print(f"  Short segment: using center [{center_start}:{center_end}], placing at [{cropped_start}:{cropped_end}]")
                    
                    if cropped_end <= mel.shape[-1]:
                        mel[:, :, cropped_start:cropped_end] += cropped_pred
                        overlap_count[:, :, cropped_start:cropped_end] += 1
                    else:
                        print(f"  Warning: cropped_end ({cropped_end}) > mel.shape[-1] ({mel.shape[-1]})")
    
    # ゼロ除算を避けるため、overlap_countが0の部分は1にする
    overlap_count = torch.clamp(overlap_count, min=1)
    mel = torch.div(mel, overlap_count)
    
    return mel

def get_audio_filelist(file):
    # トレーニングデータのファイルを読み込む
    with open(file, 'r', encoding='utf-8') as fi:
        reader = csv.reader(fi)
        next(reader)  # 1行目（カラム名）をスキップ
        training_files = [row[0]  # Audio Pathの部分（1列目）
                          for row in reader if len(row) > 0]
    return training_files

def main(args):
    # Network Builders
    builder = models.ModelBuilder()
    net_frame = builder.build_visual(
        pool_type=args.img_pool,
        weights=args.weights_frame,
        arch_frame=args.arch_frame)
    net_unet = builder.build_unet(weights=args.weights_unet)
    nets = (net_frame, net_unet)

    # Wrap networks for multiple GPUs (RealBinauralNetWrapperを使用)
    netWrapper = RealBinauralNetWrapper(nets, use_enhanced_loss=False)
    
    print(f"Using {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    netWrapper.move_to_device(args.device)  # モデルをデバイスに移動
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=args.gpu_ids)  # ラップ

    # モデルの最初のパラメータのデバイスを確認
    model_device = next(netWrapper.module.net.parameters()).device
    print(f"The model is on device: {model_device}")

    filelist = get_audio_filelist(args.list_test)

    os.makedirs(args.output_dir_left, exist_ok=True)
    os.makedirs(args.output_dir_right, exist_ok=True)

    netWrapper.eval()
    
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            print(f"Processing {i+1}/{len(filelist)}: {filename}")
            
            dataset_audio = GenAudioRealBinauralDataset(filename, args)
            loader_audio = torch.utils.data.DataLoader(
                dataset_audio,
                batch_size=1,  # バッチサイズを1に固定してエラーを回避
                shuffle=False, 
                num_workers=0,  # マルチプロセシングを無効化してデバッグを容易に
                drop_last=False
            )
            
            mel = generate(netWrapper, loader_audio, args)
            print(f"Generated mel shape: {mel.shape}")
            mel = mel.cpu()
            
            # 左右チャンネル分離
            left_mel = mel[0]  # (F, T)
            right_mel = mel[1]  # (F, T)

            # ファイル名から拡張子を除去
            basename = os.path.splitext(os.path.split(filename)[1])[0]
            output_file_left = os.path.join(args.output_dir_left, basename + '.npy')
            output_file_right = os.path.join(args.output_dir_right, basename + '.npy')
            
            # numpyファイルとして保存
            np.save(output_file_left, left_mel.numpy())
            np.save(output_file_right, right_mel.numpy())
            
            print(f"Saved: {output_file_left}")
            print(f"Saved: {output_file_right}")

if __name__ == '__main__':
    print('Initializing Real Binaural Inference Process..')

    parser = ArgParser()
    args = parser.parse_test_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))  # 設定
    torch.cuda.set_device(args.gpu_ids[0])  # 明示的にデバイスを設定
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # リアルバイノーラルデータセット用の設定
    args.data_root = getattr(args, 'data_root', '/home/h-okano/real_dataset')
    
    # 境界部分除外設定（メルスペクトログラムの両端から除外するフレーム数）
    args.crop_frames = getattr(args, 'crop_frames', 8)  # デフォルト8フレーム
    print(f"Crop frames (both ends): {args.crop_frames}")
    
    # チェックポイントパス設定
    args.weights_frame = os.path.join(args.ckpt, "frame_best.pth")
    args.weights_unet = os.path.join(args.ckpt, "unet_best.pth")
    
    # 出力ディレクトリの設定（リアルバイノーラル用）
    if not hasattr(args, 'output_dir_left'):
        args.output_dir_left = os.path.join(args.ckpt, "output_left")
    if not hasattr(args, 'output_dir_right'):
        args.output_dir_right = os.path.join(args.ckpt, "output_right")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args) 
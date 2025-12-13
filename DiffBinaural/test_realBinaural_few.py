# System libs
import os
import random
import csv
# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
from binaural_loss_enhanced import BinauralEnhancedLoss, enhanced_l1_loss
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
        
        # 改良された損失関数（テスト時は使用しない）
        self.use_enhanced_loss = use_enhanced_loss
        if use_enhanced_loss:
            self.enhanced_loss = BinauralEnhancedLoss(
                coherence_weight=0.2,
                dynamics_weight=0.1,
                stereo_weight=0.15
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



def save_comparison_plots(pred_mel, gt_mel, output_dir, basename, segment_idx):
    """予測とグラウンドトゥルースの比較プロットを保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # CPUに移動
    if torch.is_tensor(pred_mel):
        pred_mel = pred_mel.cpu().numpy()
    if torch.is_tensor(gt_mel):
        gt_mel = gt_mel.cpu().numpy()
    
    # 値の範囲を調整 (max=1, min=-9)
    pred_mel = np.clip(pred_mel, -9, 1)
    gt_mel = np.clip(gt_mel, -9, 1)
    
    # 左右チャンネル分離（pred_melは2チャンネル、gt_melも2チャンネル）
    pred_left = pred_mel[0]  # (F, T)
    pred_right = pred_mel[1] # (F, T)
    gt_left = gt_mel[0]      # (F, T)
    gt_right = gt_mel[1]     # (F, T)
    
    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 共通のカラーマップの範囲設定
    vmin, vmax = -9, 1
    
    # 左チャンネル
    im1 = axes[0, 0].imshow(pred_left, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Predicted Left - {basename}_seg{segment_idx}')
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Mel Frequency')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(gt_left, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Ground Truth Left - {basename}_seg{segment_idx}')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Mel Frequency')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 右チャンネル
    im3 = axes[1, 0].imshow(pred_right, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'Predicted Right - {basename}_seg{segment_idx}')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Mel Frequency')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(gt_right, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Ground Truth Right - {basename}_seg{segment_idx}')
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('Mel Frequency')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, f'{basename}_seg{segment_idx}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {output_path}")

def generate_and_compare(netWrapper, loader, args, max_samples=3):
    """生成とテストデータの比較を行う（少数サンプル用）"""
    torch.set_grad_enabled(False)
    
    # switch to eval mode
    netWrapper.eval()
    
    results = []
    comparison_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(comparison_dir, exist_ok=True)
    
    processed_count = 0
    
    for i, batch_data in enumerate(loader):
        if processed_count >= max_samples:
            break
            
        print(f"\nProcessing batch {i+1}/{min(len(loader), max_samples)}")
        
        # forward pass
        outputs = netWrapper.module.sample(batch_data, args)
        
        B = outputs['pred_mag'].shape[0]
        
        for j in range(B):
            if processed_count >= max_samples:
                break
                
            pred_mag = outputs['pred_mag'][j]  # (2, F, T)
            gt_mag = outputs['gt_mag'][j]      # (2, F, T)
            
            print(f"  Sample {processed_count+1}: pred_shape={pred_mag.shape}, gt_shape={gt_mag.shape}")
            
            # 比較プロット保存（シンプルに可視化のみ）
            basename = f"sample_{processed_count:03d}"
            save_comparison_plots(pred_mag, gt_mag, comparison_dir, basename, processed_count)
            
            # 簡単な統計情報のみ表示
            pred_np = pred_mag.cpu().numpy()
            gt_np = gt_mag.cpu().numpy()
            mse = np.mean((pred_np - gt_np) ** 2)
            print(f"  MSE: {mse:.6f}")
            
            processed_count += 1
    
    return processed_count

def save_simple_summary(total_samples, output_dir):
    """シンプルなサマリーを保存"""
    summary_path = os.path.join(output_dir, "visualization_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write(f"Generated visualization comparisons\n")
        f.write(f"Total samples visualized: {total_samples}\n")
        f.write(f"Mel spectrogram range: min=-9, max=1\n")
        f.write(f"Output directory: {output_dir}\n")
    
    print(f"\nVisualization summary saved: {summary_path}")
    print(f"Total samples visualized: {total_samples}")

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
    
    # 少数のファイルのみ処理
    max_files = getattr(args, 'max_files', 2)
    filelist = filelist[:max_files]
    print(f"Processing {len(filelist)} files for comparison")

    os.makedirs(args.output_dir, exist_ok=True)

    netWrapper.eval()
    
    total_samples_processed = 0
    
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(filelist)}: {filename}")
            print(f"{'='*60}")
            
            dataset_audio = GenAudioRealBinauralDataset(filename, args)
            loader_audio = torch.utils.data.DataLoader(
                dataset_audio,
                batch_size=1,  # バッチサイズを1に固定
                shuffle=False, 
                num_workers=0,  # マルチプロセシングを無効化
                drop_last=False
            )
            
            # ファイル名から拡張子を除去
            basename = os.path.splitext(os.path.split(filename)[1])[0]
            
            # ファイル別の出力ディレクトリ作成
            file_output_dir = os.path.join(args.output_dir, basename)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 一時的に出力ディレクトリを変更
            original_output_dir = args.output_dir
            args.output_dir = file_output_dir
            
            # 各ファイルから最初の3つのサンプルを可視化
            samples_count = generate_and_compare(netWrapper, loader_audio, args, max_samples=3)
            total_samples_processed += samples_count
            
            # ファイル別のサマリー保存
            save_simple_summary(samples_count, file_output_dir)
            
            # 出力ディレクトリを元に戻す
            args.output_dir = original_output_dir
    
    # 全体のサマリー保存
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    save_simple_summary(total_samples_processed, args.output_dir)

if __name__ == '__main__':
    print('Initializing Real Binaural Comparison Process..')

    parser = ArgParser()
    args = parser.parse_test_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))  # 設定
    torch.cuda.set_device(args.gpu_ids[0])  # 明示的にデバイスを設定
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # リアルバイノーラルデータセット用の設定
    args.data_root = getattr(args, 'data_root', '/home/h-okano/real_dataset')
    
    # チェックポイントパス設定
    args.weights_frame = os.path.join(args.ckpt, "frame_best.pth")
    args.weights_unet = os.path.join(args.ckpt, "unet_best.pth")
    
    # 比較用の設定
    args.max_files = getattr(args, 'max_files', 2)  # 処理するファイル数
    
    args.output_dir = os.path.join(args.ckpt, "visualization_results")
    
    print(f"Max files to process: {args.max_files}")
    print(f"Samples per file: 3 (first 3 samples)")
    print(f"Mel spectrogram range: min=-9, max=1")
    print(f"Output directory: {args.output_dir}")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args) 
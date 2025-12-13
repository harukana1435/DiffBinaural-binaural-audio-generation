# System libs
import os
import random
import time
import json

# Numerical libs
from matplotlib import pyplot as plt
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
from dataset.dataset_real_binaural import RealBinauralDataset
from modules import models
from diffusion_utils import diffusion_pytorch
from utils.helpers import AverageMeter, magnitude2heatmap, \
    istft_reconstruction, warpgrid, makedirs, save_mel_to_tensorboard, save_mel_to_tensorboard2, \
    _nested_map, save_checkpoint, load_checkpoint, scan_checkpoint, min_max_normalize, invert_min_max_normalize

from training_stabilizer import TrainingStabilizer, create_stabilized_training_loop

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Network wrapper, defines forward pass
class RealBinauralNetWrapper(torch.nn.Module):
    def __init__(self, nets, use_enhanced_loss=True):
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
        
        # 学習安定化
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
        mono_mel = batch_data['mono_mel'] # (B, C, F, T) - モノラル音声のメルスペクトログラム 
        binaural_mel = batch_data['binaural_mel'] # (B, C, F, T)
        frames = batch_data['frames'] # (B, C, L, N, H, W) - ダミーフレーム
        pos = batch_data['pos_data'] # (B, L, N, 3)
        mask = batch_data['mask'] # (B, L, N)
        pos_2d = batch_data['2d_pos_data'] # (B, L, N, 2)

        # デバッグ: 入力データのNaNチェック
        if torch.isnan(pos_2d).any():
            print(f"⚠️ NaN detected in pos_2d: {torch.isnan(pos_2d).sum()} values")
            pos_2d = torch.nan_to_num(pos_2d, nan=0.0)
            
        if torch.isnan(mono_mel).any():
            print(f"⚠️ NaN detected in mono_mel: {torch.isnan(mono_mel).sum()} values")
            
        if torch.isnan(binaural_mel).any():
            print(f"⚠️ NaN detected in binaural_mel: {torch.isnan(binaural_mel).sum()} values")
        
        
        if args.weighted_loss:
            weight = mono_mel
            weight = weight > 1e-3
        else:
            weight = torch.ones_like(mono_mel)

        # LOG magnitude normalization
        # mono_mel = torch.log1p(mono_mel)
        # binaural_mel = torch.log1p(binaural_mel)

        mono_mel = torch.clamp(mono_mel, min=self.min, max=self.max)
        binaural_mel = torch.clamp(binaural_mel, min=self.min, max=self.max)
        
        # # 0-1 normalization
        mono_mel = 2.0 * (mono_mel - self.min) / (self.max - self.min) - 1.0
        binaural_mel = 2.0 * (binaural_mel - self.min) / (self.max - self.min) - 1.0

        # detach
        mono_mel = mono_mel.detach()
        binaural_mel = binaural_mel.detach()


        # Frame feature (conditions) - 実際のフレーム特徴量または位置情報のみ
        if hasattr(self.net_frame, 'forward_multiframe'):
            feat_frames = self.net_frame.forward_multiframe(frames, pos_2d, mask)
            #feat_frames = self.net_frame.forward_multiframe(frames)
            
            # デバッグ: feat_framesのNaNチェック
            if torch.isnan(feat_frames).any():
                print(f"⚠️ NaN detected in feat_frames: {torch.isnan(feat_frames).sum()} values")
                print(f"   pos_2d range: [{pos_2d.min():.3f}, {pos_2d.max():.3f}]")
                print(f"   mask sum (True=invalid): {mask.sum()}/{mask.numel()}")
                print(f"   feat_frames stats: mean={torch.nanmean(feat_frames):.3f}, std={torch.std(feat_frames[~torch.isnan(feat_frames)]):.3f}")
        else:
            # フレーム処理が無い場合は位置情報のみ使用
            print("no frame feature")
            # feat_frames = pos_2d.mean(dim=(1, 2))  # [B, 2] 平均位置
        
        # 基本損失のみ
        loss_mel = self.sampler(binaural_mel, [mono_mel, feat_frames], cfg=True, log=False, weight=weight)
        return loss_mel

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

        # magnitude normalization
        # mono_mel = torch.log1p(mono_mel)

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
            # feat_frames = pos_2d.mean(dim=(1, 2))
        
        # DDIM sampling（validation時は25ステップ）
        preds = self.sampler.ddim_sample(
            condition=[mono_mel, feat_frames], 
            return_all_timesteps=True, 
            silence_mask_sampling=False,
            sampling_timesteps=25  # validation時は25ステップ
        )

        pred = preds[:, -1, ...]
        
        # 逆正規化
        pred = 0.5 * (pred + 1.0) * (self.max - self.min) + self.min
        pred = torch.clamp(pred, min=self.min, max=self.max)

        # pred = torch.exp(pred) - 1
        
        return {'pred_mag': pred, 'gt_mag': binaural_mel}

def calc_metrics(batch_data, outputs, args):
    """メトリクス計算"""
    l2_distance_meter = AverageMeter()

    gt_mag = outputs['gt_mag']
    pred_mag = outputs['pred_mag']
    
    gt_mag = gt_mag.to(pred_mag.device)
    B = gt_mag.shape[0]

    for j in range(B):
        l2_distance = np.linalg.norm(gt_mag[j].cpu().numpy() - pred_mag[j].cpu().numpy())
        l2_distance_meter.update(l2_distance)

    return l2_distance_meter.average()

def evaluate(netWrapper, loader, history, epoch, args, writer):
    """評価"""
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    makedirs(args.vis, remove=False)
    netWrapper.eval()

    mel_l2 = AverageMeter()
    
    # 可視化用のデータを保存
    first_batch_data = None
    first_outputs = None

    for i, batch_data in enumerate(loader):
        outputs = netWrapper.module.sample(batch_data, args)
        data_l2 = calc_metrics(batch_data, outputs, args)
        mel_l2.update(data_l2)
        
        # 最初のバッチを保存
        if i == 0:
            first_batch_data = batch_data
            first_outputs = outputs
    
    print('[Eval Summary] Epoch: {}, mel_l2: {:.4f}'.format(epoch, mel_l2.average()))
    
    history['val']['epoch'].append(epoch)
    history['val']['mel_l2'].append(mel_l2.average())
    
    # 最初のバッチを可視化
    if first_batch_data is not None:
        save_mel_to_tensorboard(first_batch_data, first_outputs, writer, epoch)
    
    if args.mode != "eval":
        writer.add_scalar('eval mel_l2', mel_l2.average(), epoch)

def train(netWrapper, loader, optimizer, history, epoch, args, writer, running_loss):
    """学習"""
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    netWrapper.train()

    torch.cuda.synchronize()
    tic = time.perf_counter()

    for i, batch_data in enumerate(tqdm(loader, desc="Training Progress", ncols=100)):
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        optimizer.zero_grad()
        err = netWrapper.forward(batch_data, args)
        err = err.mean()

        err.backward()
        
        # 改良された学習安定化
        if hasattr(netWrapper.module, 'stabilizer'):
            stabilizer_info = netWrapper.module.stabilizer.training_step(
                netWrapper, optimizer, err.item()
            )
        else:
            nn.utils.clip_grad_norm_(netWrapper.parameters(), 5.0)
        
        optimizer.step()

        running_loss += err.item()

        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_unet: {}, lr_frame: {}, loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_unet, args.lr_frame, err.item()))
            
            writer.add_scalar('training loss', running_loss / args.disp_iter, epoch * len(loader) + i)
            
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.mean().item())
            running_loss = 0.0

def basic_checkpoint(nets, history, epoch, args):
    """基本的なチェックポイント保存"""
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_frame, net_unet) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(net_frame.state_dict(), '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_unet.state_dict(), '{}/unet_{}'.format(args.ckpt, suffix_latest))    

    cur_mel_l2 = history['val']['mel_l2'][-1]
    if cur_mel_l2 < args.best_mel_l2:
        print("saving best at {} epoch".format(epoch))
        args.best_mel_l2 = cur_mel_l2
        torch.save(net_frame.state_dict(), '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_unet.state_dict(), '{}/unet_{}'.format(args.ckpt, suffix_best))

def advanced_checkpoint(nets, history, epoch, args):
    """詳細なチェックポイント保存"""
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_frame, net_unet) = nets
    
    with open(os.path.join(args.ckpt,'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    checkpoint_path_history = "{}/history_{:06d}".format(args.ckpt, epoch)
    save_checkpoint(checkpoint_path_history, {'history':history})
    checkpoint_path_model = "{}/frame_{:06d}".format(args.ckpt, epoch)
    save_checkpoint(checkpoint_path_model, net_frame.state_dict())
    checkpoint_path_model = "{}/unet_{:06d}".format(args.ckpt, epoch)
    save_checkpoint(checkpoint_path_model, net_unet.state_dict())

def create_optimizer(nets, args):
    """オプティマイザー作成"""
    (net_frame, net_unet) = nets
    param_groups = [{'params': net_unet.parameters(), 'lr': args.lr_unet},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.AdamW(param_groups)

def adjust_learning_rate(optimizer, args):
    """学習率調整"""
    args.lr_unet *= 0.95
    args.lr_frame *= 0.95
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.95
        
def decrease_learning_rate(optimizer, decay_factor=0.94):
    """学習率減衰"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def main(args):
    # Network Builders
    builder = models.ModelBuilder()
    net_frame = builder.build_visual(
        pool_type=args.img_pool,
        weights=args.weights_frame,
        arch_frame=args.arch_frame)
    net_unet = builder.build_unet(weights=args.weights_unet)
    nets = (net_frame, net_unet)

    # リアルデータセット用のデータセットとローダー
    detection_csv = os.path.join(args.data_root, "action_detection_results", "detection_results.csv")
    
    dataset_train = RealBinauralDataset(
        csv_file=args.list_train,
        data_root=args.data_root,
        detection_csv=detection_csv,
        sample_rate=22050,  # リアルデータセットは22kHz
        audio_len=args.audLen,
        n_mels=args.num_mels,
        split='train', 
        simple_frames=False
    )
    
    dataset_val = RealBinauralDataset(
        csv_file=args.list_val,
        data_root=args.data_root,
        detection_csv=detection_csv,
        sample_rate=22050,
        audio_len=args.audLen,
        n_mels=args.num_mels,
        split='val',
        simple_frames=False
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)

    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    writer = SummaryWriter(f'{args.ckpt}/runs')
    args.writer = writer

    # 改良されたNetWrapperを使用
    netWrapper = RealBinauralNetWrapper(nets, use_enhanced_loss=False)

    print(f"Using {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    
    # train_pos.pyと同じ順序: move_to_device → DataParallel
    netWrapper.move_to_device(args.device)  # モデルをデバイスに移動
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=args.gpu_ids)  # ラップ
    
    # モデルの最初のパラメータのデバイスを確認
    model_device = next(netWrapper.module.net.parameters()).device
    print(f"The model is on device: {model_device}")
    
    # GPU使用量確認（簡潔版）
    if torch.cuda.is_available():
        print("GPU使用量確認:")
        for i in range(len(args.gpu_ids)):
            gpu_allocated = torch.cuda.memory_allocated(i) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} (物理GPU {args.gpu_ids[i]}): {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved")

    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of performance
    if args.history_path and args.mode=='train':
        history = torch.load(args.history_path)['history']
        last_epoch = history['val']['epoch'][-1]+1
        for epoch in range(1, last_epoch):
            if epoch in args.lr_steps:
                adjust_learning_rate(optimizer, args)
    else:
        history = {
            'train': {'epoch': [], 'err': []},
            'val': {'epoch': [], 'err': [], 'mel_l2': []}}
        last_epoch = 1

    # Eval mode
    if args.mode == 'eval':
        args.testing = True
        evaluate(netWrapper, loader_val, history, 0, args, writer)
        print('Evaluation Done!')
        return
        
    running_loss = 0.
    
    # Training loop
    for epoch in range(last_epoch, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args, writer, running_loss)
        writer.flush()

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            args.testing = True
            evaluate(netWrapper, loader_val, history, epoch, args, writer)
            writer.flush()
            args.testing = False
            
            basic_checkpoint(nets, history, epoch, args)
            
            if epoch % (args.eval_epoch*10)==0:
                advanced_checkpoint(nets, history, epoch, args)

        # 学習率調整
        if(args.learning_rate_decrease_itr > 0 and epoch % args.learning_rate_decrease_itr == 0):
            decrease_learning_rate(optimizer, args.decay_factor)
            print('decreased learning rate by ', args.decay_factor)

    print('Training Done!')

if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    
    # リアルデータセット用の追加設定
    args.data_root = getattr(args, 'data_root', '/home/h-okano/real_dataset')
    
    # GPU設定（nvidia-smiの結果から空いているGPUを使用）
    if hasattr(args, 'gpu_ids') and isinstance(args.gpu_ids, str):
        args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    else:
        # デフォルトでGPU 5, 6を使用（nvidia-smiの結果から比較的空いている）
        args.gpu_ids = [5, 6]
        print(f"デフォルトGPU設定: {args.gpu_ids}")
        

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))  # 設定
    torch.cuda.set_device(args.gpu_ids[0])  # 明示的にデバイスを設定
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # experiment name
    if args.mode == 'train' or args.mode == 'eval':
        args.id += '-realdata'  # リアルデータセット識別子
        args.id += '-22khz'     # サンプリング周波数識別子
        args.id += '-frames{}'.format(args.num_frames)
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])
        args.id += '-lr_unet{}'.format(args.lr_unet)

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    
    if args.mode == 'train':
        if os.path.isdir(args.ckpt):
            frame_path = scan_checkpoint(args.ckpt, 'frame_')
            unet_path = scan_checkpoint(args.ckpt, 'unet_')
            history_path = scan_checkpoint(args.ckpt, 'history_')
            args.history_path = history_path
            if args.history_path:
                args.weights_unet = unet_path
                args.weights_frame = frame_path
                args.history_path = history_path
        else:    
            makedirs(args.ckpt, remove=False)
            args.history_path = None

    elif args.mode == 'eval':
        args.weights_unet = os.path.join(args.ckpt, 'unet_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")
    args.best_mel_l2 = float("inf")
    args.testing = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args) 
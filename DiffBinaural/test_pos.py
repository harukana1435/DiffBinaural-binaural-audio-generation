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
# from scipy.misc import imsave
from imageio import imwrite as imsave
from mir_eval.separation import bss_eval_sources
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Our libs
from utils.arguments import ArgParser
from dataset.genaudio_pos import GenAudioPosDataset
from modules import models
from diffusion_utils import diffusion_pytorch
from utils.helpers import AverageMeter, magnitude2heatmap, \
    istft_reconstruction, warpgrid, makedirs, save_mel_to_tensorboard, _nested_map, save_checkpoint,load_checkpoint,scan_checkpoint
import warnings
# UserWarningとFutureWarningを無視する
#warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


h = None
device = None


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net = nets
        self.sampler = diffusion_pytorch.GaussianDiffusion(
            self.net,
            image_size = 80,
            timesteps = 1000,   # number of steps
            sampling_timesteps = 50, # if ddim else None
            loss_type = 'l1',    # L1 or L2
            objective = 'pred_noise', # pred_noise or pred_x0
            beta_schedule = 'cosine', #linear or cosine or sigmoid 64×64の画像なので、コサインとした
            ddim_sampling_eta = 0,
            auto_normalize = False,
            min_snr_loss_weight=False
        )
        self.scale_factor = 0.167

    def move_to_device(self, device):
            """
            モジュール全体と内包モジュールを指定デバイスに移動する。

            Args:
                device (torch.device): 移動先のデバイス。
            """
            self.to(device)
            if hasattr(self.net_frame, "to"):
                self.net_frame.to(device)
            if hasattr(self.net, "to"):
                self.net.to(device)
            if hasattr(self.sampler, "to"):
                self.sampler.to(device)  # diffusion_pytorch.GaussianDiffusion も移動可能なら移動

            print(f"NetWrapper and its components moved to {device}")


    def forward(self, batch_data, args):
        mix_mel = batch_data['mix_mel'] # (B, C, F, T) C=1, F=64, T=64
        binaural_mel = batch_data['binaural_mel'] # (B, C, F, T) C=1, F=64, T=64
        frames = batch_data['frames'] #(B, C, L, N, H, W) L=5, N=4, C=3, H=224, W=224 
        pos = batch_data['pos_data'] # (B, L, N, 3) 距離、仰角、方位角の順番
        mask = batch_data['mask'] #(B, L, N)
        
        if args.weighted_loss:
            weight = mix_mel
            # weight = torch.clamp(weight, 1e-3, 10)
            weight = weight > 1e-3 # mix_melの値が1e-3以上のものにweightをつけている
        else: # こっち
            weight = torch.ones_like(mix_mel)

        # LOG magnitude
        #mix_mel = min_max_normalize(mix_mel, self.min, self.max)
        #binaural_mel = min_max_normalize(binaural_mel, self.min, self.max) #正規化
        
        mix_mel = torch.log1p(mix_mel) * self.scale_factor #正規化
        binaural_mel = torch.log1p(binaural_mel) * self.scale_factor #正規化

        # detach
        mix_mel = mix_mel.detach()
        binaural_mel = binaural_mel.detach()

        # Frame feature (conditions)
        feat_frames = self.net_frame.forward_multiframe(frames, pos, mask) #(B, C)
        
        # Loss
        loss_mel = self.sampler(binaural_mel, [mix_mel, feat_frames], cfg=False, log=False, weight=weight) #weightは分離音声に対して、一定のスペクトログラムはオフにする

        return loss_mel



    def sample(self, batch_data, args): #サンプルのときは最後に、hifiganに入れられるように正規化しないといけないが、hifigan側でやったほうがいいかも
        model_device = next(self.net_frame.parameters()).device
        batch_data = _nested_map(batch_data, lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x)
        mix_mel = batch_data['mix_mel'] # (B, C, F, T) C=1, F=80, T=80
        binaural_mel = batch_data['binaural_mel'] # (B, C, F, T) C=2, F=80, T=80
        frames = batch_data['frames'] #(B, L, C, H, W) L=4, C=3, H=224, W=224
        pos = batch_data['pos_data'] # (B, L, N, 3) 距離、仰角、方位角の順番
        mask = batch_data['mask'] #(B, L, N)

        # magnitude
        #mix_mel = min_max_normalize(mix_mel, self.min, self.max) #正規化
        mix_mel = torch.log1p(mix_mel) * self.scale_factor #正規化
        
        # detach
        mix_mel = mix_mel.detach()
        binaural_mel = binaural_mel.detach()
        
        # Frame feature (conditions)
        feat_frames = self.net_frame.forward_multiframe(frames, pos, mask) #(B, C)
        
        # ddim sampling
        preds = self.sampler.ddim_sample(condition=[mix_mel, feat_frames], return_all_timesteps = True, silence_mask_sampling=True)

        pred = preds[:, -1, ...]
        
        pred = pred / self.scale_factor
        pred = torch.exp(pred.abs()) - 1

        #pred = invert_min_max_normalize(pred, self.min, self.max)
        
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


    for i, batch_data in enumerate(loader):

        if mel==None:
            t=batch_data['total_time_frame'][0]
            m=args.num_mels
            device=next(netWrapper.module.net.parameters()).device
            mel = torch.zeros((2, m, t)).to(device)
            overlap_count = torch.zeros((2, m, t)).to(device)
        
        # forward pass
        outputs = netWrapper.module.sample(batch_data, args)
        
        B = outputs['pred_mag'].shape[0]

        for j in range(B):
            start_frame = batch_data['start_time_frame'][j]
            mel[:, :,start_frame:start_frame+args.num_mels] += outputs['pred_mag'][j]
            overlap_count[:, :,start_frame:start_frame+args.num_mels] += 1
    
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

    # Wrap networks for multiple GPUs
    netWrapper = NetWrapper(nets)
    
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
            dataset_audio =  GenAudioPosDataset(filename, args)
            loader_audio = torch.utils.data.DataLoader(dataset_audio,batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers),drop_last=False)
            
            mel = generate(netWrapper, loader_audio, args)
            print(mel.shape)
            mel = mel.cpu()
            
            left_mel = mel[0].squeeze(0)
            right_mel = mel[1].squeeze(0)

            output_file_left = os.path.join(args.output_dir_left, os.path.splitext(os.path.split(filename)[1])[0] + '.npy')
            output_file_right = os.path.join(args.output_dir_right, os.path.splitext(os.path.split(filename)[1])[0] + '.npy')
            torch.save(left_mel, output_file_left)
            torch.save(right_mel, output_file_right)
            print(output_file_right)


if __name__ == '__main__':
    print('Initializing Inference Process..')

    parser = ArgParser()
    args = parser.parse_test_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))  # 設定
    torch.cuda.set_device(args.gpu_ids[0])  # 明示的にデバイスを設定
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.weights_frame = os.path.join(args.ckpt, "frame_001000")
    args.weights_unet = os.path.join(args.ckpt, "unet_001000")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)


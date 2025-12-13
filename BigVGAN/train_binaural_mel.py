# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Binaural mel-spectrogram training script for BigVGAN (Stage 2)
# This script trains BigVGAN using pre-generated mel spectrograms
# Based on the original BigVGAN training implementation

import warnings

import librosa
warnings.simplefilter(action="ignore", category=FutureWarning)

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import glob
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print('TensorBoard not available, continuing without logging')
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import mel_spectrogram, MAX_WAV_VALUE

from bigvgan import BigVGAN
from discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)

from utils import (
    plot_spectrogram,
    plot_spectrogram_clipped,
    plot_spectrogram_fixed_scale,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_audio,
)
import torchaudio as ta
from tqdm import tqdm

torch.backends.cudnn.benchmark = False


def detect_silence_regions(mel_spec, threshold_db=-60, min_silence_frames=5):
    """
    Detect silence regions in mel spectrogram
    Args:
        mel_spec: mel spectrogram [B, n_mels, T]
        threshold_db: silence threshold in dB
        min_silence_frames: minimum consecutive frames to consider as silence
    Returns:
        silence_mask: binary mask [B, 1, T] where 1 = silence, 0 = active
    """
    # Convert to dB scale
    mel_db = 20 * torch.log10(torch.clamp(mel_spec, min=1e-8))
    
    # Calculate energy per frame (sum across mel bins)
    energy_per_frame = torch.mean(mel_db, dim=1, keepdim=True)  # [B, 1, T]
    
    # Create silence mask based on threshold
    silence_mask = (energy_per_frame < threshold_db).float()
    
    # Apply minimum silence length filter
    if min_silence_frames > 1:
        # Use 1D convolution to enforce minimum silence duration
        kernel = torch.ones(1, 1, min_silence_frames, device=silence_mask.device)
        silence_conv = F.conv1d(silence_mask, kernel, padding=min_silence_frames//2)
        silence_mask = (silence_conv >= min_silence_frames).float()
    
    return silence_mask


def silence_aware_loss(y_mel, y_g_hat_mel, y, y_g_hat, silence_threshold_db=-60):
    """
    Silence-aware loss that applies different penalties for silent and active regions
    """
    # Detect silence regions in ground truth
    silence_mask = detect_silence_regions(y_mel, threshold_db=silence_threshold_db)
    active_mask = 1.0 - silence_mask
    
    # Mel spectral loss with region-specific weighting
    mel_loss_base = F.l1_loss(y_mel, y_g_hat_mel, reduction='none')  # [B, n_mels, T]
    
    # Higher penalty for noise in silent regions
    silence_penalty = 3.0
    active_penalty = 1.0
    
    weighted_mel_loss = (
        mel_loss_base * silence_mask * silence_penalty +
        mel_loss_base * active_mask * active_penalty
    )
    
    mel_loss = torch.mean(weighted_mel_loss)
    
    # Waveform silence loss - penalize high energy in predicted silent regions
    if y is not None and y_g_hat is not None:
        # Convert silence mask to waveform domain
        hop_size = y.shape[-1] // y_mel.shape[-1]
        silence_mask_wave = F.interpolate(
            silence_mask, 
            size=y.shape[-1], 
            mode='nearest'
        )  # [B, 1, T_wave]
        
        # Calculate energy in predicted waveform during silent regions
        y_g_hat_energy = y_g_hat ** 2
        silence_energy_loss = torch.mean(y_g_hat_energy * silence_mask_wave)
        
        return mel_loss, silence_energy_loss * 10.0
    
    return mel_loss, torch.tensor(0.0, device=y_mel.device)


def spectral_consistency_loss(y_g_hat_mel, low_freq_weight=2.0, high_freq_weight=0.5):
    """
    Encourage spectral consistency to reduce artifacts
    """
    # Temporal smoothness loss
    temporal_diff = torch.diff(y_g_hat_mel, dim=-1)
    temporal_smoothness = torch.mean(torch.abs(temporal_diff))
    
    # Frequency smoothness loss (reduce harsh frequency discontinuities)
    freq_diff = torch.diff(y_g_hat_mel, dim=-2)
    freq_smoothness = torch.mean(torch.abs(freq_diff))
    
    # Frequency-weighted loss (prioritize low frequencies)
    n_mels = y_g_hat_mel.shape[-2]
    freq_weights = torch.linspace(low_freq_weight, high_freq_weight, n_mels, device=y_g_hat_mel.device)
    freq_weights = freq_weights.view(1, -1, 1)  # [1, n_mels, 1]
    
    weighted_spec_loss = torch.mean(torch.abs(y_g_hat_mel) * freq_weights)
    
    return temporal_smoothness * 0.1 + freq_smoothness * 0.05


def energy_regularization_loss(y_mel, y_g_hat_mel, y_g_hat):
    """
    Regularize energy distribution to match ground truth characteristics
    """
    # Total energy conservation
    gt_energy = torch.sum(y_mel, dim=(1, 2))  # [B]
    pred_energy = torch.sum(y_g_hat_mel, dim=(1, 2))  # [B]
    energy_conservation_loss = F.l1_loss(pred_energy, gt_energy)
    
    # Dynamic range preservation
    gt_max = torch.max(y_mel.view(y_mel.shape[0], -1), dim=1)[0]  # [B]
    gt_min = torch.min(y_mel.view(y_mel.shape[0], -1), dim=1)[0]  # [B]
    gt_dynamic_range = gt_max - gt_min
    
    pred_max = torch.max(y_g_hat_mel.view(y_g_hat_mel.shape[0], -1), dim=1)[0]  # [B]
    pred_min = torch.min(y_g_hat_mel.view(y_g_hat_mel.shape[0], -1), dim=1)[0]  # [B]
    pred_dynamic_range = pred_max - pred_min
    
    dynamic_range_loss = F.l1_loss(pred_dynamic_range, gt_dynamic_range)
    
    # Waveform RMS matching for overall level control
    if y_g_hat is not None:
        # Calculate RMS over time dimension
        gt_rms = torch.sqrt(torch.mean(torch.sum(y_mel**2, dim=1), dim=1))  # [B]
        pred_rms = torch.sqrt(torch.mean(y_g_hat**2, dim=(1, 2)))  # [B]
        rms_loss = F.l1_loss(pred_rms, gt_rms)
    else:
        rms_loss = torch.tensor(0.0, device=y_mel.device)
    
    return energy_conservation_loss * 0.1 + dynamic_range_loss * 0.1 + rms_loss * 0.05


def adaptive_loss_weighting(y_mel, current_step, total_steps):
    """
    Adaptive loss weighting that changes focus during training
    """
    # Early training: focus on basic reconstruction
    # Later training: focus more on silence and fine details
    
    progress = min(current_step / total_steps, 1.0)
    
    # Silence loss weight increases during training
    silence_weight = 0.5 + 1.5 * progress  # 0.5 -> 2.0
    
    # Spectral consistency weight increases later
    spectral_weight = 0.1 + 0.4 * progress  # 0.1 -> 0.5
    
    # Energy regularization weight stays moderate
    energy_weight = 0.3 + 0.2 * progress  # 0.3 -> 0.5
    
    return silence_weight, spectral_weight, energy_weight


class BinauralMelSpectrogramDataset(Dataset):
    """
    Dataset for binaural audio training using pre-generated mel spectrograms.
    Similar to BinauralSingleChannelMelDataset, treats left/right channels as separate samples.
    This doubles the dataset size: even indices = left channel, odd indices = right channel.
    """
    
    def __init__(self, mel_left_dir, mel_right_dir, audio_dir, segment_size, 
                 n_fft, num_mels, hop_size, win_size, sampling_rate, fmin, fmax, 
                 split=True, shuffle=True, device=None, fmax_loss=None):
        
        self.mel_left_dir = mel_left_dir
        self.mel_right_dir = mel_right_dir
        self.audio_dir = audio_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device
        
        # Get all mel spectrogram files from left directory
        self.mel_files_left = sorted(glob.glob(os.path.join(mel_left_dir, "*.npy")))
        
        # Find matching pairs and corresponding audio files
        self.valid_files = []
        for mel_left in self.mel_files_left:
            basename = os.path.basename(mel_left)
            mel_right = os.path.join(mel_right_dir, basename)
            if os.path.exists(mel_right):
                # Extract the base filename to find corresponding audio
                audio_basename = basename.replace('.npy', '')
                # Look for various audio extensions
                audio_extensions = ['.wav', '.mp3', '.flac']
                audio_path = None
                for ext in audio_extensions:
                    potential_path = os.path.join(audio_dir, audio_basename + ext)
                    if os.path.exists(potential_path):
                        audio_path = potential_path
                        break
                
                if audio_path:
                    self.valid_files.append((mel_left, mel_right, audio_path))
        
        print(f"Found {len(self.valid_files)} matching mel-audio files")
        print(f"Dataset size: {len(self.valid_files) * 2} samples (left + right channels)")
        
        if len(self.valid_files) == 0:
            raise ValueError(f"No matching mel-audio pairs found in {mel_left_dir}, {mel_right_dir}, {audio_dir}")
    
    def __len__(self):
        # Double the dataset size (left + right channels)
        return len(self.valid_files) * 2
    
    def __getitem__(self, index):
        try:
            # Determine which file and which channel (L/R)
            file_index = index // 2
            channel_index = index % 2  # 0 = left, 1 = right
            mel_left_path, mel_right_path, audio_path = self.valid_files[file_index]
            
            # Choose the appropriate mel spectrogram based on channel
            if channel_index == 0:  # Left channel
                mel_path = mel_left_path
                channel_name = "L"
            else:  # Right channel
                mel_path = mel_right_path
                channel_name = "R"
            
            # Load mel spectrogram for the selected channel
            mel = np.load(mel_path)
            
            # Remove first and last 8 frames to avoid noise artifacts
            frames_trimmed = 0
            if mel.shape[1] > 16:  # Ensure we have enough frames to trim
                mel = mel[:, 8:-8]
                frames_trimmed = 8  # Record how many frames we trimmed from each side
            
            # Convert to tensor
            mel = torch.from_numpy(mel).float()
            
            # Load corresponding ground truth audio
            try:
                audio_stereo, source_sampling_rate = librosa.load(audio_path, sr=None, mono=False)
                
                # Handle different audio formats (similar to BinauralSingleChannelMelDataset)
                if audio_stereo.ndim == 1:
                    # Mono file - duplicate to stereo
                    audio_stereo = np.stack([audio_stereo, audio_stereo])
                elif audio_stereo.ndim == 2:
                    if audio_stereo.shape[0] == 2:
                        # Already in (2, T) format
                        pass
                    elif audio_stereo.shape[1] == 2:
                        # Transpose from (T, 2) to (2, T)
                        audio_stereo = audio_stereo.T
                    else:
                        # Multi-channel: take first 2 channels or duplicate if only 1
                        if audio_stereo.shape[0] == 1:
                            audio_stereo = np.tile(audio_stereo, (2, 1))
                        else:
                            audio_stereo = audio_stereo[:2, :]
                
                # Extract the desired channel
                audio = audio_stereo[channel_index]  # Shape: (T,)
                
                # Trim audio to match the trimmed mel spectrogram
                if frames_trimmed > 0:
                    audio_trim_samples = frames_trimmed * self.hop_size
                    if audio.shape[0] > 2 * audio_trim_samples:  # Ensure we have enough audio to trim
                        audio = audio[audio_trim_samples:-audio_trim_samples]
                
                # Resample if necessary
                if source_sampling_rate != self.sampling_rate:
                    audio = librosa.resample(
                        audio,
                        orig_sr=source_sampling_rate,
                        target_sr=self.sampling_rate,
                    )
                
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                # Create silent audio as fallback
                audio_length = mel.shape[1] * self.hop_size
                audio = np.zeros(audio_length)
            
            # Calculate the expected audio length based on mel spectrogram length
            expected_audio_length = mel.shape[1] * self.hop_size
            
            # Adjust audio length to match mel spectrogram
            if audio.shape[0] > expected_audio_length:
                # Trim audio if too long
                audio = audio[:expected_audio_length]
            elif audio.shape[0] < expected_audio_length:
                # Pad audio if too short
                pad_size = expected_audio_length - audio.shape[0]
                audio = np.pad(audio, (0, pad_size), mode='constant')
            
            # Random segment selection for training
            if self.split and mel.shape[1] * self.hop_size >= self.segment_size:
                max_mel_start = mel.shape[1] - (self.segment_size // self.hop_size)
                mel_start = torch.randint(0, max_mel_start + 1, (1,)).item()
                mel_end = mel_start + (self.segment_size // self.hop_size)
                
                # Extract mel segment
                mel = mel[:, mel_start:mel_end]
                
                # Extract corresponding audio segment
                audio_start = mel_start * self.hop_size
                audio_end = audio_start + self.segment_size
                audio = audio[audio_start:audio_end]
            else:
                # Use full mel spectrogram and corresponding audio
                # Pad mel spectrogram and audio if too short
                min_mel_frames = self.segment_size // self.hop_size
                if mel.shape[1] < min_mel_frames:
                    pad_frames = min_mel_frames - mel.shape[1]
                    mel = F.pad(mel, (0, pad_frames))
                
                if len(audio) < self.segment_size:
                    pad_size = self.segment_size - len(audio)
                    audio = np.pad(audio, (0, pad_size))
            
            # Normalize audio (similar to BinauralSingleChannelMelDataset)
            audio = librosa.util.normalize(audio) * 0.95
            
            # Convert audio to tensor
            audio = torch.from_numpy(audio).float()
            audio = audio.unsqueeze(0)  # [1, T]
            
            # Compute mel_loss used by spectral regression objective
            mel_loss = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax_loss,
                center=False,
            )
            
            # Add channel identifier to filename
            filename_with_channel = f"{os.path.basename(audio_path)}_{channel_name}"
            
            return (mel.squeeze(), audio.squeeze(0), filename_with_channel, mel_loss.squeeze())
            
        except Exception as e:
            print(f"[WARNING] Failed to load mel-audio pair, skipping! file_index: {file_index} channel: {channel_index} Error: {e}")
            # Load a random other sample to avoid breaking the batch
            return self[torch.randint(0, len(self), (1,)).item()]





def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(
            backend=h.dist_config["dist_backend"],
            init_method=h.dist_config["dist_url"],
            world_size=h.dist_config["world_size"] * h.num_gpus,
            rank=rank,
        )

    # Set seed and device
    torch.cuda.manual_seed(h.seed)
    if h.num_gpus > 1:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank:d}")
    else:
        device = torch.device('cuda:0')

    # Initialize single-channel BigVGAN generator
    generator = BigVGAN(h).to(device)
    
    # Load pretrained weights if specified
    if a.pretrained_bigvgan:
        print(f"Loading pretrained BigVGAN weights from {a.pretrained_bigvgan}")
        state_dict = load_checkpoint(a.pretrained_bigvgan, device)
        generator.load_state_dict(state_dict['generator'])
        print("Loaded pretrained generator weights")

    # Initialize discriminators - MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)
    
    # Define additional discriminators. BigVGAN-v1 uses UnivNet's MRD as default
    # New in BigVGAN-v2: option to switch to new discriminators
    if h.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print("[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator")
        # Variable name is kept as "mrd" for backward compatibility & minimal code change
        mrd = MultiBandDiscriminator(h).to(device)
    elif h.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print("[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator")
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:  # Fallback to original MRD in BigVGAN-v1
        mrd = MultiResolutionDiscriminator(h).to(device)

    # New in BigVGAN-v2: option to switch to multi-scale L1 mel loss
    if h.get("use_multiscale_melloss", False):
        print("[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss")
        fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=h.sampling_rate
        ).to(device)  # NOTE: accepts waveform as input
    else:
        fn_mel_loss_singlescale = F.l1_loss

    # Print the model & number of parameters, and create checkpoints directory
    if rank == 0:
        print(generator)
        print(mpd)
        print(mrd)
        print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters())}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters())}")
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("Stage 2 checkpoints directory : ", a.checkpoint_path)

    # Initialize checkpoint variables
    steps = 0
    state_dict_do = None
    last_epoch = -1

    # First, try to load from stage 2 checkpoint directory (for resuming stage 2 training)
    stage2_loaded = False
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        
        if cp_g is not None and cp_do is not None:
            if rank == 0:
                print(f"Found stage 2 checkpoints, resuming from: {a.checkpoint_path}")
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            generator.load_state_dict(state_dict_g['generator'])
            mpd.load_state_dict(state_dict_do['mpd'])
            mrd.load_state_dict(state_dict_do['mrd'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']
            stage2_loaded = True
            if rank == 0:
                print(f"Resumed from step {steps}, epoch {last_epoch + 1}")

    # If no stage 2 checkpoint and load_stage1_checkpoint is True, load from stage 1
    if not stage2_loaded and a.load_stage1_checkpoint and os.path.isdir(a.stage1_checkpoint_path):
        if rank == 0:
            print(f"Loading stage 1 checkpoint from: {a.stage1_checkpoint_path}")
        cp_g_stage1 = scan_checkpoint(a.stage1_checkpoint_path, 'g_')
        cp_do_stage1 = scan_checkpoint(a.stage1_checkpoint_path, 'do_')
        
        if cp_g_stage1 is not None and cp_do_stage1 is not None:
            if rank == 0:
                print(f"Found stage 1 checkpoints: {cp_g_stage1}, {cp_do_stage1}")
            state_dict_g_stage1 = load_checkpoint(cp_g_stage1, device)
            state_dict_do_stage1 = load_checkpoint(cp_do_stage1, device)
            
            # Load only the model weights, reset training state for stage 2
            generator.load_state_dict(state_dict_g_stage1['generator'])
            mpd.load_state_dict(state_dict_do_stage1['mpd'])
            
            # Handle different discriminator naming between stage 1 and stage 2
            if 'mrd' in state_dict_do_stage1:
                mrd.load_state_dict(state_dict_do_stage1['mrd'])
            elif 'msd' in state_dict_do_stage1:
                mrd.load_state_dict(state_dict_do_stage1['msd'])
                if rank == 0:
                    print("Loaded MSD weights from stage 1 into MRD for stage 2")
            else:
                if rank == 0:
                    print("Warning: No compatible discriminator found in stage 1 checkpoint")
            
            if rank == 0:
                print("Loaded stage 1 model weights, starting stage 2 training from step 0")
            # Keep steps=0 and last_epoch=-1 for fresh start of stage 2
        else:
            if rank == 0:
                print(f"No valid stage 1 checkpoints found in {a.stage1_checkpoint_path}")
    
    if not stage2_loaded and not a.load_stage1_checkpoint:
        if rank == 0:
            print("Starting training from scratch (no checkpoint loading)")

    # Initialize DDP, optimizers, and schedulers
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    # Optimizers
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()),
                               h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    # Only load optimizer states if we're resuming stage 2 training (not starting from stage 1)
    if state_dict_do is not None and stage2_loaded:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        if rank == 0:
            print("Loaded optimizer states from stage 2 checkpoint")
    else:
        if rank == 0:
            print("Using fresh optimizer states for stage 2 training")

    # Learning rate scheduler
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # Dataset
    if rank == 0:
        print(f"Loading training data from:")
        print(f"  Left mel dir: {a.mel_left_train_dir}")
        print(f"  Right mel dir: {a.mel_right_train_dir}")
        print(f"  Audio dir: {a.audio_dir}")
    
    trainset = BinauralMelSpectrogramDataset(
        a.mel_left_train_dir, a.mel_right_train_dir, a.audio_dir,
        h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, 
        h.sampling_rate, h.fmin, h.fmax, split=True, shuffle=False if h.num_gpus > 1 else True,
        device=device, fmax_loss=h.fmax_for_loss
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                             sampler=train_sampler, batch_size=h.batch_size, pin_memory=True,
                             drop_last=True)

    if rank == 0:
        # Calculate dynamic intervals for better logging
        steps_per_epoch = len(train_loader)
        epoch_summary_interval = max(1, steps_per_epoch // 5)  # 5 times per epoch
        epoch_validation_interval = steps_per_epoch  # Once per epoch
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"TensorBoard logging every {epoch_summary_interval} steps")
        print(f"Validation every {epoch_validation_interval} steps")

        # Validation dataset (use same directories for now, or create validation split)
        validset = BinauralMelSpectrogramDataset(
            a.mel_left_val_dir if hasattr(a, 'mel_left_val_dir') and a.mel_left_val_dir else a.mel_left_train_dir,
            a.mel_right_val_dir if hasattr(a, 'mel_right_val_dir') and a.mel_right_val_dir else a.mel_right_train_dir,
            a.audio_dir, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
            h.sampling_rate, h.fmin, h.fmax, split=False, shuffle=False,
            device=device, fmax_loss=h.fmax_for_loss
        )
        
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                      sampler=None, batch_size=1, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs")) if TENSORBOARD_AVAILABLE else None
        
        if sw is not None:
            print(f"TensorBoard logs will be saved to: {os.path.join(a.checkpoint_path, 'logs')}")
            print("To view logs, run: tensorboard --logdir {}".format(os.path.join(a.checkpoint_path, "logs")))
        else:
            print("TensorBoard not available - logging disabled")

    """
    Validation loop
    """
    def validate(rank, a, h, loader):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        torch.cuda.empty_cache()
        val_err_tot = 0
        
        with torch.no_grad():
            for j, batch in enumerate(loader):
                if j >= 10:  # Limit validation samples
                    break
                    
                x, y, _, y_mel = batch
                x = x.to(device)
                y = y.to(device)
                y_mel = y_mel.to(device)
                y = y.unsqueeze(1)
                
                if hasattr(generator, "module"):
                    y_g_hat = generator.module(x)
                else:
                    y_g_hat = generator(x)
                
                # Generate mel spectrogram from generated audio
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_for_loss,
                )

                min_t = min(y_mel.size(-1), y_g_hat_mel.size(-1))
                val_err_tot += F.l1_loss(y_mel[...,:min_t], y_g_hat_mel[...,:min_t]).item()

                if j <= 4:
                    if sw is not None:
                        sw.add_audio('gt/y_{}'.format(j), y[0, 0], steps, h.sampling_rate)
                    if sw is not None:
                        # 正解メルスペクトログラム (y_mel) を表示 - 固定スケール使用 (-10 to 2)
                        sw.add_figure('gt/y_mel_spec_{}'.format(j), plot_spectrogram_fixed_scale(y_mel[0].cpu().numpy(), vmin=-10.0, vmax=2.0), steps)
                    if sw is not None:
                        # 入力メル（拡散モデル生成）も比較用に表示 - 固定スケール使用 (-10 to 2)
                        sw.add_figure('input/x_diffusion_mel_{}'.format(j), plot_spectrogram_fixed_scale(x[0].cpu().numpy(), vmin=-10.0, vmax=2.0), steps)

                    if sw is not None:
                        sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0, 0], steps, h.sampling_rate)
                    if sw is not None:
                        sw.add_figure('generated/y_hat_spec_{}'.format(j), plot_spectrogram_fixed_scale(y_g_hat_mel[0].cpu().numpy(), vmin=-10.0, vmax=2.0), steps)

            val_err_tot = val_err_tot / (j + 1)
            
            if sw is not None:
                sw.add_scalar("validation/mel_spec_error", val_err_tot, steps)

        generator.train()

    # If the checkpoint is loaded, start with validation loop
    if steps != 0 and rank == 0 and not a.debug:
        validate(rank, a, h, validation_loader)

    # Exit the script if --evaluate is set to True
    if a.evaluate:
        exit()

    generator.train()
    mpd.train()
    mrd.train()

    # Main training loop
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)  # Add channel dimension

            # Generate audio from mel spectrograms
            y_g_hat = generator(x)

            # Generate mel spectrogram from output for loss calculation
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax_for_loss,
            )

            optim_d.zero_grad()

            # Discriminator forward - MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # Discriminator forward - MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            # Set clip_grad_norm value
            clip_grad_norm = h.get("clip_grad_norm", 1000.0)  # Default to 1000

            # Whether to freeze D for initial training steps
            if steps >= a.freeze_step:
                loss_disc_all.backward()
                grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
                    mpd.parameters(), clip_grad_norm
                )
                grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
                    mrd.parameters(), clip_grad_norm
                )
                optim_d.step()
            else:
                if rank == 0:
                    print(f"[WARNING] skipping D training for the first {a.freeze_step} steps")
                grad_norm_mpd = 0.0
                grad_norm_mrd = 0.0

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss (increased weight for better reconstruction)
            lambda_melloss = h.get("lambda_melloss", 60.0)  # Increased from 45 to 60 for better mel reconstruction
            if h.get("use_multiscale_melloss", False):
                loss_mel = fn_mel_loss_multiscale(y, y_g_hat) * lambda_melloss
            else:
                # シンプルな静寂認識損失を追加
                loss_mel_base = fn_mel_loss_singlescale(y_mel, y_g_hat_mel) * lambda_melloss
                loss_mel_silence = simple_silence_aware_mel_loss(y_mel, y_g_hat_mel, 
                                                               silence_threshold_db=-50, 
                                                               silence_penalty=2.0) * (lambda_melloss * 0.3)
                loss_mel = loss_mel_base + loss_mel_silence

            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            # MRD loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            if steps >= a.freeze_step:
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )
            else:
                if rank == 0:
                    print(f"[WARNING] using regression loss only for G for the first {a.freeze_step} steps")
                loss_gen_all = loss_mel

            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), clip_grad_norm
            )
            optim_g.step()

            if rank == 0:
                # Calculate mel error for display
                with torch.no_grad():
                    mel_error = (loss_mel.item() / lambda_melloss)

                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print(
                        f"Steps: {steps:d}, "
                        f"Gen Loss Total: {loss_gen_all:4.3f}, "
                        f"Mel Error: {mel_error:4.3f}, "
                        f"s/b: {time.time() - start_b:4.3f} "
                        f"lr: {optim_g.param_groups[0]['lr']:4.7f} "
                        f"grad_norm_g: {grad_norm_g:4.3f}"
                    )

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                  {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                  {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                   'mrd': (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                                   'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                   'steps': steps, 'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    if sw is not None:
                        sw.add_scalar("training/gen_loss_total", loss_gen_all.item(), steps)
                        sw.add_scalar("training/mel_spec_error", mel_error, steps)
                        sw.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                        sw.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                        sw.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                        sw.add_scalar("training/grad_norm_mpd", grad_norm_mpd, steps)
                        sw.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                        sw.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                        sw.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                        sw.add_scalar("training/grad_norm_mrd", grad_norm_mrd, steps)
                        sw.add_scalar("training/grad_norm_g", grad_norm_g, steps)
                        sw.add_scalar("training/learning_rate_d", scheduler_d.get_last_lr()[0], steps)
                        sw.add_scalar("training/learning_rate_g", scheduler_g.get_last_lr()[0], steps)
                        sw.add_scalar("training/epoch", epoch + 1, steps)

                # Validation
                if steps % a.validation_interval == 0:
                    # Plot training input x so far used
                    for i_x in range(x.shape[0]):
                        if sw is not None:
                            sw.add_figure(
                                f"training_input/x_{i_x}",
                                plot_spectrogram_fixed_scale(x[i_x].cpu(), vmin=-10.0, vmax=2.0),
                                steps,
                            )
                            sw.add_audio(
                                f"training_input/y_{i_x}",
                                y[i_x][0],
                                steps,
                                h.sampling_rate,
                            )

                    # Validation loop
                    if not a.debug and steps != 0:
                        validate(rank, a, h, validation_loader)

            steps += 1

            # BigVGAN-v2 learning rate scheduler is changed from epoch-level to step-level
            scheduler_g.step()
            scheduler_d.step()

        if rank == 0:
            print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")


def main():
    print('Initializing Training Process for Binaural BigVGAN with Pre-generated Mel Spectrograms..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--mel_left_train_dir', default='/home/h-okano/DiffBinaural/test_results/realBinaural_left_bigvgan_train')
    parser.add_argument('--mel_right_train_dir', default='/home/h-okano/DiffBinaural/test_results/realBinaural_right_bigvgan_train')
    parser.add_argument('--mel_left_val_dir', default='/home/h-okano/DiffBinaural/test_results/realBinaural_left_bigvgan_val')
    parser.add_argument('--mel_right_val_dir', default='/home/h-okano/DiffBinaural/test_results/realBinaural_right_bigvgan_val')
    parser.add_argument('--audio_dir', default='/home/h-okano/real_dataset/processed/binaural_audios_22050Hz')
    parser.add_argument('--checkpoint_path', default='cp_bigvgan_binaural_mel_stage2_5', help='Path to save stage 2 checkpoints')
    parser.add_argument('--stage1_checkpoint_path', default='cp_bigvgan_binaural_1ch_dummy', help='Path to load stage 1 checkpoints from')
    parser.add_argument('--config', default='/home/h-okano/bigvgan/configs/bigvgan_binaural_22khz_80band_256x.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=50, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=100, type=int)
    parser.add_argument('--freeze_step', default=0, type=int, help='freeze D for the first specified steps. G only uses regression loss for these steps.')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--pretrained_bigvgan', help='Path to pretrained BigVGAN checkpoint for transfer learning')
    parser.add_argument('--load_stage1_checkpoint', default=True, type=bool, help='Whether to load stage 1 checkpoint as starting point')
    parser.add_argument('--debug', default=False, type=bool, help='debug mode. skips validation loop throughout training')
    parser.add_argument('--evaluate', default=False, type=bool, help='only run evaluation from checkpoint and exit')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f"Batch size per GPU: {h.batch_size}")
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=h.num_gpus,
            args=(
                a,
                h,
            ),
        )
    else:
        train(0, a, h)


if __name__ == '__main__':
    main() 
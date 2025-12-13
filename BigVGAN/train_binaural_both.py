# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Combined Binaural BigVGAN Training Script
# This script supports training from both audio files and pre-computed mel spectrograms
# Based on train.py and train_binaural_mel.py

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader, Dataset
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, MAX_WAV_VALUE
import librosa
import math
import numpy as np
import random
import pathlib
from tqdm import tqdm
import glob

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
from pesq import pesq
import auraloss

torch.backends.cudnn.benchmark = False


def simple_silence_aware_mel_loss(y_mel, y_g_hat_mel, silence_threshold_db=-50, silence_penalty=2.0):
    """
    Simple silence-aware mel loss that penalizes noise in silent regions
    """
    # Convert to dB scale
    y_mel_db = 20 * torch.log10(torch.clamp(y_mel, min=1e-8))
    
    # Calculate energy per frame (mean across mel bins)
    energy_per_frame = torch.mean(y_mel_db, dim=1, keepdim=True)  # [B, 1, T]
    
    # Create silence mask
    silence_mask = (energy_per_frame < silence_threshold_db).float()
    active_mask = 1.0 - silence_mask
    
    # Mel spectral loss with region-specific weighting
    mel_loss_base = F.l1_loss(y_mel, y_g_hat_mel, reduction='none')  # [B, n_mels, T]
    
    # Apply different penalties
    weighted_mel_loss = (
        mel_loss_base * silence_mask * silence_penalty +
        mel_loss_base * active_mask * 1.0
    )
    
    return torch.mean(weighted_mel_loss)


class BinauralCombinedDataset(torch.utils.data.Dataset):
    """
    Combined dataset for binaural audio training that supports both:
    1. Audio files (with real-time mel computation like BinauralSingleChannelMelDataset)
    2. Pre-computed mel spectrograms (like BinauralMelSpectrogramDataset)
    
    This version supports mel condition switching for robustness training:
    - GT mel: extracted from ground truth audio using BigVGAN's mel function
    - Pred mel: pre-computed diffusion model outputs (cached offline)
    - Probability-based switching during training for robustness
    """
    
    def __init__(
        self,
        # Audio file parameters (from BinauralSingleChannelMelDataset)
        training_files=None,
        hparams=None,
        # Mel spectrogram parameters (from BinauralMelSpectrogramDataset)
        mel_left_dir=None,
        mel_right_dir=None,
        audio_dir=None,
        # Predicted mel directories (diffusion model outputs)
        mel_pred_left_dir=None,
        mel_pred_right_dir=None,
        # Common parameters
        segment_size=None,
        n_fft=None,
        num_mels=None,
        hop_size=None,
        win_size=None,
        sampling_rate=None,
        fmin=None,
        fmax=None,
        split=True,
        shuffle=True,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True,
        # New parameter to control data source priority
        prefer_precomputed_mels=True,  # If True, prioritize pre-computed mels over audio files
        # Robustness training parameters
        current_epoch=0,
        use_pred_mel_schedule=True,  # Enable curriculum learning schedule
        recent_losses=None, # 最近の損失を記録
        loss_threshold=None, # 損失閾値
    ):
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = self.sampling_rate // 2
        self.fmax_loss = self.sampling_rate // 2
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.is_seen = is_seen
        self.hparams = hparams
        self.prefer_precomputed_mels = prefer_precomputed_mels
        
        # Robustness training parameters
        self.current_epoch = current_epoch
        self.use_pred_mel_schedule = use_pred_mel_schedule
        self.mel_pred_left_dir = mel_pred_left_dir
        self.mel_pred_right_dir = mel_pred_right_dir
        
        # Cache resampler instances for better performance
        self._resamplers = {}  # Dictionary to cache resamplers
        
        self.audio_files = []
        self.mel_files = []
        self.mel_pred_files = []  # Predicted mel files (diffusion outputs)
        self.data_sources = []  # Track which source each sample comes from
        
        # Initialize audio file data source
        if training_files is not None and len(training_files) > 0:
            self.audio_files = training_files
            random.seed(1234)
            if shuffle:
                random.shuffle(self.audio_files)
            
            print(f"[INFO] Found {len(self.audio_files)} audio files for real-time mel computation")
            # Verify audio files exist
            print("[INFO] Checking audio dataset integrity...")
            for i in tqdm(range(len(self.audio_files))):
                assert os.path.exists(self.audio_files[i]), f"{self.audio_files[i]} not found"
        
        # GT mel computation: No pre-saved GT mels, compute on-the-fly from audio
        # mel_left_dir and mel_right_dir are set to None to indicate GT mels are computed dynamically
        self.mel_files = []  # No pre-saved GT mel files
        print(f"[INFO] GT mels will be computed on-the-fly from audio files (no pre-saved GT mels)")
        
        # Initialize predicted mel data source (diffusion model outputs)
        if mel_pred_left_dir is not None and mel_pred_right_dir is not None and audio_dir is not None:
            mel_pred_files_left = sorted(glob.glob(os.path.join(mel_pred_left_dir, "*.npy")))
            valid_mel_pred_files = []
            
            for mel_pred_left in mel_pred_files_left:
                basename = os.path.basename(mel_pred_left)
                mel_pred_right = os.path.join(mel_pred_right_dir, basename)
                if os.path.exists(mel_pred_right):
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
                        valid_mel_pred_files.append((mel_pred_left, mel_pred_right, audio_path))
            
            self.mel_pred_files = valid_mel_pred_files
            print(f"[INFO] Found {len(self.mel_pred_files)} pre-computed predicted mel-audio file pairs")
        
        # Combine data sources
        self._build_combined_dataset()
        
        if self.is_seen and len(self.combined_files) > 0:
            if hasattr(self, 'audio_files') and len(self.audio_files) > 0:
                self.name = pathlib.Path(self.audio_files[0]).parts[0] if self.audio_files else "binaural"
            elif hasattr(self, 'mel_files') and len(self.mel_files) > 0:
                self.name = pathlib.Path(self.mel_files[0][2]).parts[0] if self.mel_files else "binaural"
            else:
                self.name = "binaural"
        else:
            self.name = "binaural"
        
        # 最近の損失を記録
        self.recent_losses = recent_losses if recent_losses is not None else []
        # 損失閾値
        self.loss_threshold = loss_threshold if loss_threshold is not None else 0.1
    
    def update_loss_history(self, loss_value):
        """損失履歴を更新"""
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 10:  # 直近10回の平均
            self.recent_losses.pop(0)
    
    def _build_combined_dataset(self):
        """Build combined dataset from available sources"""
        self.combined_files = []
        self.data_sources = []
        
        # Priority: Use predicted mels if available (with audio for GT), otherwise use audio files only
        if len(self.mel_pred_files) > 0:
            # Add predicted mel files with curriculum learning capability (doubles the size due to L/R channels)
            for mel_pred_left, mel_pred_right, audio_path in self.mel_pred_files:
                # Each mel file set contributes 2 samples (L and R channels)
                self.combined_files.append(('mel_pred', (mel_pred_left, mel_pred_right, audio_path), 0))  # Left channel
                self.combined_files.append(('mel_pred', (mel_pred_left, mel_pred_right, audio_path), 1))  # Right channel
                self.data_sources.extend(['mel_pred', 'mel_pred'])
            print(f"[INFO] Using predicted mels + on-the-fly GT mels (curriculum learning)")
        else:
            # Add audio files only (GT mels computed on-the-fly, doubles the size due to L/R channels)
            for audio_file in self.audio_files:
                # Each audio file contributes 2 samples (L and R channels)
                self.combined_files.append(('audio', audio_file, 0))  # Left channel
                self.combined_files.append(('audio', audio_file, 1))  # Right channel
                self.data_sources.extend(['audio', 'audio'])
            print(f"[INFO] Using audio files only (GT mels computed on-the-fly)")
        
        print(f"[INFO] Combined dataset: {len(self.combined_files)} total samples")
        print(f"  - Audio file samples: {len(self.audio_files) * 2}")
        print(f"  - Pre-computed predicted mel samples: {len(self.mel_pred_files) * 2}")
        print(f"  - GT mels: computed dynamically from audio")
        
        if len(self.combined_files) == 0:
            raise ValueError("No valid data sources found. Please provide either audio files or predicted mel directories.")
    
    def __len__(self):
        return len(self.combined_files)
    
    def get_pred_mel_probability(self):
        """より積極的なカリキュラムスケジュール"""
        if not self.use_pred_mel_schedule or not self.split:
            return 0.0
        
        epoch = self.current_epoch
        
        # 早期導入パターン
        e_start = 10    # 早めに開始
        e_mid = 30      # 早めに50%到達
        e_end = 60      # 早めに最大到達
        
        if epoch < e_start:
            return 0.0
        elif epoch < e_mid:
            progress = (epoch - e_start) / (e_mid - e_start)
            return progress * 0.5  # 0→50%
        elif epoch < e_end:
            progress = (epoch - e_mid) / (e_end - e_mid)
            return 0.5 + progress * 0.4  # 50%→90%
        else:
            return 1.0  # 予測mel 90%
    
    def update_epoch(self, epoch):
        """Update current epoch for curriculum scheduling"""
        self.current_epoch = epoch
    
    def get_resampler(self, orig_freq, new_freq):
        """Get cached resampler for performance optimization"""
        key = (orig_freq, new_freq)
        if key not in self._resamplers:
            self._resamplers[key] = ta.transforms.Resample(
                orig_freq=orig_freq,
                new_freq=new_freq,
                dtype=torch.float32
            )
        return self._resamplers[key]
    
    def __getitem__(self, index):
        try:
            data_type, data_path, channel_index = self.combined_files[index]
            
            if data_type == 'audio':
                return self._get_audio_sample(data_path, channel_index)
            elif data_type == 'mel_pred':
                return self._get_mel_pred_sample(data_path, channel_index)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
                
        except Exception as e:
            print(f"[WARNING] Failed to load sample at index {index}, skipping! Error: {e}")
            return self[random.randrange(len(self))]
    
    def _get_audio_sample(self, filename, channel_index):
        """Get sample from audio file using torchaudio for better performance"""
        # Load stereo audio using torchaudio (faster than librosa)
        audio_tensor, source_sampling_rate = ta.load(filename)
        
        # Handle different audio formats
        if audio_tensor.shape[0] == 1:
            # Mono file - duplicate to stereo
            audio_tensor = audio_tensor.repeat(2, 1)
        elif audio_tensor.shape[0] > 2:
            # Multi-channel: take first 2 channels
            audio_tensor = audio_tensor[:2, :]
        
        # Extract the desired channel and convert to numpy
        audio = audio_tensor[channel_index].numpy()  # Shape: (T,)
        
        # Main logic that uses <mel, audio> pair for training BigVGAN
        if not self.fine_tuning:
            if self.split:  # Training step
                # Obtain randomized audio chunk
                if source_sampling_rate != self.sampling_rate:
                    # Adjust segment size to crop if the source sr is different
                    target_segment_size = math.ceil(
                        self.segment_size
                        * (source_sampling_rate / self.sampling_rate)
                    )
                else:
                    target_segment_size = self.segment_size

                # Compute upper bound index for the random chunk
                random_chunk_upper_bound = max(
                    0, audio.shape[0] - target_segment_size
                )

                # Crop or pad audio to obtain random chunk with target_segment_size
                if audio.shape[0] >= target_segment_size:
                    audio_start = random.randint(0, random_chunk_upper_bound)
                    audio = audio[audio_start : audio_start + target_segment_size]
                else:
                    audio = np.pad(
                        audio,
                        (0, target_segment_size - audio.shape[0]),
                        mode="constant",
                    )

                # Resample audio chunk to self.sampling rate using cached resampler
                if source_sampling_rate != self.sampling_rate:
                    resampler = self.get_resampler(source_sampling_rate, self.sampling_rate)
                    audio_tensor_temp = torch.from_numpy(audio).unsqueeze(0)
                    audio_resampled = resampler(audio_tensor_temp)
                    audio = audio_resampled.squeeze().numpy()
                    
                    if audio.shape[0] > self.segment_size:
                        # trim last elements to match self.segment_size
                        audio = audio[: self.segment_size]

            else:  # Validation step
                # Resample full audio clip to target sampling rate using cached resampler
                if source_sampling_rate != self.sampling_rate:
                    resampler = self.get_resampler(source_sampling_rate, self.sampling_rate)
                    audio_tensor_temp = torch.from_numpy(audio).unsqueeze(0)
                    audio_resampled = resampler(audio_tensor_temp)
                    audio = audio_resampled.squeeze().numpy()
                # Trim last elements to match audio length to self.hop_size * n for evaluation
                if (audio.shape[0] % self.hop_size) != 0:
                    audio = audio[: -(audio.shape[0] % self.hop_size)]

            # Remove audio normalization as requested
            # audio = librosa.util.normalize(audio) * 0.95

            # Cast ndarray to torch tensor
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)  # [B(1), self.segment_size]

            # Compute mel spectrogram corresponding to audio
            mel = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )  # [B(1), self.num_mels, self.segment_size // self.hop_size]

        # Fine-tuning logic that uses pre-computed mel
        else:
            # For fine-tuning, assert that the waveform is in the defined sampling_rate
            assert (
                source_sampling_rate == self.sampling_rate
            ), f"For fine_tuning, waveform must be in the specified sampling rate {self.sampling_rate}, got {source_sampling_rate}"

            # Cast ndarray to torch tensor
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)  # [B(1), T_time]

            # Load pre-computed mel from disk
            mel = np.load(
                os.path.join(
                    self.base_mels_path,
                    os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                )
            )
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)  # ensure [B, C, T]

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start
                        * self.hop_size : (mel_start + frames_per_seg)
                        * self.hop_size,
                    ]

                # Pad pre-computed mel and audio to match length
                mel = torch.nn.functional.pad(
                    mel, (0, frames_per_seg - mel.size(2)), "constant"
                )
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(1)), "constant"
                )

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
        )  # [B(1), self.num_mels, self.segment_size // self.hop_size]

        # Shape sanity checks
        assert (
            audio.shape[1] == mel.shape[2] * self.hop_size
            and audio.shape[1] == mel_loss.shape[2] * self.hop_size
        ), f"Audio length must be mel frame length * hop_size. Got audio shape {audio.shape} mel shape {mel.shape} mel_loss shape {mel_loss.shape}"

        # Add channel identifier to filename
        channel_name = "L" if channel_index == 0 else "R"
        filename_with_channel = f"{os.path.basename(filename)}_{channel_name}_audio"

        return (mel.squeeze(), audio.squeeze(0), filename_with_channel, mel_loss.squeeze())
    

    
    def _get_mel_pred_sample(self, data_paths, channel_index):
        """Get sample with robustness training: randomly choose between GT and predicted mel"""
        mel_pred_left_path, mel_pred_right_path, audio_path = data_paths
        
        # Get probability of using predicted mel based on curriculum
        p_pred = self.get_pred_mel_probability()
        
        # For validation (split=False), check if GT mel directories are available
        # If GT mel dirs are None, always use pred_mel (validation with pred_mel only)
        if not self.split:
            if hasattr(self, 'mel_files') and len(self.mel_files) > 0:
                # GT mels available for validation - use GT
                use_pred_mel = False
            else:
                # No GT mels available - use pred_mel for realistic validation
                use_pred_mel = True
        else:
            # Training: use curriculum probability
            use_pred_mel = (random.random() < p_pred)
        
        # Initialize trimming variables
        frames_trimmed = 0
        
        if use_pred_mel:
            # Use predicted mel (diffusion model output)
            if channel_index == 0:  # Left channel
                mel_path = mel_pred_left_path
                channel_name = "L"
            else:  # Right channel
                mel_path = mel_pred_right_path
                channel_name = "R"
            
            # Load predicted mel spectrogram
            mel = np.load(mel_path)
            
            # Remove first and last 8 frames from predicted mel to avoid boundary artifacts
            if mel.shape[1] > 16:  # Ensure we have enough frames to trim (time axis)
                mel = mel[:, 8:-8]  # Cut 8 frames from both ends in time dimension
                frames_trimmed = 8
            
            source_type = "pred"
        else:
            # Use GT mel computed from audio using BigVGAN's mel function
            # This ensures perfect consistency with vocoder expectations
            try:
                # Load audio using torchaudio (faster and more integrated)
                audio_tensor, source_sampling_rate = ta.load(audio_path)
                
                # Handle different audio formats
                if audio_tensor.shape[0] == 1:
                    # Mono file - duplicate to stereo
                    audio_tensor = audio_tensor.repeat(2, 1)
                elif audio_tensor.shape[0] > 2:
                    # Multi-channel: take first 2 channels
                    audio_tensor = audio_tensor[:2, :]
                
                # Extract the desired channel
                audio_single_channel = audio_tensor[channel_index:channel_index+1, :]  # Keep 2D: [1, T]
                
                # Resample if necessary using cached resampler (higher quality)
                if source_sampling_rate != self.sampling_rate:
                    resampler = self.get_resampler(source_sampling_rate, self.sampling_rate)
                    audio_single_channel = resampler(audio_single_channel)
                
                # Remove audio normalization as requested
                # audio_tensor = ta.functional.normalize(audio_tensor) * 0.95
                
                # Already tensor, ready for mel computation
                audio_tensor = audio_single_channel
                
                # Compute GT mel using BigVGAN's mel function (ensures consistency)
                mel_tensor = mel_spectrogram(
                    audio_tensor,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                
                mel = mel_tensor.squeeze().numpy()
                
            except Exception as e:
                print(f"Error computing GT mel from audio {audio_path}: {e}")
                # Fallback to predicted mel if GT computation fails
                if channel_index == 0:
                    mel_path = mel_pred_left_path
                else:
                    mel_path = mel_pred_right_path
                mel = np.load(mel_path)
            
            channel_name = "L" if channel_index == 0 else "R"
            source_type = "gt"
        
        # Convert mel to tensor
        mel = torch.from_numpy(mel).float()
        
        # Load corresponding ground truth audio for loss computation
        try:
            # Load audio using torchaudio (faster and more integrated)
            audio_tensor, source_sampling_rate = ta.load(audio_path)
            
            # Handle different audio formats
            if audio_tensor.shape[0] == 1:
                # Mono file - duplicate to stereo
                audio_tensor = audio_tensor.repeat(2, 1)
            elif audio_tensor.shape[0] > 2:
                # Multi-channel: take first 2 channels
                audio_tensor = audio_tensor[:2, :]
            
            # Extract the desired channel and convert to numpy
            audio = audio_tensor[channel_index].numpy()
            
            # Resample if necessary using cached resampler (higher quality)
            if source_sampling_rate != self.sampling_rate:
                resampler = self.get_resampler(source_sampling_rate, self.sampling_rate)
                audio_resampled = resampler(audio_tensor[channel_index:channel_index+1, :])
                audio = audio_resampled.squeeze().numpy()
            
            # Trim audio to match the trimmed mel spectrogram (if predicted mel was trimmed)
            if use_pred_mel and frames_trimmed > 0:
                audio_trim_samples = frames_trimmed * self.hop_size
                if audio.shape[0] > 2 * audio_trim_samples:  # Ensure we have enough audio to trim
                    audio = audio[audio_trim_samples:-audio_trim_samples]
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Create silent audio as fallback
            audio_length = mel.shape[1] * self.hop_size
            audio = np.zeros(audio_length)
        
        # Calculate expected audio length based on mel spectrogram length
        expected_audio_length = mel.shape[1] * self.hop_size
        
        # Adjust audio length to match mel spectrogram
        if audio.shape[0] > expected_audio_length:
            audio = audio[:expected_audio_length]
        elif audio.shape[0] < expected_audio_length:
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
            min_mel_frames = self.segment_size // self.hop_size
            if mel.shape[1] < min_mel_frames:
                pad_frames = min_mel_frames - mel.shape[1]
                mel = F.pad(mel, (0, pad_frames))
            
            if len(audio) < self.segment_size:
                pad_size = self.segment_size - len(audio)
                audio = np.pad(audio, (0, pad_size))
        
        # Remove audio normalization as requested
        # audio = librosa.util.normalize(audio) * 0.95
        
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
        
        # Add channel and source type identifier to filename
        trim_suffix = f"_trim{frames_trimmed}" if frames_trimmed > 0 else ""
        filename_with_channel = f"{os.path.basename(audio_path)}_{channel_name}_mel_{source_type}{trim_suffix}"
        
        return (mel.squeeze(), audio.squeeze(0), filename_with_channel, mel_loss.squeeze())


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
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank:d}")

    # Define BigVGAN generator
    generator = BigVGAN(h).to(device)

    # Define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)

    # Define additional discriminators. BigVGAN-v1 uses UnivNet's MRD as default
    # New in BigVGAN-v2: option to switch to new discriminators: MultiBandDiscriminator / MultiScaleSubbandCQTDiscriminator
    if h.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print(
            "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        # Variable name is kept as "mrd" for backward compatibility & minimal code change
        mrd = MultiBandDiscriminator(h).to(device)
    elif h.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print(
            "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:  # Fallback to original MRD in BigVGAN-v1
        mrd = MultiResolutionDiscriminator(h).to(device)

    # New in BigVGAN-v2: option to switch to multi-scale L1 mel loss
    if h.get("use_multiscale_melloss", False):
        print(
            "[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss"
        )
        fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=h.sampling_rate
        )  # NOTE: accepts waveform as input
    else:
        fn_mel_loss_singlescale = F.l1_loss

    # Print the model & number of parameters, and create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        print(generator)
        print(mpd)
        print(mrd)
        print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters())}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters())}")
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print(f"Checkpoints directory: {a.checkpoint_path}")

    if os.path.isdir(a.checkpoint_path):
        # New in v2.1: If the step prefix pattern-based checkpoints are not found, also check for renamed files in Hugging Face Hub to resume training
        cp_g = scan_checkpoint(
            a.checkpoint_path, prefix="g_", renamed_file="bigvgan_generator.pt"
        )
        cp_do = scan_checkpoint(
            a.checkpoint_path,
            prefix="do_",
            renamed_file="bigvgan_discriminator_optimizer.pt",
        )

    # Load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    # Initialize DDP, optimizers, and schedulers
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    # Define training and validation datasets
    """
    Combined dataset supports both audio files and pre-computed mel spectrograms
    """
    
    # Get traditional audio file list (optional)
    training_filelist = []
    validation_filelist = []
    
    if a.input_training_file and os.path.exists(a.input_training_file):
        with open(a.input_training_file, 'r', encoding='utf-8') as f:
            training_filelist = [line.strip().split('|')[0] for line in f if line.strip()]
        print(f"[INFO] Found {len(training_filelist)} audio files from training file list")
    
    if a.input_validation_file and os.path.exists(a.input_validation_file):
        with open(a.input_validation_file, 'r', encoding='utf-8') as f:
            validation_filelist = [line.strip().split('|')[0] for line in f if line.strip()]
        print(f"[INFO] Found {len(validation_filelist)} audio files from validation file list")

    # Create combined training dataset
    print("[INFO] Creating combined binaural dataset with robustness training support")
    trainset = BinauralCombinedDataset(
        # Audio file parameters (for GT mel computation)
        training_files=training_filelist,
        hparams=h,
        # GT mel: computed on-the-fly from audio (no pre-saved GT mels)
        mel_left_dir=None,  # No pre-saved GT mels
        mel_right_dir=None,  # No pre-saved GT mels
        # Predicted mel spectrogram parameters (diffusion outputs)
        mel_pred_left_dir=getattr(a, 'mel_pred_left_train_dir', None),
        mel_pred_right_dir=getattr(a, 'mel_pred_right_train_dir', None),
        audio_dir=getattr(a, 'audio_dir', None),
        # Common parameters
        segment_size=h.segment_size,
        n_fft=h.n_fft,
        num_mels=h.num_mels,
        hop_size=h.hop_size,
        win_size=h.win_size,
        sampling_rate=h.sampling_rate,
        fmin=h.fmin,
        fmax=h.fmax,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
        is_seen=True,
        prefer_precomputed_mels=getattr(a, 'prefer_precomputed_mels', True),
        # Robustness training parameters
        current_epoch=0,
        use_pred_mel_schedule=getattr(a, 'use_pred_mel_schedule', True),
        recent_losses=None, # 最近の損失を記録
        loss_threshold=0.1, # 損失閾値
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        # Create combined validation dataset (pred_mel only for realistic evaluation)
        validset = BinauralCombinedDataset(
            # Audio file parameters (for GT audio loading, but GT mel computed on-the-fly)
            training_files=validation_filelist,
            hparams=h,
            # GT mel: no pre-saved GT mels (computed on-the-fly from audio when needed)
            mel_left_dir=None,
            mel_right_dir=None,
            # Predicted mel spectrogram parameters (diffusion outputs) - main source for validation
            mel_pred_left_dir=getattr(a, 'mel_pred_left_val_dir', getattr(a, 'mel_pred_left_train_dir', None)),
            mel_pred_right_dir=getattr(a, 'mel_pred_right_val_dir', getattr(a, 'mel_pred_right_train_dir', None)),
            audio_dir=getattr(a, 'audio_dir', None),
            # Common parameters
            segment_size=h.segment_size,
            n_fft=h.n_fft,
            num_mels=h.num_mels,
            hop_size=h.hop_size,
            win_size=h.win_size,
            sampling_rate=h.sampling_rate,
            fmin=h.fmin,
            fmax=h.fmax,
            split=False,
            shuffle=False,
            fmax_loss=h.fmax_for_loss,
            device=device,
            fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir,
            is_seen=True,
            prefer_precomputed_mels=getattr(a, 'prefer_precomputed_mels', True),
            # Robustness training parameters (validation uses pred_mel only)
            current_epoch=0,
            use_pred_mel_schedule=False,  # No curriculum for validation, but will use pred_mel
            recent_losses=None, # 最近の損失を記録
            loss_threshold=0.1, # 損失閾値
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        # Tensorboard logger
        sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))
        if a.save_audio:  # Also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, "samples"), exist_ok=True)

    """
    Validation loop, "mode" parameter is automatically defined as (seen or unseen)_(name of the dataset).
    If the name of the dataset contains "nonspeech", it skips PESQ calculation to prevent errors 
    """

    def validate(rank, a, h, loader, mode="seen"):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        torch.cuda.empty_cache()
        
        val_err_tot = 0
        
        with torch.no_grad():
            for j, batch in enumerate(loader):
                if j >= 10:  # Limit validation samples
                    break
                    
                x, y, _, y_mel = batch
                y = y.to(device)
                if hasattr(generator, "module"):
                    y_g_hat = generator.module(x.to(device))
                else:
                    y_g_hat = generator(x.to(device))
                y_mel = y_mel.to(device, non_blocking=True)
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

                # Log only first few samples to TensorBoard
                if j <= 4:
                    sw.add_audio(f'gt/y_{j}', y[0], steps, h.sampling_rate)
                    sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram_fixed_scale(x[0].cpu().numpy()), steps)

                    sw.add_audio(f'generated/y_hat_{j}', y_g_hat[0], steps, h.sampling_rate)
                    y_hat_spec = mel_spectrogram(
                        y_g_hat.squeeze(1),
                        h.n_fft,
                        h.num_mels,
                        h.sampling_rate,
                        h.hop_size,
                        h.win_size,
                        h.fmin,
                        h.fmax,
                    )
                    sw.add_figure(f'generated/y_hat_spec_{j}', plot_spectrogram_fixed_scale(y_hat_spec.squeeze(0).cpu().numpy()), steps)

            val_err = val_err_tot / (j + 1)
            sw.add_scalar(f"validation/mel_spec_error", val_err, steps)
            
            # Log validation dataset type for tracking
            val_dataset_type = "pred_mel" if len(loader.dataset.mel_files) == 0 else "gt_mel"
            sw.add_text(f"validation/dataset_type", val_dataset_type, steps)

        generator.train()

    # If the checkpoint is loaded, start with validation loop
    if steps != 0 and rank == 0 and not a.debug:
        if not a.skip_seen:
            # Log validation dataset type
            val_dataset_type = "pred_mel" if len(validation_loader.dataset.mel_files) == 0 else "gt_mel"
            print(f"[INFO] Initial validation with {val_dataset_type}")
            
            validate(
                rank,
                a,
                h,
                validation_loader,
                mode=f"seen_{train_loader.dataset.name}_{val_dataset_type}",
            )

    # Exit the script if --evaluate is set to True
    if a.evaluate:
        exit()

    # Main training loop
    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        # Update dataset epoch for curriculum scheduling (all ranks need this)
        train_loader.dataset.update_epoch(epoch)
        
        if rank == 0:
            start = time.time()
            p_pred = train_loader.dataset.get_pred_mel_probability()
            print(f"Epoch: {epoch + 1}, Pred mel probability: {p_pred:.3f}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
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

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

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
                print(
                    f"[WARNING] skipping D training for the first {a.freeze_step} steps"
                )
                grad_norm_mpd = 0.0
                grad_norm_mrd = 0.0

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            lambda_melloss = h.get(
                "lambda_melloss", 45.0
            )  # Defaults to 45 in BigVGAN-v1 if not set
            if h.get("use_multiscale_melloss", False):  # uses wav <y, y_g_hat> for loss
                loss_mel = fn_mel_loss_multiscale(y, y_g_hat) * lambda_melloss
            else:  # Uses mel <y_mel, y_g_hat_mel> for loss
                # Add simple silence-aware loss
                loss_mel_base = fn_mel_loss_singlescale(y_mel, y_g_hat_mel) * lambda_melloss
                loss_mel_silence = simple_silence_aware_mel_loss(y_mel, y_g_hat_mel, 
                                                               silence_threshold_db=-50, 
                                                               silence_penalty=2.0) * (lambda_melloss * 0.2)
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
                print(
                    f"[WARNING] using regression loss only for G for the first {a.freeze_step} steps"
                )
                loss_gen_all = loss_mel

            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), clip_grad_norm
            )
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to stdout
                    p_pred = train_loader.dataset.get_pred_mel_probability()
                    print(
                        f"Steps: {steps:d}, "
                        f"Gen Loss Total: {loss_gen_all:4.3f}, "
                        f"Mel Error: {mel_error:4.3f}, "
                        f"P(pred): {p_pred:.3f}, "
                        f"s/b: {time.time() - start_b:4.3f} "
                        f"lr: {optim_g.param_groups[0]['lr']:4.7f} "
                        f"grad_norm_g: {grad_norm_g:4.3f}"
                    )

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{a.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = f"{a.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to tensorboard
                    p_pred_current = train_loader.dataset.get_pred_mel_probability()
                    
                    sw.add_scalar("training/gen_loss_total", loss_gen_all.item(), steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/pred_mel_probability", p_pred_current, steps)  # Add curriculum tracking
                    sw.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                    sw.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                    sw.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                    sw.add_scalar("training/grad_norm_mpd", grad_norm_mpd, steps)
                    sw.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                    sw.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                    sw.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                    sw.add_scalar("training/grad_norm_mrd", grad_norm_mrd, steps)
                    sw.add_scalar("training/grad_norm_g", grad_norm_g, steps)
                    sw.add_scalar(
                        "training/learning_rate_d", scheduler_d.get_last_lr()[0], steps
                    )
                    sw.add_scalar(
                        "training/learning_rate_g", scheduler_g.get_last_lr()[0], steps
                    )
                    sw.add_scalar("training/epoch", epoch + 1, steps)

                # Validation
                if steps % a.validation_interval == 0:
                    # Plot training input x so far used
                    for i_x in range(x.shape[0]):
                        sw.add_figure(
                            f"training_input/x_{i_x}",
                            plot_spectrogram(x[i_x].cpu()),
                            steps,
                        )
                        sw.add_audio(
                            f"training_input/y_{i_x}",
                            y[i_x][0],
                            steps,
                            h.sampling_rate,
                        )

                    # Seen and unseen speakers validation loops
                    if not a.debug and steps != 0:
                        # Log validation dataset type
                        val_dataset_type = "pred_mel" if len(validation_loader.dataset.mel_files) == 0 else "gt_mel"
                        print(f"[INFO] Running validation with {val_dataset_type}")
                        
                        validate(
                            rank,
                            a,
                            h,
                            validation_loader,
                            mode=f"seen_{train_loader.dataset.name}_{val_dataset_type}",
                        )

            steps += 1

            # BigVGAN-v2 learning rate scheduler is changed from epoch-level to step-level
            scheduler_g.step()
            scheduler_d.step()

        if rank == 0:
            print(
                f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n"
            )


def main():
    print("Initializing Combined Binaural BigVGAN Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)

    # Audio file parameters (from train.py)
    parser.add_argument('--input_wavs_dir', default='/home/h-okano/DiffBinaural/FairPlay/binaural_audios_22050Hz')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='/home/h-okano/bigvgan/filelists/training_binaural.txt')
    parser.add_argument('--input_validation_file', default='/home/h-okano/bigvgan/filelists/validation_binaural.txt')
    parser.add_argument('--list_input_unseen_validation_file', default=[], nargs='*')
    parser.add_argument('--list_input_unseen_wavs_dir', default=[], nargs='*')

    # Note: GT mel parameters removed - GT mels are computed on-the-fly from audio files
    
    # Pre-computed predicted mel parameters (diffusion model outputs)
    parser.add_argument('--mel_pred_left_train_dir', default="/home/h-okano/DiffBinaural/test_results/realBinaural_left_train_mix", help='Directory containing predicted mel spectrograms (left channel) for training')
    parser.add_argument('--mel_pred_right_train_dir', default="/home/h-okano/DiffBinaural/test_results/realBinaural_right_train_mix", help='Directory containing predicted mel spectrograms (right channel) for training')
    parser.add_argument('--mel_pred_left_val_dir', default="/home/h-okano/DiffBinaural/test_results/realBinaural_left_val_mix", help='Directory containing predicted mel spectrograms (left channel) for validation')
    parser.add_argument('--mel_pred_right_val_dir', default="/home/h-okano/DiffBinaural/test_results/realBinaural_right_val_mix", help='Directory containing predicted mel spectrograms (right channel) for validation')
    
    parser.add_argument('--audio_dir', default='/home/h-okano/real_dataset/processed/binaural_audios_22050Hz')

    # Data source preference
    parser.add_argument('--prefer_precomputed_mels', default=True, type=bool, 
                       help='If True, prioritize pre-computed mels over real-time audio processing')
    
    # Robustness training parameters
    parser.add_argument('--use_pred_mel_schedule', default=True, type=bool,
                       help='Enable curriculum learning schedule for predicted mel usage')
    parser.add_argument('--disable_audio_normalization', default=True, type=bool,
                       help='Disable audio normalization during training')

    # Training parameters
    parser.add_argument("--checkpoint_path", default="/home/h-okano/bigvgan/cp_bigvgan_binaural_mix_comb")
    parser.add_argument("--config", default="/home/h-okano/bigvgan/configs/bigvgan_binaural_22khz_80band_256x.json")

    parser.add_argument("--training_epochs", default=100000, type=int)
    parser.add_argument("--stdout_interval", default=50, type=int)
    parser.add_argument("--checkpoint_interval", default=3000, type=int)
    parser.add_argument("--summary_interval", default=50, type=int)
    parser.add_argument("--validation_interval", default=600, type=int)

    parser.add_argument(
        "--freeze_step",
        default=0,
        type=int,
        help="freeze D for the first specified steps. G only uses regression loss for these steps.",
    )

    parser.add_argument("--fine_tuning", default=False, type=bool)

    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="debug mode. skips validation loop throughout training",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        type=bool,
        help="only run evaluation from checkpoint and exit",
    )
    parser.add_argument(
        "--eval_subsample",
        default=5,
        type=int,
        help="subsampling during evaluation loop",
    )
    parser.add_argument(
        "--skip_seen",
        default=False,
        type=bool,
        help="skip seen dataset. useful for test set inference",
    )
    parser.add_argument(
        "--save_audio",
        default=False,
        type=bool,
        help="save audio of test set inference to disk",
    )

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    build_env(a.config, "config.json", a.checkpoint_path)

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


if __name__ == "__main__":
    main() 
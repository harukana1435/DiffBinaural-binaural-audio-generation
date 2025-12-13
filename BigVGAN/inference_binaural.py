# Binaural inference script for BigVGAN
# Based on the original BigVGAN inference implementation

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
import librosa
import numpy as np
from utils import load_checkpoint
from meldataset import get_mel_spectrogram
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan_binaural import BinauralBigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def inference_binaural(a, h):
    """
    Perform binaural inference using BinauralBigVGAN.
    
    Expects input directory to contain either:
    1. Paired mel spectrograms: filename_left.npy, filename_right.npy
    2. Stereo wav files (will be split into left/right channels)
    """
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    
    with torch.no_grad():
        # Case 1: Process paired mel spectrograms
        if hasattr(a, 'input_mels_dir') and a.input_mels_dir:
            process_paired_mel_spectrograms(generator, a)
        
        # Case 2: Process stereo wav files
        elif hasattr(a, 'input_wavs_dir') and a.input_wavs_dir:
            process_stereo_wav_files(generator, a, h)
        
        # Case 3: Process separate left/right mel spectrograms from diffusion model output
        elif hasattr(a, 'input_left_mels') and hasattr(a, 'input_right_mels'):
            process_separate_mel_inputs(generator, a)
        
        else:
            raise ValueError("No valid input method specified. Use --input_mels_dir, --input_wavs_dir, or --input_left_mels/--input_right_mels")


def process_paired_mel_spectrograms(generator, a):
    """Process paired mel spectrogram files (filename_left.npy, filename_right.npy)"""
    print("Processing paired mel spectrograms...")
    
    files = os.listdir(a.input_mels_dir)
    left_files = [f for f in files if f.endswith('_left.npy')]
    
    for left_file in left_files:
        base_name = left_file.replace('_left.npy', '')
        right_file = f"{base_name}_right.npy"
        
        if right_file not in files:
            print(f"Warning: Missing right channel for {base_name}, skipping...")
            continue
            
        # Load mel spectrograms
        mel_left = np.load(os.path.join(a.input_mels_dir, left_file))
        mel_right = np.load(os.path.join(a.input_mels_dir, right_file))
        
        # Convert to tensors
        mel_left = torch.FloatTensor(mel_left).to(device)
        mel_right = torch.FloatTensor(mel_right).to(device)
        
        # Ensure correct dimensions (add batch dimension if needed)
        if mel_left.dim() == 2:
            mel_left = mel_left.unsqueeze(0)
        if mel_right.dim() == 2:
            mel_right = mel_right.unsqueeze(0)
            
        # Generate binaural audio
        y_g_hat = generator(mel_left, mel_right)
        
        # Process output
        binaural_audio = y_g_hat.squeeze(0)  # Remove batch dimension
        left_audio = binaural_audio[0].cpu().numpy()
        right_audio = binaural_audio[1].cpu().numpy()
        
        # Save as stereo wav
        stereo_audio = np.stack([left_audio, right_audio], axis=0)
        stereo_audio = stereo_audio * MAX_WAV_VALUE
        stereo_audio = stereo_audio.astype(np.int16)
        
        output_path = os.path.join(a.output_dir, f"{base_name}_binaural.wav")
        write(output_path, generator.h.sampling_rate, stereo_audio.T)
        
        print(f"Generated: {output_path}")


def process_stereo_wav_files(generator, a, h):
    """Process stereo wav files by splitting into left/right channels"""
    print("Processing stereo wav files...")
    
    filelist = os.listdir(a.input_wavs_dir)
    wav_files = [f for f in filelist if f.endswith(('.wav', '.flac', '.mp3'))]
    
    for filename in wav_files:
        # Load stereo audio
        audio_path = os.path.join(a.input_wavs_dir, filename)
        wav, sr = librosa.load(audio_path, sr=h.sampling_rate, mono=False)
        
        if wav.ndim == 1:
            print(f"Warning: {filename} is mono, duplicating to stereo...")
            wav = np.stack([wav, wav], axis=0)
        elif wav.shape[0] > 2:
            print(f"Warning: {filename} has >2 channels, using first 2...")
            wav = wav[:2]
            
        # Split into left and right channels
        left_wav = torch.FloatTensor(wav[0]).to(device)
        right_wav = torch.FloatTensor(wav[1]).to(device)
        
        # Compute mel spectrograms
        mel_left = get_mel_spectrogram(left_wav.unsqueeze(0), h)
        mel_right = get_mel_spectrogram(right_wav.unsqueeze(0), h)
        
        # Generate binaural audio
        y_g_hat = generator(mel_left, mel_right)
        
        # Process output
        binaural_audio = y_g_hat.squeeze(0)
        left_audio = binaural_audio[0].cpu().numpy()
        right_audio = binaural_audio[1].cpu().numpy()
        
        # Save as stereo wav
        stereo_audio = np.stack([left_audio, right_audio], axis=0)
        stereo_audio = stereo_audio * MAX_WAV_VALUE
        stereo_audio = stereo_audio.astype(np.int16)
        
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(a.output_dir, f"{base_name}_reconstructed_binaural.wav")
        write(output_path, h.sampling_rate, stereo_audio.T)
        
        print(f"Generated: {output_path}")


def process_separate_mel_inputs(generator, a):
    """Process separate left and right mel spectrogram inputs (e.g., from diffusion model)"""
    print("Processing separate mel inputs...")
    
    # Load mel spectrograms
    mel_left = np.load(a.input_left_mels)
    mel_right = np.load(a.input_right_mels)
    
    # Convert to tensors
    mel_left = torch.FloatTensor(mel_left).to(device)
    mel_right = torch.FloatTensor(mel_right).to(device)
    
    # Ensure correct dimensions
    if mel_left.dim() == 2:
        mel_left = mel_left.unsqueeze(0)
    if mel_right.dim() == 2:
        mel_right = mel_right.unsqueeze(0)
        
    # Generate binaural audio
    y_g_hat = generator(mel_left, mel_right)
    
    # Process output
    binaural_audio = y_g_hat.squeeze(0)
    left_audio = binaural_audio[0].cpu().numpy()
    right_audio = binaural_audio[1].cpu().numpy()
    
    # Save as stereo wav
    stereo_audio = np.stack([left_audio, right_audio], axis=0)
    stereo_audio = stereo_audio * MAX_WAV_VALUE
    stereo_audio = stereo_audio.astype(np.int16)
    
    output_path = os.path.join(a.output_dir, "generated_binaural.wav")
    write(output_path, generator.h.sampling_rate, stereo_audio.T)
    
    print(f"Generated: {output_path}")


def main():
    print('Initializing Binaural Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', required=True, help='Path to the binaural generator checkpoint')
    parser.add_argument('--config_file', help='Path to config file (JSON)')
    parser.add_argument('--output_dir', default='generated_files_binaural', help='Output directory')
    
    # Input options (specify one)
    parser.add_argument('--input_mels_dir', help='Directory containing paired mel spectrograms (_left.npy, _right.npy)')
    parser.add_argument('--input_wavs_dir', help='Directory containing stereo wav files')
    parser.add_argument('--input_left_mels', help='Path to left channel mel spectrogram (.npy)')
    parser.add_argument('--input_right_mels', help='Path to right channel mel spectrogram (.npy)')
    
    parser.add_argument('--use_cuda_kernel', action='store_true', help='Use optimized CUDA kernel for inference')
    
    a = parser.parse_args()

    # Load config
    if a.config_file:
        with open(a.config_file) as f:
            data = f.read()
        global h
        h = AttrDict(json.loads(data))
    else:
        # Try to load config from checkpoint directory
        checkpoint_dir = os.path.dirname(a.checkpoint_file)
        config_path = os.path.join(checkpoint_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                data = f.read()
            h = AttrDict(json.loads(data))
        else:
            raise ValueError("Config file not found. Specify --config_file or place config.json in checkpoint directory")

    torch.manual_seed(h.seed)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    inference_binaural(a, h)


if __name__ == '__main__':
    main() 
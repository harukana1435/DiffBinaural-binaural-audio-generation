# Inference script for DiffBinaural generated mel spectrograms
# Processes separate left/right mel spectrogram directories

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
import librosa
import numpy as np
from utils import load_checkpoint
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan_binaural import BinauralBigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def process_diffbinaural_mels(generator, a):
    """
    Process DiffBinaural generated mel spectrograms from separate left/right directories.
    
    Args:
        generator: BinauralBigVGAN model
        a: Arguments containing left_mels_dir and right_mels_dir paths
    """
    print("Processing DiffBinaural generated mel spectrograms...")
    print(f"Left mels directory: {a.left_mels_dir}")
    print(f"Right mels directory: {a.right_mels_dir}")
    
    # Get all .npy files from left directory
    left_files = [f for f in os.listdir(a.left_mels_dir) if f.endswith('.npy')]
    left_files.sort()
    
    # Get all .npy files from right directory  
    right_files = [f for f in os.listdir(a.right_mels_dir) if f.endswith('.npy')]
    right_files.sort()
    
    print(f"Found {len(left_files)} left mel files and {len(right_files)} right mel files")
    
    processed_count = 0
    
    for left_file in left_files:
        # Find corresponding right file
        base_name = os.path.splitext(left_file)[0]
        right_file = f"{base_name}.npy"
        
        if right_file not in right_files:
            print(f"Warning: Missing right channel for {base_name}, skipping...")
            continue
            
        try:
            # Load mel spectrograms
            left_path = os.path.join(a.left_mels_dir, left_file)
            right_path = os.path.join(a.right_mels_dir, right_file)
            
            print(f"Processing: {base_name}")
            
            mel_left = np.load(left_path)
            mel_right = np.load(right_path)
            
            print(f"  Left mel shape: {mel_left.shape}")
            print(f"  Right mel shape: {mel_right.shape}")
            
            # Convert to tensors
            mel_left = torch.FloatTensor(mel_left).to(device)
            mel_right = torch.FloatTensor(mel_right).to(device)
            
            # Ensure correct dimensions (add batch dimension if needed)
            if mel_left.dim() == 2:
                mel_left = mel_left.unsqueeze(0)
            if mel_right.dim() == 2:
                mel_right = mel_right.unsqueeze(0)
                
            print(f"  Tensor shapes - Left: {mel_left.shape}, Right: {mel_right.shape}")
            
            # Generate binaural audio
            with torch.no_grad():
                y_g_hat = generator(mel_left, mel_right)
            
            # Process output
            binaural_audio = y_g_hat.squeeze(0)  # Remove batch dimension
            left_audio = binaural_audio[0].cpu().numpy()
            right_audio = binaural_audio[1].cpu().numpy()
            
            print(f"  Generated audio - Left: {left_audio.shape}, Right: {right_audio.shape}")
            
            # Save as stereo wav
            stereo_audio = np.stack([left_audio, right_audio], axis=0)
            stereo_audio = stereo_audio * MAX_WAV_VALUE
            stereo_audio = np.clip(stereo_audio, -32767, 32767)  # Prevent clipping
            stereo_audio = stereo_audio.astype(np.int16)
            
            output_path = os.path.join(a.output_dir, f"{base_name}_binaural.wav")
            write(output_path, h.sampling_rate, stereo_audio.T)
            
            print(f"  Generated: {output_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
    
    print(f"\nCompleted processing {processed_count} files")


def inference_diffbinaural(a, h):
    """
    Perform binaural inference using BinauralBigVGAN on DiffBinaural generated mels.
    """
    print("Initializing BinauralBigVGAN...")
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    print(f"Loading checkpoint from: {a.checkpoint_file}")
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    os.makedirs(a.output_dir, exist_ok=True)
    print(f"Output directory: {a.output_dir}")

    generator.eval()
    generator.remove_weight_norm()
    
    with torch.no_grad():
        process_diffbinaural_mels(generator, a)


def main():
    print('Initializing DiffBinaural Mel Spectrogram to Audio Inference Process..')

    parser = argparse.ArgumentParser(description='Generate binaural audio from DiffBinaural mel spectrograms')
    parser.add_argument('--checkpoint_file', required=True, 
                       help='Path to the binaural BigVGAN generator checkpoint')
    parser.add_argument('--config_file', 
                       help='Path to config file (JSON). If not specified, will look in checkpoint directory')
    parser.add_argument('--left_mels_dir', required=True,
                       help='Directory containing left channel mel spectrograms (.npy files)')
    parser.add_argument('--right_mels_dir', required=True,
                       help='Directory containing right channel mel spectrograms (.npy files)')
    parser.add_argument('--output_dir', default='generated_binaural_audio', 
                       help='Output directory for generated audio files')
    parser.add_argument('--use_cuda_kernel', action='store_true', 
                       help='Use optimized CUDA kernel for inference (faster)')
    
    a = parser.parse_args()

    # Validate input directories
    if not os.path.exists(a.left_mels_dir):
        raise ValueError(f"Left mels directory not found: {a.left_mels_dir}")
    if not os.path.exists(a.right_mels_dir):
        raise ValueError(f"Right mels directory not found: {a.right_mels_dir}")

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
            print(f"Loading config from: {config_path}")
            with open(config_path) as f:
                data = f.read()
            h = AttrDict(json.loads(data))
        else:
            raise ValueError("Config file not found. Specify --config_file or place config.json in checkpoint directory")

    torch.manual_seed(h.seed)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print configuration
    print(f"Sampling rate: {h.sampling_rate}")
    print(f"Hop length: {h.hop_length}")
    print(f"Win length: {h.win_length}")
    print(f"N FFT: {h.n_fft}")
    print(f"N mels: {h.n_mels}")

    inference_diffbinaural(a, h)


if __name__ == '__main__':
    main() 
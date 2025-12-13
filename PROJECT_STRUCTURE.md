# Project Structure

This document provides a detailed overview of the project structure and file organization.

## Directory Tree

```
DiffBinaural: binaural audio generation/
│
├── README.md                    # Main project documentation
├── QUICKSTART.md                # Quick start guide for new users
├── PROJECT_STRUCTURE.md         # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── DiffBinaural/                # Stage 1: Diffusion model
│   ├── README.md                # DiffBinaural documentation
│   │
│   ├── modules/                 # Neural network modules
│   │   ├── models.py            # Model builder and factory
│   │   ├── unet.py              # UNet architecture for diffusion
│   │   ├── audioVisual_model.py # Visual encoder (ResNet-based)
│   │   ├── attention.py         # Attention mechanisms
│   │   ├── networks.py          # Network utilities
│   │   └── norms.py             # Normalization layers
│   │
│   ├── dataset/                 # Dataset loaders
│   │   ├── __init__.py
│   │   ├── base.py              # Base dataset class
│   │   ├── fairplay_pos.py      # FairPlay dataset with position
│   │   ├── fairplay_pos_left.py # FairPlay left channel
│   │   ├── fairplay_pos_right.py# FairPlay right channel
│   │   ├── dataset_real_binaural.py  # RealBinaural dataset
│   │   ├── dataset_real_binaural_mix.py  # Mixed dataset
│   │   ├── genaudio_*.py        # Audio generation datasets
│   │   └── video_transforms.py  # Video preprocessing
│   │
│   ├── diffusion_utils/         # Diffusion model implementation
│   │   └── diffusion_pytorch.py # DDPM/DDIM implementation
│   │
│   ├── utils/                   # Utility functions
│   │   ├── arguments.py         # Argument parser
│   │   └── helpers.py           # Helper functions (mel, STFT, etc.)
│   │
│   ├── configs/                 # Configuration files
│   │   └── advanced_diffusion_config.py
│   │
│   ├── train_fairplay.py        # Training script for FairPlay
│   ├── train_realBinaural.py    # Training script for RealBinaural
│   ├── test_fairplay.py         # Testing script for FairPlay
│   ├── test_realBinaural.py     # Testing script for RealBinaural
│   ├── test_realBinaural_few.py # Testing on few samples
│   ├── test_pos.py              # Testing with position encoding
│   ├── training_stabilizer.py   # Training stabilization utilities
│   ├── position_utils.py        # Position encoding utilities
│   ├── evaluate_binaural_22050.py      # Binaural audio evaluation
│   └── evaluate_mel_spectrogram_rmse.py # Mel-spectrogram evaluation
│
└── BigVGAN/                     # Stage 2: Neural vocoder
    ├── README.md                # BigVGAN documentation
    ├── LICENSE                  # NVIDIA BigVGAN license
    ├── README_original.md       # Original BigVGAN README
    │
    ├── configs/                 # Configuration files
    │   ├── bigvgan_22khz_80band.json
    │   └── bigvgan_binaural_22khz_80band_256x.json
    │
    ├── alias_free_activation/   # Anti-aliasing modules
    │   ├── __init__.py
    │   ├── act.py               # Activation functions
    │   ├── filter.py            # Anti-aliasing filters
    │   └── resample.py          # Resampling operations
    │
    ├── bigvgan.py               # BigVGAN generator
    ├── discriminators.py        # Multi-scale discriminators
    ├── loss.py                  # Loss functions
    ├── meldataset.py            # Mel-spectrogram dataset
    ├── env.py                   # Environment setup
    ├── utils.py                 # Utility functions
    ├── activations.py           # Activation functions
    │
    ├── train_binaural_mel.py    # Stage 2a: Train with GT mels
    ├── train_binaural_both.py   # Stage 2b: Scheduled sampling
    ├── inference_binaural.py    # Binaural inference
    ├── inference_diffbinaural_mels.py  # Inference with DiffBinaural mels
    └── inference_e2e.py         # End-to-end inference
```

## Key Files Description

### Root Level

- **README.md**: Comprehensive project documentation with architecture overview, training instructions, and usage examples
- **QUICKSTART.md**: Step-by-step guide for getting started quickly
- **requirements.txt**: All Python package dependencies
- **LICENSE**: MIT License with acknowledgments
- **.gitignore**: Excludes checkpoints, data, logs, and temporary files

### DiffBinaural/ (Stage 1)

#### Training Scripts
- `train_fairplay.py`: Train on FairPlay dataset (synthetic binaural audio)
- `train_realBinaural.py`: Train on RealBinaural dataset (recorded binaural audio)

#### Testing Scripts
- `test_fairplay.py`: Generate binaural mels for FairPlay test set
- `test_realBinaural.py`: Generate binaural mels for RealBinaural test set
- `test_realBinaural_few.py`: Quick testing on a few samples
- `test_pos.py`: Test with position encoding

#### Evaluation Scripts
- `evaluate_binaural_22050.py`: Evaluate binaural audio quality (PESQ, STOI, etc.)
- `evaluate_mel_spectrogram_rmse.py`: Evaluate mel-spectrogram accuracy (RMSE, L1, L2)

#### Core Modules
- `modules/models.py`: ModelBuilder class for creating networks
- `modules/unet.py`: UNet architecture for diffusion model
- `modules/audioVisual_model.py`: Visual encoder (ResNet18/50)
- `diffusion_utils/diffusion_pytorch.py`: DDPM/DDIM implementation

#### Datasets
- `dataset/fairplay_pos.py`: FairPlay dataset with 2D position encoding
- `dataset/dataset_real_binaural.py`: RealBinaural dataset loader
- `dataset/genaudio_*.py`: Audio generation datasets for inference

#### Utilities
- `utils/arguments.py`: Command-line argument parser
- `utils/helpers.py`: Mel-spectrogram, STFT, visualization utilities
- `training_stabilizer.py`: Gradient clipping, NaN detection
- `position_utils.py`: Position encoding functions

### BigVGAN/ (Stage 2)

#### Training Scripts
- `train_binaural_mel.py`: Standard training with pre-computed mels
- `train_binaural_both.py`: Advanced training with scheduled sampling

#### Inference Scripts
- `inference_binaural.py`: Generate binaural audio from mel-spectrograms
- `inference_diffbinaural_mels.py`: Inference using DiffBinaural outputs
- `inference_e2e.py`: End-to-end mono → binaural pipeline

#### Core Modules
- `bigvgan.py`: BigVGAN generator architecture
- `discriminators.py`: Multi-Period and Multi-Resolution Discriminators
- `loss.py`: Generator and discriminator losses
- `meldataset.py`: Dataset for mel-spectrogram training

#### Alias-Free Activation
- `alias_free_activation/act.py`: Snake activation with anti-aliasing
- `alias_free_activation/filter.py`: Low-pass filters
- `alias_free_activation/resample.py`: Upsampling/downsampling

#### Configuration
- `configs/bigvgan_binaural_22khz_80band_256x.json`: Main config for 22kHz binaural audio

## Data Flow

### Training Pipeline

```
1. Raw Data
   ├── Monaural audio (22kHz)
   ├── Binaural audio (22kHz, stereo)
   └── Video frames (224x224)
   
2. Stage 1: DiffBinaural Training
   ├── Input: Mono mel + Visual features
   ├── Output: Binaural mel (L+R)
   └── Checkpoints: unet_best.pth, frame_best.pth
   
3. Stage 1: Generate Mels for Stage 2
   ├── Run: test_realBinaural.py
   ├── Output: Left/Right mel-spectrograms (.npy)
   └── Used for BigVGAN training
   
4. Stage 2a: BigVGAN Pre-training
   ├── Input: Ground-truth mels from audio
   ├── Output: High-quality audio waveforms
   └── Checkpoints: g_*.pth, do_*.pth
   
5. Stage 2b: BigVGAN Fine-tuning
   ├── Input: Mix of GT and predicted mels
   ├── Output: Robust binaural audio
   └── Final model for inference
```

### Inference Pipeline

```
1. Input
   ├── Monaural audio file (.wav)
   └── Video file (.mp4) or frames
   
2. Preprocessing
   ├── Extract mel-spectrogram from audio
   ├── Extract frames from video
   └── Compute position encoding
   
3. Stage 1: DiffBinaural
   ├── Load: unet_best.pth, frame_best.pth
   ├── Generate: Binaural mel (L+R)
   └── DDIM sampling (25 steps)
   
4. Stage 2: BigVGAN
   ├── Load: g_best.pth
   ├── Input: Left mel, Right mel
   ├── Generate: Left audio, Right audio
   └── Combine: Stereo binaural audio
   
5. Output
   └── Binaural audio file (.wav, 22kHz, stereo)
```

## File Naming Conventions

### Checkpoints
- `unet_latest.pth`: Latest UNet weights
- `unet_best.pth`: Best UNet weights (lowest validation loss)
- `frame_latest.pth`: Latest visual encoder weights
- `frame_best.pth`: Best visual encoder weights
- `g_XXXXXXXX`: BigVGAN generator checkpoint at step XXXXXXXX
- `do_XXXXXXXX`: BigVGAN discriminator+optimizer checkpoint

### Generated Files
- `{video_id}_{timestamp}_left.npy`: Left channel mel-spectrogram
- `{video_id}_{timestamp}_right.npy`: Right channel mel-spectrogram
- `{video_id}_{timestamp}.wav`: Generated binaural audio

### Logs
- `checkpoints/{experiment_id}/runs/`: TensorBoard logs
- `checkpoints/{experiment_id}/training_history.json`: Training metrics

## Configuration Files

### DiffBinaural Config
Located in `DiffBinaural/configs/advanced_diffusion_config.py`
- Diffusion parameters (timesteps, schedule)
- Model architecture (channels, layers)
- Training hyperparameters

### BigVGAN Config
Located in `BigVGAN/configs/bigvgan_binaural_22khz_80band_256x.json`
- Audio parameters (sample rate, FFT size, hop size)
- Generator architecture (upsample rates, channels)
- Discriminator settings
- Training hyperparameters (learning rate, batch size)

## Important Notes

### What's Included
✅ All training and inference code
✅ Dataset loaders for FairPlay and RealBinaural
✅ Evaluation scripts
✅ Configuration files
✅ Comprehensive documentation

### What's NOT Included (by design)
❌ Trained model checkpoints (too large)
❌ Dataset files (audio, video, images)
❌ Generated outputs
❌ Training logs and TensorBoard files
❌ Cached data and temporary files

### To Use This Repository
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your dataset (see README.md)
3. Train Stage 1: `python DiffBinaural/train_realBinaural.py ...`
4. Generate mels: `python DiffBinaural/test_realBinaural.py ...`
5. Train Stage 2: `python BigVGAN/train_binaural_mel.py ...`
6. Fine-tune: `python BigVGAN/train_binaural_both.py ...`
7. Inference: Use scripts in `BigVGAN/inference_*.py`

## Additional Resources

- **Main README**: Comprehensive overview and usage
- **DiffBinaural README**: Detailed Stage 1 documentation
- **BigVGAN README**: Detailed Stage 2 documentation
- **QUICKSTART**: Step-by-step tutorial

For questions, please refer to the documentation or open an issue on GitHub.



# DiffBinaural: Binaural Audio Generation

This repository contains the implementation of DiffBinaural, a diffusion-based model for binaural audio generation from monaural audio and visual information.

## Overview

DiffBinaural is a two-stage approach for generating high-quality binaural audio:

1. **Stage 1 (DiffBinaural)**: A diffusion model that generates binaural mel-spectrograms from monaural audio and visual features
2. **Stage 2 (BigVGAN)**: A neural vocoder that converts mel-spectrograms to high-quality binaural audio waveforms

## Project Structure

```
DiffBinaural: binaural audio generation/
├── DiffBinaural/              # Stage 1: Diffusion model for binaural mel generation
│   ├── modules/               # Network architectures (UNet, visual encoders)
│   ├── dataset/               # Dataset loaders for FairPlay and RealBinaural
│   ├── utils/                 # Helper functions and argument parsers
│   ├── diffusion_utils/       # Diffusion model implementation
│   ├── configs/               # Configuration files
│   ├── train_fairplay.py      # Training script for FairPlay dataset
│   ├── train_realBinaural.py  # Training script for RealBinaural dataset
│   ├── test_fairplay.py       # Testing script for FairPlay dataset
│   ├── test_realBinaural.py   # Testing script for RealBinaural dataset
│   └── training_stabilizer.py # Training stabilization utilities
│
├── BigVGAN/                   # Stage 2: Neural vocoder
│   ├── alias_free_activation/ # Anti-aliasing activation functions
│   ├── configs/               # BigVGAN configuration files
│   ├── train_binaural_mel.py  # Training with pre-computed mels (Stage 2)
│   ├── train_binaural_both.py # Training with scheduled sampling
│   ├── bigvgan.py             # BigVGAN generator architecture
│   ├── discriminators.py      # Multi-scale discriminators
│   ├── loss.py                # Loss functions
│   ├── meldataset.py          # Mel-spectrogram dataset
│   └── utils.py               # Utility functions
│
└── README.md                  # This file
```

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.13.0
- librosa >= 0.8.1
- numpy
- scipy
- tensorboard
- soundfile
- matplotlib
- pesq
- auraloss
- tqdm
- nnAudio
- ninja

### Hardware Requirements

- GPU with at least 16GB VRAM (24GB+ recommended for training)
- Multi-GPU support available for distributed training

## Dataset Preparation

### FairPlay Dataset

The FairPlay dataset should be organized as follows:

```
FairPlay/
├── audio/
│   ├── mono/          # Monaural audio files
│   └── binaural/      # Binaural audio files
├── video/
│   └── frames/        # Extracted video frames
└── metadata.csv       # Metadata with audio-video pairs
```

### RealBinaural Dataset

The RealBinaural dataset should be organized as follows:

```
real_dataset/
├── processed/
│   ├── mono_audios_22050Hz/      # Monaural audio files
│   ├── binaural_audios_22050Hz/  # Binaural audio files
│   └── frames/                    # Video frames
├── action_detection_results/
│   └── detection_results.csv      # Action detection metadata
└── splits/
    ├── train.csv                  # Training split
    └── val.csv                    # Validation split
```

## Training

### Stage 1: DiffBinaural Training

#### Training on FairPlay Dataset

```bash
cd DiffBinaural

python train_fairplay.py \
    --id fairplay_experiment \
    --list_train /path/to/fairplay/train.csv \
    --list_val /path/to/fairplay/val.csv \
    --ckpt ./checkpoints \
    --num_gpus 2 \
    --batch_size_per_gpu 8 \
    --num_epoch 1000 \
    --lr_unet 0.0001 \
    --lr_frame 0.0001 \
    --eval_epoch 10
```

#### Training on RealBinaural Dataset

```bash
cd DiffBinaural

python train_realBinaural.py \
    --id realBinaural_experiment \
    --list_train /path/to/real_dataset/splits/train.csv \
    --list_val /path/to/real_dataset/splits/val.csv \
    --data_root /path/to/real_dataset \
    --ckpt ./checkpoints \
    --num_gpus 2 \
    --batch_size_per_gpu 8 \
    --num_epoch 1000 \
    --lr_unet 0.0001 \
    --lr_frame 0.0001 \
    --eval_epoch 10
```

### Stage 2: BigVGAN Training

#### Stage 2a: Training with Pre-computed Mels (Recommended)

First, generate mel-spectrograms using the trained DiffBinaural model:

```bash
cd DiffBinaural

python test_realBinaural.py \
    --id realBinaural_experiment \
    --mode test \
    --weights_unet ./checkpoints/unet_best.pth \
    --weights_frame ./checkpoints/frame_best.pth \
    --output_dir ./generated_mels
```

Then train BigVGAN on the generated mels:

```bash
cd BigVGAN

python train_binaural_mel.py \
    --mel_left_train_dir /path/to/generated_mels/left/train \
    --mel_right_train_dir /path/to/generated_mels/right/train \
    --mel_left_val_dir /path/to/generated_mels/left/val \
    --mel_right_val_dir /path/to/generated_mels/right/val \
    --audio_dir /path/to/real_dataset/processed/binaural_audios_22050Hz \
    --checkpoint_path ./checkpoints/bigvgan_stage2 \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 2000
```

#### Stage 2b: Training with Scheduled Sampling (Advanced)

For improved robustness, use scheduled sampling which gradually transitions from ground-truth mels to predicted mels:

```bash
cd BigVGAN

python train_binaural_both.py \
    --mel_left_train_dir /path/to/generated_mels/left/train \
    --mel_right_train_dir /path/to/generated_mels/right/train \
    --mel_pred_left_dir /path/to/predicted_mels/left/train \
    --mel_pred_right_dir /path/to/predicted_mels/right/train \
    --audio_dir /path/to/real_dataset/processed/binaural_audios_22050Hz \
    --checkpoint_path ./checkpoints/bigvgan_scheduled \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 2000 \
    --use_scheduled_sampling
```

**Scheduled Sampling Strategy:**
- Starts with 100% ground-truth mels
- Gradually increases the probability of using predicted mels
- Helps the model become robust to imperfect inputs from DiffBinaural
- Reduces the train-test mismatch problem

## Testing and Evaluation

### Testing DiffBinaural (Stage 1)

```bash
cd DiffBinaural

python test_realBinaural.py \
    --id realBinaural_experiment \
    --mode eval \
    --weights_unet ./checkpoints/unet_best.pth \
    --weights_frame ./checkpoints/frame_best.pth \
    --list_val /path/to/real_dataset/splits/val.csv
```

### Evaluation Metrics

The repository includes evaluation scripts for:

```bash
# Evaluate mel-spectrogram quality
python evaluate_mel_spectrogram_rmse.py \
    --pred_dir ./generated_mels \
    --gt_dir ./ground_truth_mels

# Evaluate binaural audio quality (22kHz)
python evaluate_binaural_22050.py \
    --pred_dir ./generated_audio \
    --gt_dir ./ground_truth_audio
```

## Key Features

### DiffBinaural (Stage 1)

- **Diffusion-based generation**: Uses DDIM sampling for efficient inference
- **Visual conditioning**: Incorporates visual features from video frames
- **Position encoding**: Supports 2D position encoding for spatial audio
- **Training stabilization**: Includes gradient clipping and loss monitoring
- **Multi-dataset support**: Works with both FairPlay and RealBinaural datasets

### BigVGAN (Stage 2)

- **High-quality vocoder**: Generates 22kHz binaural audio
- **Scheduled sampling**: Novel training strategy for robustness
- **Silence-aware loss**: Reduces artifacts in silent regions
- **Multi-scale discriminators**: Ensures high-quality audio generation
- **Alias-free activation**: Reduces aliasing artifacts

## Training Tips

1. **Stage 1 (DiffBinaural)**:
   - Start with a smaller learning rate (1e-4) for stability
   - Use gradient clipping to prevent exploding gradients
   - Monitor mel-spectrogram L2 distance during validation
   - Train for at least 500 epochs for good convergence

2. **Stage 2 (BigVGAN)**:
   - Pre-train with ground-truth mels first (train_binaural_mel.py)
   - Then fine-tune with scheduled sampling (train_binaural_both.py)
   - Use the silence-aware loss to reduce noise in quiet regions
   - Monitor both mel-spectrogram error and discriminator losses

3. **Multi-GPU Training**:
   - Use `--num_gpus` to specify the number of GPUs
   - Adjust `--batch_size_per_gpu` based on your GPU memory
   - Total batch size = num_gpus × batch_size_per_gpu

## Citation

If you use this code in your research, please cite:

```bibtex
@article{diffbinaural2024,
  title={DiffBinaural: Diffusion-based Binaural Audio Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project includes code from:
- BigVGAN: [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN) (MIT License)

Please refer to individual LICENSE files for details.

## Acknowledgments

- BigVGAN implementation is based on NVIDIA's BigVGAN
- Diffusion model implementation is inspired by Denoising Diffusion Probabilistic Models (DDPM)
- FairPlay dataset for binaural audio research

## Contact

For questions or issues, please open an issue on GitHub.


# DiffBinaural-binaural-audio-generation

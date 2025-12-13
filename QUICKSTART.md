# Quick Start Guide

This guide will help you get started with DiffBinaural quickly.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DiffBinaural.git
cd "DiffBinaural: binaural audio generation"
```

### 2. Create a conda environment

```bash
conda create -n diffbinaural python=3.9
conda activate diffbinaural
```

### 3. Install PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install other dependencies

```bash
pip install -r requirements.txt
```

## Quick Training Example

### Prepare Your Data

Organize your dataset as follows:

```
your_dataset/
├── processed/
│   ├── mono_audios_22050Hz/
│   │   ├── video001.wav
│   │   └── video002.wav
│   ├── binaural_audios_22050Hz/
│   │   ├── video001.wav
│   │   └── video002.wav
│   └── frames/
│       ├── video001/
│       │   ├── frame_000001.jpg
│       │   └── ...
│       └── video002/
│           └── ...
└── splits/
    ├── train.csv
    └── val.csv
```

**train.csv format**:
```csv
video_id,start_time,end_time,action_label
video001,0.0,10.0,speaking
video002,5.0,15.0,playing_instrument
```

### Stage 1: Train DiffBinaural

```bash
cd DiffBinaural

python train_realBinaural.py \
    --id my_first_experiment \
    --list_train /path/to/your_dataset/splits/train.csv \
    --list_val /path/to/your_dataset/splits/val.csv \
    --data_root /path/to/your_dataset \
    --ckpt ./checkpoints \
    --num_gpus 1 \
    --batch_size_per_gpu 4 \
    --num_epoch 100 \
    --lr_unet 0.0001 \
    --lr_frame 0.0001 \
    --eval_epoch 10
```

**Training tips**:
- Start with fewer epochs (100) to verify everything works
- Reduce `batch_size_per_gpu` if you run out of memory
- Monitor TensorBoard: `tensorboard --logdir checkpoints/my_first_experiment/runs`

### Stage 2: Generate Mel-Spectrograms

After training DiffBinaural, generate mel-spectrograms for BigVGAN training:

```bash
python test_realBinaural.py \
    --id my_first_experiment \
    --mode test \
    --weights_unet ./checkpoints/my_first_experiment/unet_best.pth \
    --weights_frame ./checkpoints/my_first_experiment/frame_best.pth \
    --list_val /path/to/your_dataset/splits/train.csv \
    --data_root /path/to/your_dataset \
    --output_dir ./generated_mels/train

# Also generate for validation set
python test_realBinaural.py \
    --id my_first_experiment \
    --mode test \
    --weights_unet ./checkpoints/my_first_experiment/unet_best.pth \
    --weights_frame ./checkpoints/my_first_experiment/frame_best.pth \
    --list_val /path/to/your_dataset/splits/val.csv \
    --data_root /path/to/your_dataset \
    --output_dir ./generated_mels/val
```

### Stage 3: Train BigVGAN

```bash
cd ../BigVGAN

# Stage 2a: Train with ground-truth mels
python train_binaural_mel.py \
    --mel_left_train_dir ../DiffBinaural/generated_mels/train/left \
    --mel_right_train_dir ../DiffBinaural/generated_mels/train/right \
    --mel_left_val_dir ../DiffBinaural/generated_mels/val/left \
    --mel_right_val_dir ../DiffBinaural/generated_mels/val/right \
    --audio_dir /path/to/your_dataset/processed/binaural_audios_22050Hz \
    --checkpoint_path ./checkpoints/stage2a \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 500

# Stage 2b: Fine-tune with scheduled sampling
python train_binaural_both.py \
    --mel_left_train_dir ../DiffBinaural/generated_mels/train/left \
    --mel_right_train_dir ../DiffBinaural/generated_mels/train/right \
    --mel_pred_left_dir ../DiffBinaural/generated_mels/train/left \
    --mel_pred_right_dir ../DiffBinaural/generated_mels/train/right \
    --audio_dir /path/to/your_dataset/processed/binaural_audios_22050Hz \
    --checkpoint_path ./checkpoints/stage2b \
    --stage1_checkpoint_path ./checkpoints/stage2a \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 1000 \
    --load_stage1_checkpoint True
```

## Inference

### Full Pipeline: Mono Audio + Video → Binaural Audio

```python
import torch
import numpy as np
from DiffBinaural.modules import models
from DiffBinaural.diffusion_utils import diffusion_pytorch
from BigVGAN.bigvgan import BigVGAN
from BigVGAN.env import AttrDict
import json

# 1. Load DiffBinaural
builder = models.ModelBuilder()
net_frame = builder.build_visual(
    pool_type='avgpool',
    arch_frame='resnet18'
)
net_unet = builder.build_unet()

net_frame.load_state_dict(torch.load('DiffBinaural/checkpoints/frame_best.pth'))
net_unet.load_state_dict(torch.load('DiffBinaural/checkpoints/unet_best.pth'))

sampler = diffusion_pytorch.GaussianDiffusion(
    net_unet,
    image_size=80,
    timesteps=1000,
    sampling_timesteps=25,
    loss_type='l1',
    objective='pred_noise',
    beta_schedule='cosine'
)

net_frame.cuda().eval()
net_unet.cuda().eval()

# 2. Load BigVGAN
with open('BigVGAN/configs/bigvgan_binaural_22khz_80band_256x.json') as f:
    h = AttrDict(json.load(f))

bigvgan = BigVGAN(h).cuda()
checkpoint = torch.load('BigVGAN/checkpoints/stage2b/g_best')
bigvgan.load_state_dict(checkpoint['generator'])
bigvgan.eval()

# 3. Inference
with torch.no_grad():
    # Load your mono mel and video frames
    mono_mel = torch.randn(1, 1, 80, 80).cuda()  # Replace with actual data
    frames = torch.randn(1, 3, 5, 4, 224, 224).cuda()  # Replace with actual frames
    pos_2d = torch.randn(1, 5, 4, 2).cuda()  # Replace with actual positions
    mask = torch.zeros(1, 5, 4).bool().cuda()
    
    # Generate binaural mel with DiffBinaural
    feat_frames = net_frame.forward_multiframe(frames, pos_2d, mask)
    binaural_mel = sampler.ddim_sample(
        condition=[mono_mel, feat_frames],
        return_all_timesteps=False,
        sampling_timesteps=25
    )
    
    # Split into left and right channels
    mel_left = binaural_mel[:, 0:1, :, :]  # [1, 1, 80, T]
    mel_right = binaural_mel[:, 1:2, :, :]  # [1, 1, 80, T]
    
    # Generate audio with BigVGAN
    audio_left = bigvgan(mel_left.squeeze(1))  # [1, 1, T*256]
    audio_right = bigvgan(mel_right.squeeze(1))  # [1, 1, T*256]
    
    # Combine into stereo
    audio_stereo = torch.cat([audio_left, audio_right], dim=1)  # [1, 2, T*256]
    
    # Save
    import soundfile as sf
    audio_np = audio_stereo[0].cpu().numpy().T  # [T*256, 2]
    sf.write('output_binaural.wav', audio_np, 22050)
```

## Troubleshooting

### Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch_size_per_gpu 2  # Instead of 8
```

**Solution 2**: Use gradient checkpointing (modify code)

**Solution 3**: Use smaller model
```bash
--arch_frame resnet18  # Instead of resnet50
```

### Training is slow

**Solution 1**: Use multiple GPUs
```bash
--num_gpus 2
```

**Solution 2**: Reduce validation frequency
```bash
--eval_epoch 20  # Instead of 10
```

**Solution 3**: Use mixed precision training (modify code to add `torch.cuda.amp`)

### Poor audio quality

**Solution 1**: Train longer
```bash
--num_epoch 1000  # Instead of 100
```

**Solution 2**: Use larger batch size
```bash
--batch_size_per_gpu 16  # If you have enough memory
```

**Solution 3**: Check data quality
- Verify audio files are not corrupted
- Ensure proper stereo channel order
- Check mel-spectrogram normalization

## Next Steps

1. **Experiment with hyperparameters**:
   - Learning rates
   - Number of diffusion steps
   - Model architectures

2. **Try different datasets**:
   - FairPlay dataset
   - Your own recorded binaural audio

3. **Evaluate results**:
   ```bash
   cd DiffBinaural
   python evaluate_binaural_22050.py \
       --pred_dir ./generated_audio \
       --gt_dir /path/to/ground_truth
   ```

4. **Visualize results**:
   - Use TensorBoard to monitor training
   - Compare mel-spectrograms visually
   - Listen to generated audio samples

## Getting Help

- Check the detailed READMEs in `DiffBinaural/` and `BigVGAN/` directories
- Review the main `README.md` for architecture details
- Open an issue on GitHub if you encounter problems

## Citation

If you use this code, please cite:

```bibtex
@article{diffbinaural2024,
  title={DiffBinaural: Diffusion-based Binaural Audio Generation},
  author={Your Name},
  year={2024}
}
```



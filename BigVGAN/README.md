# BigVGAN for Binaural Audio Generation

This directory contains the BigVGAN neural vocoder adapted for binaural audio generation. BigVGAN serves as Stage 2 of the DiffBinaural pipeline, converting mel-spectrograms to high-quality binaural audio waveforms.

## Overview

BigVGAN is a high-fidelity neural vocoder that has been adapted for binaural audio generation. This implementation includes:

1. **Standard mel-to-waveform training** (`train_binaural_mel.py`)
2. **Scheduled sampling training** (`train_binaural_both.py`) - Our novel contribution

## Key Features

### 1. Scheduled Sampling Strategy

The scheduled sampling approach addresses the train-test mismatch problem in two-stage models:

- **Problem**: During training, BigVGAN sees perfect ground-truth mels. During inference, it receives imperfect mels from DiffBinaural.
- **Solution**: Gradually transition from ground-truth mels to predicted mels during training.

**Implementation** (`train_binaural_both.py`):
```python
# Curriculum learning schedule
epoch_progress = current_epoch / total_epochs
pred_mel_probability = min(0.8, epoch_progress * 0.8)  # 0% → 80%

# Randomly choose between GT and predicted mels
if random.random() < pred_mel_probability:
    mel_input = predicted_mel  # From DiffBinaural
else:
    mel_input = ground_truth_mel  # From audio
```

**Benefits**:
- Reduces artifacts when using DiffBinaural outputs
- Improves robustness to mel-spectrogram imperfections
- Better generalization to unseen data

### 2. Silence-Aware Loss

Special loss function to reduce noise in silent regions:

```python
def simple_silence_aware_mel_loss(y_mel, y_g_hat_mel, 
                                  silence_threshold_db=-50, 
                                  silence_penalty=2.0):
    """
    Applies higher penalty for errors in silent regions
    Reduces background noise and artifacts
    """
    # Detect silent regions
    silence_mask = detect_silence(y_mel, threshold_db)
    
    # Apply weighted loss
    loss = mel_loss * silence_mask * silence_penalty + \
           mel_loss * active_mask * 1.0
    
    return loss.mean()
```

### 3. Binaural Single-Channel Training

Treats left and right channels as independent samples:
- Doubles the effective dataset size
- Each channel is processed separately
- Maintains binaural characteristics while simplifying training

## Training Stages

### Stage 2a: Pre-training with Ground-Truth Mels

First, train BigVGAN with perfect mel-spectrograms extracted from ground-truth audio:

```bash
python train_binaural_mel.py \
    --mel_left_train_dir /path/to/gt_mels/left/train \
    --mel_right_train_dir /path/to/gt_mels/right/train \
    --audio_dir /path/to/binaural_audio \
    --checkpoint_path ./checkpoints/stage2a_gt_mels \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 1000
```

**Purpose**: Learn basic mel-to-waveform mapping with clean data.

### Stage 2b: Fine-tuning with Scheduled Sampling

Then, fine-tune with scheduled sampling using DiffBinaural outputs:

```bash
python train_binaural_both.py \
    --mel_left_train_dir /path/to/gt_mels/left/train \
    --mel_right_train_dir /path/to/gt_mels/right/train \
    --mel_pred_left_dir /path/to/diffbinaural_mels/left/train \
    --mel_pred_right_dir /path/to/diffbinaural_mels/right/train \
    --audio_dir /path/to/binaural_audio \
    --checkpoint_path ./checkpoints/stage2b_scheduled \
    --stage1_checkpoint_path ./checkpoints/stage2a_gt_mels \
    --config ./configs/bigvgan_binaural_22khz_80band_256x.json \
    --training_epochs 2000 \
    --load_stage1_checkpoint True
```

**Purpose**: Adapt to imperfect mels from DiffBinaural while maintaining quality.

## Configuration Files

### `bigvgan_binaural_22khz_80band_256x.json`

Main configuration for binaural audio at 22kHz:

```json
{
  "resblock": "1",
  "num_gpus": 2,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "adam_b1": 0.8,
  "adam_b2": 0.99,
  "lr_decay": 0.999,
  "seed": 1234,
  
  "upsample_rates": [8, 8, 2, 2],
  "upsample_kernel_sizes": [16, 16, 4, 4],
  "upsample_initial_channel": 512,
  
  "resblock_kernel_sizes": [3, 7, 11],
  "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
  
  "num_mels": 80,
  "n_fft": 1024,
  "hop_size": 256,
  "win_size": 1024,
  
  "sampling_rate": 22050,
  "fmin": 0,
  "fmax": 11025,
  "fmax_for_loss": 11025
}
```

## Training Scripts

### `train_binaural_mel.py`

Standard training with pre-computed mel-spectrograms.

**Key parameters**:
- `--mel_left_train_dir`: Directory with left channel mels
- `--mel_right_train_dir`: Directory with right channel mels
- `--audio_dir`: Directory with ground-truth binaural audio
- `--checkpoint_path`: Where to save checkpoints
- `--config`: Path to configuration JSON
- `--training_epochs`: Number of training epochs (default: 2000)
- `--checkpoint_interval`: Save checkpoint every N steps (default: 5000)
- `--validation_interval`: Run validation every N steps (default: 100)

**Features**:
- Silence-aware mel loss
- Multi-scale discriminators (MPD + MRD)
- Gradient clipping for stability
- TensorBoard logging

### `train_binaural_both.py`

Advanced training with scheduled sampling.

**Additional parameters**:
- `--mel_pred_left_dir`: Directory with predicted left channel mels (from DiffBinaural)
- `--mel_pred_right_dir`: Directory with predicted right channel mels (from DiffBinaural)
- `--stage1_checkpoint_path`: Path to stage 2a checkpoint
- `--load_stage1_checkpoint`: Whether to load stage 2a weights (default: True)

**Scheduled Sampling Schedule**:
```python
# Epoch 0-200: 0% → 20% predicted mels
# Epoch 200-500: 20% → 50% predicted mels
# Epoch 500-1000: 50% → 80% predicted mels
# Epoch 1000+: 80% predicted mels (plateau)
```

## Model Architecture

### Generator (BigVGAN)

- **Input**: Mel-spectrogram [B, 80, T]
- **Output**: Waveform [B, 1, T×256]
- **Upsampling**: 4 stages (8×8×2×2 = 256× total)
- **Residual blocks**: Multi-receptive field fusion
- **Activation**: Anti-aliased Snake activation

### Discriminators

1. **Multi-Period Discriminator (MPD)**
   - Analyzes audio at different periods
   - Captures pitch and periodicity

2. **Multi-Resolution Discriminator (MRD)**
   - Analyzes audio at multiple resolutions
   - Captures temporal structure

## Loss Functions

### Generator Loss

```python
loss_gen = loss_mel + loss_gen_mpd + loss_gen_mrd + loss_fm_mpd + loss_fm_mrd
```

Where:
- `loss_mel`: L1 mel-spectrogram reconstruction loss (λ=60)
- `loss_gen_*`: Adversarial losses from discriminators
- `loss_fm_*`: Feature matching losses

### Discriminator Loss

```python
loss_disc = loss_disc_mpd + loss_disc_mrd
```

Standard GAN hinge loss for both discriminators.

## Inference

To generate binaural audio from mel-spectrograms:

```python
import torch
from bigvgan import BigVGAN
from env import AttrDict
import json

# Load config
with open('configs/bigvgan_binaural_22khz_80band_256x.json') as f:
    config = json.load(f)
h = AttrDict(config)

# Load model
generator = BigVGAN(h).cuda()
checkpoint = torch.load('checkpoints/g_best')
generator.load_state_dict(checkpoint['generator'])
generator.eval()

# Generate audio
with torch.no_grad():
    mel = torch.randn(1, 80, 100).cuda()  # Example mel
    audio = generator(mel)  # [1, 1, 25600]
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints/stage2b_scheduled/logs
```

**Key metrics to monitor**:
- `training/mel_spec_error`: Mel reconstruction error (should decrease)
- `training/gen_loss_total`: Total generator loss
- `training/disc_loss_mpd`: MPD discriminator loss
- `training/disc_loss_mrd`: MRD discriminator loss
- `validation/mel_spec_error`: Validation mel error

### Expected Training Behavior

**Stage 2a (GT mels)**:
- Mel error: 0.5 → 0.1 (first 500 epochs)
- Discriminator losses stabilize around 1.0-2.0
- Audio quality improves rapidly

**Stage 2b (Scheduled sampling)**:
- Mel error may initially increase slightly (0.1 → 0.15)
- Gradually decreases as model adapts (0.15 → 0.12)
- Audio quality with DiffBinaural inputs improves significantly

## Troubleshooting

### Issue: High-frequency noise in generated audio

**Solution**:
- Increase `silence_penalty` in silence-aware loss
- Reduce learning rate
- Check mel-spectrogram normalization

### Issue: Muffled audio

**Solution**:
- Verify `fmax` and `fmax_for_loss` match training data
- Check if mel-spectrograms are properly normalized
- Increase training epochs

### Issue: Training instability

**Solution**:
- Reduce learning rate (try 5e-5)
- Increase gradient clipping threshold
- Use `--freeze_step` to delay discriminator training

## Performance

### Computational Requirements

- **Training**: 2× NVIDIA RTX 3090 (24GB each)
- **Batch size**: 16 per GPU (32 total)
- **Training time**: ~3-5 days for 2000 epochs
- **Inference**: Real-time on single GPU

### Audio Quality

- **Sample rate**: 22,050 Hz
- **Mel bands**: 80
- **Hop size**: 256 samples (~11.6 ms)
- **Receptive field**: ~1 second

## Citation

This implementation is based on NVIDIA's BigVGAN:

```bibtex
@inproceedings{lee2023bigvgan,
  title={BigVGAN: A Universal Neural Vocoder with Large-Scale Training},
  author={Lee, Sang-gil and Kim, Wei-Ning and Ping, Wei and Ginsburg, Boris and Catanzaro, Bryan},
  booktitle={ICLR},
  year={2023}
}
```

## License

This code is based on NVIDIA BigVGAN, which is licensed under the MIT License. See LICENSE file for details.



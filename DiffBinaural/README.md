# DiffBinaural: Diffusion Model for Binaural Audio Generation

This directory contains the DiffBinaural model implementation, which serves as Stage 1 of the binaural audio generation pipeline. DiffBinaural generates binaural mel-spectrograms from monaural audio and visual information using a diffusion-based approach.

## Overview

DiffBinaural uses a conditional diffusion model to generate binaural mel-spectrograms. The model takes:
- **Input**: Monaural audio mel-spectrogram + Visual features from video frames
- **Output**: Binaural mel-spectrogram (left and right channels)

## Model Architecture

### Core Components

1. **Visual Encoder** (`modules/audioVisual_model.py`)
   - Extracts features from video frames
   - Supports multi-frame temporal modeling
   - Includes 2D position encoding for spatial audio

2. **UNet Backbone** (`modules/unet.py`)
   - Conditional diffusion model
   - Takes monaural mel + visual features as conditions
   - Generates binaural mel-spectrograms

3. **Diffusion Process** (`diffusion_utils/diffusion_pytorch.py`)
   - DDIM sampling for efficient inference
   - Cosine noise schedule
   - 1000 diffusion steps, 25 sampling steps

## Training Scripts

### `train_fairplay.py`

Training script for the FairPlay dataset.

**Usage**:
```bash
python train_fairplay.py \
    --id fairplay_exp \
    --list_train /path/to/fairplay/train.csv \
    --list_val /path/to/fairplay/val.csv \
    --ckpt ./checkpoints \
    --num_gpus 2 \
    --batch_size_per_gpu 8 \
    --num_epoch 1000 \
    --lr_unet 0.0001 \
    --lr_frame 0.0001 \
    --eval_epoch 10 \
    --learning_rate_decrease_itr 100 \
    --decay_factor 0.94
```

**Key Parameters**:
- `--id`: Experiment identifier
- `--list_train`: Path to training CSV file
- `--list_val`: Path to validation CSV file
- `--ckpt`: Checkpoint directory
- `--num_gpus`: Number of GPUs to use
- `--batch_size_per_gpu`: Batch size per GPU
- `--num_epoch`: Total training epochs
- `--lr_unet`: Learning rate for UNet
- `--lr_frame`: Learning rate for visual encoder
- `--eval_epoch`: Evaluate every N epochs
- `--learning_rate_decrease_itr`: Decrease LR every N epochs
- `--decay_factor`: LR decay factor (default: 0.94)

**Dataset Format** (FairPlay):
```csv
audio_path,video_path,duration
/path/to/audio1.wav,/path/to/video1.mp4,10.0
/path/to/audio2.wav,/path/to/video2.mp4,8.5
```

### `train_realBinaural.py`

Training script for the RealBinaural dataset (recorded binaural audio).

**Usage**:
```bash
python train_realBinaural.py \
    --id realBinaural_exp \
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

**Additional Parameters**:
- `--data_root`: Root directory of RealBinaural dataset
- `--audLen`: Audio length in seconds (default: 0.64s)
- `--num_mels`: Number of mel bands (default: 80)

**Dataset Format** (RealBinaural):
```csv
video_id,start_time,end_time,action_label
video001,0.0,10.0,speaking
video002,5.0,15.0,playing_instrument
```

The dataset loader automatically finds corresponding:
- Monaural audio: `{data_root}/processed/mono_audios_22050Hz/{video_id}.wav`
- Binaural audio: `{data_root}/processed/binaural_audios_22050Hz/{video_id}.wav`
- Video frames: `{data_root}/processed/frames/{video_id}/frame_{:06d}.jpg`

## Testing Scripts

### `test_fairplay.py`

Generate binaural mel-spectrograms for FairPlay dataset.

**Usage**:
```bash
python test_fairplay.py \
    --id fairplay_exp \
    --mode test \
    --weights_unet ./checkpoints/unet_best.pth \
    --weights_frame ./checkpoints/frame_best.pth \
    --list_val /path/to/fairplay/test.csv \
    --output_dir ./generated_mels/fairplay
```

### `test_realBinaural.py`

Generate binaural mel-spectrograms for RealBinaural dataset.

**Usage**:
```bash
python test_realBinaural.py \
    --id realBinaural_exp \
    --mode test \
    --weights_unet ./checkpoints/unet_best.pth \
    --weights_frame ./checkpoints/frame_best.pth \
    --list_val /path/to/real_dataset/splits/test.csv \
    --data_root /path/to/real_dataset \
    --output_dir ./generated_mels/realBinaural
```

**Output Format**:
- Left channel mels: `{output_dir}/left/{video_id}_{timestamp}.npy`
- Right channel mels: `{output_dir}/right/{video_id}_{timestamp}.npy`

## Dataset Loaders

### FairPlay Dataset (`dataset/fairplay_pos.py`)

Loads FairPlay dataset with:
- Binaural audio (stereo)
- Monaural audio (downmixed)
- Video frames (5 frames per sample)
- 2D position information

**Features**:
- Automatic audio resampling to 22kHz
- Mel-spectrogram extraction (80 bands)
- Frame extraction at 5 FPS
- Data augmentation (random cropping)

### RealBinaural Dataset (`dataset/dataset_real_binaural.py`)

Loads real recorded binaural audio with:
- Binaural audio (recorded with binaural microphones)
- Monaural audio (center channel or downmixed)
- Video frames
- Action detection metadata

**Features**:
- 22kHz audio processing
- Synchronized audio-video loading
- Action-based temporal segmentation
- Robust frame loading with fallbacks

## Diffusion Model Details

### Noise Schedule

Uses cosine schedule for better sample quality:

```python
beta_schedule = 'cosine'
timesteps = 1000  # Training steps
sampling_timesteps = 25  # Inference steps (DDIM)
```

### Loss Function

L1 loss between predicted and target mel-spectrograms:

```python
loss = sampler(
    binaural_mel,  # Target
    [mono_mel, visual_features],  # Conditions
    cfg=True,  # Classifier-free guidance
    weight=weight  # Optional weighting
)
```

### Sampling Process

DDIM sampling for fast inference:

```python
pred = sampler.ddim_sample(
    condition=[mono_mel, visual_features],
    return_all_timesteps=False,
    sampling_timesteps=25  # 25 steps instead of 1000
)
```

## Training Stabilization

### `training_stabilizer.py`

Includes utilities for stable training:

1. **Gradient Clipping**
   ```python
   nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
   ```

2. **NaN Detection**
   - Monitors for NaN in inputs, features, and outputs
   - Replaces NaN with zeros to prevent crashes

3. **Learning Rate Scheduling**
   - Exponential decay every N epochs
   - Default: 0.94 decay factor every 100 epochs

## Normalization

### Mel-Spectrogram Normalization

```python
# Clamp to range
mel = torch.clamp(mel, min=-12, max=2.5)

# Normalize to [-1, 1]
mel = 2.0 * (mel - min_val) / (max_val - min_val) - 1.0
```

### Denormalization for Output

```python
# Denormalize from [-1, 1]
mel = 0.5 * (mel + 1.0) * (max_val - min_val) + min_val
mel = torch.clamp(mel, min=-12, max=2.5)
```

## Evaluation Metrics

### `evaluate_mel_spectrogram_rmse.py`

Evaluates mel-spectrogram quality:

```bash
python evaluate_mel_spectrogram_rmse.py \
    --pred_dir ./generated_mels \
    --gt_dir ./ground_truth_mels
```

**Metrics**:
- RMSE (Root Mean Square Error)
- L1 distance
- L2 distance
- Per-frequency band analysis

### `evaluate_binaural_22050.py`

Evaluates binaural audio quality:

```bash
python evaluate_binaural_22050.py \
    --pred_dir ./generated_audio \
    --gt_dir ./ground_truth_audio
```

**Metrics**:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- Mel-cepstral distortion

## Model Checkpoints

### Checkpoint Format

```
checkpoints/
├── frame_latest.pth       # Latest visual encoder weights
├── unet_latest.pth        # Latest UNet weights
├── frame_best.pth         # Best visual encoder weights
├── unet_best.pth          # Best UNet weights
├── frame_000100           # Checkpoint at epoch 100
├── unet_000100
└── training_history.json  # Training metrics
```

### Loading Checkpoints

```python
# Load best model
net_frame.load_state_dict(torch.load('checkpoints/frame_best.pth'))
net_unet.load_state_dict(torch.load('checkpoints/unet_best.pth'))
```

### Resume Training

Training automatically resumes from the latest checkpoint if found:

```bash
# Will automatically load latest checkpoint
python train_realBinaural.py --id realBinaural_exp --ckpt ./checkpoints
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints/realBinaural_exp/runs
```

**Logged Metrics**:
- Training loss (per iteration)
- Validation mel L2 distance (per epoch)
- Learning rates (UNet and visual encoder)
- Mel-spectrogram visualizations

### Expected Training Behavior

**Initial epochs (1-100)**:
- Loss: 0.5 → 0.2
- Mel L2: 200 → 100
- Rapid improvement in quality

**Middle epochs (100-500)**:
- Loss: 0.2 → 0.1
- Mel L2: 100 → 50
- Gradual refinement

**Late epochs (500-1000)**:
- Loss: 0.1 → 0.05
- Mel L2: 50 → 30
- Fine-tuning details

## Troubleshooting

### Issue: NaN in training

**Symptoms**: Loss becomes NaN, training crashes

**Solutions**:
- Reduce learning rate (try 5e-5)
- Check input data for NaN values
- Increase gradient clipping threshold
- Verify normalization ranges

### Issue: Poor binaural separation

**Symptoms**: Left and right channels sound similar

**Solutions**:
- Train longer (>500 epochs)
- Increase visual feature importance
- Check if binaural audio is properly loaded
- Verify stereo channel order

### Issue: Artifacts in silent regions

**Symptoms**: Noise during quiet parts

**Solutions**:
- Use weighted loss (set `--weighted_loss`)
- Increase training data
- Post-process with noise gate

## Advanced Features

### Position Encoding

2D position encoding for spatial audio:

```python
pos_2d = batch_data['2d_pos_data']  # [B, L, N, 2]
# L: number of frames
# N: number of objects
# 2: (x, y) coordinates
```

### Multi-Frame Processing

Process multiple frames for temporal context:

```python
frames = batch_data['frames']  # [B, C, L, N, H, W]
# L: 5 frames (default)
# N: 4 objects per frame
# H, W: 224x224 (ResNet input size)
```

### Mask Handling

Handle variable number of objects:

```python
mask = batch_data['mask']  # [B, L, N]
# True: invalid object (padding)
# False: valid object
```

## Performance

### Computational Requirements

- **Training**: 2× NVIDIA RTX 3090 (24GB each)
- **Batch size**: 16 (8 per GPU)
- **Training time**: ~2-3 days for 1000 epochs
- **Inference**: ~0.1s per sample on single GPU

### Memory Usage

- **FairPlay**: ~8GB per GPU (batch size 8)
- **RealBinaural**: ~10GB per GPU (batch size 8)

## Tips for Best Results

1. **Data Preparation**:
   - Ensure audio is properly aligned with video
   - Use high-quality binaural recordings
   - Verify stereo channel order (L/R)

2. **Training**:
   - Start with pretrained visual encoder
   - Use learning rate warmup
   - Monitor validation metrics closely

3. **Inference**:
   - Use DDIM with 25 steps for speed
   - Can increase to 50 steps for quality
   - Batch multiple samples for efficiency

## Citation

If you use DiffBinaural in your research, please cite:

```bibtex
@article{diffbinaural2024,
  title={DiffBinaural: Diffusion-based Binaural Audio Generation from Monaural Audio and Visual Information},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```



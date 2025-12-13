# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan import BigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def detect_and_exclude_zero_frames(mel_spec, zero_threshold=1e-10):
    """
    ゼロフレームを検出し、非ゼロフレームのみを抽出する
    
    Args:
        mel_spec: (num_mels, time_frames) のメルスペクトログラム
        zero_threshold: ゼロとみなす閾値
        
    Returns:
        filtered_mel: 非ゼロフレームのみのメルスペクトログラム
        zero_mask: ゼロフレームの位置 (True=ゼロフレーム)
        nonzero_indices: 非ゼロフレームの元インデックス
    """
    # 各フレームの絶対値合計を計算
    frame_sums = np.sum(np.abs(mel_spec), axis=0)
    zero_mask = frame_sums <= zero_threshold
    
    if not np.any(zero_mask):
        # ゼロフレームがない場合はそのまま返す
        nonzero_indices = np.arange(mel_spec.shape[1])
        return mel_spec, zero_mask, nonzero_indices
    
    print("Zero frames detected: {} out of {} ({:.2f}%)".format(
        np.sum(zero_mask), len(zero_mask), np.sum(zero_mask) / len(zero_mask) * 100))
    
    # 非ゼロフレームのインデックスを取得
    nonzero_indices = np.where(~zero_mask)[0]
    
    # 非ゼロフレームのみを抽出
    filtered_mel = mel_spec[:, nonzero_indices]
    
    print("Filtered mel shape: {} -> {}".format(mel_spec.shape, filtered_mel.shape))
    
    return filtered_mel, zero_mask, nonzero_indices


def reconstruct_audio_with_silence(filtered_audio, zero_mask, nonzero_indices, hop_size, original_length):
    """
    フィルタリングされた音声を元の長さに復元し、ゼロフレーム部分を静音にする
    
    Args:
        filtered_audio: 非ゼロフレームから生成された音声 (samples,)
        zero_mask: ゼロフレームのマスク (time_frames,)
        nonzero_indices: 非ゼロフレームの元インデックス
        hop_size: ホップサイズ
        original_length: 元の音声の長さ（サンプル数）
        
    Returns:
        restored_audio: 元の長さに復元された音声
    """
    # 元の長さの音声配列を0で初期化
    restored_audio = np.zeros(original_length, dtype=filtered_audio.dtype)
    
    # 各非ゼロフレームから生成された音声を元の位置に配置
    for i, original_frame_idx in enumerate(nonzero_indices):
        # フィルタリングされた音声の対応する部分を取得
        start_sample_filtered = i * hop_size
        end_sample_filtered = min((i + 1) * hop_size, len(filtered_audio))
        
        # 元の音声での対応する位置を計算
        start_sample_original = original_frame_idx * hop_size
        end_sample_original = min((original_frame_idx + 1) * hop_size, original_length)
        
        # フィルタリングされた音声から元の位置にコピー
        filtered_segment_length = end_sample_filtered - start_sample_filtered
        original_segment_length = end_sample_original - start_sample_original
        
        # 長さを調整してコピー
        copy_length = min(filtered_segment_length, original_segment_length)
        if copy_length > 0:
            restored_audio[start_sample_original:start_sample_original + copy_length] = \
                filtered_audio[start_sample_filtered:start_sample_filtered + copy_length]
    
    return restored_audio


def inference_binaural(a, h):
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    # Get list of files from left channel directory
    left_filelist = os.listdir(a.input_mels_left_dir)
    
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    
    with torch.no_grad():
        for i, filename in enumerate(left_filelist):
            # Load left and right mel spectrograms
            left_mel_path = os.path.join(a.input_mels_left_dir, filename)
            right_mel_path = os.path.join(a.input_mels_right_dir, filename)
            
            # Check if corresponding right channel file exists
            if not os.path.exists(right_mel_path):
                print("Warning: Right channel file not found for {}, skipping...".format(filename))
                continue
            
            # Load the mel spectrograms in .npy format
            x_left_orig = np.load(left_mel_path)
            x_right_orig = np.load(right_mel_path)
            
            # ゼロフレームを検出・除外
            if a.interpolate_zero_frames:
                print("Processing: {}".format(filename))
                x_left_filtered, zero_mask_left, nonzero_indices_left = detect_and_exclude_zero_frames(x_left_orig)
                x_right_filtered, zero_mask_right, nonzero_indices_right = detect_and_exclude_zero_frames(x_right_orig)
                
                x_left = x_left_filtered
                x_right = x_right_filtered
                
                # 元の音声長を計算
                original_audio_length = x_left_orig.shape[1] * h.hop_size
            else:
                x_left = x_left_orig
                x_right = x_right_orig
                zero_mask_left = np.zeros(x_left.shape[1], dtype=bool)
                zero_mask_right = np.zeros(x_right.shape[1], dtype=bool)
                nonzero_indices_left = np.arange(x_left.shape[1])
                nonzero_indices_right = np.arange(x_right.shape[1])
                original_audio_length = x_left.shape[1] * h.hop_size
            
            # Convert to torch tensors and move to device
            x_left = torch.FloatTensor(x_left).to(device)
            x_right = torch.FloatTensor(x_right).to(device)
            
            # Ensure proper dimensions (add batch dimension if needed)
            if len(x_left.shape) == 2:
                x_left = x_left.unsqueeze(0)
            if len(x_right.shape) == 2:
                x_right = x_right.unsqueeze(0)

            # Generate audio for left and right channels separately
            y_left = generator(x_left)
            y_right = generator(x_right)

            # Combine left and right channels into stereo audio
            audio_left_filtered = y_left.squeeze().cpu().numpy()
            audio_right_filtered = y_right.squeeze().cpu().numpy()
            
            # ゼロフレーム部分を静音として元の長さに復元
            if a.interpolate_zero_frames:
                audio_left = reconstruct_audio_with_silence(
                    audio_left_filtered, zero_mask_left, nonzero_indices_left, h.hop_size, original_audio_length)
                audio_right = reconstruct_audio_with_silence(
                    audio_right_filtered, zero_mask_right, nonzero_indices_right, h.hop_size, original_audio_length)
            else:
                audio_left = audio_left_filtered
                audio_right = audio_right_filtered
            
            # Create stereo audio by stacking left and right channels
            # Shape: (2, num_samples) where [0] is left, [1] is right
            stereo_audio = np.stack([audio_left, audio_right], axis=0)
            
            # Apply volume normalization
            stereo_audio = stereo_audio * MAX_WAV_VALUE
            stereo_audio = stereo_audio.astype("int16")
            
            # Transpose to (num_samples, 2) format for wav file
            stereo_audio = stereo_audio.T

            output_file = os.path.join(
                a.output_dir, os.path.splitext(filename)[0] + "_binaural_generated.wav"
            )
            write(output_file, h.sampling_rate, stereo_audio)
            print("Generated: {}".format(output_file))


def main():
    print("Initializing Binaural Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mels_left_dir", default="/home/h-okano/DiffBinaural/test_results/realBinaural_left_test_mix", 
                       help="Directory containing left channel mel spectrograms")
    parser.add_argument("--input_mels_right_dir", default="/home/h-okano/DiffBinaural/test_results/realBinaural_right_test_mix",
                       help="Directory containing right channel mel spectrograms")
    parser.add_argument("--output_dir", default="generated_realbinaural_files_mix")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)
    parser.add_argument("--interpolate_zero_frames", action="store_true", default=True,
                       help="Exclude zero frames during inference and restore as silence to avoid artifacts")

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference_binaural(a, h)


if __name__ == "__main__":
    main()

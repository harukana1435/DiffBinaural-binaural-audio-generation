#!/usr/bin/env python3
"""
メルスペクトログラムファイルからRMSE評価スクリプト
保存されたメルスペクトログラム(.npy)ファイルから直接RMSEを計算
"""

import os
import argparse
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import librosa
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}

def safe_statistics(data_list, metric_name):
    """
    Safely calculate statistics, filtering out NaN and infinite values.
    """
    # Convert to numpy array and filter out invalid values
    data_array = np.array(data_list)
    valid_data = data_array[np.isfinite(data_array)]
    
    if len(valid_data) == 0:
        print(f"Warning: No valid values for {metric_name}")
        return 0.0, 0.0, 0.0
    
    if len(valid_data) != len(data_array):
        print(f"Warning: {len(data_array) - len(valid_data)} invalid values filtered out for {metric_name}")
    
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0
    stderr_val = std_val / np.sqrt(len(valid_data)) if len(valid_data) > 0 else 0.0
    
    return float(mean_val), float(std_val), float(stderr_val)

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, center=False):
    """22050Hz用のメルスペクトログラム計算"""
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    device_key = str(y.device)
    
    if device_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=0.0, fmax=sampling_rate/2)
        mel_basis[device_key] = torch.from_numpy(mel).float().to(y.device)
    
    if device_key not in hann_window:
        hann_window[device_key] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[device_key],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[device_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def generate_gt_mel_spectrogram(audio_file, sr=22050, mel=80):
    """音声ファイルからメルスペクトログラムを生成"""
    # サンプリングレートに応じてSTFTパラメータを調整
    if sr == 16000:
        n_fft = 512
        hop_size = 160
        win_size = 512
    else:  # sr == 22050 or other rates
        n_fft = 1024
        hop_size = 256
        win_size = 1024
    
    # 音声を読み込み
    binaural_audio, _ = librosa.load(audio_file, sr=sr, mono=False)
    
    # 長さを調整（元のコードと同じ処理）
    binaural_audio = binaural_audio[:, 8*256:-8*256]
    
    # 左チャンネル
    left_channel = torch.FloatTensor(torch.from_numpy(binaural_audio[0, :]).unsqueeze(0))
    left_mel = mel_spectrogram(left_channel, n_fft, mel, sr, hop_size, win_size)
    
    # 右チャンネル
    right_channel = torch.FloatTensor(torch.from_numpy(binaural_audio[1, :]).unsqueeze(0))
    right_mel = mel_spectrogram(right_channel, n_fft, mel, sr, hop_size, win_size)
    
    # (1, mel_bins, time_frames) -> (mel_bins, time_frames)に変換
    return left_mel.squeeze(0).numpy(), right_mel.squeeze(0).numpy()

def calculate_mel_rmse_from_files(pred_left_file, pred_right_file, gt_left_mel, gt_right_mel):
    """メルスペクトログラムファイルからRMSEを計算"""
    try:
        # 予測メルスペクトログラムを読み込み
        pred_left_mel = np.load(pred_left_file)
        pred_right_mel = np.load(pred_right_file)
        
        # 前後8フレームをカット（音声の8*256サンプルに対応）
        pred_left_mel = pred_left_mel[:, 8:-8]
        pred_right_mel = pred_right_mel[:, 8:-8]
        gt_left_mel = gt_left_mel[:, 8:-8]
        gt_right_mel = gt_right_mel[:, 8:-8]
        
        # 形状を確認して調整
        min_time_frames = min(pred_left_mel.shape[1], pred_right_mel.shape[1], 
                             gt_left_mel.shape[1], gt_right_mel.shape[1])
        
        pred_left_mel = pred_left_mel[:, :min_time_frames]
        pred_right_mel = pred_right_mel[:, :min_time_frames]
        gt_left_mel = gt_left_mel[:, :min_time_frames]
        gt_right_mel = gt_right_mel[:, :min_time_frames]
        
        # TensorFlowテンソルに変換
        pred_left_tensor = torch.FloatTensor(pred_left_mel)
        pred_right_tensor = torch.FloatTensor(pred_right_mel)
        gt_left_tensor = torch.FloatTensor(gt_left_mel)
        gt_right_tensor = torch.FloatTensor(gt_right_mel)
        
        # 各チャンネルのRMSEを計算
        left_rmse = torch.sqrt(F.mse_loss(gt_left_tensor, pred_left_tensor))
        right_rmse = torch.sqrt(F.mse_loss(gt_right_tensor, pred_right_tensor))
        
        # 平均RMSE
        avg_rmse = (left_rmse + right_rmse) / 2.0
        
        return float(avg_rmse), float(left_rmse), float(right_rmse)
        
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        return None, None, None

def find_matching_files(base_filename, left_dir, right_dir):
    """ベースファイル名から対応する左右のメルスペクトログラムファイルを見つける"""
    # basketball_m40_1.npy のような形式
    left_file = os.path.join(left_dir, base_filename)
    right_file = os.path.join(right_dir, base_filename)
    
    if os.path.exists(left_file) and os.path.exists(right_file):
        return left_file, right_file
    else:
        return None, None

def find_matching_gt_file(base_filename, gt_dir):
    """ベースファイル名から対応するGT音声ファイルを見つける"""
    # basketball_m40_1.npy -> basketball_0_1.wav
    # "m40" を "0" に置き換え
    gt_filename = base_filename.replace('_m40_', '_0_').replace('.npy', '.wav')
    gt_path = os.path.join(gt_dir, gt_filename)
    
    if os.path.exists(gt_path):
        return gt_path
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate mel spectrogram RMSE from .npy files")
    parser.add_argument('--pred_left_dir', default='/home/h-okano/DiffBinaural/test_results/realBinaural_left_test_comp',
                       help='Directory containing predicted left channel mel spectrograms')
    parser.add_argument('--pred_right_dir', default='/home/h-okano/DiffBinaural/test_results/realBinaural_right_test_comp',
                       help='Directory containing predicted right channel mel spectrograms')
    parser.add_argument('--gt_dir', default='/home/h-okano/real_dataset/processed/binaural_audios_22050Hz_comp',
                       help='Directory containing ground truth binaural audio files')
    parser.add_argument('--audio_sampling_rate', default=22050, type=int, help='Audio sampling rate')
    parser.add_argument('--output_csv', default='/home/h-okano/DiffBinaural/mel_rmse_evaluation_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--num_mels', default=80, type=int, help='Number of mel bands')
    
    args = parser.parse_args()
    
    # 結果リスト
    results = []
    
    # 予測ファイルを取得（左チャンネルディレクトリから）
    pred_files = glob.glob(os.path.join(args.pred_left_dir, '*.npy'))
    
    # # basketball_m40_0 から basketball_m40_10 までをフィルタリング
    # target_files = [f'basketball_m40_{i}.npy' for i in range(11)]  # 0から10まで
    # pred_files = [f for f in pred_files if os.path.basename(f) in target_files]
    # pred_files = sorted(pred_files)  # ソート
    
    # print(f"Found {len(pred_files)} prediction files (basketball_m40_0 to basketball_m40_10)")
    # print(f"Target files: {[os.path.basename(f) for f in pred_files]}")
    
    # GTメルスペクトログラムのキャッシュ
    gt_mel_cache = {}
    
    for pred_left_file in tqdm(pred_files, desc="Evaluating"):
        base_filename = os.path.basename(pred_left_file)
        
        # 対応する右チャンネルファイルを見つける
        pred_left_path, pred_right_path = find_matching_files(base_filename, args.pred_left_dir, args.pred_right_dir)
        if pred_left_path is None or pred_right_path is None:
            print(f"Warning: No matching right channel file for {base_filename}")
            continue
        
        # 対応するGTファイルを見つける
        gt_file = find_matching_gt_file(base_filename, args.gt_dir)
        if gt_file is None:
            print(f"Warning: No GT file found for {base_filename}")
            continue
        
        try:
            # GTメルスペクトログラムを生成またはキャッシュから取得
            if gt_file not in gt_mel_cache:
                gt_left_mel, gt_right_mel = generate_gt_mel_spectrogram(
                    gt_file, sr=args.audio_sampling_rate, mel=args.num_mels
                )
                gt_mel_cache[gt_file] = (gt_left_mel, gt_right_mel)
            else:
                gt_left_mel, gt_right_mel = gt_mel_cache[gt_file]
            
            # RMSEを計算
            avg_rmse, left_rmse, right_rmse = calculate_mel_rmse_from_files(
                pred_left_path, pred_right_path, gt_left_mel, gt_right_mel
            )
            
            if avg_rmse is not None:
                # 結果を保存
                results.append({
                    'filename': base_filename,
                    'mel_rmse_avg': avg_rmse,
                    'mel_rmse_left': left_rmse,
                    'mel_rmse_right': right_rmse
                })
            
        except Exception as e:
            print(f"Error processing {base_filename}: {e}")
            continue
    
    # 結果をDataFrameに変換
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No valid results found!")
        return
    
    # 統計を計算・表示
    print("\n" + "="*60)
    print("MEL SPECTROGRAM RMSE EVALUATION RESULTS")
    print("="*60)
    
    metrics = ['mel_rmse_avg', 'mel_rmse_left', 'mel_rmse_right']
    
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].values
            mean_val, std_val, stderr_val = safe_statistics(values, metric)
            print(f"{metric}: {mean_val:.6f} ± {std_val:.6f} (SE: {stderr_val:.6f})")
    
    # CSVに保存
    df.to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to: {args.output_csv}")
    
    # 統計サマリーも保存
    summary_file = args.output_csv.replace('.csv', '_summary.csv')
    summary_data = []
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].values
            mean_val, std_val, stderr_val = safe_statistics(values, metric)
            summary_data.append({
                'metric': metric,
                'mean': mean_val,
                'std': std_val,
                'stderr': stderr_val,
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")

if __name__ == '__main__':
    main()

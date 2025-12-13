#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バイノーラル音声評価スクリプト（22050Hz用）
生成されたバイノーラル音声と本物のバイノーラル音声を比較して各種メトリクスを計算
"""

import os
import librosa
import argparse
import numpy as np
from numpy import linalg as LA
from scipy.signal import hilbert
import statistics as stat
import torch
from librosa.filters import mel as librosa_mel_fn
import mir_eval
import torch.nn.functional as F
import pandas as pd
import glob
from tqdm import tqdm

mel_basis = {}
hann_window = {}

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

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

def align_signals(gt_binaural, predicted_binaural):
    """
    相互相関を用いて2つのバイノーラル信号の時間軸を合わせる。
    """
    # 入力チェック（最低限）
    if gt_binaural.ndim != 2 or predicted_binaural.ndim != 2 or gt_binaural.shape[0] != 2 or predicted_binaural.shape[0] != 2:
        raise ValueError("入力は(2, N)の形状を持つnumpy配列である必要があります。")
    if gt_binaural.shape[1] == 0 or predicted_binaural.shape[1] == 0:
         print("警告: 一方または両方の信号が空です。アライメントせずに返します。")
         min_len = min(gt_binaural.shape[1], predicted_binaural.shape[1])
         return gt_binaural[:, :min_len], predicted_binaural[:, :min_len], 0

    # アライメントには片方のチャンネル（例：左チャンネル）を使用
    gt_channel = gt_binaural[0, :]
    pred_channel = predicted_binaural[0, :]

    # 相互相関を計算
    correlation = np.correlate(gt_channel, pred_channel, mode='full')
    lag_samples = np.argmax(correlation) - (len(pred_channel) - 1)

    print(f"検出されたラグ: {lag_samples} サンプル")

    # ラグに基づいて信号をスライスしてアライメント
    if lag_samples > 0:
        if lag_samples >= predicted_binaural.shape[1]:
             print(f"警告: ラグ({lag_samples})が予測信号長({predicted_binaural.shape[1]})以上です。")
             predicted_aligned = predicted_binaural[:, :0]
             gt_aligned = gt_binaural[:, :predicted_binaural.shape[1] - lag_samples] if predicted_binaural.shape[1] - lag_samples > 0 else gt_binaural[:, :0]
        else:
            predicted_aligned = predicted_binaural[:, lag_samples:]
            gt_aligned = gt_binaural[:, :predicted_binaural.shape[1] - lag_samples]

    elif lag_samples < 0:
        abs_lag = abs(lag_samples)
        if abs_lag >= gt_binaural.shape[1]:
             print(f"警告: ラグ({abs_lag})がGT信号長({gt_binaural.shape[1]})以上です。")
             gt_aligned = gt_binaural[:, :0]
             predicted_aligned = predicted_binaural[:, :gt_binaural.shape[1] - abs_lag] if gt_binaural.shape[1] - abs_lag > 0 else predicted_binaural[:, :0]
        else:
             gt_aligned = gt_binaural[:, abs_lag:]
             predicted_aligned = predicted_binaural[:, :gt_binaural.shape[1] - abs_lag]
    else:
        gt_aligned = gt_binaural
        predicted_aligned = predicted_binaural

    # スライス後、最終的に長さを揃える
    min_len = min(gt_aligned.shape[1], predicted_aligned.shape[1])
    if min_len <= 0:
         print("警告: アライメント後の信号長が0以下です。空の配列を返します。")
         return gt_binaural[:,:0], predicted_binaural[:,:0], lag_samples

    gt_aligned = gt_aligned[:, :min_len]
    predicted_aligned = predicted_aligned[:, :min_len]

    return gt_aligned, predicted_aligned, lag_samples

def compute_sar_sir_sdr(predicted_binaural, gt_binaural):
    """
    Computes the SAR, SIR, and SDR for the predicted binaural signal.
    """
    # Ensure both signals are the same length
    min_length = min(predicted_binaural.shape[1], gt_binaural.shape[1])
    predicted_binaural = predicted_binaural[:, :min_length]
    gt_binaural = gt_binaural[:, :min_length]

    try:
        # Compute SDR, SIR, SAR
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(gt_binaural, predicted_binaural)
        # Return the average over both channels
        return float(np.mean(sar)), float(np.mean(sir)), float(np.mean(sdr))
    except Exception as e:
        print(f"Error computing SDR: {e}")
        return 0.0, 0.0, 0.0

def STFT_RMSE_distance(predicted_binaural, gt_binaural, sr=22050):
    """STFT RMSE距離を計算（サンプリングレートに応じてパラメータ調整）"""
    # サンプリングレートに応じてSTFTパラメータを調整
    if sr == 16000:
        n_fft = 512
        hop_length = 160
        win_length = 400
    else:  # sr == 22050 or other rates
        n_fft = 512
        hop_length = 256
        win_length = 512
    
    # Channel 1
    predicted_spect_channel1 = librosa.core.stft(
        np.asfortranarray(predicted_binaural[0,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    gt_spect_channel1 = librosa.core.stft(
        np.asfortranarray(gt_binaural[0,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.sqrt(np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2)))

    # Channel 2
    predicted_spect_channel2 = librosa.core.stft(
        np.asfortranarray(predicted_binaural[1,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    gt_spect_channel2 = librosa.core.stft(
        np.asfortranarray(gt_binaural[1,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    
    real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
    predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
    gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
    channel2_distance = np.sqrt(np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2)))

    # Average the distance between two channels
    stft_rmse_distance = (channel1_distance + channel2_distance) / 2.0
    return float(stft_rmse_distance)

def STFT_phase_and_magnitude_RMSE_distance(predicted_binaural, gt_binaural, sr=22050):
    """STFT位相・振幅RMSE距離を計算"""
    # サンプリングレートに応じてSTFTパラメータを調整
    if sr == 16000:
        n_fft = 512
        hop_length = 160
        win_length = 400
    else:  # sr == 22050 or other rates
        n_fft = 512
        hop_length = 256
        win_length = 512
    
    def calculate_distance(predicted_spect, gt_spect):
        # Extract magnitude and phase
        predicted_magnitude = np.abs(predicted_spect)
        gt_magnitude = np.abs(gt_spect)
        
        predicted_phase = np.angle(predicted_spect)
        gt_phase = np.angle(gt_spect)
        
        # Magnitude RMSE distance (only magnitude part)
        magnitude_distance = np.sqrt(np.mean(np.power((predicted_magnitude - gt_magnitude), 2)))
        
        # Phase RMSE distance (only phase part)
        phase_distance = np.sqrt(np.mean(np.power((predicted_phase - gt_phase), 2)))
        
        return magnitude_distance, phase_distance
    
    # Channel 1
    predicted_spect_channel1 = librosa.core.stft(
        np.asfortranarray(predicted_binaural[0,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    gt_spect_channel1 = librosa.core.stft(
        np.asfortranarray(gt_binaural[0,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    
    # Channel 2
    predicted_spect_channel2 = librosa.core.stft(
        np.asfortranarray(predicted_binaural[1,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    gt_spect_channel2 = librosa.core.stft(
        np.asfortranarray(gt_binaural[1,:]), 
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    
    # Calculate the magnitude and phase distances for both channels
    mag_dist_channel1, phase_dist_channel1 = calculate_distance(predicted_spect_channel1, gt_spect_channel1)
    mag_dist_channel2, phase_dist_channel2 = calculate_distance(predicted_spect_channel2, gt_spect_channel2)
    
    # Averaging the distances for both channels
    total_magnitude_distance = (mag_dist_channel1 + mag_dist_channel2) / 2.0
    total_phase_distance = (phase_dist_channel1 + phase_dist_channel2) / 2.0
    
    return float(total_magnitude_distance), float(total_phase_distance)

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

def MEL_RMSE_distance(predicted_binaural, gt_binaural, mel=80, sr=22050):
    """メルスペクトログラムRMSE距離を計算"""
    # サンプリングレートに応じてSTFTパラメータを調整
    if sr == 16000:
        n_fft = 512
        hop_size = 160
        win_size = 512
    else:  # sr == 22050 or other rates
        n_fft = 1024
        hop_size = 256
        win_size = 1024
    
    #channel1
    predicted_spect_channel1 = torch.FloatTensor(torch.from_numpy(predicted_binaural[0, :]).unsqueeze(0))
    predicted_spct_channnel1_mel = mel_spectrogram(predicted_spect_channel1, n_fft, mel, sr, hop_size, win_size)
    gt_spect_channel1 = torch.FloatTensor(torch.from_numpy(gt_binaural[0, :]).unsqueeze(0))
    gt_spct_channnel1_mel = mel_spectrogram(gt_spect_channel1, n_fft, mel, sr, hop_size, win_size)
    channel1_l2_distance = torch.sqrt(F.mse_loss(gt_spct_channnel1_mel, predicted_spct_channnel1_mel))
    
    predicted_spect_channel2 = torch.FloatTensor(torch.from_numpy(predicted_binaural[1, :]).unsqueeze(0))
    predicted_spct_channnel2_mel = mel_spectrogram(predicted_spect_channel2, n_fft, mel, sr, hop_size, win_size)
    gt_spect_channel2 = torch.FloatTensor(torch.from_numpy(gt_binaural[1, :]).unsqueeze(0))
    gt_spct_channnel2_mel = mel_spectrogram(gt_spect_channel2, n_fft, mel, sr, hop_size, win_size)
    channel2_l2_distance = torch.sqrt(F.mse_loss(gt_spct_channnel2_mel, predicted_spct_channnel2_mel))
    
    #average the distance between two channels
    mel_rmse_distance = (channel1_l2_distance + channel2_l2_distance) / 2.0
    return float(mel_rmse_distance)

def Envelope_distance(predicted_binaural, gt_binaural):
    """エンベロープ距離を計算"""
    # Channel 1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    
    min_length = min(len(gt_env_channel1), len(pred_env_channel1))
    gt_env_channel1 = gt_env_channel1[:min_length]
    pred_env_channel1 = pred_env_channel1[:min_length]
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    # Channel 2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
    
    min_length = min(len(gt_env_channel2), len(pred_env_channel2))
    gt_env_channel2 = gt_env_channel2[:min_length]
    pred_env_channel2 = pred_env_channel2[:min_length]
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    envelope_distance = (channel1_distance + channel2_distance) / 2.0
    return float(envelope_distance)

def calculate_snr(clean_audio, noisy_audio):
    
    """Calculates the Signal-to-Noise Ratio (SNR) in dB for binaural audio.

    Args:
        clean_audio: Clean binaural audio signal (2, N) NumPy array.
        noisy_audio: Noisy binaural audio signal (2, N) NumPy array.

    Returns:
        SNR in dB (float). Returns -np.inf if the signal power is zero.
    """
    # Ensure both signals have the same length
    min_length = min(clean_audio.shape[1], noisy_audio.shape[1])
    clean_audio = clean_audio[:, :min_length]
    noisy_audio = noisy_audio[:, :min_length]
    
    # Calculate power for both channels combined
    signal_power = np.sum(clean_audio**2)
    noise_power = np.sum((noisy_audio - clean_audio)**2)

    if signal_power == 0:
        return -np.inf

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def compute_iacc(binaural_audio, frame_size=1024, hop_length=512):
    """
    Computes the Inter-aural Cross-Correlation (IACC) for binaural audio.
    
    Parameters:
    - binaural_audio: (2, N) numpy array, where N is the number of samples
    - frame_size: Size of each frame for STFT analysis
    - hop_length: Hop length for STFT analysis
    
    Returns:
    - iacc_mean: Mean IACC value across all frames
    - iacc_values: Array of IACC values for each frame
    """
    left_channel = binaural_audio[0, :]
    right_channel = binaural_audio[1, :]
    
    # Ensure both channels have the same length
    min_length = min(len(left_channel), len(right_channel))
    left_channel = left_channel[:min_length]
    right_channel = right_channel[:min_length]
    
    # Calculate number of frames
    n_frames = (min_length - frame_size) // hop_length + 1
    iacc_values = []
    
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_size
        
        left_frame = left_channel[start_idx:end_idx]
        right_frame = right_channel[start_idx:end_idx]
        
        # Calculate cross-correlation
        cross_corr = np.correlate(left_frame, right_frame, mode='full')
        
        # Calculate auto-correlations
        left_auto_corr = np.correlate(left_frame, left_frame, mode='full')
        right_auto_corr = np.correlate(right_frame, right_frame, mode='full')
        
        # Get the maximum values (at zero lag)
        max_cross_corr = np.max(np.abs(cross_corr))
        max_left_auto = np.max(left_auto_corr)
        max_right_auto = np.max(right_auto_corr)
        
        # Calculate IACC
        denominator = np.sqrt(max_left_auto * max_right_auto)
        if denominator > 1e-10:  # Use small threshold to avoid numerical issues
            iacc = max_cross_corr / denominator
            # Clamp IACC to valid range [0, 1]
            iacc = np.clip(iacc, 0.0, 1.0)
        else:
            iacc = 0.0
            
        iacc_values.append(iacc)
    
    iacc_values = np.array(iacc_values)
    iacc_mean = np.mean(iacc_values)
    
    return iacc_mean, iacc_values

def compute_iacc_difference(predicted_binaural, gt_binaural, frame_size=1024, hop_length=512):
    """
    Computes the difference in IACC between predicted and ground truth binaural audio.
    
    Parameters:
    - predicted_binaural: (2, N) numpy array, predicted binaural signal
    - gt_binaural: (2, N) numpy array, ground truth binaural signal
    - frame_size: Size of each frame for STFT analysis
    - hop_length: Hop length for STFT analysis
    
    Returns:
    - iacc_diff_mean: Mean difference in IACC values
    - pred_iacc_mean: Mean IACC of predicted signal
    - gt_iacc_mean: Mean IACC of ground truth signal
    """
    # Ensure both signals have the same length
    min_length = min(predicted_binaural.shape[1], gt_binaural.shape[1])
    predicted_binaural = predicted_binaural[:, :min_length]
    gt_binaural = gt_binaural[:, :min_length]
    
    # Calculate IACC for both signals
    pred_iacc_mean, pred_iacc_values = compute_iacc(predicted_binaural, frame_size, hop_length)
    gt_iacc_mean, gt_iacc_values = compute_iacc(gt_binaural, frame_size, hop_length)
    
    # Calculate frame-wise differences
    min_frames = min(len(pred_iacc_values), len(gt_iacc_values))
    pred_iacc_values = pred_iacc_values[:min_frames]
    gt_iacc_values = gt_iacc_values[:min_frames]
    
    iacc_diff_values = np.abs(pred_iacc_values - gt_iacc_values)
    iacc_diff_mean = np.mean(iacc_diff_values)
    
    return iacc_diff_mean, pred_iacc_mean, gt_iacc_mean

def compute_ild_error(predicted_binaural, gt_binaural, frame_size=1024, hop_length=512, sr=22050):
    """
    Computes frequency-dependent Inter-aural Level Difference (ILD) error with perceptual weighting.
    Higher frequencies (>1.5kHz) are weighted more heavily as they are more important for ILD perception.
    
    Parameters:
    - predicted_binaural: (2, N) numpy array, predicted binaural signal
    - gt_binaural: (2, N) numpy array, ground truth binaural signal
    - frame_size: Size of each frame for STFT analysis (default: 1024)
    - hop_length: Hop length for STFT analysis (default: 512)
    - sr: Sampling rate
    
    Returns:
    - ild_error_mean: Mean frequency-weighted ILD error in dB
    - pred_ild_mean: Mean ILD of predicted signal
    - gt_ild_mean: Mean ILD of ground truth signal
    """
    # Ensure both signals have the same length
    min_length = min(predicted_binaural.shape[1], gt_binaural.shape[1])
    predicted_binaural = predicted_binaural[:, :min_length]
    gt_binaural = gt_binaural[:, :min_length]
    
    # Define frequency bands for ILD computation (higher frequencies emphasized)
    # Bands: 0-500Hz, 500-1000Hz, 1000-1500Hz, 1500-3000Hz, 3000-6000Hz, 6000-11025Hz
    nyquist = sr / 2
    freq_bands = [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 3000),
        (3000, 6000),
        (6000, nyquist)
    ]
    
    # Perceptual weights: higher frequencies more important for ILD
    # Based on duplex theory: ILD dominant above 1.5kHz
    band_weights = np.array([0.5, 0.7, 1.0, 1.5, 1.5, 1.3])
    band_weights = band_weights / np.sum(band_weights)  # Normalize
    
    n_fft = frame_size
    n_frames = (min_length - frame_size) // hop_length + 1
    
    pred_ild_weighted = []
    gt_ild_weighted = []
    
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_size
        
        # Extract frames
        left_pred_frame = predicted_binaural[0, start_idx:end_idx]
        right_pred_frame = predicted_binaural[1, start_idx:end_idx]
        left_gt_frame = gt_binaural[0, start_idx:end_idx]
        right_gt_frame = gt_binaural[1, start_idx:end_idx]
        
        # Compute STFT for frequency-dependent analysis
        pred_left_stft = np.fft.rfft(left_pred_frame * np.hanning(frame_size))
        pred_right_stft = np.fft.rfft(right_pred_frame * np.hanning(frame_size))
        gt_left_stft = np.fft.rfft(left_gt_frame * np.hanning(frame_size))
        gt_right_stft = np.fft.rfft(right_gt_frame * np.hanning(frame_size))
        
        freqs = np.fft.rfftfreq(frame_size, 1/sr)
        
        pred_ild_frame = 0.0
        gt_ild_frame = 0.0
        
        # Compute ILD for each frequency band with weighting
        for (f_low, f_high), weight in zip(freq_bands, band_weights):
            # Find frequency bins in this band
            freq_mask = (freqs >= f_low) & (freqs < f_high)
            
            if not np.any(freq_mask):
                continue
            
            # Calculate energy in this band for each channel
            pred_left_energy = np.sum(np.abs(pred_left_stft[freq_mask])**2)
            pred_right_energy = np.sum(np.abs(pred_right_stft[freq_mask])**2)
            gt_left_energy = np.sum(np.abs(gt_left_stft[freq_mask])**2)
            gt_right_energy = np.sum(np.abs(gt_right_stft[freq_mask])**2)
            
            eps = 1e-10
            # ILD in dB for this band
            pred_ild_band = 10 * np.log10((pred_left_energy + eps) / (pred_right_energy + eps))
            gt_ild_band = 10 * np.log10((gt_left_energy + eps) / (gt_right_energy + eps))
            
            # Accumulate weighted ILD
            pred_ild_frame += weight * pred_ild_band
            gt_ild_frame += weight * gt_ild_band
        
        pred_ild_weighted.append(pred_ild_frame)
        gt_ild_weighted.append(gt_ild_frame)
    
    pred_ild_weighted = np.array(pred_ild_weighted)
    gt_ild_weighted = np.array(gt_ild_weighted)
    
    # Calculate statistics
    pred_ild_mean = np.mean(pred_ild_weighted)
    gt_ild_mean = np.mean(gt_ild_weighted)
    
    # Calculate ILD error
    ild_error_values = np.abs(pred_ild_weighted - gt_ild_weighted)
    ild_error_mean = np.mean(ild_error_values)
    
    return ild_error_mean, pred_ild_mean, gt_ild_mean

def compute_itd_error(predicted_binaural, gt_binaural, frame_size=1024, hop_length=512, sr=22050):
    """
    Computes ITD error using GCC-PHAT (Generalized Cross-Correlation with Phase Transform) method
    with physiological delay constraints and subsample interpolation.
    
    GCC-PHAT is more robust to reverberation and noise than simple cross-correlation.
    
    Parameters:
    - predicted_binaural: (2, N) numpy array, predicted binaural signal
    - gt_binaural: (2, N) numpy array, ground truth binaural signal
    - frame_size: Size of each frame for analysis (default: 1024)
    - hop_length: Hop length for analysis (default: 512)
    - sr: Sampling rate
    
    Returns:
    - itd_error_mean: Mean ITD error in samples
    - itd_error_ms_mean: Mean ITD error in milliseconds
    - pred_itd_mean: Mean ITD of predicted signal (in samples)
    - gt_itd_mean: Mean ITD of ground truth signal (in samples)
    """
    # Ensure both signals have the same length
    min_length = min(predicted_binaural.shape[1], gt_binaural.shape[1])
    predicted_binaural = predicted_binaural[:, :min_length]
    gt_binaural = gt_binaural[:, :min_length]
    
    # Physiological ITD constraint: typical head diameter ~17cm => max ITD ~0.7ms
    # At 22050Hz: 0.7ms = 15.4 samples. Use ±20 samples as safe margin
    max_itd_samples = int(0.001 * sr)  # ±1ms window (22 samples at 22050Hz)
    
    n_frames = (min_length - frame_size) // hop_length + 1
    pred_itd_values = []
    gt_itd_values = []
    
    def compute_gcc_phat_itd(left_frame, right_frame, max_delay):
        """
        Compute ITD using GCC-PHAT method with subsample interpolation.
        
        GCC-PHAT normalizes by magnitude spectrum, emphasizing phase differences.
        This makes it robust to room reverberation and spectral coloring.
        """
        # FFT of both channels
        n = len(left_frame)
        left_fft = np.fft.fft(left_frame, n=2*n)  # Zero-pad for linear convolution
        right_fft = np.fft.fft(right_frame, n=2*n)
        
        # Cross-power spectrum
        cross_spectrum = left_fft * np.conj(right_fft)
        
        # PHAT weighting: normalize by magnitude
        eps = 1e-10
        gcc_phat_spectrum = cross_spectrum / (np.abs(cross_spectrum) + eps)
        
        # Inverse FFT to get correlation
        gcc_phat = np.fft.ifft(gcc_phat_spectrum)
        gcc_phat = np.real(gcc_phat)
        
        # Rearrange to center zero-lag
        gcc_phat = np.concatenate([gcc_phat[len(gcc_phat)//2:], gcc_phat[:len(gcc_phat)//2]])
        
        # Apply physiological delay window constraint
        center = len(gcc_phat) // 2
        window_start = max(0, center - max_delay)
        window_end = min(len(gcc_phat), center + max_delay + 1)
        
        # Find peak within window
        windowed_gcc = gcc_phat[window_start:window_end]
        if len(windowed_gcc) == 0:
            return 0.0
        
        peak_idx_local = np.argmax(windowed_gcc)
        peak_idx_global = window_start + peak_idx_local
        
        # Subsample interpolation using parabolic interpolation
        # Improves accuracy beyond integer sample precision
        if peak_idx_global > 0 and peak_idx_global < len(gcc_phat) - 1:
            # Three-point parabolic interpolation
            alpha = gcc_phat[peak_idx_global - 1]
            beta = gcc_phat[peak_idx_global]
            gamma = gcc_phat[peak_idx_global + 1]
            
            # Parabola vertex offset
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-10:
                p = 0.5 * (alpha - gamma) / denom
                # Limit interpolation offset to [-0.5, 0.5]
                p = np.clip(p, -0.5, 0.5)
            else:
                p = 0.0
            
            itd_refined = (peak_idx_global - center) + p
        else:
            itd_refined = float(peak_idx_global - center)
        
        return itd_refined
    
    # Process each frame
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_size
        
        # Extract frames
        left_pred_frame = predicted_binaural[0, start_idx:end_idx]
        right_pred_frame = predicted_binaural[1, start_idx:end_idx]
        left_gt_frame = gt_binaural[0, start_idx:end_idx]
        right_gt_frame = gt_binaural[1, start_idx:end_idx]
        
        # Apply window to reduce spectral leakage
        window = np.hanning(frame_size)
        left_pred_frame = left_pred_frame * window
        right_pred_frame = right_pred_frame * window
        left_gt_frame = left_gt_frame * window
        right_gt_frame = right_gt_frame * window
        
        # Compute ITD using GCC-PHAT
        pred_itd = compute_gcc_phat_itd(left_pred_frame, right_pred_frame, max_itd_samples)
        gt_itd = compute_gcc_phat_itd(left_gt_frame, right_gt_frame, max_itd_samples)
        
        pred_itd_values.append(pred_itd)
        gt_itd_values.append(gt_itd)
    
    pred_itd_values = np.array(pred_itd_values)
    gt_itd_values = np.array(gt_itd_values)
    
    # Calculate mean ITD
    pred_itd_mean = np.mean(pred_itd_values)
    gt_itd_mean = np.mean(gt_itd_values)
    
    # Calculate ITD error
    itd_error_values = np.abs(pred_itd_values - gt_itd_values)
    itd_error_mean = np.mean(itd_error_values)
    
    # Convert to milliseconds
    itd_error_ms_mean = itd_error_mean / sr * 1000
    
    return itd_error_mean, itd_error_ms_mean, pred_itd_mean, gt_itd_mean

def find_matching_gt_file(generated_filename, gt_dir):
    """生成ファイルに対応するGTファイルを見つける"""
    # basketball_m40_10_binaural_generated.wav -> basketball_m40_10.wav
    base_name = generated_filename.replace('_binaural_generated.wav', '.wav')
    gt_path = os.path.join(gt_dir, base_name)
    
    if os.path.exists(gt_path):
        return gt_path
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate binaural audio quality")
    parser.add_argument('--generated_dir', default='/home/h-okano/bigvgan/generated_realbinaural_files_mix',
                       help='Directory containing generated binaural audio files')
    parser.add_argument('--gt_dir', default='/home/h-okano/real_dataset/processed/binaural_audios_22050Hz',
                       help='Directory containing ground truth binaural audio files')
    parser.add_argument('--audio_sampling_rate', default=22050, type=int, help='Audio sampling rate')
    parser.add_argument('--normalization', default=False, action='store_true', help='Apply normalization')
    parser.add_argument('--output_csv', default='/home/h-okano/DiffBinaural/fairplay_evaluation_results.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # メトリクスリスト
    results = []
    
    # 生成ファイルを取得
    generated_files = glob.glob(os.path.join(args.generated_dir, '*_binaural_generated.wav'))
    print(f"Found {len(generated_files)} generated files")
    
    # 10%だけ評価（最初の26ファイル）
    num_files_to_process = max(1, len(generated_files)) # 10%だけ評価したいなら// 10
    generated_files = generated_files[:num_files_to_process]
    print(f"Processing {len(generated_files)} files (10% sample)")
    
    for generated_file in tqdm(generated_files, desc="Evaluating"):
        filename = os.path.basename(generated_file)
        
        # 対応するGTファイルを見つける
        gt_file = find_matching_gt_file(filename, args.gt_dir)
        if gt_file is None:
            print(f"Warning: No GT file found for {filename}")
            continue
        
        try:
            # 音声を読み込み
            predicted_binaural, _ = librosa.load(generated_file, sr=args.audio_sampling_rate, mono=False)
            gt_binaural, _ = librosa.load(gt_file, sr=args.audio_sampling_rate, mono=False)
            
            # 長さを揃える
            predicted_binaural = predicted_binaural[:, 8*256:-8*256]
            gt_binaural = gt_binaural[:, 8*256:-8*256]
            min_length = min(predicted_binaural.shape[1], gt_binaural.shape[1])
            predicted_binaural = predicted_binaural[:, :min_length]
            gt_binaural = gt_binaural[:, :min_length]
            
            if args.normalization:
                predicted_binaural = normalize(predicted_binaural)
                gt_binaural = normalize(gt_binaural)
            
            # 各メトリクスを計算
            mel_distance = MEL_RMSE_distance(predicted_binaural, gt_binaural, sr=args.audio_sampling_rate)
            stft_distance = STFT_RMSE_distance(predicted_binaural, gt_binaural, sr=args.audio_sampling_rate)
            envelope_distance = Envelope_distance(predicted_binaural, gt_binaural)
            mag_distance, phase_distance = STFT_phase_and_magnitude_RMSE_distance(predicted_binaural, gt_binaural, sr=args.audio_sampling_rate)
            sar, sir, sdr = compute_sar_sir_sdr(predicted_binaural, gt_binaural)
            snr = calculate_snr(gt_binaural, predicted_binaural)
            iacc_distance, pred_iacc, gt_iacc = compute_iacc_difference(predicted_binaural, gt_binaural)
            ild_error, pred_ild, gt_ild = compute_ild_error(predicted_binaural, gt_binaural, sr=args.audio_sampling_rate)
            itd_error_samples, itd_error_ms, pred_itd, gt_itd = compute_itd_error(predicted_binaural, gt_binaural, sr=args.audio_sampling_rate)
            
            # 結果を保存
            results.append({
                'filename': filename,
                'mel_rmse_distance': mel_distance,
                'stft_rmse_distance': stft_distance,
                'envelope_distance': envelope_distance,
                'magnitude_rmse_distance': mag_distance,
                'phase_rmse_distance': phase_distance,
                'sar': sar,
                'sir': sir,
                'sdr': sdr,
                'snr': snr,
                'iacc_distance': iacc_distance,
                'predicted_iacc': pred_iacc,
                'gt_iacc': gt_iacc,
                'ild_error': ild_error,
                'predicted_ild': pred_ild,
                'gt_ild': gt_ild,
                'itd_error_samples': itd_error_samples,
                'itd_error_ms': itd_error_ms,
                'predicted_itd': pred_itd,
                'gt_itd': gt_itd
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # 結果をDataFrameに変換
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No valid results found!")
        return
    
    # 統計を計算・表示
    print("\n" + "="*60)
    print("BINAURAL AUDIO EVALUATION RESULTS")
    print("="*60)
    
    metrics = ['mel_rmse_distance', 'stft_rmse_distance', 'envelope_distance', 
               'magnitude_rmse_distance', 'phase_rmse_distance', 'sar', 'sir', 'sdr', 'snr',
               'iacc_distance', 'predicted_iacc', 'gt_iacc',
               'ild_error', 'predicted_ild', 'gt_ild',
               'itd_error_samples', 'itd_error_ms', 'predicted_itd', 'gt_itd']
    
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
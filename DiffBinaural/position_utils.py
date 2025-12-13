"""
Position Utilities for DiffBinaural
位置情報変換・処理ユーティリティ（2次元位置対応）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

class Position2DConverter:
    """
    3次元位置情報を2次元に変換するユーティリティ（絶対に必要）
    既存の3D位置データを2D位置データに効果的に変換
    """
    
    @staticmethod
    def angle_to_2d(angle_degrees: Union[float, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """角度（度）を2次元位置座標に変換"""
        if isinstance(angle_degrees, torch.Tensor):
            rad = torch.deg2rad(angle_degrees)
            x = torch.sin(rad)  # 左右位置 (-1 to 1)
            y = torch.cos(rad)  # 前後位置 (-1 to 1)
            return torch.stack([x, y], dim=-1)
        else:
            rad = np.radians(angle_degrees)
            x = np.sin(rad)
            y = np.cos(rad)
            return np.stack([x, y], axis=-1)
    
    @staticmethod
    def cartesian_3d_to_2d(pos_3d: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """3次元直交座標を2次元に投影"""
        if isinstance(pos_3d, torch.Tensor):
            # XZ平面への投影（Y軸は高さとして無視）
            x, z = pos_3d[..., 0], pos_3d[..., 2]
            # 正規化
            norm = torch.sqrt(x**2 + z**2 + 1e-8)
            return torch.stack([x / norm, z / norm], dim=-1)
        else:
            x, z = pos_3d[..., 0], pos_3d[..., 2]
            norm = np.sqrt(x**2 + z**2 + 1e-8)
            return np.stack([x / norm, z / norm], axis=-1)
    
    @staticmethod
    def spherical_to_2d(azimuth: Union[float, np.ndarray, torch.Tensor], 
                       elevation: Union[float, np.ndarray, torch.Tensor] = None) -> Union[np.ndarray, torch.Tensor]:
        """球座標系（方位角・仰角）を2次元位置に変換"""
        # 仰角が提供されない場合は水平面投影
        if elevation is None:
            return Position2DConverter.angle_to_2d(azimuth)
        
        if isinstance(azimuth, torch.Tensor):
            az_rad = torch.deg2rad(azimuth)
            el_rad = torch.deg2rad(elevation)
            # 水平面への投影（仰角を考慮した重み付き）
            cos_el = torch.cos(el_rad)
            x = torch.sin(az_rad) * cos_el
            y = torch.cos(az_rad) * cos_el
            return torch.stack([x, y], dim=-1)
        else:
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            cos_el = np.cos(el_rad)
            x = np.sin(az_rad) * cos_el
            y = np.cos(az_rad) * cos_el
            return np.stack([x, y], axis=-1)

class Position2DEmbedding(nn.Module):
    """
    2次元位置情報のニューラルネットワーク埋め込み（既存システムへの追加）
    """
    
    def __init__(self, embed_dim: int = 64, max_freq: float = 10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # 位置エンコーディング用の周波数
        freqs = torch.exp(torch.linspace(0, np.log(max_freq), embed_dim // 4))
        self.register_buffer('freqs', freqs)
        
        # 位置情報を高次元に変換
        self.pos_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    
    def forward(self, pos_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_2d: [batch_size, 2] or [batch_size, seq_len, 2]
        Returns:
            embedded position: [batch_size, embed_dim] or [batch_size, seq_len, embed_dim]
        """
        original_shape = pos_2d.shape
        if len(original_shape) == 3:
            batch_size, seq_len = original_shape[:2]
            pos_2d = pos_2d.view(-1, 2)
        
        # サイン・コサイン位置エンコーディング
        x, y = pos_2d[:, 0:1], pos_2d[:, 1:2]
        
        # 各周波数でのサイン・コサイン
        x_embed = torch.cat([torch.sin(x * self.freqs), torch.cos(x * self.freqs)], dim=-1)
        y_embed = torch.cat([torch.sin(y * self.freqs), torch.cos(y * self.freqs)], dim=-1)
        
        # 結合と投影
        pos_embed = torch.cat([x_embed, y_embed], dim=-1)
        pos_embed = self.pos_proj(pos_embed)
        
        # 元の形状に戻す
        if len(original_shape) == 3:
            pos_embed = pos_embed.view(batch_size, seq_len, -1)
        
        return pos_embed

class BinauraPositionProcessor:
    """
    バイノーラル音響に特化した位置処理（HRTF近似）
    """
    
    @staticmethod
    def compute_itd_factor(pos_2d: Union[np.ndarray, torch.Tensor], 
                          head_radius: float = 0.0875) -> Union[np.ndarray, torch.Tensor]:
        """両耳時間差（ITD）の計算係数"""
        # 単純化されたITDモデル（Woodworthの式を近似）
        if isinstance(pos_2d, torch.Tensor):
            angle = torch.atan2(pos_2d[..., 0], pos_2d[..., 1])  # 方位角
            itd_factor = torch.sin(angle) * head_radius / 343.0  # 音速343m/s
            return itd_factor
        else:
            angle = np.arctan2(pos_2d[..., 0], pos_2d[..., 1])
            itd_factor = np.sin(angle) * head_radius / 343.0
            return itd_factor
    
    @staticmethod
    def compute_ild_factor(pos_2d: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """両耳強度差（ILD）の計算係数"""
        if isinstance(pos_2d, torch.Tensor):
            # 角度に基づく簡単なILD近似
            angle = torch.atan2(pos_2d[..., 0], pos_2d[..., 1])
            ild_factor = torch.sin(angle) * 0.5  # -0.5 to 0.5の範囲
            return ild_factor
        else:
            angle = np.arctan2(pos_2d[..., 0], pos_2d[..., 1])
            ild_factor = np.sin(angle) * 0.5
            return ild_factor
    
    @staticmethod
    def apply_binaural_weighting(mono_spec: torch.Tensor, 
                                pos_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """モノラルスペクトログラムにバイノーラル重み付けを適用"""
        batch_size = mono_spec.shape[0]
        
        # ITD・ILD係数計算
        itd_factor = BinauraPositionProcessor.compute_itd_factor(pos_2d)
        ild_factor = BinauraPositionProcessor.compute_ild_factor(pos_2d)
        
        # 左右チャンネルの重み
        left_weight = 1.0 + ild_factor.unsqueeze(-1).unsqueeze(-1)
        right_weight = 1.0 - ild_factor.unsqueeze(-1).unsqueeze(-1)
        
        # 重み付き適用
        left_spec = mono_spec * left_weight
        right_spec = mono_spec * right_weight
        
        return left_spec, right_spec

def normalize_position_data(pos_data: Union[np.ndarray, torch.Tensor], 
                           method: str = 'unit_circle') -> Union[np.ndarray, torch.Tensor]:
    """位置データの正規化（既存システムとの互換性確保）"""
    if method == 'unit_circle':
        # 単位円に正規化
        if isinstance(pos_data, torch.Tensor):
            norm = torch.sqrt(torch.sum(pos_data**2, dim=-1, keepdim=True))
            return pos_data / (norm + 1e-8)
        else:
            norm = np.sqrt(np.sum(pos_data**2, axis=-1, keepdims=True))
            return pos_data / (norm + 1e-8)
    
    elif method == 'minmax':
        # [-1, 1] の範囲に正規化
        if isinstance(pos_data, torch.Tensor):
            pos_min = torch.min(pos_data, dim=-2, keepdim=True)[0]
            pos_max = torch.max(pos_data, dim=-2, keepdim=True)[0]
            return 2 * (pos_data - pos_min) / (pos_max - pos_min + 1e-8) - 1
        else:
            pos_min = np.min(pos_data, axis=-2, keepdims=True)
            pos_max = np.max(pos_data, axis=-2, keepdims=True)
            return 2 * (pos_data - pos_min) / (pos_max - pos_min + 1e-8) - 1
    
    return pos_data

# 既存システムとの統合用ヘルパー関数
def convert_existing_pos_to_2d(pos_3d_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """既存の3D位置データを2Dに変換"""
    return Position2DConverter.cartesian_3d_to_2d(pos_3d_data)

def create_position_embedding_layer(embed_dim: int = 64):
    """既存システムに追加できる位置埋め込み層を作成"""
    return Position2DEmbedding(embed_dim) 
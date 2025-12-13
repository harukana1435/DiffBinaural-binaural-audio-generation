import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from modules.attention import MaskedAttention

# helpers functions

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet(nn.Module):
    def __init__(self, original_resnet,pool_type='maxpool', use_transformer=False):
        super(Resnet, self).__init__()
        self.pool_type = pool_type
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.Transformer(d_model=512, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=512, batch_first=True)

    def forward(self, x, pool=True):
        x = self.features(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x[:, 0:1, ...]
        x = x.permute(0, 2, 1, 3, 4)

        x = torch.mean(x, dim=(3,4))

        # transformer
        if self.use_transformer:
            x = self.transformer(x.transpose(1,2), x.transpose(1,2)).transpose(1,2)

        if not pool:
            return x

        x = torch.mean(x, dim=2)

        x = x.view(B, C)
        return x

class Clip(nn.Module):
    def __init__(self, model, pool_type='maxpool', use_transformer=False):
        super(Clip, self).__init__()
        self.pool_type = pool_type
        self.model = model
        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.temporal_transformer = nn.Transformer(d_model=512, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=2048, batch_first=True)

            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B, T, C)

        # transformer
        # x = self.temporal_transformer(x.transpose(1,2)).transpose(1,2)
        x = self.temporal_transformer(x, x).transpose(1,2) #(B, C, T)
        
        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = torch.mean(x, 2)
        elif self.pool_type == 'maxpool':
            x = torch.max(x, 2)[0]

        return x


class Clip(nn.Module):
    def __init__(self, model, pool_type='maxpool', use_transformer=False):
        super(Clip, self).__init__()
        self.pool_type = pool_type
        self.model = model
        self.emb_dim = 512
        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,    # 埋め込み次元
            nhead=8,                  # ヘッド数
            dim_feedforward=2048,     # フィードフォワードネットワークのサイズ
            batch_first=True          # バッチサイズが最初に来る場合
        )
            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B, T, C)

        # transformer
        x = self.temporal_transformer_encoder(x) #(B, T, C)
        
        x = torch.mean(x, dim=1) #(B, 512)

        return x
    
    
    
    
class Clip_Pos(nn.Module):
    def __init__(self, model, pool_type='maxpool', dropout = 0.1):
        super(Clip_Pos, self).__init__()
        self.pool_type = pool_type
        self.model = model
        self.max_sources = 4
        
        self.emb_dim = 512
        
        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.pos_emb = SinusoidalPosEmb(64)
        self.pos_emb_mlp = nn.Linear(192, 1024)
        self.pos_emb_act = nn.GELU()
        
       
        
        self.pos_attention = MaskedAttention(query_dim=self.emb_dim, heads=8, dim_head=64) #query_dimは音源の最大値であるN=4がはいる。
        self.pos_layer1 = nn.LayerNorm(self.emb_dim)
        self.pos_ff = PositionwiseFeedForward(self.emb_dim, self.emb_dim*4)
        self.pos_layer2 = nn.LayerNorm(self.emb_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,    # 埋め込み次元
            nhead=8,                  # ヘッド数
            dim_feedforward=2048,     # フィードフォワードネットワークのサイズ
            batch_first=True          # バッチサイズが最初に来る場合
        )
            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pos, mask):
        (B, C, T, N, H, W) = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B * T * N, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B*T, N, C)

        pos = pos.view(B*T*N*3)
        pos = self.pos_emb(pos)
        pos = pos.view(B*T*N, -1)
        pos = self.pos_emb_mlp(pos)
        pos = self.pos_emb_act(pos)
        pos = pos.view(B*T, N, 1024)
        scale, shift = pos.chunk(2, dim=2)
        x = x * (scale + 1) + shift
        
        mask = mask.view(B*T, N)
        pos_attn = self.pos_attention(x, mask) #(B*T, N, 512)
        x = x + self.pos_dropout(pos_attn)
        x = self.pos_layer1(x)
        ff_output = self.pos_ff(x)
        x = x + self.pos_dropout(ff_output)
        x = self.pos_layer2(x)
        
        x = torch.max(x, dim=1)[0] #(B*T, 512)
        
        x = x.view(B, T, 512)
        
        # transformer
        x = self.temporal_transformer_encoder(x)
        
        x = torch.mean(x, dim=1) #(B, 512)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the model (also the input and output dimension).
            d_ff (int): The dimension of the feed-forward hidden layer.
            dropout (float): Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor, shape [batch_size, seq_len, d_model]

        Returns:
            Tensor: Output tensor, shape [batch_size, seq_len, d_model]
        """
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Clip_Pos2D(nn.Module):
    def __init__(self, model, pool_type='maxpool', dropout = 0.1):
        super(Clip_Pos2D, self).__init__() # Corrected super call
        self.pool_type = pool_type
        self.model = model
        self.max_sources = 4

        self.emb_dim = 512 # Dimension of CLIP image features

        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False

        # Sinusoidal embeddings for elevation and azimuth
        self.pos_emb_dim = 64 # Dimension for each angle's sinusoidal embedding
        self.pos_emb_ele = SinusoidalPosEmb(self.pos_emb_dim) # Elevation embedding
        self.pos_emb_azi = SinusoidalPosEmb(self.pos_emb_dim) # Azimuth embedding

        # Separate MLPs for scale (from elevation) and shift (from azimuth)
        self.mlp_scale = nn.Sequential(
            nn.Linear(self.pos_emb_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.emb_dim) # Output 512 dims for scale
        )
        self.mlp_shift = nn.Sequential(
            nn.Linear(self.pos_emb_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.emb_dim) # Output 512 dims for shift
        )

        # Attention and feedforward layers for refining features after positional modulation
        self.pos_attention = MaskedAttention(query_dim=self.emb_dim, heads=8, dim_head=64)
        self.pos_layer1 = nn.LayerNorm(self.emb_dim)
        self.pos_ff = PositionwiseFeedForward(self.emb_dim, self.emb_dim*4)
        self.pos_layer2 = nn.LayerNorm(self.emb_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,    # 埋め込み次元
            nhead=8,                  # ヘッド数
            dim_feedforward=2048,     # フィードフォワードネットワークのサイズ
            batch_first=True          # バッチサイズが最初に来る場合
        )
            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pos, mask):
        (B, C, T, N, H, W) = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B * T * N, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B*T, N, C) # x shape: (B*T, N, 512)

        # --- Positional Embedding and Modulation ---
        # Assuming pos shape is (B, T, N, 2) where pos[..., 0] is elevation, pos[..., 1] is azimuth
        pos_ele = pos[..., 0] # Elevation (B, T, N)
        pos_azi = pos[..., 1] # Azimuth (B, T, N)

        # Apply sinusoidal embeddings
        # Reshape angles to (B*T*N) before passing to embedding
        emb_ele = self.pos_emb_ele(pos_ele.reshape(-1)) # (B*T*N, 64)
        emb_azi = self.pos_emb_azi(pos_azi.reshape(-1)) # (B*T*N, 64)

        # Calculate scale from elevation embedding and shift from azimuth embedding
        scale_flat = self.mlp_scale(emb_ele) # (B*T*N, 512)
        shift_flat = self.mlp_shift(emb_azi) # (B*T*N, 512)

        # Reshape scale and shift to match x's dimensions for broadcasting
        scale = scale_flat.view(B*T, N, self.emb_dim) # (B*T, N, 512)
        shift = shift_flat.view(B*T, N, self.emb_dim) # (B*T, N, 512)

        # Apply scale and shift to image features x
        x = x * (scale + 1) + shift
        # --- End Positional Embedding and Modulation ---

        mask = mask.view(B*T, N)
        # Apply attention mechanism using the modulated features
        pos_attn = self.pos_attention(x, mask) #(B*T, N, 512)
        x = x + self.pos_dropout(pos_attn)
        x = self.pos_layer1(x)
        ff_output = self.pos_ff(x)
        x = x + self.pos_dropout(ff_output)
        x = self.pos_layer2(x)
        
        x = torch.max(x, dim=1)[0] #(B*T, 512)
        
        x = x.view(B, T, 512)
        
        # transformer
        x = self.temporal_transformer_encoder(x)
        
        x = torch.mean(x, dim=1) #(B, 512)

        return x



import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def sinusoidal_position_encoding(L: int, D: int, device=None):
    """(L, D) の標準的な正弦位置埋め込み（相対順序性付与のため）。"""
    if device is None:
        device = torch.device('cpu')
    pe = torch.zeros(L, D, device=device)
    position = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, D)


# ----------------------------
# Set Transformer Aggregation (順序不変, マスク安全)
# ----------------------------

class SetTransformerAggregation(nn.Module):
    """
    音源の順序に依存しない集約モジュール。
    - シード (learnable query) とクロスアテンションで N を 1 に集約
    - 「そのフレームに有効音源が1つもない」ケースでは学習可能 Null を返して NaN を回避

    引数 mask: (B, N) で True=無効（音源なし）, False=有効
    """
    def __init__(self, emb_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_seeds = 1

        self.seed = nn.Parameter(torch.randn(1, self.num_seeds, emb_dim))
        self.null_embed = nn.Parameter(torch.zeros(1, emb_dim))
        nn.init.normal_(self.null_embed, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            emb_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, N, E)
        mask: (B, N)  True=無効, False=有効
        Returns: (B, E)
        """
        B, N, E = x.shape
        valid = ~mask.bool()             # True=有効
        all_invalid = ~valid.any(dim=1)  # (B,)

        out = self.null_embed.expand(B, E).clone()  # 既定は Null

        if (~all_invalid).any():
            x_v = x[~all_invalid]          # (Bv, N, E)
            m_v = valid[~all_invalid]      # (Bv, N) True=有効
            
            # 各バッチで少なくとも1つの有効音源があることを確認
            batch_has_valid = m_v.any(dim=1)  # (Bv,)
            
            if batch_has_valid.any():
                seeds = self.seed.expand(x_v.size(0), -1, -1)  # (Bv,1,E)
                kpm = ~m_v                                      # True=PAD(無効)

                attn_out, _ = self.cross_attention(
                    query=seeds, key=x_v, value=x_v, key_padding_mask=kpm
                )  # (Bv,1,E)
                seeds = self.layer_norm1(seeds + self.dropout(attn_out))
                ffn_out = self.ffn(seeds)
                seeds = self.layer_norm2(seeds + self.dropout(ffn_out))  # (Bv,1,E)
                out_valid = seeds.squeeze(1)  # (Bv,E)
                
                # 有効なバッチのみ更新
                valid_indices = torch.where(~all_invalid)[0][batch_has_valid]
                out[valid_indices] = out_valid[batch_has_valid]

        return out  # (B,E)


# ----------------------------
# 学習注意プーリング（任意）
# ----------------------------

class AttentiveTemporalPool(nn.Module):
    """時間方向の学習注意プーリング（マスク対応）"""
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, seq: torch.Tensor, mask_invalid: torch.Tensor) -> torch.Tensor:
        """
        seq:         (B,T,E)
        mask_invalid:(B,T) True=無効（フレーム無効）, False=有効
        """
        B, T, E = seq.shape
        q = self.query.expand(B, -1, -1)                          # (B,1,E)
        scores = torch.matmul(q, seq.transpose(1, 2)).squeeze(1)  # (B,T)
        scores = scores.masked_fill(mask_invalid.bool(), float('-inf'))
        attn = torch.softmax(scores, dim=-1)                      # (B,T)
        pooled = (seq * attn.unsqueeze(-1)).sum(dim=1)            # (B,E)
        return pooled


# ----------------------------
# Main Module
# ----------------------------

class Clip_Pos2D_Concat(nn.Module):
    """
    Clip_Pos2D の変種: 位置情報（elevation, azimuth）をCLIP特徴に直接連結する方式
    
    主な変更点:
      - 正弦波埋め込み + scale/shift 変調 → CLIP特徴の後ろに位置情報を直接連結（512 + 2 = 514次元）
      - 514次元を512次元に投影
      - その他のアーキテクチャ（MaskedAttention、PositionwiseFeedForward等）は Clip_Pos2D と同じ
    """
    def __init__(self, model, pool_type='maxpool', dropout=0.1):
        super(Clip_Pos2D_Concat, self).__init__()
        self.pool_type = pool_type
        self.model = model
        self.max_sources = 4

        self.emb_dim = 512  # Dimension of CLIP image features
        self.concat_dim = 514  # CLIP (512) + position (2)

        # Freeze CLIP model
        for param in self.model.parameters():
            param.requires_grad = False

        # 514次元（CLIP + 位置情報）を512次元に投影
        self.pos_projection = nn.Sequential(
            nn.Linear(self.concat_dim, self.emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention and feedforward layers for refining features after positional concatenation
        self.pos_attention = MaskedAttention(query_dim=self.emb_dim, heads=8, dim_head=64)
        self.pos_layer1 = nn.LayerNorm(self.emb_dim)
        self.pos_ff = PositionwiseFeedForward(self.emb_dim, self.emb_dim * 4)
        self.pos_layer2 = nn.LayerNorm(self.emb_dim)
        self.pos_dropout = nn.Dropout(dropout)

        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)
        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x

    def forward_multiframe(self, x, pos, mask):
        """
        Args:
            x:    (B, C, T, N, H, W) - 画像入力
            pos:  (B, T, N, 2) - 位置情報 [elevation, azimuth]
            mask: (B, T, N) - True=無効（音源なし）, False=有効（音源あり）
        Returns:
            (B, 512) - 最終的な特徴ベクトル
        """
        (B, C, T, N, H, W) = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B * T * N, C, H, W)

        # CLIP特徴抽出
        x = self.model.encode_image(x)  # (B*T*N, 512)

        (_, C) = x.size()
        x = x.view(B, T, N, C)  # (B, T, N, 512)

        # 位置情報を直接連結
        # pos shape: (B, T, N, 2) where pos[..., 0] is elevation, pos[..., 1] is azimuth
        x_concat = torch.cat([x, pos], dim=-1)  # (B, T, N, 514)

        # 514次元を512次元に投影
        x = self.pos_projection(x_concat)  # (B, T, N, 512)

        # Reshape for attention processing
        x = x.view(B * T, N, self.emb_dim)  # (B*T, N, 512)
        mask = mask.view(B * T, N)

        # Apply attention mechanism
        pos_attn = self.pos_attention(x, mask)  # (B*T, N, 512)
        x = x + self.pos_dropout(pos_attn)
        x = self.pos_layer1(x)
        ff_output = self.pos_ff(x)
        x = x + self.pos_dropout(ff_output)
        x = self.pos_layer2(x)

        # Max pooling over sources (N dimension)
        x = torch.max(x, dim=1)[0]  # (B*T, 512)

        x = x.view(B, T, 512)

        # Temporal transformer
        x = self.temporal_transformer_encoder(x)

        # Mean pooling over time
        x = torch.mean(x, dim=1)  # (B, 512)

        return x


class Clip_Pos2D_Enhanced(nn.Module):
    """
    目的:
      - CLIP画像特徴 (512) に画素座標 (x,y)∈[-1,1] を「そのまま後方に連結」して 514 次元トークンを構成
      - 音源間(N)の Transformer + SetTransformer で各フレーム t の集合を 1 トークンへ
      - 時間方向(T)の Transformer で時系列関係をモデル化
      - 最終出力は 512 次元

    マスク仕様:
      - 入力 mask は True=無効（音源なし）, False=有効（音源あり）
    """
    def __init__(
        self,
        model,                          # CLIP 互換: encode_image, encode_text
        dropout: float = 0.1,
        num_heads: int = 8,
        num_source_layers: int = 2,
        num_temporal_layers: int = 2,
        temporal_pool: str = "mean",    # ["mean","attn"]
        use_time_pe: bool = True,
        normalize_clip: bool = True,
    ):
        super().__init__()
        self.model = model
        self.emb_dim = 512
        self.concat_dim = 514           # 512 + 2(x,y)
        self.temporal_pool = temporal_pool
        self.use_time_pe = use_time_pe
        self.normalize_clip = normalize_clip

        # Freeze CLIP
        for p in self.model.parameters():
            p.requires_grad = False

        # 514 -> 512 投影（Transformer適合）
        self.token_proj = nn.Sequential(
            nn.Linear(self.concat_dim, self.emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.emb_dim),
        )

        # 音源間 Transformer
        src_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            src_layer, num_layers=num_source_layers
        )

        # 集合（N -> 1）集約
        self.set_aggregator = SetTransformerAggregation(
            emb_dim=self.emb_dim, num_heads=num_heads, dropout=dropout
        )

        # 時間方向 Transformer
        tmp_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            tmp_layer, num_layers=num_temporal_layers
        )

        # Temporal Pooling
        if temporal_pool == "attn":
            self.temporal_pooler = AttentiveTemporalPool(self.emb_dim)

        # 出力整形
        self.final_projection = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.LayerNorm(self.emb_dim),
        )

        # 全フレーム無効バッチ用の Null
        self.null_temporal = nn.Parameter(torch.zeros(1, self.emb_dim))
        nn.init.normal_(self.null_temporal, std=0.02)

    # ---- CLIP passthrough ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.model.encode_image(x)  # (B,512)
        if self.normalize_clip:
            feats = F.normalize(feats, dim=-1)
        return feats

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.model.encode_text(x)  # (B,512)
        if self.normalize_clip:
            feats = F.normalize(feats, dim=-1)
        return feats

    # ---- Multi-frame x Multi-source ----
    def forward_multiframe(self, x: torch.Tensor, pos_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C, T, N, H, W)   # CLIP前処理済み画像
            pos_xy: (B, T, N, 2)         # 画像座標 (x,y) ∈ [-1,1]
            mask:   (B, T, N)            # True=無効（音源なし）, False=有効（音源あり）
        Return:
            (B, 512)
        """
        B, C, T, N, H, W = x.shape
        device = x.device
        mask = mask.bool()  # True=無効

        # 1) CLIP特徴
        x_btN = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * T * N, C, H, W)
        with torch.no_grad():
            clip = self.model.encode_image(x_btN)          # (B*T*N,512)
        clip = clip.view(B, T, N, self.emb_dim)            # (B,T,N,512)
        if self.normalize_clip:
            clip = F.normalize(clip, dim=-1)

        # 2) (512 + 2) 連結 → 514
        assert pos_xy.shape == (B, T, N, 2), f"pos_xy must be (B,T,N,2), got {pos_xy.shape}"
        token_514 = torch.cat([clip, pos_xy], dim=-1)      # (B,T,N,514)
        
        # デバッグ: NaNチェック
        if torch.isnan(clip).any():
            print(f"⚠️ NaN in CLIP features: {torch.isnan(clip).sum()}")
        if torch.isnan(pos_xy).any():
            print(f"⚠️ NaN in pos_xy: {torch.isnan(pos_xy).sum()}")
        if torch.isnan(token_514).any():
            print(f"⚠️ NaN in token_514: {torch.isnan(token_514).sum()}")

        # 3) 514 -> 512 投影
        token_512 = self.token_proj(token_514)             # (B,T,N,512)
        
        # デバッグ: 投影後のNaNチェック
        if torch.isnan(token_512).any():
            print(f"⚠️ NaN in token_512 after projection: {torch.isnan(token_512).sum()}")
            print(f"   token_514 stats: min={token_514.min():.3f}, max={token_514.max():.3f}")
            print(f"   pos_xy range: [{pos_xy.min():.3f}, {pos_xy.max():.3f}]")

        # ===== 音源間 Transformer =====
        # (B,T,N,E) -> (B*T,N,E)
        src_bt = token_512.view(B * T, N, self.emb_dim)
        src_kpm = mask.view(B * T, N)                      # ★True=PAD（無効）: 反転しない！
        
        # デバッグ: Transformer入力のNaNチェック
        if torch.isnan(src_bt).any():
            print(f"⚠️ NaN in src_bt (before source_encoder): {torch.isnan(src_bt).sum()}")
            # NaN回避：NaNをゼロで置き換え
            src_bt = torch.nan_to_num(src_bt, nan=0.0)
            
        # 安全なTransformer処理：全てマスクされた行の処理
        # src_kpm: (B*T, N) - True=PADする（無効）
        all_masked_rows = src_kpm.all(dim=1)  # (B*T,) - その時刻の全音源が無効
        
        if all_masked_rows.any():
            # 一部の行が全てマスクされている場合の安全処理
            src_encoded = torch.zeros_like(src_bt)  # 初期化
            
            # 有効な行のみTransformerを適用
            valid_rows = ~all_masked_rows
            if valid_rows.any():
                src_bt_valid = src_bt[valid_rows]
                src_kpm_valid = src_kpm[valid_rows]
                src_encoded_valid = self.source_encoder(
                    src_bt_valid, src_key_padding_mask=src_kpm_valid
                )
                src_encoded[valid_rows] = src_encoded_valid
        else:
            # 通常処理：全ての行に有効な音源がある
            src_encoded = self.source_encoder(
                src_bt, src_key_padding_mask=src_kpm
            )                                                  # (B*T,N,512)
        
        # デバッグ: Transformer出力のNaNチェック
        if torch.isnan(src_encoded).any():
            print(f"⚠️ NaN in src_encoded (after source_encoder): {torch.isnan(src_encoded).sum()}")
            print(f"   src_kpm shape: {src_kpm.shape}, True count: {src_kpm.sum()}")
            # 全てが無効な場合の安全処理
            src_encoded = torch.nan_to_num(src_encoded, nan=0.0)

        # 4) N -> 1 集約（SetTransformerAggregation は mask=True=無効 前提）
        agg_bt = self.set_aggregator(src_encoded, src_kpm)  # (B*T,512)
        
        # デバッグ: 集約後のNaNチェックと安全処理
        if torch.isnan(agg_bt).any():
            print(f"⚠️ NaN in agg_bt (SetTransformerAggregation output): {torch.isnan(agg_bt).sum()}")
            print(f"   src_encoded has NaN: {torch.isnan(src_encoded).any()}")
            print(f"   src_kpm (True=invalid): {src_kpm.sum()}/{src_kpm.numel()}")
            # NaN回避
            agg_bt = torch.nan_to_num(agg_bt, nan=0.0)
            
        agg = agg_bt.view(B, T, self.emb_dim)               # (B,T,512)

        # ===== 時間方向 Transformer =====
        frame_valid = (~mask).any(dim=2)                   # (B,T) その時刻に有効音源が1つ以上
        frame_kpm   = ~frame_valid                         # True=PAD（そのフレームは全無効）
        time_in = agg
        if self.use_time_pe:
            pe = sinusoidal_position_encoding(T, self.emb_dim, device=device)  # (T,512)
            time_in = time_in + pe.unsqueeze(0)

        batch_valid = frame_valid.any(dim=1)               # (B,) 少なくとも1フレーム有効？
        out = self.null_temporal.expand(B, self.emb_dim).clone()

        if batch_valid.any():
            time_in_v = time_in[batch_valid]               # (Bv,T,512)
            tmp_kpm_v = frame_kpm[batch_valid]             # True=PAD（全無効フレーム）
            tmp_encoded = self.temporal_encoder(
                time_in_v, src_key_padding_mask=tmp_kpm_v
            )                                              # (Bv,T,512)

            # 5) 時間プーリング（マスク付き）
            if self.temporal_pool == "attn":
                pooled = self.temporal_pooler(tmp_encoded, tmp_kpm_v)  # (Bv,512)
            else:
                w = frame_valid[batch_valid].float().unsqueeze(-1)      # ★有効=1
                denom = w.sum(dim=1).clamp_min(1.0)
                pooled = (tmp_encoded * w).sum(dim=1) / denom           # (Bv,512)

            out[batch_valid] = pooled

        # 6) 最終 512 次元整形
        out = self.final_projection(out)                   # (B,512)
        if self.normalize_clip:
            out = F.normalize(out, dim=-1)
        return out
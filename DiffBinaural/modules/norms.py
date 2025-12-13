import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F



class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))  # スケールパラメータ
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))  # シフトパラメータ
        self.register_buffer("running_mean", torch.zeros(1, dim, 1, 1))  # 移動平均
        self.register_buffer("running_var", torch.ones(1, dim, 1, 1))    # 移動分散
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:  # 訓練モード
            # バッチ次元 (dim=0) で統計量を計算
            batch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            batch_var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

            # 移動平均と移動分散を更新
            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var

            # 正規化
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:  # 推論モード
            # 移動平均と移動分散を使用
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # スケールとシフトを適用
        return x_hat * self.gamma + self.beta


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
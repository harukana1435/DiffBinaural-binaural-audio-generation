import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from modules.attention import *
from modules.norms import *


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
    

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False)) #第1引数では次元をどう減らすか、第2引数はどんな操作を行うか
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        # B, _ = x.shape
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1) #-1をしなくてよさそう
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        #sin_emb = emb.sin()
        #cos_emb = emb.cos()
        #emb = torch.stack((sin_emb, cos_emb), dim=-1).view(B, -1)

        return emb


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, dropout = 0.1): # dim_out=64, dim=64, time_emb_dim=256 time_mlpの後
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2) # 128に変換
        ) if exists(time_emb_dim) else None
        
        self.input_layer = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            WeightStandardizedConv2d(dim, dim_out, kernel_size=3, padding = 1), #64が64になる
            nn.GroupNorm(groups, dim_out)
        )
        
        self.output_layer = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            WeightStandardizedConv2d(dim_out, dim_out, kernel_size=3, padding = 1), #64が64になる
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() #dimとdim_outが異なっていれば、1×1の畳み込みをdim→dim_outにする

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.emb_layers) and exists(time_emb): #確定で徹
            time_emb = self.emb_layers(time_emb) #b 128に変換される
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1) #b 64 が2つできる

        h = self.input_layer(x)
        
        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        
        h = self.output_layer(h)

        return h + self.res_conv(x)



# model
class Unet(nn.Module):
    def __init__(
        self,
        dim, # →　64
        init_dim = None,  # →64
        out_dim = None, # →　2
        dim_mults=(1, 2, 4),
        channels = 3, # →　2
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels # 1
        self.self_condition = self_condition #  Trueで確定
        input_channels = channels+1 if self_condition else channels # 入力は2つのスペクトログラムだから

        init_dim = default(init_dim, dim) #init_dimが定義されてないので、128となっている
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1) #2チャンネルを128チャンネルに線形変換している (B, 2, 128, 128)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # [128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:])) # [(128, 256), (256, 512)]

        res_block = partial(ResnetBlock, groups = resnet_block_groups) #特定の関数やクラスの一部の引数を固定して、新しい関数やクラスを作成できる
        
        attn_block = partial(AttentionBlock, n_heads=4, d_head=32, groups= 8)

        # time embeddings

        time_dim = dim * 4 #512
        context_dim = 512 #視覚特徴

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim #64

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim), #64→512
            nn.GELU(), #標準正規分布の累積分布関数を使用する。負の値が非常に大きい場合は0になる、正の値はそのまま通す。負の値も微妙に通すのがいいらしい。
            nn.Linear(time_dim, time_dim) #512→512
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                res_block(dim_in, dim_in, time_emb_dim = time_dim), # 64, 64, 256
                attn_block(dim_in, time_emb_dim = time_dim, context_dim=context_dim),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
 
        # # baseline
        self.mid_block1 = res_block(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = MiddleAttentionBlock(mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = res_block(mid_dim, mid_dim, time_emb_dim = time_dim)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                res_block(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_block(dim_out, time_emb_dim = time_dim, context_dim=context_dim),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))


        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = res_block(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        nn.init.kaiming_normal_(self.final_conv.weight)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of trainable parameters: {self.num_params}")

    def forward(self, x, time, x_self_cond = None, mix_t = None, visual_feat = None): #xはB×1×256×256　visual_featはB×512×Tになっている
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        t = self.time_mlp(time)
        
        c = t

        r = x.clone()
        
        

        h = []

            
        for res_block, attn_block, downsample in self.downs:
            x = res_block(x, time_emb=c)

            x = attn_block(x, context=visual_feat, time_emb=c)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, time_emb=c)
        x = self.mid_attn(x, time_emb=c)
        x = self.mid_block2(x, time_emb=c)

            
        for res_block, attn_block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = res_block(x, time_emb=c)

            x = attn_block(x, context=visual_feat, time_emb=c)

            x = upsample(x)    

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, c)
        x = self.final_conv(x)
        return x
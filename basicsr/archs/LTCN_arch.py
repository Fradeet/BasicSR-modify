'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

# from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import init
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFreeLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBiasLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):

    def __init__(self, dim, layer_norm_type):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # return to_4d(self.body(to_3d(x)), h, w)
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################

class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


##########################################################################
########图像归一化处理################################################

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


########################################################################
####attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)



##---------------ECAAttention ----------
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


class ESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


##########################################################################

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        # 如果k是整数，p为k与2整除后向下取整；如果k是列表等，p对应的是列表中每个元素整除2。
    return p


##########################################################################


# Block
# 卷积层，仅用于计算 padding，并返回卷积层
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)


########################LFE#########################################

class LFEB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        dim_out=n_feats
        expansion_rate = 1
        shrinkage_rate = 0.25
        hidden_dim = int(expansion_rate * dim_out)
        self.conv_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.GELU = nn.GELU()
        self.dwconv_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride = 1, padding = 1, groups = hidden_dim)
        self.SE=SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate)
        self.conv_3 = nn.Conv2d(hidden_dim, dim_out, 1)

    def forward(self, x):
        x_original = x.clone()
        x = self.GELU(self.conv_1(x))
        x = self.GELU(self.dwconv_2(x))
        x = self.SE(x)
        x = self.conv_3(x)
        x = x + x_original
        return x

#################################################################################################

class LSTB(nn.Module):
    def __init__(
            self,
            dim,
            depth=2,
            num_heads=8,
            ffn_expansion_factor=2.66,
            qkv_bias=True):
        super().__init__()
        # # local feature extraction
        # self.LFE = LFEB(n_feats=dim)
        self.MIRB = MIRB(n_feats=dim)
        # Glable feature extraction
        self.GFE = TransformerBlock(dim=dim,
                                    num_heads=num_heads,
                                    bias=qkv_bias,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    LayerNorm_type='WithBias')


        # self.esa = ESA(dim)

    def forward(self, x, pre_attn=None, RAA=None):
        # local feature extraction
        # x = self.LFE(x)
        #
        # Glable
        x = self.GFE(x)

        x = self.MIRB(x)
        # x = self.esa(x)

        return x


class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        # self.norm = LayerNorm(n_feats, data_format='channels_first')
        # self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3, groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats, ffn_expansion_factor, res_scale=1.0):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([
            LSTB(dim=n_feats,
                 depth=2,
                 num_heads=4,
                 ffn_expansion_factor=ffn_expansion_factor,
                 qkv_bias=True
                 ) \
            for m in range(n_resblocks)])

        # self.body_t = LKAT(n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)

        # x = self.body_t(res) + x

        return x


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


# 改进的倒置残差块
class MIRB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.GELU = nn.GELU()
        self.conv_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.deconv_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats)
        # self.deconv_2 = DepthwiseSeparableConv(n_feats, n_feats)
        self.conv_3 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        x_original = x.clone()
        x = self.GELU(self.conv_1(x))
        x = self.GELU(self.deconv_2(x))
        x = self.conv_3(x)
        x = x + x_original
        return x


# From CSDN
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


@ARCH_REGISTRY.register()
class LTCN(nn.Module):
    def __init__(
            self,
            int_channels=3,
            out_channels=3,
            n_resblocks=4,
            n_resgroups=1,
            ffn_expansion_factor=2.66,
            n_feats=50,
            scale=2,
            res_scale=1.0
    ):
        super().__init__()

        # res_scale = res_scale
        self.n_resgroups = n_resgroups

        self.sub_mean = MeanShift(1.0)
        self.head = nn.Conv2d(int_channels, n_feats, 3, 1, 1)

        # define body module
        self.body = nn.ModuleList([
            ResGroup(
                n_resblocks, n_feats, ffn_expansion_factor, res_scale=res_scale)
            for i in range(n_resgroups)])

        if self.n_resgroups > 1:
            self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            Upsamplers.PixelShuffleDirect(scale=scale, num_feat=int_channels, num_out_ch=out_channels)
        )
        self.add_mean = MeanShift(1.0, sign=1)
        self.conv33 = nn.Conv2d(n_feats, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x_original = x.clone()
        x = self.head(x)
        for i in self.body:
            x = i(x)
        x = self.conv33(x)
        x = x + x_original
        x = self.tail(x)
        x = self.add_mean(x)
        return x

    def visual_feature(self, x):
        fea = []
        x = self.head(x)
        res = x

        for i in self.body:
            temp = res
            res = i(res)
            fea.append(res)

        res = self.body_t(res) + x

        x = self.tail(res)
        return x, fea

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

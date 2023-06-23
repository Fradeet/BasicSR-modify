import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from collections import OrderedDict

from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import to_2tuple, trunc_normal_
from .restormer_arch import TransformerBlock as RestormerTransformerBlock
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from .biformer_arch import BiLevelRoutingAttention , Block

# HNCT 中有 HBCT，HBCT 中有 ESA，SwinT

# 当前的修改内容
# 移植 Reformer 的 TransformerBlock 代码  (import)，导入新增参数
#  修改了 BasicLayer 内的导入
# 清理无用传入参数


# HNCT
@ARCH_REGISTRY.register()
class HNCTModifyKe(nn.Module):
    # 什么代码，为什么原来的 scale 是一个定值？
    # PyTorch nn 初始化函数
    def __init__(self, in_nc=3, nf=48, num_modules=4, out_nc=3, upscale=2,num_heads=4,qkv_bias=False,ffn_expansion_factor=2.66,LayerNorm_type='WithBias'):
        super(HNCTModifyKe, self).__init__()

        # B 是一个自定义的模块，用于构建网络
        self.forwardFeedConv = conv_layer(in_nc, nf, kernel_size=3)

        # 定义了如下网络结构：
        self.HBCT1 = HBCT(in_channels=nf,num_heads=num_heads,bias=qkv_bias,ffn_expansion_factor=ffn_expansion_factor,LayNorm_type=LayerNorm_type)
        self.HBCT2 = HBCT(in_channels=nf,num_heads=num_heads,bias=qkv_bias,ffn_expansion_factor=ffn_expansion_factor,LayNorm_type=LayerNorm_type)
        self.HBCT3 = HBCT(in_channels=nf,num_heads=num_heads,bias=qkv_bias,ffn_expansion_factor=ffn_expansion_factor,LayNorm_type=LayerNorm_type)
        self.HBCT4 = HBCT(in_channels=nf,num_heads=num_heads,bias=qkv_bias,ffn_expansion_factor=ffn_expansion_factor,LayNorm_type=LayerNorm_type)
        self.HBCT5 = HBCT(in_channels=nf, num_heads=num_heads, bias=qkv_bias, ffn_expansion_factor=ffn_expansion_factor,LayNorm_type=LayerNorm_type)
        self.HBCT6 = HBCT(in_channels=nf, num_heads=num_heads, bias=qkv_bias, ffn_expansion_factor=ffn_expansion_factor,LayNorm_type=LayerNorm_type)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0
        self.MIRB = MIRB(n_feats=nf)

    # PyTorch nn 前向传播函数
    def forward(self, input):
        out_fea = self.forwardFeedConv(input)
        out_B1 = self.HBCT1(out_fea)
        out_B2 = self.HBCT2(out_B1)
        out_B3 = self.HBCT3(out_B2)
        out_B4 = self.HBCT4(out_B3)
        out_B5 = self.HBCT5(out_B4)
        out_B6 = self.HBCT6(out_B5)
        # torch.cat 是 PyTorch 的拼接函数
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4,out_B5,out_B6], dim=1))
        # out_lr = self.LR_conv(out_B) + out_fea
        out_lr = self.MIRB(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output

    # def set_scale(self, scale_idx):
    #     self.scale_idx = scale_idx


class HBCT(nn.Module):

    def __init__(self, in_channels, distillation_rate=0.25,num_heads=4,bias=False,ffn_expansion_factor=2.66,LayNorm_type='WithBias'):
        super(HBCT, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.esa = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        # self.sparatt = Spartial_Attention.Spartial_Attention()
        # self.swinT = TransformerBlock(dim=in_channels,
        #                               num_heads=num_heads,
        #                               bias=bias,
        #                               ffn_expansion_factor=ffn_expansion_factor,
        #                               LayerNorm_type=LayNorm_type)

        self.bifo = Block(dim=64, drop_path=0, layer_scale_init_value=-1, topk=1, num_heads=8, n_win=2, qk_dim=64, kv_per_win=-1, kv_downsample_ratio=4, kv_downsample_kernel=4, kv_downsample_mode="identity", param_attention="qkv", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False, side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=True)

    def forward(self, input):
        input = self.esa2(input)
        # input = self.swinT(input)
        input = self.bifo(input)
        out_fused = self.esa(self.c1_r(input))
        return out_fused


class ESA(nn.Module):

    def __init__(self, n_feats, conv):
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


# SwinT
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
        # self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=8, qk_dim=dim,
        #                                     # qk_scale=,
        #                                     kv_per_win=-1,
        #                                     kv_downsample_ratio=1,
        #                                     kv_downsample_kernel=1,
        #                                     kv_downsample_mode='identity',
        #                                     topk=1, param_attention='qkv', param_routing=False,
        #                                     diff_routing=False, soft_routing=False,
        #                                     side_dwconv=5,
        #                                     auto_pad=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x_original = x.clone()
        x = self.norm1(x)
        # 需要转换为 NHWC 格式
        x = self.attn(x) + x_original
        x_original = x
        x = x_original + self.ffn(self.norm2(x))

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

#         return to_4d(self.body(x), h, w)


class PatchEmbed(nn.Module):

    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# Block
# 卷积层，仅用于计算 padding，并返回卷积层
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)
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

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc,
               out_nc,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,
               bias=True,
               pad_type='zero',
               norm_type=None,
               act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

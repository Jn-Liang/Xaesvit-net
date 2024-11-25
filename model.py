import time

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

# 定义RepConv层
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=1, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # 移除kernel_size的断言
        # assert kernel_size == 3

        padding_11 = padding - kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, bias=bias, groups=groups)

        # 初始化为0，这意味着开始时不会改变权重
        nn.init.constant_(self.conv1x1.weight, 0)

        # 等效转换层
        self.switch = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        else:
            # 训练阶段使用复合卷积
            x = self.conv(x) + self.switch * self.conv1x1(x)
            return x

# 定义空间通道混合注意力CBAM
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # hidden_dim = int(round(inp * expand_ratio))
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                                                  groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = conv_2d(embed_dim, 1 + 2 * embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out


class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
                                           LinearSelfAttention(embed_dim, attn_dropout), nn.Dropout(dropout))
        self.pre_norm_ffn = nn.Sequential(nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
                                          conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True,
                                                  norm=False, act=True), nn.Dropout(dropout),
                                          conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True,
                                                  norm=False, act=False), nn.Dropout(dropout))

    def forward(self, x):
        # self attention
        x = x + self.pre_norm_attn(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlockv3_v2(nn.Module):
    def __init__(self, inp, attn_dim, ffn_multiplier, attn_blocks, patch_size):
        super(MobileViTBlockv3_v2, self).__init__()
        self.patch_h, self.patch_w = patch_size

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', RepConv(inp, inp, kernel_size=3, stride=1, padding=1))
        self.local_rep.add_module('conv_1x1', RepConv(inp, attn_dim, kernel_size=1, stride=1, padding=0))

        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(attn_dim, ffn_dim))
        self.global_rep.add_module('LayerNorm2D', nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1))

        self.conv_proj = RepConv(2 * attn_dim, inp, kernel_size=1, stride=1, padding=0)
        self.cbam = CBAM(gate_channels=inp, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.unfold(feature_map, kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w))
        patches = patches.reshape(batch_size, in_channels, self.patch_h * self.patch_w, -1)
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(patches, output_size=output_size, kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w))
        return feature_map

    def forward(self, x):
        res = x.clone()
        fm_conv = self.local_rep(x)
        x, output_size = self.unfolding_pytorch(fm_conv)
        x = self.global_rep(x)
        x = self.folding_pytorch(patches=x, output_size=output_size)
        x = self.conv_proj(torch.cat((x, fm_conv), dim=1))
        x = self.cbam(x)  # 应用CBAM模块
        x = x + res
        return x


class MobileViTv3_v2(nn.Module):
    def __init__(self, image_size, width_multiplier, num_classes, patch_size=(2, 2)):
        """
        Implementation of MobileViTv3 based on v2
        """
        super().__init__()
        # check image size
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0
        assert width_multiplier in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

        # model size
        channels = []
        channels.append(int(max(16, min(64, 32 * width_multiplier))))
        channels.append(int(64 * width_multiplier))
        channels.append(int(128 * width_multiplier))
        channels.append(int(256 * width_multiplier))
        channels.append(int(384 * width_multiplier))
        channels.append(int(512 * width_multiplier))
        attn_dim = []
        attn_dim.append(int(128 * width_multiplier))
        attn_dim.append(int(192 * width_multiplier))
        attn_dim.append(int(256 * width_multiplier))

        # default shown in paper
        ffn_multiplier = 2
        mv2_exp_mult = 2

        self.conv_1 = conv_2d(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult))
        self.layer_2 = nn.Sequential(InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
                                     InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult))
        self.layer_3 = nn.Sequential(InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
                                     MobileViTBlockv3_v2(channels[3], attn_dim[0], ffn_multiplier, 2,
                                                         patch_size=patch_size))
        self.layer_4 = nn.Sequential(InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
                                     MobileViTBlockv3_v2(channels[4], attn_dim[1], ffn_multiplier, 4,
                                                         patch_size=patch_size))
        self.layer_5 = nn.Sequential(InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
                                     MobileViTBlockv3_v2(channels[5], attn_dim[2], ffn_multiplier, 3,
                                                         patch_size=patch_size))
        self.out1 = nn.Linear(channels[-1], num_classes, bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # FF head
        x = torch.mean(x, dim=[-2, -1])
        x = self.out1(x)

        return x


if __name__ == '__main__':
    image_size = (224, 224)
    num_classes = 3
    width_multiplier = 1
    model = MobileViTv3_v2(image_size, width_multiplier, num_classes, patch_size=(2, 2))

    pth_path = r"/美度模型/backbone/checkpoint_ema_best.pt"
    # pth_path = r"E:\论文\审美认知加工\2024010405MBEgAL\MobileViTv3-PyTorch-master\MobileViTv3-PyTorch-master\mobilevitv3_1_0_0\checkpoint_ema_best.pt"

    checkpoint = torch.load(pth_path, map_location='cpu')
    # model.load_state_dict(checkpoint, strict=True)

    model.eval()
    model.cuda(0)
    time_sum = 0
    for i in range(25):
        data = torch.rand(1, 3, 224, 224)
        data = data.cuda(0)
        start = time.time()
        out = model(data)

        if i == 0:
            continue
        time_sum += time.time() - start
        print('time', time_sum / i)

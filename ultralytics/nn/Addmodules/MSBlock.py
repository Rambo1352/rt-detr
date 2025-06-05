import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BasicBlock_MSBlock', 'BottleNeck_MSBlock']

class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None):
        super(ConvNormLayer, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups)

        self.norm = nn.BatchNorm2d(ch_out)


    def forward(self, inputs):

        out = self.conv(inputs)
        out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out

class SELayer(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction_ratio, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k) -> None:
        super().__init__()

        self.in_conv = Conv(inc, ouc, 1)
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))


class MSBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3,
                 in_down_ratio=2.) -> None:
        super().__init__()

        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        self.in_conv = Conv(inc, in_channel)

        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(int(layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)

        self.attention = None

    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out



class BasicBlock_MSBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 act='relu',
                 variant='b',
                 att=False):
        super(BasicBlock_MSBlock, self).__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2d(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            act='relu')
        """↓ 替换了此处的ConvNormLayer ↓"""
        # 有需要的上面的也可以替换
        self.branch2b = MSBlock(
            ch_out,
            ch_out)
        #         self.branch2b = ConvNormLayer(
        #             ch_in=ch_out,
        #             ch_out=ch_out,
        #             filter_size=3,
        #             stride=1,
        #             act=None)
        """↑ 替换了此处的ConvNormLayer ↑"""


        self.att = att
        if self.att:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)

        if self.att:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = out + short
        out = F.relu(out)

        return out


class BottleNeck_MSBlock(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', att=False):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        """↓ 替换了此处的ConvNormLayer ↓"""
        # 有需要的上面的也可以替换
        self.branch2c = MSBlock(width, ch_out * self.expansion)
        # self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        """↑ 替换了此处的ConvNormLayer ↑"""

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.att = att
        if self.att:
            self.se = SELayer(ch_out)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.att:
            out = self.se(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = F.relu(out)

        return out

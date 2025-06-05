import torch.nn as nn

__all__ = ['LSKA', 'BasicBlock_LSKA', 'BottleNeck_LSKA']

class LSKA(nn.Module):
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn



from collections import OrderedDict
import torch.nn.functional as F


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



class BasicBlock_LSKA(nn.Module):
    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 act='relu',
                 variant='b',
                 att=False):
        super(BasicBlock_LSKA, self).__init__()
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

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None)

        self.att = att
        if self.att:
            self.se = LSKA(ch_out)

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


class BottleNeck_LSKA(nn.Module):
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
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

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
            self.se = LSKA(ch_out * 4)

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

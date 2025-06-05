import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from .DLKAttention import deformable_LKA
from torchvision.ops import Conv2dNormActivation
__all__ = ['vanillanet_dlka02_5']

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    返回 fuse 后的 (weight, bias)，但不改原模块
    """
    W = conv.weight   # (out, in, k, k)
    if conv.bias is not None:
        b = conv.bias
    else:
        b = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)

    rm = bn.running_mean
    rv = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(rv + eps)
    w_bn = (gamma / std).reshape(-1, 1, 1, 1)
    W_f = W * w_bn
    b_f = beta + (b - rm) * gamma / std
    return W_f, b_f

class IWAct(nn.Module):
    """In-Place ReLU + depthwise  conv 激活，支持 deploy fuse."""
    def __init__(self, channels, act_radius=3, deploy=False):
        super().__init__()
        self.deploy = deploy
        kernel_size = act_radius*2+1
        if not deploy:
            self.alpha = nn.BatchNorm2d(channels, eps=1e-6)
            self.w = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))
            trunc_normal_(self.w, std=.02)
        else:
            # deploy 模式直接用 conv depthwise
            self.conv = nn.Conv2d(channels, channels, kernel_size,
                                  padding=act_radius, groups=channels, bias=True)

    def forward(self, x):
        x = torch.relu_(x)
        if self.deploy:
            return self.conv(x)
        else:
            # depthwise conv + BN
            y = nn.functional.conv2d(x, self.w, padding=self.w.shape[-1]//2, groups=x.shape[1])
            return self.alpha(y)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        # fuse BN 到 w、b
        w_f, b_f = fuse_conv_bn(conv=nn.Conv2d(self.w.shape[1], self.w.shape[0],
                                               kernel_size=self.w.shape[-1],
                                               padding=self.w.shape[-1]//2,
                                               groups=self.w.shape[0], bias=True),
                                 bn=self.alpha)
        # 这里 conv.weight,bias 直接赋值
        self.conv = nn.Conv2d(self.w.shape[0], self.w.shape[0],
                              kernel_size=self.w.shape[-1],
                              padding=self.w.shape[-1]//2,
                              groups=self.w.shape[0], bias=True)
        self.conv.weight.data.copy_(w_f)
        self.conv.bias.data.copy_(b_f)
        # 清理
        del self.w, self.alpha
        self.deploy = True

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, act_radius=3, stride=2, deploy=False, ada_pool=None, att=False):
        super().__init__()
        self.deploy = deploy
        self.att = att

        # ---- training mode 下：
        if not deploy:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 1, bias=False),
                nn.BatchNorm2d(in_ch, eps=1e-6),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-6),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)

        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool2d(stride)

        self.act = IWAct(out_ch, act_radius, deploy)
        self.att_layer = deformable_LKA(out_ch) if att else nn.Identity()

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv2(self.conv1(x))
        x = self.pool(x)
        x = self.act(x)
        x = self.att_layer(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return

        # fuse conv1+bn1
        c1 = self.conv1[0]; b1 = self.conv1[1]
        W1, b_1 = fuse_conv_bn(c1, b1)

        # fuse conv2+bn2
        c2 = self.conv2[0]; b2 = self.conv2[1]
        W2, b_2 = fuse_conv_bn(c2, b2)

        # 矩阵相乘，得到一个新的 fused 1×1 conv
        # W1: (Cmid, Cin,1,1) -> flat1: (Cmid, Cin)
        # W2: (Cout, Cmid,1,1) -> flat2: (Cout, Cmid)
        flat1 = W1.flatten(2).squeeze(-1)  # (Cmid, Cin)
        flat2 = W2.flatten(2).squeeze(-1)  # (Cout, Cmid)
        W_new = torch.matmul(flat2, flat1) # (Cout, Cin)
        b_new = b_2 + flat2.matmul(b_1)    # (Cout,)

        # 构建 deploy conv
        in_ch = flat1.shape[1]
        out_ch= flat2.shape[0]
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.conv.weight.data.copy_(W_new.view(out_ch, in_ch, 1, 1))
        self.conv.bias.data.copy_(b_new)

        # 切换激活
        self.act.switch_to_deploy()

        # 删除训练用子模块
        del self.conv1, self.conv2
        self.deploy = True

class VanillaNet(nn.Module):
    def __init__(self, in_ch=3, depths=[2,2,2], dims=[256, 512, 1024, 2048],
                 act_radius=3, strides=[2,2,2], ada_pool=None, deploy=False):
        super().__init__()
        self.deploy = deploy

        # Stem: conv4×4 + activation
        d0 = dims[0]
        if not deploy:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_ch, d0, 4, 4, bias=False),
                nn.BatchNorm2d(d0, eps=1e-6),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(d0, d0, 1, 1, bias=False),
                nn.BatchNorm2d(d0, eps=1e-6),
                IWAct(d0, act_radius, deploy=False)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, d0, 4, 4, bias=True),
                IWAct(d0, act_radius, deploy=True)
            )

        # Stages
        self.blocks = nn.ModuleList()
        for i, s in enumerate(strides):
            ada = ada_pool[i] if ada_pool else None
            self.blocks.append(Block(dims[i], dims[i+1],
                                     act_radius, s,
                                     deploy, ada))

        # 权重初始化
        self.apply(self._init_weights)
        # 直接用 dims 即可
        self.width_list = dims

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = []
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
        feats.append(x)

        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return

        # fuse stem1
        # 拆成两个 conv+bn + activation 再合并为一个 conv+act
        # 先 fuse 前两个 conv+bn → W_a, b_a；再 fuse 后两个 conv+bn → W_b, b_b；
        # 然后串成一个 conv:   W = W_b @ W_a, b = b_b + W_b @ b_a
        seq = self.stem1
        # first conv4×4 + BN
        c0, bn0 = seq[0], seq[1]
        W0, b0 = fuse_conv_bn(c0, bn0)

        # second 1×1 + BN
        c1, bn1 = seq[3], seq[4]
        W1, b1 = fuse_conv_bn(c1, bn1)

        # 串联
        flat0 = W0.flatten(2).squeeze(-1)  # (C1, Cin)
        flat1 = W1.flatten(2).squeeze(-1)  # (C1, C1)
        W_stem = torch.matmul(flat1, flat0)
        b_stem = b1 + flat1.matmul(b0)

        # 重建 stem
        C1, Cin = flat0.shape
        self.stem = nn.Sequential(
            nn.Conv2d(Cin, C1, 4, 4, bias=True),
            IWAct(C1, seq[-1].w.shape[-1]//2, deploy=True)
        )
        self.stem[0].weight.data.copy_(W_stem.view(C1, Cin, 4, 4))
        self.stem[0].bias .data.copy_(b_stem)

        # 清理
        del self.stem1

        # switch 所有 blocks
        for blk in self.blocks:
            blk.switch_to_deploy()


        self.deploy = True

def vanillanet_dlka02_5(**kwargs):
    # dims: [128*4,256*4,512*4,1024*4] = [512,1024,2048,4096]
    return VanillaNet(in_ch=3,
                      dims=[512,1024,2048,4096],
                      strides=[2,2,2],
                      ada_pool=None,
                      act_radius=3,
                      deploy=kwargs.pop('deploy', False),
                      **kwargs)

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import math
import cv2
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from imageio import imsave

def Norm(input):
    mean = np.mean(input)
    std = np.std(input)
    # print((input == mean))
    return (input-mean)/std

def getOperator(inputMaps):
    b, c, h, w = inputMaps.shape
    inputMapsOne = torch.zeros(inputMaps.shape)
    inputMapsOne = inputMaps.clone()
    inputMapsTwo = torch.zeros(inputMaps.shape)
    inputMapsTwo = inputMaps.clone()
    inputMapsThree = torch.zeros(inputMaps.shape)
    inputMapsThree = inputMaps.clone()
    inputMapsFour = torch.zeros(inputMaps.shape)
    inputMapsFour = inputMaps.clone()
    inputMapsFive = torch.zeros(inputMaps.shape)
    inputMapsFive = inputMaps.clone()
    #a = torch.zeros(inputMaps.shape)
    for i in range(b):
        # Sobel
        x = cv2.Sobel(inputMapsOne.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, 1, 0)
        y = cv2.Sobel(inputMapsOne.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, 0, 1)
        #absX = cv2.convertScaleAbs(x) 
        #absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        dst = Norm(dst)
        #a = torch.tensor(dst).permute(2, 0, 1).clone()
        inputMapsOne[i, :, :, :] = Variable(torch.tensor(dst).permute(2, 0, 1).clone())

        # Scharr
        x = cv2.Scharr(inputMapsTwo.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, 1, 0)
        y = cv2.Scharr(inputMapsTwo.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, 0, 1)
        #absX = cv2.convertScaleAbs(x)
        #absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        dst = Norm(dst)
        inputMapsTwo[i, :, :, :] = Variable(torch.tensor(dst).permute(2, 0, 1).clone())
        #inputMapsTwo.permute(0, 2, 3, 1)[i, :, :, :] = Variable(torch.from_numpy(dst))

        # Laplacian
        # ksize = 1, 3, 5, 7 
        dst = cv2.Laplacian(inputMapsThree.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, ksize=3)
        # dst = cv2.convertScaleAbs(gray_lap)
        dst = Norm(dst)
        inputMapsThree[i, :, :, :] = Variable(torch.tensor(dst).permute(2, 0, 1).clone())
 
        # Roberts
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(inputMapsFour.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, kernelx)
        y = cv2.filter2D(inputMapsFour.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, kernely)
        
        #absX = cv2.convertScaleAbs(x)
        #absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        dst = Norm(dst)
        inputMapsFour[i, :, :, :] = Variable(torch.tensor(dst).permute(2, 0, 1).clone())

        # Prewitt
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

        x = cv2.filter2D(inputMapsFive.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, kernelx)
        y = cv2.filter2D(inputMapsFive.permute(0, 2, 3, 1)[i, :, :, :].cpu().detach().numpy(), -1, kernely)
        
        # absX = cv2.convertScaleAbs(x)
        # absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        dst = Norm(dst)
        inputMapsFive[i, :, :, :] = Variable(torch.tensor(dst).permute(2, 0, 1).clone())

    inputMaps = torch.cat((inputMapsOne, inputMapsTwo, inputMapsThree, inputMapsFour, inputMapsFive), dim = 1)
    return inputMaps

class NEW_BLOCK(nn.Module):
    def __init__(self, channel):
        super(NEW_BLOCK, self).__init__()
        #self.b, self.c, self.h, self.w = input_maps.shape
        self.c = channel
        self.conv1 = nn.Conv2d(in_channels=5*self.c, out_channels=self.c, kernel_size=3, padding=1)
        self.pooling1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)
        self.pooling2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)
        self.pooling3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)
        
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=2, stride=2)
        
        self.conv6 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)
        
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)

    def forward(self, input_maps):
        #orign_maps = torch.zeros(input_maps.shape)
        b, c, h, w = input_maps.shape
        orign_maps = input_maps
        input_maps = getOperator(input_maps)
        input_maps = self.conv1(input_maps)
        for i in range(c):
            inputMap = torch.zeros(input_maps.shape)
        h1 = input_maps.shape[2]
        w1 = input_maps.shape[3]
        input_maps = self.pooling1(input_maps)
        input_maps = self.conv2(input_maps)
        h2 = input_maps.shape[2]
        w2 = input_maps.shape[3]
        input_maps = self.pooling2(input_maps)
        input_maps = self.conv3(input_maps)
        h3 = input_maps.shape[2]
        w3 = input_maps.shape[3]
        input_maps = self.pooling3(input_maps)
        input_maps = self.conv4(input_maps)
        input_maps = self.deconv1(input_maps)
        if input_maps.shape[2] != h3:
            input_maps = F.interpolate(input_maps, size=[h3, w3], mode='nearest')
        input_maps = self.conv5(input_maps)
        input_maps = self.deconv2(input_maps)
        if input_maps.shape[2] != h2:
            input_maps = F.interpolate(input_maps, size=[h2, w2], mode='nearest')
        input_maps = self.conv6(input_maps)
        input_maps = self.deconv3(input_maps)
        if input_maps.shape[2] != h1:
            input_maps = F.interpolate(input_maps, size=[h1, w1], mode='nearest')
        input_maps = self.conv7(input_maps)
        
        return orign_maps*input_maps

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


model_cfgs = {
    "regnetx_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    "regnetx_400mf": _mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    "regnetx_600mf": _mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    "regnetx_800mf": _mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    "regnetx_1.6gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    "regnetx_3.2gf": _mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    "regnetx_4.0gf": _mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    "regnetx_6.4gf": _mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    "regnetx_8.0gf": _mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    "regnetx_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    "regnetx_16gf": _mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    "regnetx_32gf": _mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    "regnety_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    "regnety_400mf": _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    "regnety_600mf": _mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    "regnety_800mf": _mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    "regnety_1.6gf": _mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    "regnety_3.2gf": _mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    "regnety_4.0gf": _mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    "regnety_6.4gf": _mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    "regnety_8.0gf": _mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    "regnety_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    "regnety_16gf": _mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    "regnety_32gf": _mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25)
}


def generate_width_depth(wa, w0, wm, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert wa > 0 and w0 > 0 and wm > 1 and w0 % q == 0
    widths_cont = np.arange(depth) * wa + w0
    width_exps = np.round(np.log(widths_cont / w0) / np.log(wm))
    widths_j = w0 * np.power(wm, width_exps)
    widths_j = np.round(np.divide(widths_j, q)) * q
    num_stages, max_stage = len(np.unique(widths_j)), width_exps.max() + 1
    assert num_stages == int(max_stage)
    assert num_stages == 4
    widths = widths_j.astype(int).tolist()
    return widths, num_stages


def adjust_width_groups_comp(widths: list, groups: list):
    """Adjusts the compatibility of widths and groups."""
    groups = [min(g, w_bot) for g, w_bot in zip(groups, widths)]
    # Adjust w to an integral multiple of g
    widths = [int(round(w / g) * g) for w, g in zip(widths, groups)]
    return widths, groups


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 368,
                 out_unit: int = 1000,
                 output_size: tuple = (1, 1),
                 drop_ratio: float = 0.25):
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(in_features=in_unit, out_features=out_unit)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

__all__ = ['MCALayer', 'MCAGate']


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out

class Bottleneck(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1)

        self.conv2 = ConvBNAct(in_c=out_c,
                               out_c=out_c,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_c // group_width)
        if se_ratio > 0:
            self.se = SqueezeExcitation(in_c, out_c, se_ratio)
        else:
            self.se = nn.Identity()
        self.conv3 = ConvBNAct(in_c=out_c, out_c=out_c, kernel_s=1, act=None)
        self.ac3 = nn.ReLU(inplace=True)
        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()
        if (in_c != out_c) or (stride != 1):
            self.downsample = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()
        self.myblock = NEW_BLOCK(24)
        # self.myblock = MCALayer(inp=24)
        # self.myblock = SEModule(channels=48, reduction=2)


    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        # print(x)
        x = self.conv1(x)#Y200:24 100 100;56 50 50;152 25 25;152 13 13;152 13 13;152 13 13
        #Y400:48 100 100;104 50 50;104 25 25;104 25 25;208 25 25;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;440 13 13;
        #X200:24 100 100;56 50 50;152 25 25;152 13 13;152 13 13;152 13 13;368 13 13;
        # Best
        if x.shape[1] == 24 and x.shape[2] == 100:
            x = self.myblock(x)

        x = self.conv2(x)#24 50 50;56 25 25;152 13 13;152 13 13;152 13 13;152 13 13
        #48 50 50;104 25 25;104 25 25;104 25 25;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;440 7 7;
        #X200:24 50 50;56 25 25;152 13 13;152 13 13;152 13 13;152 13 13;368 7 7;
        # if x.shape[1] == 104 and x.shape[2] == 25:
        #     x = self.myblock(x)
        x = self.se(x)#24 50 50;56 25 25;152 13 13;152 13 13;152 13 13;152 13 13
        #48 50 50;104 25 25;104 25 25;104 25 25;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;440 7 7;
        #X200:24 50 50;56 25 25;152 13 13;152 13 13;152 13 13;152 13 13;368 7 7;

        x = self.conv3(x)#24 50 50;56 25 25;152 13 13;152 13 13;152 13 13;152 13 13
        #48 50 50;104 25 25;104 25 25;104 25 25;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;208 13 13;440 7 7;
        #X200:24 50 50;56 25 25;152 13 13;152 13 13;152 13 13;152 13 13;368 7 7;

        x = self.dropout(x)

        shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.ac3(x)
        return x


class RegStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float):
        super(RegStage, self).__init__()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_c = in_c if i == 0 else out_c

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_c=block_in_c,
                                       out_c=out_c,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.children():
            x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    and refer to: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py
    """

    def __init__(self,
                 cfg: dict,
                 in_c: int = 3,
                 num_classes: int = 1000,
                 zero_init_last_bn: bool = True):
        super(RegNet, self).__init__()

        # RegStem
        stem_c = cfg["stem_width"]
        self.stem = ConvBNAct(in_c, out_c=stem_c, kernel_s=3, stride=2, padding=1)

        # build stages
        input_channels = stem_c
        stage_info = self._build_stage_info(cfg)
        for i, stage_args in enumerate(stage_info):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(in_c=input_channels, **stage_args))
            input_channels = stage_args["out_c"]
 
        # RegHead
        self.head = RegHead(in_unit=input_channels, out_unit=num_classes)

        # initial weights
        print(self.modules())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def forward(self, x: Tensor) -> Tensor:

        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def _build_stage_info(cfg: dict):
        wa, w0, wm, d = cfg["wa"], cfg["w0"], cfg["wm"], cfg["depth"]
        widths, num_stages = generate_width_depth(wa, w0, wm, d)

        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_widths, stage_groups = adjust_width_groups_comp(stage_widths, stage_groups)

        info = []
        for i in range(num_stages):
            info.append(dict(out_c=stage_widths[i],
                             depth=stage_depths[i],
                             group_width=stage_groups[i],
                             se_ratio=cfg["se_ratio"]))

        return info


def create_regnet(model_name="RegNetX_200MF", num_classes=6):
    model_name = model_name.lower().replace("-", "_")
    if model_name not in model_cfgs.keys():
        print("support model name: \n{}".format("\n".join(model_cfgs.keys())))
        raise KeyError("not support model name: {}".format(model_name))

    model = RegNet(cfg=model_cfgs[model_name], num_classes=num_classes)
    return model
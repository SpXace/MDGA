from PIL import Image
import math
import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import torch.nn.functional as F
import numpy as np
from imageio import imsave
from matplotlib import pyplot as plt
from torch.autograd import Variable
def visulize_attention_ratio(img_path, attention_mask, ratio=0.5, cmap="jet"):
    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

def Norm(input):
    mean = np.mean(input)
    std = np.std(input)
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
        #inputMapsThree.permute(0, 2, 3, 1)[i, :, :, :] = Variable(torch.from_numpy(dst))

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
        #inputMapsFour.permute(0, 2, 3, 1)[i, :, :, :] = Variable(torch.from_numpy(dst))

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
        #inputMapsFive.permute(0, 2, 3, 1)[i, :, :, :] = Variable(torch.from_numpy(dst))
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
        orign_maps = input_maps
        input_maps = getOperator(input_maps)
        input_maps = self.conv1(input_maps)
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

class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    message = "basic"

    def __init__(self, in_channels, out_channels, strides, is_se=True):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels, 16)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return F.relu(out)

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

class BottleNeck(nn.Module):
    """
    BottleNeck block for RestNet-50, ResNet-101, ResNet-152
    """
    message = "bottleneck"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)

        # fit input with residual output
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        print(self.shortcut(x))
        return F.relu(out)

class ResNet(nn.Module):
    """
    building ResNet_34
    """

    def __init__(self, block: object, groups: object, num_classes=1000) -> object:
        super(ResNet, self).__init__()
        self.channels = 64  # out channels from the first convolutional layer
        self.block = block

        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(7)
        patches = 512 if self.block.message == "basic" else 512 * 4
        self.fc = nn.Linear(patches, num_classes)  # for 224 * 224 input size
        self.myBlock = NEW_BLOCK(64)
         # self.myBlock = MCALayer(inp=64)
        # self.myBlock = SEModule(channels=64, reduction=2)

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        
        # 18：64 100 100；
        # 34：64
        # out = self.myBlock(out)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.conv2_x(out)

        #18：64 50 50
        #34：64
        out = self.myBlock(out)
        out = self.conv3_x(out)
        #18：256 25 25  
        
        out = self.conv4_x(out)
        #18：512 13 13；
        #34：256
        
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out

def ResNet18(num_classes=6):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=6):
    return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], num_classes=num_classes)

def ResNet50(num_classes=6):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes)

def ResNet101(num_classes=6):
    return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], num_classes=num_classes)

def ResNet152(num_classes=6):
    return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], num_classes=num_classes)

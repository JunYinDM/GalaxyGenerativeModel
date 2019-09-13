from __future__ import print_function

import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset , DataLoader


import math
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm




model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BlurPool(nn.Module):
    def __init__(self, stride):
        super(BlurPool, self).__init__()
        self.kernel = nn.Parameter(torch.from_numpy((np.array([[1, 4, 6, 4, 1],
                                              [4, 16, 24, 16, 4],
                                              [6, 24, 36, 24, 6],
                                              [4, 16, 24, 16, 4],
                                              [1, 4, 6, 4, 1]])/256.0).astype('float32')),
                                   requires_grad=False).view(1, 1, 5, 5)
        self.stride = stride

    def forward(self, x):
        num_dims = x.size(1)
        kernel = self.kernel.repeat(num_dims, 1, 1, 1).to(x.device)
        x = F.conv2d(x, kernel, groups=num_dims, stride=self.stride, padding=2)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, anti_alias=False):
        super(BasicBlock, self).__init__()

        if anti_alias and stride != 1:
            self.conv1 = nn.Sequential(conv3x3(inplanes, planes, 1),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(inplace=True),
                                       BlurPool(stride=stride))
        else:
            self.conv1 = nn.Sequential(conv3x3(inplanes, planes, stride),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(inplace=True))

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model   


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batch_size, num_dims, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X C X N
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        q_k = math.sqrt(num_dims // 2)
        attention = F.softmax(energy/q_k, dim=2)  # BX (N) X (N)
        proj_value = x.view(batch_size, num_dims, -1)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, num_dims, width, height)

        out = self.value_conv(out)

        out = self.bn(out)

        out = out + x
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, in_dims=128):
        super(PositionalEncoding, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dims+2, in_dims, 1, stride=1, bias=False),
                                  nn.BatchNorm2d(in_dims),
                                  nn.ReLU())

    def forward(self, x):
        batch_size, num_dims, width, height = x.size()
        width_axis = torch.arange(-width//2, width//2, step=1, dtype=x.dtype,
                                  device=x.device).view(1, 1, width, 1).repeat(1, 1, 1, height)
        height_axis = torch.arange(-height//2, height//2, step=1, dtype=x.dtype,
                                   device=x.device).view(1, 1, 1, height).repeat(1, 1, width, 1)
        axis = torch.cat((width_axis, height_axis), dim=1).repeat(batch_size, 1, 1, 1)
        x = torch.cat((x, axis), dim=1)
        x = self.conv(x)
        return x
class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=128, zero_init_residual=True, non_local=False,
                 anti_alias=False):
        super(ResNet, self).__init__()
        self.anti_alias = anti_alias
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], anti_alias=anti_alias)
        if non_local:
            self.layer1 = nn.Sequential(self.layer1,
                                        SelfAttn(64))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, anti_alias=anti_alias)
        if non_local:
            self.layer2 = nn.Sequential(self.layer2,
                                        SelfAttn(128))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, anti_alias=anti_alias)
        if non_local:
            self.layer3 = nn.Sequential(self.layer3,
                                        SelfAttn(256))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, anti_alias=anti_alias)
        if non_local:
            self.layer4 = nn.Sequential(self.layer4,
                                        SelfAttn(512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, SelfAttn):
                    nn.init.constant_(m.bn.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, anti_alias=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, 1),
                              BlurPool(stride=stride))
                if anti_alias else conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, anti_alias=anti_alias))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.num_parameters = args.num_parameters
        self.nn = nn.Sequential(nn.Conv2d(1+self.num_parameters, 64, 3, stride=2, bias=False),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(64, 128, 3, stride=2, bias=False),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(128, 256, 3, stride=2, bias=False),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(256, 512, 3, stride=2, bias=False),
                                nn.BatchNorm2d(512),
                                nn.LeakyReLU(0.2, True),
                                nn.Conv2d(512, 1, 3, stride=1, bias=True))

    def forward(self, x):
        return self.nn(x)    
    
    
    
# fully connected     
class autoencoder_0(nn.Module):
    def __init__(self):
        super(autoencoder_0, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(96 * 96, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),

            nn.Linear(64, 10),
            nn.ReLU(True))
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 96 * 96),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# conv
class autoencoder_1(nn.Module):
    def __init__(self):
        super(autoencoder_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 10, stride=2, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
    
  
 #conv with latent dim = 9 

class autoencoder_2(nn.Module):
    def __init__(self):
        super(autoencoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), 
            
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  
            
            nn.Conv2d(8, 2, 3, stride=1, padding=1),  # b, 2,
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 2,
            
            nn.Conv2d(2, 1, 3, stride=2, padding=1),  # b, 1, 3, 3 
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(1, 2, 3, stride=1, padding =1 ),  # b, 2, 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2, 8, 3, stride=1),  # b, 8, 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16,  
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 8, 2, stride=3),  # b, 8,
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=3),  # b, 1, 96, 96
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    

class autoencoder_3(nn.Module):
    def __init__(self):
        super(autoencoder_3, self).__init__()     # 1, 96, 96 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=3 ),  # 4, 32, 32, 
            nn.ReLU(True),
            
            nn.Conv2d(4, 8, 3, stride=3, padding=2),  # 8, 12 , 12 
            nn.ReLU(True),
            
            nn.Conv2d(8, 4, 3, stride=3),  #  4, 4 ,4 
            nn.ReLU(True),
            
            nn.Conv2d(4, 1, 2, stride=2, padding=1),  # 1, 3, 3 
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(1, 2, 3, stride=1, padding =1 ),  # b, 2, 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(2, 4, 3, stride=1),  # b, 8, 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4, 8, 3, stride=2),  # b, 16,  
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8, 4, 2, stride=3),  # b, 8,
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 3, stride=3),  # b, 1, 96, 96
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    



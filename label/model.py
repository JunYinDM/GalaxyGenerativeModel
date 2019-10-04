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




class autoencoder_linear(nn.Module):
    def __init__(self):
        super(autoencoder_linear, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64, 256),
            nn.ReLU(True),
            nn.Linear(256, 9))
        self.decoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 64))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class autoencoder_999(nn.Module):   # 
    def __init__(self):            #  1x 64 x 64 
        
        super(autoencoder_999, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 64 * 64 * 64  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),   # 64 * 31 * 31 
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 *16 * 16 
            nn.ReLU(True),
            nn.BatchNorm2d(128), 
            nn.MaxPool2d(2, stride=1),  # 128 * 15 * 15 
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 256 * 15 * 15 
            nn.ReLU(True),
            nn.BatchNorm2d(256), 
            nn.MaxPool2d(2, stride=1),  # b, 256, 14, 14 
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # b, 14, 14 
            nn.BatchNorm2d(256), 
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),  # b, 2,  
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2),  # b, 8, 55, 55 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=1),  # b, 16, 
            nn.BatchNorm2d(32), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=1),  # b, 1,  
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x,z
    
class autoencoder_333(nn.Module):   # 
    def __init__(self):            #  1x 64 x 64 
        
        super(autoencoder_333, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 64 * 64 * 64  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),   # 64 * 31 * 31 
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 *16 * 16 
            nn.ReLU(True),
            nn.BatchNorm2d(128), 
            nn.MaxPool2d(2, stride=1),  # 128 * 15 * 15 
            
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b,  * 15 * 15 
            nn.ReLU(True),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2, stride=1),  # b, 256, 14, 14 
            
            nn.Conv2d(64, 1, 3, stride=1, padding=1),  # b, 1  x 14, 14 
            nn.BatchNorm2d(1), 
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 3, stride=2),  # b, 2,  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 8, 55, 55 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1),  # b, 16, 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=1),  # b, 1,  
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x,z
    
    
        
    
        

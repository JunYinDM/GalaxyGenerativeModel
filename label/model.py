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
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # b,256*  14, 14 
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
    

class autoencoder_333_2(nn.Module):   # 
    def __init__(self):            #  1x 64 x 64 
        
        super(autoencoder_333_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 64 * 64 * 64  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),   # 64 * 31 * 31   (31 -2)/2 +1 
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 *16 * 16 
            nn.ReLU(True),
            nn.BatchNorm2d(128), 
            nn.MaxPool2d(2, stride=1),  # 128 * 15 * 15 
            
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b, 64 * 15 * 15 
            nn.ReLU(True),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2, stride=1),  # b, 64, 14, 14 
            
            nn.Conv2d(64, 1, 3, stride=1, padding=1),  # b, 1  x 14, 14 
           # nn.Linear(14*14, 49)

            
        #    nn.BatchNorm2d(1), 
         #   nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 3, stride=2),  # b, 64,  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 64, 55, 55 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1),  # 128, 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=1),  # b, 64,  
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x,z   
    
    
class vae_501(nn.Module):  
    def __init__(self):            #  1x 64 x 64 
        
        super(vae_501, self).__init__()
        
        self.fc11 = nn.Linear(14*14, 14)
        self.fc12 = nn.Linear(14*14, 14)

        self.fc21 = nn.Linear(14, 14*14)
        
        self.enc = nn.Sequential(
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
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 3, stride=2),  # b, 2,  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 8, 55, 55 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1),  # b, 16, 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=1),  # b, 1,  64 x 64 
        )

        
        
    def encoder(self, x):
        h1=self.enc(x)
      #  print("h1", h1.shape)
        h2=h1.view(-1,14*14)
        
        return  self.fc11(h2), self.fc12(h2)
        
    def reparametrize(self, mu, logvar):  # mu, sigma --> mu + sigma * N(0,1)
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
        
        
    def decoder (self, z):
        
        h3= self.fc21(z)
        h4=h3.view(-1,1,14,14)

        return self.dec(h4)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    
    
    
class autoencoder_1014(nn.Module):   # 
    def __init__(self):            #  1x 64 x 64 
        
        super(autoencoder_1014, self).__init__()
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
            
            nn.Conv2d(64, 4, 3, stride=1, padding=1),  # b, 4  x 14, 14 
           # nn.BatchNorm2d(1), 
            #nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, 3, stride=2),  # b, ,  
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
    

    
class autoencoder_1015(nn.Module):   # 
    def __init__(self):            #  1x 64 x 64 
        
        super(autoencoder_1015, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 64 * 64 * 64  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),   # 64 * 31 * 31   (31 -2)/2 +1 
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 *16 * 16 
            nn.ReLU(True),
            nn.BatchNorm2d(128), 
            nn.MaxPool2d(2, stride=1),  # 128 * 15 * 15 
            
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b, 64 * 15 * 15 
            nn.ReLU(True),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2, stride=1),  # b, 64, 14, 14 
            
            nn.Conv2d(64, 1, 3, stride=1, padding=1),  # b, 1  x 14, 14 
        #    nn.Linear(14*14, 14*14)

            
        #    nn.BatchNorm2d(1), 
         #   nn.ReLU(True)
        )
        
        self.lin_1= nn.Linear(14*14, 14*14)
        self.lin_2= nn.Linear(14*14, 14*14)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 3, stride=2),  # b, 64,  
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 64, 55, 55 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1),  # 128, 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=1),  # b, 64,  

        )

    def forward(self, x):
        z_1 = self.encoder(x)
        z = self.lin_1(z_1.view(z_1.size(0),14*14))
        z_3 = self.lin_2(z)
        
        x=self.decoder(z_3.view(z.size(0),1,14,14))
        return x,z   
    

    


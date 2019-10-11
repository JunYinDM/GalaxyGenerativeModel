from __future__ import print_function

import os
import sys
import glob
import h5py
import numpy as np
import math


import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset , DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorboard
import tensorboardX
from torch.utils.tensorboard import SummaryWriter


from log import Logger
from data import  trainDataset, testDataset, trainlabelDataset,testlabelDataset
from util import r2, mse, rmse, mae, pp_mse, pp_rmse, pp_mae
from model import autoencoder_999, autoencoder_333_2

def to_img(x):   # image size 
    x = x.view(x.size(0), 1, 64, 64)
    return x




if not os.path.exists('./gal_img1001'):
    os.mkdir('./gal_img1001')

    
dataset= trainlabelDataset()
dataloader= DataLoader(dataset=dataset, batch_size=64,shuffle=True,drop_last=True)

test_dataset = testlabelDataset()
test_dataloader= DataLoader(dataset=test_dataset, batch_size=64,shuffle=True,drop_last=True)


writer = SummaryWriter("run1001/expody807",)  ################################################### change name 

num_epochs =20000
batch_size = 64
learning_rate = 1e-4
weight=1e1

model = autoencoder_333_2().cuda()   ############################################################## AE model 
model.load_state_dict(torch.load('gal_img1001/exp483_8490.pth'))    ###

criterion_mean = nn.L1Loss(reduction='mean')
criterion_none = nn.L1Loss(reduction='none')



#scheduler 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000,10000], gamma=0.1)




for epoch in range(num_epochs):
    total_loss = 0.0
    total_mse = 0.0 
    total_recon=0.0
    total_latent=0.0
    
    num_examples = 0.0
    test_num_examples=0.0
    
    
    test_total_loss = 0.0    
    test_total_mse=0.0
    test_total_recon=0.0
    test_total_latent=0.0
    model.train()
    for data in dataloader:
        img,label= [x.type(torch.float32).cuda() for x in data]
        img = img.view(img.size(0), 1,64,64)

       # print(img.shape)
       # print("",img.sum())
       # print("",img[0].sum())
        # forward
        output, z = model(img)
        z=z.view(z.size(0),14*14)
       # print("output ",output.shape)
       # print("z ",z.shape)
    ################################################## Loss function with regularizing Z ########################
        
        loss= (( criterion_none(output, img)/(img.sum(dim=3).sum(dim=2).sum(dim=1))).mean()) + (criterion_none(z[:,:7], label)).mean() /weight  + 1e-7*torch.norm(z[:,7:],p=1)

        loss_recon=(( criterion_none(output, img)/(img.sum(dim=3).sum(dim=2).sum(dim=1))).mean())
        loss_latent=(criterion_none(z[:,:7], label)).mean() /weight
        
        
        MSE_loss = nn.MSELoss()(output, img)
        batch_size = img.size(0)
        total_loss += loss.item() * batch_size
        total_mse += MSE_loss.item() * batch_size
        total_recon+= loss_recon.item() * batch_size
        total_latent+= loss_latent.item() * batch_size

        num_examples += batch_size

        
        optimizer.zero_grad()
    # backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    model.eval()
    for data in test_dataloader:
        test_img,test_label= [x.type(torch.float32).cuda() for x in data]

        test_img = test_img.view(test_img.size(0), 1,64,64)
       # print(img.shape)
        

        # forward
        test_output,test_z = model(test_img)
        test_z=test_z.view(test_z.size(0),14*14)
                
       #print("output ",output.shape)
       # test_loss = criterion(test_output, test_img) + criterion(z[:,:7], label) /(1e)  #  + 1e-5*  criterion(test_z[:,:7], test_label) 
        test_loss= (( criterion_none(test_output, test_img)/(test_img.sum(dim=3).sum(dim=2).sum(dim=1))).mean())  + (criterion_none(test_z[:,:7], test_label)  /weight).mean() 
        
        test_loss_recon= (( criterion_none(test_output, test_img)/(test_img.sum(dim=3).sum(dim=2).sum(dim=1))).mean()) 
        test_loss_latent= (criterion_none(test_z[:,:7], test_label)  /weight).mean() 
        
        test_MSE_loss = nn.MSELoss()(test_output, test_img)
        batch_size = test_img.size(0)
        test_total_loss += test_loss.item() * batch_size
        test_total_mse += test_MSE_loss.item() * batch_size
        test_total_recon+= test_loss_recon.item() * batch_size
        test_total_latent+= test_loss_latent.item() * batch_size


        test_num_examples += batch_size

    writer.add_scalar('Loss/train',total_loss / num_examples,epoch)
    writer.add_scalar('Mse/train', total_mse / num_examples,epoch)   
    writer.add_scalar('Recon/train', total_recon / num_examples,epoch)        
    writer.add_scalar('Latent/train', total_latent / num_examples,epoch)        
    writer.add_scalar('Loss/test',test_total_loss / test_num_examples,epoch)
    writer.add_scalar('Mse/test', test_total_mse / test_num_examples,epoch)
    writer.add_scalar('Recon/test', test_total_recon / test_num_examples,epoch)        
    writer.add_scalar('Latent/test', test_total_latent / test_num_examples,epoch)   
    '''
    print('epoch [{}/{}], loss:{:.6f}, MSE_loss:{:.6f}, recon_loss:{:.6f}, latent_loss:{:.6f}'
          .format(epoch + 1, num_epochs, total_loss / num_examples, total_mse/ num_examples, total_recon / num_examples, total_latent/ num_examples))     
    
    print('epoch [{}/{}], test_loss:{:.6f}, test_MSE_loss:{:.6f}, test_recon_loss:{:.6f}, test_latent_loss:{:.6f}'
          .format(epoch + 1, num_epochs, test_total_loss / num_examples, test_total_mse/ num_examples, test_total_recon / num_examples, test_total_latent/ num_examples))   
          '''

    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        test_x = to_img(test_img.cpu().data)    ########## change name 
        test_x_hat = to_img(test_output.cpu().data)
        torch.save(x, './gal_img1001/expody807_x_{}.pt'.format(epoch))
        torch.save(x_hat, './gal_img1001/expody807_x_hat_{}.pt'.format(epoch))
        torch.save(test_x, './gal_img1001/expody807_test_x_{}.pt'.format(epoch))
        torch.save(test_x_hat, './gal_img1001/expody807_test_x_hat_{}.pt'.format(epoch))
        torch.save(model.state_dict(), './gal_img1001/expody807_{}.pth'.format(epoch))     
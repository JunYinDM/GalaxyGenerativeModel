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
import tensorboard
import tensorboardX
from torch.utils.tensorboard import SummaryWriter


from log import Logger
from data import gDataset, trainDataset, testDataset
from util import r2, mse, rmse, mae, pp_mse, pp_rmse, pp_mae
from model import autoencoder_1111,autoencoder_999


if not os.path.exists('./gal_img924'):
    os.mkdir('./gal_img924')

    
    
def to_img(x):   # image size 56 * 56 
    x = x.view(x.size(0), 1, 56, 56)
    return x

dataset= trainDataset()
dataloader= DataLoader(dataset=dataset, batch_size=64,shuffle=True,drop_last=True)

test_dataset = testDataset()
test_dataloader= DataLoader(dataset=test_dataset, batch_size=64,shuffle=True,drop_last=True)


writer = SummaryWriter("run917/924_exp1111_1e-7",)  ################################################### change name 

num_epochs =200000
batch_size = 64
learning_rate = 1e-4



model = autoencoder_999().cuda()   ############################################################## AE model 
criterion = nn.L1Loss()

#scheduler 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,20000,50000], gamma=0.1)




for epoch in range(num_epochs):
    total_loss = 0.0
    total_mse = 0.0 
    num_examples = 0.0
    test_num_examples=0.0
    
    
    test_total_loss = 0.0    
    test_total_mse=0.0
    
    model.train()
    i=0
    for data in dataloader:
        img = data
        img = img.type(torch.float32)
        img = img.view(img.size(0), 1,56,56)
        img = img.cuda()
      #  print(img.shape)
        

        # forward
        output,z = model(img)
       # output, z,latent = model(img)
      #  print('z size',z.size)
        
        latent = z.mean(axis=2, keepdim=True).mean(axis=3, keepdim=True)  # change from 256x3x3 to 256 x1 x1 
      #  print("latent size",latent.size())

        label =torch.ones([64,5,1,1]).cuda()
     #   print("label size",label.size())

        diff= (latent[:,:5,:,:]-label) 
     #   print(diff.size())
    #    i=i+1
     #   print("the",i,"th iteration")
                
    #    print("output ",output.shape)
    ################################################## Loss function with regularizing Z ########################
        loss = criterion(output, img) + 1e-7*torch.norm(z,p=1) + 1e-7*torch.norm(diff,p=1)
        
        
        MSE_loss = nn.MSELoss()(output, img)
        batch_size = img.size(0)
     #   print("batch_size",batch_size)
        total_loss += loss.item() * batch_size
        total_mse += MSE_loss.item() * batch_size
        num_examples += batch_size

        
        optimizer.zero_grad()
    # backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    model.eval()
    for data in test_dataloader:
        test_img = data
        test_img = test_img.type(torch.float32)
        test_img = test_img.view(test_img.size(0), 1,56,56)
        test_img = test_img.cuda()
       # print(img.shape)
        

        # forward
     #   test_output,z_output,latent_output = model(test_img)
        test_output,z_output = model(test_img)
        
        
       #print("output ",output.shape)
        test_loss = criterion(test_output, test_img) 
        
        
        test_MSE_loss = nn.MSELoss()(test_output, test_img)
        batch_size = test_img.size(0)
        test_total_loss += test_loss.item() * batch_size
        test_total_mse += test_MSE_loss.item() * batch_size
        test_num_examples += batch_size

        
    writer.add_scalar('Loss/train',total_loss / num_examples,epoch)
    writer.add_scalar('Mse/train', total_mse / num_examples,epoch)        
    writer.add_scalar('Loss/test',test_total_loss / test_num_examples,epoch)
    writer.add_scalar('Mse/test', test_total_mse / test_num_examples,epoch)
    

    
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, total_loss / num_examples, total_mse/ num_examples))    
    print(' epoch [{}/{}],test_loss:{:.4f}, test_MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, test_total_loss / test_num_examples, test_total_mse/ test_num_examples))

    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        test_x = to_img(test_img.cpu().data)    ########## change name 
        test_x_hat = to_img(test_output.cpu().data)
        torch.save(x, './gal_img924/exp1111_x_{}.pt'.format(epoch))
        torch.save(x_hat, './gal_img924/exp1111_x_hat_{}.pt'.format(epoch))
        torch.save(test_x, './gal_img924/exp1111_test_x_{}.pt'.format(epoch))
        torch.save(test_x_hat, './gal_img924/exp1111_test_x_hat_{}.pt'.format(epoch))
        torch.save(model.state_dict(), './gal_img924/exp1111_{}.pth'.format(epoch))       
           
    










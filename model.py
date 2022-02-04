 
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
#from data import   trainlabelDataset_reduced,testlabelDataset_reduced
from util import r2, mse, rmse, mae, pp_mse, pp_rmse, pp_mae
#from model import autoencoder_999, autoencoder_333_2,autoencoder_1015




################    some initiliazations  

expname='cae_9098'

num_epochs =40000
batch_size = 64
learning_rate = 1e-3
#weight=1e3
reg=1e-3
################    




def to_img(x):   # image size 
    x = x.view(x.size(0), 1, 64, 64)
    return x


if not os.path.exists('./gal_img1107'):
    os.mkdir('./gal_img1107')
    

if not os.path.exists('./run1107'):
    os.mkdir('./run1107')    


    

class trainlabelDataset_cae(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
        
        
        f= h5py.File('train_yay_preproc.h5','r')
        #f= h5py.File('train_preproc.h5','r')
        image = f['img'][:]    
        cat = f['cat'][:]
        label = f['label'][:]
        snr = f['snr'][:]
        sigma = f['sigma'][:]
        image_nonoise = f['img_nonoise'][:]    
        image_withnoise = f['img_withnoise'][:]    

        f.close()
        
        
        
        
        image.astype('float32')
        image_nonoise.astype('float32')
        image_withnoise.astype('float32')
        cat.astype('float32')
        label.astype('float32')
        snr.astype('float32')
        sigma.astype('float32')

        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.imagenonoise= torch.from_numpy(image_nonoise[:])
        self.imagewithnoise= torch.from_numpy(image_withnoise[:])
        self.cat= torch.from_numpy(cat[:])
        self.label=torch.from_numpy(label[:])
        self.snr=torch.from_numpy(snr[:])
        self.sigma=torch.from_numpy(sigma[:])

        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.imagenonoise[index],self.cat[index], self.label[index], self.snr[index] , self.sigma[index] 

    
class testlabelDataset_cae(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
        
        f= h5py.File('test_yay_preproc.h5','r')
        #f= h5py.File('test_preproc.h5','r')
        image = f['img'][:]    
        cat = f['cat'][:]
        label = f['label'][:]
        snr = f['snr'][:]
        sigma = f['sigma'][:]
        image_nonoise = f['img_nonoise'][:]    
        image_withnoise = f['img_withnoise'][:]    
 
        f.close()
        
        
        
        image.astype('float32')
        image_nonoise.astype('float32')
        image_withnoise.astype('float32')
        cat.astype('float32')
        label.astype('float32')
        snr.astype('float32')
        sigma.astype('float32')
    
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.imagenonoise= torch.from_numpy(image_nonoise[:])
        self.imagewithnoise= torch.from_numpy(image_withnoise[:])
        self.cat= torch.from_numpy(cat[:])
        self.label=torch.from_numpy(label[:])
        self.snr=torch.from_numpy(snr[:])
        self.sigma=torch.from_numpy(sigma[:])

        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.imagenonoise[index],self.cat[index], self.label[index], self.snr[index] , self.sigma[index] 
        
    

    

class autoencoder_1110_con(nn.Module):   # 
    def __init__(self):            #  4x 64 x 64 
        
        super(autoencoder_1110_con, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, 3, stride=1),  # 128 * 62 * 62 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, 3, stride=1),  #   256 * 60 * 60 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, stride=2),  #  512 * 29 * 29 
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            
            nn.Conv2d(512, 1024, 3, stride=1),  #  1024 * 27 * 27 
            nn.BatchNorm2d(1024), 
            nn.ReLU(True),
            
            
            nn.Conv2d(1024, 512, 3, stride=1),  #  512 * 25*25
            nn.BatchNorm2d(512),     
            nn.ReLU(True),
            
            nn.Conv2d(512, 256, 3, stride=1, padding=1 ),  #  256 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            
            
            nn.Conv2d(256, 128, 3, stride=2, padding=2),  # b, 128 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
            
            
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b, 64 x 16 x16 
            nn.BatchNorm2d(64),       
            nn.ReLU(True),
            
            
            nn.Conv2d(64, 1, 3, stride=1), ) # b,    1 * 12 *12          )
       
    
        self.lin_1= nn.Linear(12*12+1, 12*12+1)
        self.lin_2= nn.Linear(12*12+1, 14*14)


        self.decoder = nn.Sequential(

            
            
            nn.ConvTranspose2d(1, 128, 3, stride=2,padding=1),  # b, 64,  29,29 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 256, 3, stride=2),  # b, 64, 59, 59 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 512, 3, stride=1),  # b, 64, 59, 59 
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, stride=1),  # b, 64, 59, 59 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),            
            
            nn.ConvTranspose2d(256, 128, 3, stride=1),  # 128, 61, 61
            nn.BatchNorm2d(128), 
            nn.ReLU(True),              ##### add in attention  
            
            nn.ConvTranspose2d(128, 1, 3, stride=1),  # b, 1,64,64          
        )
        
        
        
        
    def forward(self, x,img_orig,sigma):
        z_1 = self.encoder(x)
        
        # add the average pixel value as the last neuron in the fully connected layer 
        #img 64x64, normalize by sigma

        
        avg=  img_orig.sum(dim=3).sum(dim=2).sum(dim=1)/64/64/sigma
        
        trained_=z_1.view(z_1.size(0),12*12)
        
        
        avg_flux= avg.view(x.size(0),1)
    
        z_2= torch.cat((trained_,avg_flux),dim=1)
        
        
        
        
        
        
        z = self.lin_1(z_2)
        
        z_3 = self.lin_2(z)
        
        x=self.decoder(z_3.view(z.size(0),1,14,14))
        
        return  x,z   
    
    
    


    
    
dataset= trainlabelDataset_cae()
dataloader= DataLoader(dataset=dataset, batch_size=64,shuffle=True,drop_last=True,)


test_dataset = testlabelDataset_cae()
test_dataloader= DataLoader(dataset=test_dataset, batch_size=64,shuffle=True,drop_last=True)



writer = SummaryWriter('run1107/'+expname)  ################################################### change name 





model = autoencoder_1110_con().cuda()   ############################################################## AE model 
#model.load_state_dict(torch.load('gal_img1107/cae_9031_4_train_best.pth'))    

criterion_none = nn.L1Loss(reduction='none')
criterion_mean = nn.L1Loss(reduction='mean')
criterion_none_mse = nn.MSELoss(reduction='none')
criterion_mean_mse = nn.MSELoss(reduction='mean')


#scheduler 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,2000], gamma=0.1)

best_test_loss = 1e9


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
        img_orig,img,label,snr,sigma= [x.type(torch.float32).cuda() for x in data]
        
        img_orig = img_orig.view(img_orig.size(0), 1,64,64)
        img = img.view(img.size(0), 6,64,64)

       # print(img.shape)
       # print("",img.sum())
       # print("",img[0].sum())
        # forward
        output, z = model(img,img_orig,sigma)
        z=z.view(z.size(0),12*12+1)
       # print("output ",output.shape)
       # print("z ",z.shape)
    ################################################## Loss function with regularizing Z ########################
        flux_for_scaling=img_orig.sum(dim=3).sum(dim=2).sum(dim=1)
        
        loss=  (criterion_none(output, img_orig).sum(dim=3).sum(dim=2).sum(dim=1)/(flux_for_scaling+sigma**2)
).mean() + ( criterion_none(z[:,:5], label).sum(dim=1)* snr/1e1).mean()  +( criterion_none(z[:,1:3], label[:,1:3]).sum(dim=1)* snr/1.5e1).mean()+1e-3*torch.norm(z[:,5:],p=1)
       

        loss_recon=(criterion_none(output, img_orig).sum(dim=3).sum(dim=2).sum(dim=1)/ flux_for_scaling).mean()
        loss_latent=(criterion_none(z[:,:5], label)).mean() 
        
        
        MSE_loss = nn.MSELoss()(output, img_orig)
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
        
       # print("running")
        
        
               
   

    model.eval()
    for data in test_dataloader:
        test_img_orig,test_img,test_label,snr,sigma= [x.type(torch.float32).cuda() for x in data]
        
        test_img_orig = test_img_orig.view(test_img_orig.size(0), 1,64,64)
        test_img = test_img.view(img.size(0), 6,64,64)

       # print(img.shape)
       # print("",img.sum())
       # print("",img[0].sum())
        # forward
        test_output, test_z = model(test_img,test_img_orig,sigma)
        test_z=test_z.view(test_z.size(0),12*12+1)
       # print("output ",output.shape)
       # print("z ",z.shape)
    ################################################## Loss function with regularizing Z ########################
        test_flux_for_scaling=test_img_orig.sum(dim=3).sum(dim=2).sum(dim=1)
        
        test_loss=  (criterion_none(test_output, test_img_orig).sum(dim=3).sum(dim=2).sum(dim=1)/(flux_for_scaling+sigma**2)).mean() + ( criterion_none(test_z[:,:5], test_label).sum(dim=1)* snr/1e1).mean()  +( criterion_none(test_z[:,1:3], test_label[:,1:3]).sum(dim=1)* snr/1.5e1).mean()+1e-3*torch.norm(test_z[:,5:],p=1)

        
        
        test_loss_recon=(criterion_none(test_output, test_img_orig).sum(dim=3).sum(dim=2).sum(dim=1)/ test_flux_for_scaling).mean()
        test_loss_latent=(criterion_none(test_z[:,:5], test_label)).mean() 
        
        
        test_MSE_loss = nn.MSELoss()(test_output, test_img_orig)
        batch_size = test_img.size(0)
        test_total_loss += test_loss.item() * batch_size
        test_total_mse += test_MSE_loss.item() * batch_size
        test_total_recon+= test_loss_recon.item() * batch_size
        test_total_latent+= test_loss_latent.item() * batch_size
        test_num_examples += batch_size
        
        
    if epoch  == 1:
        print("running starts")
        
    if epoch % 10 == 0:
        x = to_img(img_orig.cpu().data)
        x_hat = to_img(output.cpu().data)
        test_x = to_img(test_img_orig.cpu().data)    ########## change name 
        test_x_hat = to_img(test_output.cpu().data)        
        torch.save(x, './gal_img1107/'+expname+'_x_{}.pt'.format(epoch))
        torch.save(x_hat, './gal_img1107/'+expname+'_x_hat_{}.pt'.format(epoch))
        torch.save(test_x, './gal_img1107/'+expname+'_test_x_{}.pt'.format(epoch))
        torch.save(test_x_hat, './gal_img1107/'+expname+'_test_x_hat_{}.pt'.format(epoch))
        torch.save(model.state_dict(), './gal_img1107/'+expname+'_train_{}.pth'.format(epoch))  
        
    if best_test_loss > test_total_loss:
        best_test_loss = test_total_loss
        x = to_img(img_orig.cpu().data)
        x_hat = to_img(output.cpu().data)
        test_x = to_img(test_img_orig.cpu().data)    ########## change name 
        test_x_hat = to_img(test_output.cpu().data)        
        torch.save(x, './gal_img1107/'+expname+'_x_best.pt')
        torch.save(x_hat, './gal_img1107/'+expname+'_x_hat_best.pt')
        torch.save(test_x, './gal_img1107/'+expname+'_test_x_best.pt')
        torch.save(test_x_hat, './gal_img1107/'+expname+'_test_x_hat_best.pt')
        torch.save(model.state_dict(), './gal_img1107/'+expname+'_train_best.pth')  
        
        
    
    writer.add_scalar('Loss/train',total_loss / num_examples,epoch)
    writer.add_scalar('Mse/train', total_mse / num_examples,epoch)   
    writer.add_scalar('Recon/train', total_recon / num_examples,epoch)        
    writer.add_scalar('Latent/train', total_latent / num_examples,epoch)        
    writer.add_scalar('Loss/test',test_total_loss / test_num_examples,epoch)
    writer.add_scalar('Mse/test', test_total_mse / test_num_examples,epoch)
    writer.add_scalar('Recon/test', test_total_recon / test_num_examples,epoch)        
    writer.add_scalar('Latent/test', test_total_latent / test_num_examples,epoch)    
    
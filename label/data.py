import torch
from torch.autograd import Variable
from torch.utils.data import Dataset , DataLoader
import h5py
import numpy as np


class gDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('galaxy.h5','r')
        image = h5['img'][:]
        h5.close()
        
        image.astype('float32')
        self.len = image.shape[0]
        self.img = torch.from_numpy(image[:])
        

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.img[index]
    
    
    
class trainDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('train.h5','r')
        image = h5['img'][:]
        h5.close()
        
        image.astype('float32')
        self.len = image.shape[0]
        self.img = torch.from_numpy(image[:])
        

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.img[index]

    
class testDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('test.h5','r')
        image = h5['img'][:]
        h5.close()
        
        image.astype('float32')
        self.len = image.shape[0]
        self.img = torch.from_numpy(image[:])
        

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.img[index]    
    

    
    
    
class trainlabelDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('train.h5','r')
        image = h5['img'][:]/(10e6)
        gal_flux = f['gal_flux'][:]
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
     
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
        gal_q.astype('float32')
        gal_beta.astype('float32')
        
        self.len = image.shape[0]
        sel.image= torch.from_numpy(image[:])
        self.gal_flux = torch.from_numpy(gal_flux[:])
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
        self.gal_q = torch.from_numpy(gal_q[:])
        self.gal_beta = torch.from_numpy(gal_beta[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.img[index], self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.gal_q[index],self.gal_beta[index]
    
    
class testlabelDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('test.h5','r')
        image = h5['img'][:]/(10e6)
        gal_flux = f['gal_flux'][:]
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
     
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
        gal_q.astype('float32')
        gal_beta.astype('float32')
        
        self.len = image.shape[0]
        sel.image= torch.from_numpy(image[:])
        self.gal_flux = torch.from_numpy(gal_flux[:])
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
        self.gal_q = torch.from_numpy(gal_q[:])
        self.gal_beta = torch.from_numpy(gal_beta[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.img[index], self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.gal_q[index],self.gal_beta[index]
    
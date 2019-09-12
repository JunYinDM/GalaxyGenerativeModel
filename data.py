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
    

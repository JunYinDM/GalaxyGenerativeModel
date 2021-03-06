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
        self.image = torch.from_numpy(image[:])
        

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index]
    
    
    
class trainDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('train.h5','r')
        image = h5['img'][:]
        h5.close()
        
        image.astype('float32')
        self.len = image.shape[0]
        self.image = torch.from_numpy(image[:])
        

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index]

    
class testDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        h5 = h5py.File('test.h5','r')
        image = h5['img'][:]
        h5.close()
        
        image.astype('float32')
        self.len = image.shape[0]
        self.image = torch.from_numpy(image[:])
        

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index]    
    

    
    
    
class trainlabelDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f = h5py.File('train.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
        gal_q.astype('float32')
        gal_beta.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
        self.gal_q = torch.from_numpy(gal_q[:])
        self.gal_beta = torch.from_numpy(gal_beta[:])
        
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.gal_q[index],self.gal_beta[index]])/3
    
class testlabelDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test.h5','r')
        image = f['img'][:]/(1.5e6)  
        gal_flux = f['gal_flux'][:]/(1.5e6)
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
        gal_q.astype('float32')
        gal_beta.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
        self.gal_q = torch.from_numpy(gal_q[:])
        self.gal_beta = torch.from_numpy(gal_beta[:])
        
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.gal_q[index],self.gal_beta[index]])/3
    
    
 

     
        
        
        
class testlabelDataset_8(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test_10e8.h5','r')
        image = f['img'][:]/(1.5e6)  
        gal_flux = f['gal_flux'][:]/(1.5e6)
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
        gal_q.astype('float32')
        gal_beta.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
        self.gal_q = torch.from_numpy(gal_q[:])
        self.gal_beta = torch.from_numpy(gal_beta[:])
        
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.gal_q[index],self.gal_beta[index]])/3
    
    
    
    
    

    
class trainlabelDataset_8(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f = h5py.File('train_10e8.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
        gal_q.astype('float32')
        gal_beta.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
        self.gal_q = torch.from_numpy(gal_q[:])
        self.gal_beta = torch.from_numpy(gal_beta[:])
        
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.gal_q[index],self.gal_beta[index]])/3
    
     
        
        
        


    
    
class trainlabelDataset_stoke(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f = h5py.File('train.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        Q= np.cos(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        U= np.sin(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        V= np.sin(2 * np.arctan(gal_q )) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
      #  gal_q.astype('float32')
      #  gal_beta.astype('float32')
        Q.astype('float32') 
        U.astype('float32')
        V.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
       # self.gal_q = torch.from_numpy(gal_q[:])
       # self.gal_beta = torch.from_numpy(gal_beta[:])
        self.Q = torch.from_numpy(Q[:])
        self.U = torch.from_numpy(U[:])
        self.V = torch.from_numpy(V[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.Q[index],self.U[index],self.V[index]])
    

    
    
class testlabelDataset_stoke(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test.h5','r')
        image = f['img'][:]/(1.5e6)  
        gal_flux = f['gal_flux'][:]/(1.5e6)
        bulge_re = f['bulge_re'][:]
        disk_n = f['disk_n'][:]
        disk_r0 = f['disk_r0'][:]
        bulge_frac= f['bulge_frac'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        Q= np.cos(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        U= np.sin(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        V= np.sin(2 * np.arctan(gal_q )) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        disk_n.astype('float32')
        disk_r0.astype('float32')
        bulge_frac.astype('float32')
      #  gal_q.astype('float32')
      #  gal_beta.astype('float32')
        Q.astype('float32')
        U.astype('float32')
        V.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.disk_n = torch.from_numpy(disk_n[:])
        self.disk_r0 = torch.from_numpy(disk_r0[:])
        self.bulge_frac = torch.from_numpy(bulge_frac[:])
       # self.gal_q = torch.from_numpy(gal_q[:])
       # self.gal_beta = torch.from_numpy(gal_beta[:])
        self.Q = torch.from_numpy(Q[:])
        self.U = torch.from_numpy(U[:])
        self.V = torch.from_numpy(V[:])
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_re[index],self.disk_n[index],self.disk_r0[index], self.bulge_frac[index],self.Q[index],self.U[index],self.V[index]])
    
    
    

class trainlabelDataset_stoke_1ser(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f = h5py.File('train_1ser.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        bulge_n = f['bulge_n'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        Q= np.cos(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        U= np.sin(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        V= np.sin(2 * np.arctan(gal_q )) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        bulge_n.astype('float32')
        Q.astype('float32') 
        U.astype('float32')
        V.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log10(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.bulge_n = torch.from_numpy(bulge_n[:])

        self.Q = torch.from_numpy(Q[:])
        self.U = torch.from_numpy(U[:])
        self.V = torch.from_numpy(V[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_n[index],self.bulge_re[index],self.Q[index],self.U[index],self.V[index]])
    

    
    
class testlabelDataset_stoke_1ser(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test_1ser.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        bulge_n = f['bulge_n'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        Q= np.cos(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        U= np.sin(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        V= np.sin(2 * np.arctan(gal_q )) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        bulge_n.astype('float32')
        Q.astype('float32') 
        U.astype('float32')
        V.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log10(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.bulge_n = torch.from_numpy(bulge_n[:])

        self.Q = torch.from_numpy(Q[:])
        self.U = torch.from_numpy(U[:])
        self.V = torch.from_numpy(V[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_n[index],self.bulge_re[index],self.Q[index],self.U[index],self.V[index]])
    




    
class testlabelDataset_stoke_1ser_10e7_8(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test_2_1ser.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        bulge_n = f['bulge_n'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        Q= np.cos(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        U= np.sin(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        V= np.sin(2 * np.arctan(gal_q )) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        bulge_n.astype('float32')
        Q.astype('float32') 
        U.astype('float32')
        V.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log10(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.bulge_n = torch.from_numpy(bulge_n[:])

        self.Q = torch.from_numpy(Q[:])
        self.U = torch.from_numpy(U[:])
        self.V = torch.from_numpy(V[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_n[index],self.bulge_re[index],self.Q[index],self.U[index],self.V[index]])
    
    

    
    
    
class testlabelDataset_stoke_1ser_10e8(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test_3_1ser.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        bulge_n = f['bulge_n'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        Q= np.cos(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        U= np.sin(2 * gal_beta) *np.cos(2 * np.arctan(gal_q )) +1
        V= np.sin(2 * np.arctan(gal_q )) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        bulge_n.astype('float32')
        Q.astype('float32') 
        U.astype('float32')
        V.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log10(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.bulge_n = torch.from_numpy(bulge_n[:])

        self.Q = torch.from_numpy(Q[:])
        self.U = torch.from_numpy(U[:])
        self.V = torch.from_numpy(V[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_n[index],self.bulge_re[index],self.Q[index],self.U[index],self.V[index]])    
    
    
    
class trainlabelDataset_reduced(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f = h5py.File('train_1ser.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        bulge_n = f['bulge_n'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        g1= (1-gal_q )/(1+gal_q) *np.cos(2 * gal_beta) +1
        g2= (1-gal_q )/(1+gal_q) *np.sin(2 * gal_beta) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        bulge_n.astype('float32')
        g1.astype('float32') 
        g2.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log10(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.bulge_n = torch.from_numpy(bulge_n[:])

        self.g1 = torch.from_numpy(g1[:])
        self.g2 = torch.from_numpy(g2[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_n[index],self.bulge_re[index],self.g1[index],self.g2[index]])
    

    
    
class testlabelDataset_reduced(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
                
        f= h5py.File('test_1ser.h5','r')
        image = f['img'][:]/(1.5e6)    # max-min normalize the image 
        gal_flux = f['gal_flux'][:]/(1.5e6)  
        bulge_re = f['bulge_re'][:]
        bulge_n = f['bulge_n'][:]
        gal_q = f['gal_q'][:]
        gal_beta = f['gal_beta'][:]
        f.close()
        
        g1= (1-gal_q )/(1+gal_q) *np.cos(2 * gal_beta) +1
        g2= (1-gal_q )/(1+gal_q) *np.sin(2 * gal_beta) +1
        
        image.astype('float32')
        gal_flux.astype('float32')
        bulge_re.astype('float32')
        bulge_n.astype('float32')
        g1.astype('float32') 
        g2.astype('float32')
        
        self.len = image.shape[0]
        self.image= torch.from_numpy(image[:])
        self.gal_flux = torch.log10(torch.from_numpy(gal_flux[:]))
        self.bulge_re = torch.from_numpy(bulge_re[:])
        self.bulge_n = torch.from_numpy(bulge_n[:])

        self.g1 = torch.from_numpy(g1[:])
        self.g2 = torch.from_numpy(g2[:])
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        return self.image[index], np.asarray([self.gal_flux[index],self.bulge_n[index],self.bulge_re[index],self.g1[index],self.g2[index]])
    






    
    
if __name__  == '__main__':
    dataset = trainlabelDataset()
    print(len(dataset))
    print(len(dataset.__getitem__(5)))
    print(dataset.__getitem__(5)[0].shape)
    print(dataset.__getitem__(5)[1].shape)

    print(dataset.__getitem__(5))
    
if __name__  == '__main__':
    dataset = trainlabelDataset()
    print(len(dataset))
    print(len(dataset.__getitem__(5)))
    print(dataset.__getitem__(5)[0].shape)
    print(dataset.__getitem__(5)[1].shape)

    print(dataset.__getitem__(5))        
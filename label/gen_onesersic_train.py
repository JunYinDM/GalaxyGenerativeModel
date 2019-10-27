import sys
import os
import math
import logging
import galsim
import matplotlib.pyplot as plt
import h5py
import numpy as np 
import random 

img_ = np.zeros((10000,64,64))

gal_flux_ = np.zeros(10000)
bulge_re_ =np.zeros(10000)
bulge_n_ =np.zeros(10000)
gal_q_ = np.zeros(10000)
gal_beta_ = np.zeros(10000)



for i in range(10000): 
# fixed parameters 
    image_size = 64        # n x n pixels
    pixel_scale = 0.23     # arcsec / pixel
    random_seed = 1314662
    rng = galsim.BaseDeviate(random_seed+1)
    psf_beta = 2       #moffat parameter 
    psf_re=1         # moffat scale radius in arcsec 
    noise=300   # S/N ~ 8 and above 
    
    
    #parameter random generations  
    gal_flux = 1e6 * random.uniform(1,100)        # ADU  ("Analog-to-digital units", the units of the numbers on a CCD)
    bulge_re =random.uniform(0,3)         # arcsec
    bulge_n = random.uniform(2.5,3.5)          # Fixed 
    
    gal_q = random.uniform(0.2,0.7)       # (axis ratio 0 < q < 1)
    gal_beta = random.uniform(0,3.14)        #  radians 
 
    
    gal = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
    gal = gal.withFlux(gal_flux)
    gal_shape = galsim.Shear(q=gal_q, beta=gal_beta*galsim.radians)
    gal = gal.shear(gal_shape)
    psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)   
    final = galsim.Convolve([psf, gal])
    image = galsim.ImageF(image_size, image_size,scale=pixel_scale)
    final.drawImage(image=image)
    image.addNoise(galsim.GaussianNoise(sigma=noise))  #add noise 

    
    img_[i]= np.abs(image.array)  # to eliminate negative pixel value 
    gal_flux_[i] = gal_flux
    bulge_re_[i] =bulge_re
    bulge_n_[i] =bulge_n
    gal_q_[i] =gal_q
    gal_beta_[i] = gal_beta
    
    plt.imshow(img_[0])

    
    if i % 500 == 0:
        print(i)

    

    
    

with h5py.File("train_1ser.h5", "w") as fnew:
    fnew.create_dataset('img', data=img_)
    fnew.create_dataset('gal_flux', data=gal_flux_)
    fnew.create_dataset('bulge_n', data=bulge_n_)   
    fnew.create_dataset('bulge_re', data=bulge_re_)
    fnew.create_dataset('gal_q', data=gal_q_)
    fnew.create_dataset('gal_beta', data=gal_beta_)

print("yayyyyy, finish running")    

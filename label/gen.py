import sys
import os
import math
import logging
import galsim
import matplotlib.pyplot as plt
import h5py
import numpy as np 
import random 

img_ = np.zeros((3000,64,64))

gal_flux_ = np.zeros(3000)
bulge_re_ =np.zeros(3000)
disk_n_ =np.zeros(3000)
disk_r0_ = np.zeros(3000)
bulge_frac_ =np.zeros(3000)

    
gal_q_ = np.zeros(3000)
gal_beta_ = np.zeros(3000)
atmos_e_ =np.zeros(3000)
atmos_beta_ =np.zeros(3000)

wcs_g1_ = np.zeros(3000)
wcs_g2_ =np.zeros(3000)



for i in range(3000): 
# fixed parameters 
    image_size = 64        # n x n pixels
    pixel_scale = 0.23     # arcsec / pixel
 #   random_seed = 1314662
    rng = galsim.BaseDeviate(random_seed+1)
    bulge_n = 3.5          # Fixed 
    psf_beta = 2       #moffat parameter 
    psf_re=1         # moffat scale radius in arcsec 
    
    #parameter random generations  
    gal_flux = 1e6* random.uniform(1,100)        # ADU  ("Analog-to-digital units", the units of the numbers on a CCD)
    bulge_re = random.uniform(0,3)         # arcsec
    disk_n = random.uniform(1,2)          #
    disk_r0 = random.uniform(0,1)        # arcsec (corresponds to half_light_radius of ~3.7 arcsec)
    bulge_frac = random.uniform(0,1)    # 0 ~ 1 

    
    gal_q = random.uniform(0.2,0.7)       # (axis ratio 0 < q < 1)
    gal_beta = random.uniform(0,3.14)        #  radians 
 
    
    bulge = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
    disk = galsim.Sersic(disk_n, scale_radius=disk_r0)
    gal = bulge_frac * bulge + (1-bulge_frac) * disk
    gal = gal.withFlux(gal_flux)
    gal_shape = galsim.Shear(q=gal_q, beta=gal_beta*galsim.radians)
    gal = gal.shear(gal_shape)
    psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)   
    final = galsim.Convolve([psf, gal])
    image = galsim.ImageF(image_size, image_size,scale=pixel_scale)
    final.drawImage(image=image)


    
    img_[i]= image.array
    gal_flux_[i] = gal_flux
    bulge_re_[i] =bulge_re
    disk_n_[i] =disk_n
    disk_r0_[i] = disk_r0
    bulge_frac_[i] =bulge_frac
    gal_q_[i] =gal_q
    gal_beta_[i] = gal_beta


    if i % 500 == 0:
        print(i)

    

    
    
with h5py.File("test.h5", "w") as fnew:
    fnew.create_dataset('img', data=img_)
    fnew.create_dataset('gal_flux', data=gal_flux_)
    fnew.create_dataset('bulge_re', data=bulge_re_)
    fnew.create_dataset('disk_n', data=disk_n_)   
    fnew.create_dataset('disk_r0', data=disk_r0_)
    fnew.create_dataset('bulge_frac', data=bulge_frac_)
    fnew.create_dataset('gal_q', data=gal_q_)
    fnew.create_dataset('gal_beta', data=gal_beta_)
    fnew.create_dataset('atmos_e', data=atmos_e_)
    fnew.create_dataset('atmos_beta', data = atmos_beta_)
    fnew.create_dataset('wcs_g1', data=wcs_g1_)
    fnew.create_dataset('wcs_g2', data=wcs_g2_)
print("yayyyyy, finish running")

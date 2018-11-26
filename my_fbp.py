# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:02:26 2018

@author: ZMJ
"""
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fftshift,fft,ifft

theta_step=1
def pad_img(img):
  s1,s2=img.size
  diag=int(np.ceil(np.sqrt(s1**2+s2**2)))
  img_pad=Image.new("L",(diag,diag))
  start_x=int(round((diag-s1)/2))
  start_y=int(round((diag-s2)/2))
  img_pad.paste(img,(start_x,start_y))
  return img_pad,start_x,start_y

def projection(img,thetas):
  num_theta=len(thetas)
  sinogram=np.zeros((img.size[1],num_theta))
  for i,theta in enumerate(thetas):
    rot_img=img.rotate(theta,resample=Image.BICUBIC)
    sinogram[:,i]=np.sum(rot_img,axis=0)
  return sinogram

def filter_projection(sinogram):##ramp filter
  a=0.1
  size,num_thetas=sinogram.shape
  step=2*np.pi/size
  w=np.arange(-np.pi,np.pi,step)
  if len(w)<size:
    w=np.concatenate([w,w[-1]+step])
  rn1=np.abs(2/a*np.sin(a*w/2))
  rn2=np.sin(a*w/2)/(a*w/2)
  r=rn1*(rn2)**2
  
  filter_=fftshift(r)
  filter_sinogram=np.zeros((size,num_thetas))
  for i in range(num_thetas):
    proj_fft=fft(sinogram[:,i])
    filter_proj=proj_fft*filter_
    filter_sinogram[:,i]=np.real(ifft(filter_proj))
  return filter_sinogram

def back_projection(sinogram,thetas):
  size_=sinogram.shape[0]
  new_size=int(np.ceil(np.sqrt(size_**2+size_**2)))
  start=int(round((new_size-size_)/2))
  recon_img=Image.new("L",(new_size,new_size))  
  
  for i,theta in enumerate(thetas):
    recon_img=recon_img.rotate(theta)
    tmp1=sinogram[:,i]
    tmp2=np.zeros(shape=(new_size))
    tmp2[start:-start-1]=tmp1
    tmp=np.repeat(np.expand_dims(tmp2,1),new_size,axis=1).T
    recon_img+=tmp
    recon_img=Image.fromarray(recon_img)
    recon_img=recon_img.rotate(-theta)
  recon_img=np.array(recon_img)
  return recon_img[start:-start-1,start:-start-1]

def back_projection2(sinogram,thetas):
  size_=sinogram.shape[0]
  recon_img=np.zeros((size_,size_))  

  for i,theta in enumerate(thetas):
    tmp1=sinogram[:,i]
  
    tmp=np.repeat(np.expand_dims(tmp1,1),size_,axis=1).T  
    tmp=Image.fromarray(tmp)

    tmp=tmp.rotate(theta,expand=0)

    recon_img+=tmp

  return np.flipud(recon_img)

img=Image.open("phantom.png").convert("L")
img_pad,start_x,start_y=pad_img(img)###for rotate propose 
thetas=np.arange(0,181,theta_step)
sinogram=projection(img_pad,thetas)
filter_sino=filter_projection(sinogram)
recon_img=back_projection2(sinogram,thetas)
recon_img=recon_img[start_x:-start_x,start_y:-start_y]
recon_img=np.round((recon_img-np.min(recon_img))/np.ptp(recon_img)*255)

fig=plt.figure()
fig.add_subplot(121)
plt.title("Phantom")
plt.imshow(img,cmap=plt.get_cmap("gray"))
fig.add_subplot(122)
plt.title("Filtered Backprojection")
plt.imshow(recon_img,cmap=plt.get_cmap("gray"))
plt.show()
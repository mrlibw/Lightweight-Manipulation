import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time
import torchvision.transforms as T
from kornia.morphology import dilation, erosion

def matching_histogram(img, ref, min = 0.0, max = 1.0, int_out=False):
    """
    the input should have this dimension [*,H,W]
    * = can be any tensor size such as [Batch,Channel] or just [Channel]
    H = height
    W = weight
    H and W of ref and img may not match but the min,max, and bins across 
    all batch and channel should be the same
    """
    if int_out:
        max = max + 1
    # reshape the image for easier handling
    img_shp = img.shape
    ref_shp = ref.shape
    img = img.view(-1,img_shp[-2]*img_shp[-1])
    ref = ref.view(-1,ref_shp[-2]*ref_shp[-1])
    # doing sorting instead of histogram
    sort_val,sort_idx = torch.sort(img, descending=False)
    sort_new_val,_ = torch.sort(ref, descending=False)
    # normalize the image size if it is different
    if (img_shp!=ref_shp):
        max = ref_shp[-2]*ref_shp[-1]
        step = max / (img_shp[-2]*img_shp[-1])
        idx = torch.arange(0,max,step).to(torch.int).to(img.device)
        sort_new_val = torch.index_select(sort_new_val, 1 , idx)
    # inserting the value
    img_new = torch.zeros_like(img)
    img_new.scatter_(1, sort_idx,sort_new_val)
    img_new = img_new.reshape(*img_shp)
    if int_out:
        img_new.floor_()
    return img_new

#SSIM
def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def ssim1(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):

    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, ssim_score
    
    return ret

def norm(img, rng=(0, 255)):
    r_min = img.min()
    r_max = img.max()
    t_min = rng[0]
    t_max = rng[1]
    return (img - r_min) / (r_max - r_min) * (t_max-t_min) + t_min
  

def hist_batch_mask(real_img, fake_img, segment == False):
  #Compute the histogram matched real image
  hist_real = matching_histogram(real_img, fake_img, min = 0.0, max = 255.0, int_out=True)
  #Gaussian filter
  real_batch = T.GaussianBlur(kernel_size = 5, sigma=1)(hist_real)
  fake_batch = T.GaussianBlur(kernel_size = 5, sigma=1)(fake_img)
  
  #Compute difference
  masked_batch = torch.abs(torch.subtract(real_batch, fake_batch))
  # remove large (white) values by setting them to 0
  masked_batch[masked_batch > 160] = 0
  # keep large (white) values by setting anything below to 0
  masked_batch = masked_batch > 45
  # add the the three channels
  masked_batch = masked_batch.sum(1)
  # reshape after the dropped dimension when summing
  masked_batch = torch.reshape(masked_batch, (masked_batch.size()[0], 1, masked_batch.size()[1], masked_batch.size()[2]))
  # set to 1 any values above 0
  masked_batch = masked_batch > 0
  # cast to int and apply erosion
  masked_batch = erosion(masked_batch.long(), torch.ones(2, 2).cuda()) # 3, 3
  # apply dilation
  masked_batch = dilation(masked_batch, torch.ones(7, 7).cuda()) # 7, 7
  # invert mask
  masked_batch = 1 - masked_batch
  
  #If we want to use SSIM and Chan_Vese model
  if segment:
    from .Chan_vese import chan_vese1
    score, diff = ssim1(real_batch, fake_batch, full = True, val_range = 255)
    diff_norm = norm(diff)
    diff_sum = diff_norm.sum(0).sum(0)
    diff_sum_norm = norm(diff_sum)
    seg = chan_vese1(diff_sum_norm, image_dimension = 2)
    masked_batch = torch.mul(masked_batch, seg)
   
  return masked_batch




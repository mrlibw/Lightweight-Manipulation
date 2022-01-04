"""Compute binary mask using Histogram Matching approach."""

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

def norm(img, rng=(0, 255)):
    r_min = img.min()
    r_max = img.max()
    t_min = rng[0]
    t_max = rng[1]
    return (img - r_min) / (r_max - r_min) * (t_max-t_min) + t_min
  
def hist_batch_mask(real_img, fake_img):
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
    masked_batch = masked_batch > 32.5
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
    
    return masked_batch

#compute MSE
def M_loss(real, fake):
    m_loss = torch.mean(((real - fake)**2))
    return m_loss

#compute mask score
def mask_score():
    real_mask = torch.mul(real, mask)
    fake_mask = torch.mul(fake, mask)
    return M_loss(real_mask, fake_mask)

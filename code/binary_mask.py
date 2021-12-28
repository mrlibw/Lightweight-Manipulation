import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from kornia.morphology import dilation, erosion
import torch


def get_batch_mask(real_batch, fake_batch):
    
    THRESH_LARGE = 180 # 180
    THRESH_SMALL = 60 # 65
    
    # subtract and take absolute difference
    masked_batch = torch.abs(torch.subtract(real_batch, fake_batch))
    # remove large (white) values by setting them to 0
    masked_batch[masked_batch > THRESH_LARGE] = 0
    # keep large (white) values by setting anything below to 0
    masked_batch = masked_batch > THRESH_SMALL
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
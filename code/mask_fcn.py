"""Compute the binary mask using the pre-trained FCN model."""

from torchvision import models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import pickle as pkl
import os

def get_batch_mask(real_batch, model, show=False):

    real_batch = (real_batch + 1.0) * 127.5
    real_batch = real_batch / 255.

    fcn = model
    
    # normalize with mean and std of ImageNet
    trf = T.Compose([T.Resize(224),
                     T.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])
                    ])
                    
    # bring back image to the size of the images in Lightweight GAN
    trf_back = T.Compose([T.Resize(256)])

    trf_real_batch = trf(real_batch.float())

    out = fcn(trf_real_batch)['out']

    # output pixel class with highest probability
    out = torch.argmax(out, dim=1, keepdim=True)

    # bird class
    out = (out == 3).float()
    
    out = trf_back(out)
    
    # invert mask to ignore bird and retain background
    out = (1 - out).float()

    if show:
        with open('out.pkl', 'wb') as f:
            pkl.dump(out.cpu().detach().numpy(), f)
        print("done")
    
    return out

#compute MSE
def M_loss(real, fake):
    m_loss = torch.mean(((real - fake)**2))
    return m_loss

#compute mask score
def mask_score():
    real_mask = torch.mul(real, mask)
    fake_mask = torch.mul(fake, mask)
    return M_loss(real_mask, fake_mask)

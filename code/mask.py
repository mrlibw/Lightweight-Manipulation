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
    
    trf = T.Compose([T.Resize(224),
                     T.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])
                    ])
                    
    trf_back = T.Compose([T.Resize(256)])

    trf_real_batch = trf(real_batch.float())

    out = fcn(trf_real_batch)['out']

    out = torch.argmax(out, dim=1, keepdim=True)

    # bird class
    out = (out == 3).float()
    
    out = trf_back(out)
    
    out = (1 - out).float()

    if show:
        with open('out.pkl', 'wb') as f:
            pkl.dump(out.cpu().detach().numpy(), f)
        print("done")
    
    return out

import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from pdb import set_trace as db



class SharpDetector:
    def __init__(self):
        # --- initialize filters ---
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise ValueError("cuda is unavailable now.")
        self.laplacian_filter = torch.FloatTensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3).to(self.device)
        
    def get_mask(self, image):
        s = 1
        pad = 1
        gray = self.getGrayImage(image)
        img_lap = torch.nn.functional.conv2d(input=gray,
                                        weight=Variable(self.laplacian_filter),
                                        stride=s,
                                        padding=pad)
        out = self.blurring(img_lap)
        # thresholding and mask
        out = self.thresholding(out)
        return out 

    def getGrayImage(self, rgbImg):
        gray = (rgbImg[:,0,:,:] + rgbImg[:,1,:,:] + rgbImg[:,2,:,:]) / 3.0
        gray = torch.unsqueeze(gray, 1)
        return gray

    def blurring(self, laplased_image, sigma=5, min_abs=0.5/255):
        # --- sigma is the size of kernel of Blurr filter ---
        abs_image = torch.abs(laplased_image).to(torch.float32)  # convert to absolute values
        abs_image[abs_image < min_abs] = min_abs 
        # print(sigma)
        blurred_img = self.BlurLayer(abs_image, k_size=sigma)
        return blurred_img

    def BlurLayer(self, img, k_size=5, s=1, pad=2):
        _blur_filter = torch.ones([k_size, k_size]).to(self.device)
        blur_filter = _blur_filter.view(1,1,k_size, k_size) / (k_size**2)
        # gray = getGrayImage(img)
        img_blur = torch.nn.functional.conv2d(input=img,
                                            weight=Variable(blur_filter),
                                            stride=s,
                                            padding=pad)

        return img_blur
    
    def thresholding(self, img):
        bs, ch, h, w = img.shape
        _var = img.mean(dim=(2,3))
        _var = _var.view(bs, 1, 1, 1)
        var_tensor = _var.repeat_interleave(dim=2, 
                            repeats=h).repeat_interleave(dim=3, repeats=w)
        mask = (img > var_tensor).type(torch.FloatTensor).to('cuda')
        # db()
        return mask

if __name__=="__main__":
    # --- 画像のロード ---
    # img_id = 5
    img_path = "/home/isi/prog/Lightweight-Manipulation/data/birds/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    im = np.array(Image.open(img_path))
    
    img = torch.tensor(im) / 255.0
    img = torch.tensor(img.transpose(1, 2).transpose(0, 1))
    # img = img.transpose(0,2)  # 転置
    shp = img.shape
    img = img.view(1, *shp)  # バッチ方向を作成、ここ普通にmax255
    # ---- バッチ方向にrepeat ----
    img = torch.repeat_interleave(img, dim=0, repeats=3)
    print("img shape: ", img.shape)
    # --- to GPU ---
    img = img.to('cuda')

    model = SharpDetector()
    out = model.get_mask(img)

    np_out = out.cpu().detach().numpy()
    plt.imshow(np_out[0, 0,:,:], cmap='gray')
    plt.savefig('./blurred_test.png')
    plt.clf()



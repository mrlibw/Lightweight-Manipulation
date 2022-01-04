import os
import torch
from PIL import Image
import numpy as np

if __name__=="__main__":
    laplacian_filter = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    dirname = '../output/netG_epoch_100_ori/valid'
    n_files = [name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname,name))]
    sharpness = torch.zeros(len(n_files))
    print("number of files is {}".format(len(n_files)))
    for i,filename in enumerate(os.listdir(dirname)):
        full_filename = os.path.join(dirname, filename)
        if not (os.path.isfile(full_filename)):
            continue
        if "real" in full_filename:
            continue
        with Image.open(full_filename) as imI:
            imNP = np.array(imI)
            
            img = torch.tensor(imNP) / 255.0
            img = torch.tensor(img.transpose(1, 2).transpose(0, 1))
            img = torch.mean(img,0).unsqueeze(0)
            # img = img.transpose(0,2)  # 転置
            shp = img.shape
            img = img.view(1, *shp)  # バッチ方向を作成、ここ普通にmax255

            out = torch.nn.functional.conv2d(input=img, weight=laplacian_filter, stride=1, padding=1)

            sharpness[i] = torch.mean(torch.abs(out))
        if (i%1000==0):
            print(i)
    print(torch.mean(sharpness))
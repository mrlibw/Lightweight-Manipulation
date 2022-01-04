import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    
    with torch.no_grad():
    
        N = len(imgs)

        assert batch_size > 0
        assert N > batch_size

        # Set up dtype
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval();
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x)

        # Get predictions
        preds = torch.zeros([N, 1000]).type(dtype)

        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            #print(batch.shape)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
            print(i)

        # Now compute the mean kl-div
        split_scores = torch.zeros([splits]).type(dtype)

        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = torch.mean(part, axis=0)
            scores = torch.sum( part * torch.log(part/py[None,:]), axis=1)
            split_scores[k] = (torch.exp(torch.mean(scores)))

    return torch.mean(split_scores), torch.std(split_scores)


if __name__ == '__main__':
    
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    
    # class IgnoreLabelDataset(torch.utils.data.Dataset):
    #     def __init__(self, orig):
    #         self.orig = orig
    # 
    #     def __getitem__(self, index):
    #         return self.orig[index][0]
    # 
    #     def __len__(self):
    #         return len(self.orig)
    # 
    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Scale(32),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )
    # 
    # IgnoreLabelDataset(cifar)
    # 
    # print ("Calculating Inception Score...")
    # print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=16, resize=True, splits=10))
 
    import os
    from torchvision.io import read_image
    from PIL import Image
    import numpy as np
    
    class ImgDataset(torch.utils.data.Dataset):
        def __init__(self, dirloc, lenght, transform=None):
            self.dirloc = dirloc
            self.lenght = lenght
            self.transform = transform

        def __getitem__(self, idx):
            #img_path = dirname + "/" + str(idx) + "_fake.png"
            img_path = dirname + "/single_s" + str(idx) + ".png"
            if not (os.path.isfile(img_path)):
                print("Error no file")
            # with Image.open(img_path) as im:
            #     image = np.array(im)
            image = read_image(img_path)
            image = image/255.0
            if self.transform:
                image = self.transform(image)
            return image

        def __len__(self):
            return self.lenght 
    
    dirname = '../output/saved/Manigan/valid'
    n_files = 14660#29280
    transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = None
    print ("Calculating Inception Score...")
    print (inception_score(ImgDataset(dirname, n_files, transform=transform), cuda=True, batch_size=16, resize=True, splits=10))

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3
from pdb import set_trace as db


class FidCalculater:
    def __init__(self, n_file):
        # self.which_dataset = which_dataset  # original dataset or generated dataset
        dims = 2048  # default
        self.start_idx = 0
        self.pred_arr_org = np.empty((n_file, dims))  # this is ndarray, not Tensor!
        self.pred_arr_fake = np.empty((n_file, dims))  # this is ndarray, not Tensor!


        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to('cuda')
        
    def accumulate_pred_from_batch(self, batch_org, batch_fake):
        batch_org = batch_org.to('cuda')
        batch_fake = batch_fake.to('cuda')
        assert batch_org.shape == batch_fake.shape  # make sure that we're receiving the same shape of batch for real and fake
        
        with torch.no_grad():
            pred_org = self.model(batch_org)[0]
            pred_fake = self.model(batch_fake)[0]


        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred_org.size(2) != 1 or pred_org.size(3) != 1:
            pred_org = adaptive_avg_pool2d(pred_org, output_size=(1, 1))
        if pred_fake.size(2) != 1 or pred_fake.size(3) != 1:
            pred_fake = adaptive_avg_pool2d(pred_fake, output_size=(1, 1))

        pred_org = pred_org.squeeze(3).squeeze(2).cpu().numpy()
        pred_fake = pred_fake.squeeze(3).squeeze(2).cpu().numpy()
        try:
            self.pred_arr_org[self.start_idx:self.start_idx + pred_org.shape[0]] = pred_org
            self.pred_arr_fake[self.start_idx:self.start_idx + pred_fake.shape[0]] = pred_fake
        except:
            db()

        self.start_idx = self.start_idx + pred_org.shape[0]


        # return self.pred_arr_org, self.pred_arr_fake

    def calculate_fid(self):
        self.mu_org = np.mean(self.pred_arr_org, axis=0)
        self.mu_fake = np.mean(self.pred_arr_fake, axis=0)

        self.sigma_org = np.cov(self.pred_arr_org, rowvar=False)
        self.sigma_fake = np.cov(self.pred_arr_fake, rowvar=False)
        
        fid_value = self.calculate_frechet_distance(self.mu_org, \
            self.sigma_org, self.mu_fake, self.sigma_fake)
        self.start_idx = 0
        return fid_value


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


if __name__ == '__main__':
    pass
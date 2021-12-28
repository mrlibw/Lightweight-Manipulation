import logging

import numpy as np
import torch as th

from typing import Callable, List, Optional, Tuple, Union

import kornia.filters as fil
import numpy as np
import torch

try:
    from kornia.filters import filter3D
except ImportError:
    # polyfill kornia == 0.2.2 without filter3D
    from .polyfill import filter3D


def convolve(im: torch.Tensor,
             kern: torch.Tensor,
             image_dimension: int,
             reshape: bool = True) -> torch.Tensor:
    """Convolves an 2d/3d image with specified kernel.

    Args:
        im (torch.Tensor): the input tensor with shape of
            :math:`(C, D, H, W)` or  :math:`(D, H, W)` for 3d image and
            :math:`(C, H, W)` or  :math:`(H, W)` for 2d image.
        kern (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kD, kH, kW)` for 3d image
          and :math:`(kH, kW)` for 2d image.
        image_dimension (int): dimension of input image (2 or 3).
        reshape (bool, optional): reshape the output into the shape of input
          image.

    Return:
        torch.Tensor: result image with the same shape as input
          if reshape = True,
        or in :math:`(B, C, D, H, W)` or :math:`(B, C, H, W)`
          if reshape = False.
    """
    if len(kern.shape) == image_dimension:
        kern = kern.unsqueeze(0)
    org_shape = im.shape
    if image_dimension == 3:
        if len(im.shape) == 3:
            im = im.unsqueeze(0).unsqueeze(0)  # im: D, H, W
        elif len(im.shape) == 4:
            im = im.unsqueeze(0)  # im: C, D, H, W
        else:
            raise ValueError(
                f'im.shape {repr(im.shape)} does not match specified \
                image dimension {image_dimension}.')
        if reshape:
            return filter3D(im, kern).reshape(org_shape)
        else:
            return filter3D(im, kern)
    elif image_dimension == 2:
        if len(im.shape) == 2:
            im = im.unsqueeze(0).unsqueeze(0)  # im: H, W
        elif len(im.shape) == 3:
            im = im.unsqueeze(0)  # im: C, H, W
        else:
            raise ValueError(
                f'im.shape {repr(im.shape)} does not match specified \
                image dimension {image_dimension}.')
        if reshape:
            return fil.filter2D(im, kern).reshape(org_shape)
        else:
            return fil.filter2D(im, kern)
    else:
        raise ValueError(f'Unsupported image dimension {image_dimension}')


def gradient_magnitude_gaussian(
    im: torch.Tensor,
    sigma: Union[int, float, Tuple[Union[int, float]], List[Union[int,
                                                                  float]]],
    image_dimension: int,
    kernel_size: Optional[Union[int, Tuple[int], List[int]]] = None
) -> torch.Tensor:
    """Calculate gradient magnitude of an image convolved with gaussian kernel.

    Args:
        im (torch.Tensor): the input tensor with shape of
            :math:`(C, D, H, W)` or  :math:`(D, H, W)` for 3d image and
            :math:`(C, H, W)` or  :math:`(H, W)` for 2d image.
        sigma (int, float, list of float, tuple of float): Standard deviation
            of gaussian kernel.
            Single `int` or `float` value will be considered as
            isotropic distribution.
        image_dimension (int): dimension of input image (2 or 3).
        sigma (int, list of int, tuple of int): Shape of gaussian kernel,
            must be odd.
            Choose wisely since this value influences performance greatly.
            Single `int` value will be considered as isotropic kernel size.

    Return:
        torch.Tensor: An image with shape :math:`(D, H, W)`
            if `image_dimension == 3`
            or shape :math:`(H, W)` if `image_dimension == 2`.
    """
    if type(sigma) is not tuple and type(sigma) is not list:
        sigma = (sigma, ) * image_dimension
    if kernel_size is None:
        kernel_size = [int(np.ceil(_i) * 6 - 1) for _i in sigma]
    elif type(kernel_size) is not list and type(kernel_size) is not tuple:
        kernel_size = (kernel_size, ) * image_dimension
    if len(sigma) != image_dimension:
        raise ValueError(
            f"Length of sigma '{repr(sigma)}' must be the same as \
                image_dimension '{image_dimension}'. ")
    if len(kernel_size) != image_dimension:
        raise ValueError(
            f"Length of kernel_size '{repr(kernel_size)}' must be \
                the same as image_dimension '{image_dimension}'. ")
    if any([i % 2 == 0 for i in kernel_size]):
        raise ValueError(f"kernel_size '{repr(kernel_size)}' \
            must be all odd.")
    gaussian_kernel = gaussian_kernel_generator(kernel_size, sigma)
    gaussian_kernel.to(im)
    if image_dimension == 3:
        gradient = fil.spatial_gradient3d(
            convolve(im, gaussian_kernel, image_dimension, reshape=False))
    elif image_dimension == 2:
        gradient = fil.spatial_gradient(
            convolve(im, gaussian_kernel, image_dimension, reshape=False))
    else:
        raise ValueError(f'Unsupported image_dimension {image_dimension}.')
    return torch.norm(gradient, dim=2).sum(dim=(0, 1))


def gaussian_kernel_generator(kernel_size, sigma):
    """N-D Gaussian kernel generator."""
    # no-qa: E501
    # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size])
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * np.sqrt(2 * np.pi)) * \
            torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    return kernel


def edge_indicator_function(im: torch.Tensor,
                            sigma: Union[int, float, Tuple[Union[int, float]],
                                         List[Union[int, float]]],
                            dim: int,
                            gaussian: Callable = gradient_magnitude_gaussian):
    """Bounded recipocal of squared gaussian gradient magnitude as edge
    indicator function."""
    res = gaussian(im, sigma, dim)
    res.pow_(2).add_(1).reciprocal_()
    return res

logr = logging.getLogger('ictm')


def chan_vese1(im,
              image_dimension,
              mask=None,
              tau=0.02,
              lambd=0.05,
              max_step=100,
              kernel_size=9):
    """Chan vese model solved with iterative convolution-thresholding method
    (ICTM)."""
    def chan_vese_F(im, seg, eps=1e-5):
        C = seg.mul(im).sum().div(seg.sum() + eps)
        return (im - C).pow_(2)

    kern = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                     (tau, ) * image_dimension).to(im)
    u = im
    steps = 0
    if mask is not None:
        mask = mask.to(th.bool)
        u.mul_(mask)

    while steps <= max_step:
        phi = chan_vese_F(im, u) - chan_vese_F(im, 1-u) +  \
                lambd * np.sqrt(np.pi / tau) * \
                convolve((1 - 2*u), kern, image_dimension).squeeze()
        u_next = phi.lt(0).to(im)
        if mask is not None:
            u_next.mul_(mask)
        if u_next.ne(u).sum() == 0:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.to(th.uint8)


def geodesic_active_contour(im,
                            init_seg,
                            image_dimension,
                            mask=None,
                            sigma=1.,
                            tau=2.,
                            lambd=-0.2,
                            max_step=10,
                            kernel_size=9):
    """Geodesic active contour model solved with iterative convolution-
    thresholding method (ICTM)."""
    # https://arxiv.org/pdf/2007.00525.pdf
    if type(tau) is not list and type(tau) is not tuple:
        tau = (tau, ) * image_dimension
    g = edge_indicator_function(
        im,
        sigma,
        image_dimension,
        gaussian=lambda x, y, z: gradient_magnitude_gaussian(
            x, y, z, kernel_size=kernel_size))
    if mask is not None:
        g.mul_(mask)
        mask = mask.to(th.bool)
    g_sqrt = g.sqrt()
    kern = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                     tau).to(g)
    steps = 0
    u = init_seg.gt(0).to(g)
    while steps <= max_step:
        phi = g_sqrt * convolve(g_sqrt *
                                (1 - 2 * u), kern, image_dimension) + lambd * g
        u_next = phi.lt(0)
        if mask is not None:
            u_next.mul_(mask)
        if u_next.ne(u).sum() == 0:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.to(th.uint8)


def general_geodesic_active_contour(g,
                                    init_seg,
                                    image_dimension,
                                    mask=None,
                                    tau=2.,
                                    lambd=-0.2,
                                    max_step=10,
                                    kernel_size=9):
    """Geodesic active contour model solved with iterative convolution-
    thresholding method (ICTM)."""
    if type(tau) is not list and type(tau) is not tuple:
        tau = (tau, ) * image_dimension
    if mask is not None:
        g.mul_(mask)
        mask = mask.to(th.bool)
    g_sqrt = g.sqrt()
    kern = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                     tau).to(g)
    steps = 0
    u = init_seg.gt(0).to(g)
    while steps <= max_step:
        phi = g_sqrt * convolve(g_sqrt *
                                (1 - 2 * u), kern, image_dimension) + lambd * g
        u_next = phi.lt(0)
        if mask is not None:
            u_next.mul_(mask)
        if u_next.ne(u).sum() == 0:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.to(th.uint8)


def local_intensity_fitting(im,
                            init_seg,
                            image_dimension,
                            sigma=1.5,
                            tau=1.,
                            lambd=90,
                            mu1=1.,
                            mu2=1.,
                            eps=0.01,
                            max_step=30,
                            kernel_size=9):
    """Local intensity fitting model solved with iterative convolution-
    thresholding method (ICTM)."""
    kern_sigma = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                           (sigma, ) * image_dimension).to(im)
    kern_tau = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                         (tau, ) * image_dimension).to(im)
    u = ((1 - init_seg.gt(0).to(im)) * (1 - eps * 2) + eps)
    steps = 0
    im_sigma = convolve(im, kern_sigma, image_dimension)
    one_sigma = convolve(th.ones_like(im), kern_sigma, image_dimension)
    while steps <= max_step:
        Ik = im * u
        c1 = convolve(u, kern_sigma, image_dimension)
        c2 = convolve(Ik, kern_sigma, image_dimension)
        f1 = c2 / c1
        f2 = (im_sigma - c2) / (one_sigma - c1)
        phi1 = mu1 * convolve(
            f1.pow(2), kern_sigma, image_dimension) - 2 * mu1 * im * convolve(
                f1, kern_sigma, image_dimension) + lambd * convolve(
                    1 - u, kern_tau, image_dimension)
        phi2 = mu2 * convolve(
            f2.pow(2), kern_sigma, image_dimension) - 2 * mu2 * im * convolve(
                f2, kern_sigma, image_dimension) + lambd * convolve(
                    u, kern_tau, image_dimension)
        u_next = phi1.le(phi2).to(im) * (1 - eps * 2) + eps
        if (u_next - u).abs().sum() < eps:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.le(eps).to(th.uint8)

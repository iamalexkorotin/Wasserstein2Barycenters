import os, sys
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import sqrtm

import os, sys
import argparse
import collections
from scipy.io import savemat
from tqdm import trange
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import multiprocessing
import itertools

import torch
from PIL import Image
sys.path.append("..")

import gc

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def score_forward_maps(benchmark, D_list, score_size=1024):
    assert (benchmark.bar_maps is not None) and (benchmark.bar_sampler is not None)
    L2_UVP = []
    Y = benchmark.bar_sampler.sample(score_size).detach()
    for n in range(benchmark.num):
        X = benchmark.samplers[n].sample(score_size)
        X_push = D_list[n].push(X).detach()
        with torch.no_grad():
            X_push_true = benchmark.bar_maps[n](X)
            L2_UVP.append(
                100 * (((X_push - X_push_true) ** 2).sum(dim=1).mean() / benchmark.bar_sampler.var).item()
            )
    return L2_UVP

def score_pushforwards(benchmark, D_list, score_size=128*1024, batch_size=1024):
    assert (benchmark.bar_sampler is not None)
    BW2_UVP = []
    if score_size < batch_size:
        batch_size = score_size
    num_chunks = score_size // batch_size
    
    for n in range(benchmark.num):
        X_push = np.vstack([
            D_list[n].push(benchmark.samplers[n].sample(batch_size)).cpu().detach().numpy()
            for _ in range(num_chunks)
        ])
        X_push_cov = np.cov(X_push.T)
        X_push_mean = np.mean(X_push, axis=0)   
        UVP = 100 * calculate_frechet_distance(
            X_push_mean, X_push_cov,
            benchmark.bar_sampler.mean, benchmark.bar_sampler.cov,
        ) / benchmark.bar_sampler.var
        BW2_UVP.append(UVP)
    return BW2_UVP

def score_cycle_consistency(benchmark, D_list, D_conj_list, score_size=1024):
    cycle_UVP = []
    for n in range(benchmark.num):
        X = benchmark.samplers[n].sample(score_size)
        X_push = D_list[n].push(X).detach()
        X_push.requires_grad_(True)
        X_push_inv = D_conj_list[n].push(X_push).detach()
        with torch.no_grad():
            cycle_UVP.append(
                100 * (((X - X_push_inv) ** 2).sum(dim=1).mean() / benchmark.samplers[n].var).item()
            )
    return cycle_UVP

def score_congruence(benchmark, D_conj_list, score_size=1024):
    assert benchmark.bar_sampler is not None
    Y = benchmark.bar_sampler.sample(score_size)
    Y_sum = torch.zeros_like(Y).detach()
    for n in range(benchmark.num):
        Y_push = D_conj_list[n].push(Y).detach()
        with torch.no_grad():
            Y_sum += benchmark.alphas[n] * Y_push
    return 100 * (((Y - Y_sum) ** 2).sum(dim=1).mean() / benchmark.bar_sampler.var).item()
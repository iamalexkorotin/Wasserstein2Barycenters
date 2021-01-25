import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import sqrtm
from .tools import calculate_frechet_distance
from tqdm import tqdm_notebook as tqdm
from . import distributions

def symmetrize(X):
    return np.real((X + X.T) / 2)

def get_barycenter_cov(covs, alphas, max_iter=1000, tol=1e-8, verbose=True):
    # Iterative computation of barycenter's covariance
    # matrix via fixed-point approach
    bar_cov = np.eye(covs[0].shape[0], dtype=np.float32)
    
    for iteration in tqdm(range(max_iter)) if verbose else range(max_iter):
        bar_cov_old = bar_cov
        root_bar_cov = symmetrize(sqrtm(bar_cov))
        inv_root_bar_cov = symmetrize(np.linalg.inv(root_bar_cov))

        # To remove cycle Batch sqrtm required (does it exist?)
        inner_sum = 0.
        for k in range(len(alphas)):
            inner_sum += alphas[k] * symmetrize(sqrtm(root_bar_cov @ covs[k] @ root_bar_cov))
        inner_sum = symmetrize(inner_sum)
        inner_sum = inner_sum @ inner_sum
        bar_cov = symmetrize(inv_root_bar_cov @ inner_sum @ inv_root_bar_cov)
        if np.max(np.abs((bar_cov - bar_cov_old))) < tol:
            break
            
    return bar_cov

def get_linear_transport(mean1, cov1, mean2, cov2):
    root_cov1 = symmetrize(sqrtm(cov1))
    inv_root_cov1 = symmetrize(np.linalg.inv(root_cov1))
    weight = inv_root_cov1 @ symmetrize(sqrtm(root_cov1 @ cov2 @ root_cov1)) @ inv_root_cov1
    bias = mean2 - weight @ mean1
    return weight, bias


class Benchmark:
    pass

class Wasserstein2BarycenterBenchmark(Benchmark):
    def __init__(
        self, samplers, bar_sampler=None, alphas=None,
        compute_gaussian=True, max_iter=1000, tol=1e-6,
        device='cuda', requires_grad=False, verbose=False
    ):
        super(Wasserstein2BarycenterBenchmark, self).__init__()
        self.verbose = verbose
        self.dim = samplers[0].dim
        self.num = len(samplers)
        if alphas is not None:
            self.alphas = alphas
        else:
            self.alphas = np.ones(self.num, dtype=np.float32) / self.num
        self.device = device
        self.requires_grad = requires_grad
        
        self.samplers = samplers
        
        self.gauss_bar_sampler = None
        self.bar_sampler = bar_sampler
        self.bar_maps = None
        self.bar_maps_inv = None
        self.bar_cost = None
        
        if compute_gaussian:
            self._compute_gaussian_barycenter(max_iter=max_iter, tol=tol)
        
    def _compute_gaussian_barycenter(self, max_iter=1000, tol=1e-6):
        if self.verbose:
            print(f'Computing Gaussian Barycenter Covariance, max_iter={max_iter}')
        gauss_bar_cov = get_barycenter_cov(
            [sampler.cov for sampler in self.samplers], self.alphas,
            max_iter, tol, verbose=self.verbose
        )
        self.gauss_bar_sampler = distributions.NormalSampler(
            np.zeros(self.dim, dtype=np.float32), cov=gauss_bar_cov,
            device=self.device, requires_grad=self.requires_grad
        )
        
        if self.verbose:
            print('Computing the Gaussian Barycenter Functional')   
        self.gauss_bar_cost = np.sum([self.alphas[n] * calculate_frechet_distance(
            self.samplers[n].mean, self.samplers[n].cov,
            self.gauss_bar_sampler.mean, self.gauss_bar_sampler.cov,
        ) for n in range(self.num)])
        
        self.gauss_bar_maps_inv, self.gauss_bar_maps = [], []
        for n in tqdm(range(self.num)) if self.verbose else range(self.num):
            weight_inv, bias_inv = get_linear_transport(
                self.gauss_bar_sampler.mean, self.gauss_bar_sampler.cov,
                self.samplers[n].mean, self.samplers[n].cov,
            )
            map_inv = nn.Linear(self.dim, self.dim).to(self.device)
            map_inv.weight.data = torch.tensor(weight_inv, device=self.device)
            map_inv.bias.data = torch.tensor(bias_inv, device=self.device)
            self.gauss_bar_maps_inv.append(map_inv)
            
            weight, bias = get_linear_transport(
                self.samplers[n].mean, self.samplers[n].cov,
                self.gauss_bar_sampler.mean, self.gauss_bar_sampler.cov,
            )
            map_fwd = nn.Linear(self.dim, self.dim).to(self.device)
            map_fwd.weight.data = torch.tensor(weight, device=self.device)
            map_fwd.bias.data = torch.tensor(bias, device=self.device)
            self.gauss_bar_maps.append(map_fwd)
        
class LocationScatterBenchmark(Wasserstein2BarycenterBenchmark):
    def __init__(
        self, sampler, means, covs, alphas=None,
        compute_barycenter=True, max_iter=1000, tol=1e-6,
        device='cuda', requires_grad=False, verbose=False
    ):
        samplers = []
        for mean, cov in zip(means, covs):
            weight, bias = get_linear_transport(sampler.mean, sampler.cov, mean, cov)
            samplers.append(
                distributions.LinearTransformer(
                    weight, bias, requires_grad=requires_grad
                ).fit(sampler)
            )
            
        super(LocationScatterBenchmark, self).__init__(
            samplers, alphas=alphas,
            compute_gaussian=compute_barycenter, max_iter=max_iter, tol=tol,
            device=device, requires_grad=requires_grad, verbose=verbose
        )
        
        if compute_barycenter:
            self.bar_cost = self.gauss_bar_cost
            self.bar_maps = self.gauss_bar_maps
            self.bar_maps_inv = self.gauss_bar_maps_inv
            
            weight, bias = get_linear_transport(
                sampler.mean, sampler.cov,
                self.gauss_bar_sampler.mean, self.gauss_bar_sampler.cov
            )
            self.bar_sampler = distributions.LinearTransformer(
                weight, bias,
                requires_grad=self.requires_grad,
                device=self.device
            ).fit(sampler)
            
class EigenWarpBenchmark(LocationScatterBenchmark):
    def __init__(
        self, sampler, num=3, min_eig=0.5, max_eig=2., shift=0., alphas=None,
        compute_barycenter=True, max_iter=1000, tol=1e-6,
        device='cuda', requires_grad=False, verbose=False
    ):
        self.num = num
        self.dim = sampler.dim
        self.min_eig, self.max_eig = min_eig, max_eig
        self.shift = shift
        self.verbose = verbose
        means = self.shift * np.random.normal(size=(self.num, self.dim)).astype(np.float32)
        covs = np.zeros((self.num, self.dim, self.dim), dtype=np.float32)
        
        if self.verbose:
            print('Generating Covariance Matrices')
        for n in range(self.num):
            rotation = ortho_group.rvs(self.dim)
            weight = rotation @ np.diag(np.exp(np.linspace(np.log(min_eig), np.log(max_eig), self.dim)))
            covs[n] = weight @ weight.T
            
        super(EigenWarpBenchmark, self).__init__(
            sampler, means, covs, alphas=alphas,
            compute_barycenter=compute_barycenter, max_iter=max_iter, tol=tol,
            device=device, requires_grad=requires_grad, verbose=verbose
        )
        
# class RotatedGaussiansBenchmark(Wasserstein2BarycenterBenchmark):
#     def __init__(
#         self,
#         dim=2, count=2, alphas=None,
#         eig=(0.5, 2.), shift=3.,
#         max_iter=1000, tol=1e-6,
#         verbose=False,
#         device='cuda',
#         dtype=torch.float32,
#         requires_grad=False
#     ):
#         super(RotatedGaussiansBenchmark, self).__init__(
#             dim, count, alphas,
#             device=device, 
#             requires_grad=requires_grad
#         )
        
#         self.eig = eig
#         self.shift = shift
#         self.verbose = verbose
        
#         means = self.shift * np.random.normal(size=(self.count, self.dim)).astype(np.float32)
#         means -= (means.T * self.alphas).sum(axis=1)
        
#         transforms = np.zeros((self.count, self.dim, self.dim), dtype=np.float32)
        
#         if self.verbose:
#             print('Generating Covariance Matrices')
#         for k in range(self.count):
#             rotation = ortho_group.rvs(self.dim)
#             transforms[k] = rotation @ np.diag(np.exp(np.linspace(np.log(eig[0]), np.log(eig[1]), self.dim)))
    
        
#         if self.verbose:
#             print('Initializing samplers')
#         self.samplers = [
#             distributions.NormalSampler(
#                 means[k], weight=transforms[k],
#                 device=self.device, requires_grad=self.requires_grad
#             ) for k in range(count)
#         ]
#         self._compute_barycenter(max_iter, tol)
        
#     def _compute_barycenter(self, max_iter=1000, tol=1e-6):
#         if self.verbose:
#             print(f'Computing Barycenter Covariance, max_iter={max_iter}')
#         bar_cov = get_barycenter_cov(
#             [sampler.cov for sampler in self.samplers], self.alphas,
#             max_iter, tol, verbose=self.verbose
#         )
#         self.bar_sampler = distributions.NormalSampler(
#             np.zeros(self.dim, dtype=np.float32), cov=bar_cov,
#             device=self.device, requires_grad=self.requires_grad
#         )
        
#         if self.verbose:
#             print('Computing inverse and forward maps to barycenter')        
#         self.bar_maps_inv, self.bar_maps = [], []
#         self.bar_cost = 0.
#         for k in tqdm(range(self.count)) if self.verbose else range(self.count):
#             weight, bias, weight_inv, bias_inv = get_linear_transport(
#                 self.samplers[k].mean, self.samplers[k].cov,
#                 self.bar_sampler.mean, self.bar_sampler.cov,
#             )
#             map_inv = nn.Linear(self.dim, self.dim).to(self.device)
#             map_inv.weight.data = torch.tensor(weight_inv, device=self.device)
#             map_inv.bias.data = torch.tensor(bias_inv, device=self.device)
#             self.bar_maps_inv.append(map_inv)
            
#             map_fwd = nn.Linear(self.dim, self.dim).to(self.device)
#             map_fwd.weight.data = torch.tensor(weight, device=self.device)
#             map_fwd.bias.data = torch.tensor(bias, device=self.device)
#             self.bar_maps.append(map_fwd)
            
#             self.bar_cost += self.alphas[k] * calculate_frechet_distance(
#                 self.samplers[k].mean, self.samplers[k].cov,
#                 self.bar_sampler.mean, self.bar_sampler.cov,
#             )
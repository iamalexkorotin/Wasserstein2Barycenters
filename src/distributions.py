import torch
import numpy as np
from scipy.linalg import sqrtm
import sklearn.datasets
import random

def symmetrize(X):
    return np.real((X + X.T) / 2)

class Sampler:
    def __init__(
        self, device='cuda',
        requires_grad=False,
    ):
        self.device = device
        self.requires_grad = requires_grad
    
    def sample(self, batch_size=5):
        pass
    
    def _estimate_mean(self, num_samples=100000):
        batch = self.sample(num_samples).cpu().detach().numpy()
        self.mean = batch.mean(axis=0).astype(np.float32)
    
    def _estimate_cov(self, num_samples=100000):
        batch = self.sample(num_samples).cpu().detach().numpy()
        self.cov = np.cov(batch.T).astype(np.float32)
        self.var = np.trace(self.cov)
    
class StandardNormalSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda',
        requires_grad=False
    ):
        super(StandardNormalSampler, self).__init__(device, requires_grad)
        self.dim = dim
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = np.eye(self.dim, dtype=np.float32)
        self.var = self.dim
        
    def sample(self, batch_size=10):
        return torch.randn(
            batch_size, self.dim,
            device=self.device,
            requires_grad=self.requires_grad
        )
    
class NormalSampler(Sampler):
    def __init__(
        self, mean, cov=None, weight=None, device='cuda',
        requires_grad=False
    ):
        super(NormalSampler, self).__init__(device=device, requires_grad=requires_grad)
        self.mean = np.array(mean, dtype=np.float32)
        self.dim = self.mean.shape[0]
        
        if weight is not None:
            weight = np.array(weight, dtype=np.float32)
        
        if cov is not None:
            self.cov = np.array(cov, dtype=np.float32)
        elif weight is not None:
            self.cov = weight @ weight.T
        else:
            self.cov = np.eye(self.dim, dtype=np.float32)
            
        if weight is None:
            weight = symmetrize(sqrtm(self.cov))
            
        self.var = np.trace(self.cov)
        
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(self.mean, device=self.device, dtype=torch.float32)

    def sample(self, batch_size=4):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch.requires_grad_(self.requires_grad)
        return batch
    
class CubeUniformSampler(Sampler):
    def __init__(
        self, dim=1, centered=False, normalized=False, device='cuda',
        requires_grad=False
    ):
        super(CubeUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = np.eye(self.dim, dtype=np.float32) if self.normalized else np.eye(self.dim, dtype=np.float32) / 12
        self.mean = np.zeros(self.dim, dtype=np.float32) if self.centered else .5 * np.ones(self.dim, dtype=np.float32)
        
        self.bias = torch.tensor(self.mean, device=self.device)
        
    def sample(self, batch_size=10):
        return np.sqrt(self.var) * (torch.rand(
            batch_size, self.dim, device=self.device,
            requires_grad=self.requires_grad
        ) - .5) / np.sqrt(self.dim / 12)  + self.bias

class BoxUniformSampler(Sampler):
    # A uniform box with axes components and the range on each
    # axis i is [a_min[i], a_max[i]].
    def __init__(
        self, components, a_min, a_max, estimate_size=100000,
        device='cuda', requires_grad=False
    ):
        super(BoxUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = components.shape[1]
        self.components = torch.from_numpy(components).float().to(device=device)
        self.a_min = torch.from_numpy(a_min).float().to(device=device)
        self.a_max = torch.from_numpy(a_max).float().to(device=device)
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
    
    def sample(self, batch_size):
        with torch.no_grad():
            batch = torch.rand(
                batch_size, self.dim,
                device=self.device
            )
            batch = (torch.unsqueeze(self.a_min, 0) + 
                     batch * torch.unsqueeze(self.a_max - self.a_min, 0))
            batch = torch.matmul(batch, self.components)
            return torch.tensor(
                batch, device=self.device,
                requires_grad=self.requires_grad
            )
        
class EmpiricalSampler(Sampler):
    def __init__(
        self, data, estimate_size=100000,
        device='cuda', requires_grad=False
    ):
        super(EmpiricalSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        # data is a np array NxD
        self.dim = data.shape[1]
        self.num_points = data.shape[0]
        self.data = torch.from_numpy(data).float().to(device=device)
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size):
        inds = torch.randperm(self.num_points)
        if batch_size <= self.num_points:
            inds = inds[:batch_size]
        else:
            additional_inds = torch.randint(0, self.num_points, (batch_size - self.num_points))
            inds = torch.cat([inds, additional_inds], dim=0)
        inds_repeated = torch.unsqueeze(inds, 1).repeat(1, self.dim)
        batch = torch.gather(self.data, 0, inds_repeated.to(device=self.device))
        return torch.tensor(
                batch, device=self.device,
                requires_grad=self.requires_grad
        )
    
class TensorDatasetSampler(Sampler):
    def __init__(
        self, dataset, transform=None, storage='cpu', storage_dtype=torch.float,
        device='cuda', requires_grad=False, estimate_size=100000,
    ):
        super(TensorDatasetSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.storage = storage
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda t: t
            
        self.storage_dtype = storage_dtype
        
        self.dataset = torch.tensor(
            dataset, device=storage, dtype=storage_dtype, requires_grad=False
        )  
        
        self.dim = self.sample(1).shape[1]
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size) 
        
    def sample(self, batch_size=10):
        if batch_size:
            ind = random.choices(range(len(self.dataset)), k=batch_size)
        else:
            ind = range(len(self.dataset))
            
        with torch.no_grad():
            batch = self.transform(torch.tensor(
                self.dataset[ind], device=self.device,
                dtype=torch.float32, requires_grad=False
            ))
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch
        
    
class BallCrustUniformSampler(Sampler):
    def __init__(
        self, dim=2, r_min=0.8, r_max=1.2, estimate_size=100000,
        device='cuda', requires_grad=False
    ):
        super(BallCrustUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = dim
        assert r_min >= 0
        assert r_min < r_max
        self.r_min, self.r_max = r_min, r_max
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(
                batch_size, self.dim,
                device=self.device
            )
            batch /= torch.norm(batch, dim=1)[:, None]
            ratio = (1 - (self.r_max - self.r_min) / self.r_max) ** self.dim
            r = (torch.rand(
                batch_size, device=self.device
            ) * (1 - ratio) + ratio) ** (1. / self.dim)
        
        return torch.tensor(
            (batch.transpose(0, 1) * r * self.r_max).transpose(0, 1),
            device=self.device,
            requires_grad=self.requires_grad
        )
    
class MixN2GaussiansSampler(Sampler):
    def __init__(self, n=5, std=1, step=9, device='cuda', estimate_size=100000,
        requires_grad=False
    ):
        super(MixN2GaussiansSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        
        self.dim = 2
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        centers = np.stack([xx, yy]).reshape(2, -1).T
        self.centers = torch.tensor(
            centers,
            device=self.device,
            dtype=torch.float32
        )
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device
        )
        indices = random.choices(range(len(self.centers)), k=batch_size)
        batch *= self.std
        batch += self.step * self.centers[indices, :]
        return torch.tensor(
            batch, device=self.device,
            requires_grad=self.requires_grad
        )
    
class CubeCrustUniformSampler(Sampler):
    def __init__(
        self, dim=2, r_min=0.8, r_max=1.2, estimate_size=100000, device='cuda',
        requires_grad=False
    ):
        super(CubeCrustUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = dim
        assert r_min >= 0
        assert r_min < r_max
        self.r_min, self.r_max = r_min, r_max
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = 2 * torch.rand(
                batch_size, self.dim,
                device=self.device
            ) - 1
            axes = torch.randint(0, self.dim, size=(batch_size, 1), device=self.device)
            batch.scatter_(
                1, axes, 
                2 * ((batch.gather(1, axes) > 0)).type(torch.float32) - 1
            )
            
            ratio = (1 - (self.r_max - self.r_min) / self.r_max) ** self.dim
            r = (torch.rand(
                batch_size, device=self.device
            ) * (1 - ratio) + ratio) ** (1. / self.dim)
        
        return torch.tensor(
            (batch.transpose(0, 1) * self.r_max * r).transpose(0, 1),
            device=self.device, 
            requires_grad=self.requires_grad
        )
    
class SwissRollSampler(Sampler):
    def __init__(
        self, estimate_size=100000, device='cuda', requires_grad=False
    ):
        super(SwissRollSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = 2
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        batch = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype(np.float32)[:, [0, 2]] / 7.5
        return torch.tensor(
            batch, device=self.device,
            requires_grad=self.requires_grad
        )
    
class Mix8GaussiansSampler(Sampler):
    def __init__(
        self, with_central=False, std=1, r=12,
        estimate_size=100000, 
        device='cuda', requires_grad=False
    ):
        super(Mix8GaussiansSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(
            centers, device=self.device
        )
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(
                batch_size, self.dim,
                device=self.device
            )
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch

    
class Transformer(object):
    def __init__(
        self, device='cuda',
        requires_grad=False
    ):
        self.device = device
        self.requires_grad = requires_grad
        
class LinearTransformer(Transformer):
    def __init__(
        self, weight, bias=None, base_sampler=None,
        device='cuda',
        requires_grad=False
    ):
        super(LinearTransformer, self).__init__(
            device=device,
            requires_grad=requires_grad
        )
        
        self.fitted = False
        self.dim = weight.shape[0]
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32, requires_grad=False)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32, requires_grad=False)
        else:
            self.bias = torch.zeros(self.dim, device=device, dtype=torch.float32, requires_grad=False)
        
        
        if base_sampler is not None:
            self.fit(base_sampler)

        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        weight, bias = self.weight.cpu().numpy(), self.bias.cpu().numpy()
        
        self.mean = weight @ self.base_sampler.mean + bias
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)
        
        self.fitted = True
        return self
        
    def sample(self, batch_size=4):
        assert self.fitted == True
        
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, requires_grad=False
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch = batch.detach()
        batch.requires_grad_(self.requires_grad)
        return batch
    
class StandardNormalScaler(Transformer):
    def __init__(
        self, base_sampler=None, device='cuda', requires_grad=False
    ):
        super(StandardNormalScaler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        if base_sampler is not None:
            self.fit(base_sampler)
        
    def fit(self, base_sampler, batch_size=1000):
        self.base_sampler = base_sampler
        self.dim = self.base_sampler.dim
        
        self.bias = torch.tensor(
            self.base_sampler.mean, device=self.device, dtype=torch.float32
        )
        
        weight = symmetrize(np.linalg.inv(sqrtm(self.base_sampler.cov)))
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)
        
        return self
        
    def sample(self, batch_size=10):
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, requires_grad=False
        )
        with torch.no_grad():
            batch -= self.bias
            batch @= self.weight
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        return quad + linear
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)
    
class Conv2dConvexQuadratic(nn.Module):
    '''Convolutional Input-Convex Quadratic Layer'''
    def __init__(
        self, in_channels, out_channels, kernel_size, rank,
        stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
    ):
        super(Conv2dConvexQuadratic, self).__init__()
        
        assert rank > 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
    
        self.quadratic_decomposed = nn.Conv2d(
            in_channels, out_channels * rank, kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False,
            padding_mode=self.padding_mode
        )
        
        self.linear = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode
        )
        
    def forward(self, input):
        output = (self.quadratic_decomposed(input) ** 2)
        n, c, h, w = output.size()
        output = output.reshape(n, c // self.rank, self.rank, h, w).sum(2)
        output += self.linear(input)
        return output
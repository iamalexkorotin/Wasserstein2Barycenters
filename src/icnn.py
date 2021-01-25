import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvexQuadratic, Conv2dConvexQuadratic
    
class DenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', dropout=0.03,
        strong_convexity=1e-6
    ):
        super(DenseICNN, self).__init__()
        
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation
        self.rank = rank
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
                nn.Dropout(dropout)
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.Dropout(dropout)
            )
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output    
    
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
              
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)

class ConvICNN128(nn.Module):
    def __init__(self, channels=3):
        super(ConvICNN128, self).__init__()

        self.first_linear = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
        )
        
        self.first_squared = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        )
        
        self.convex = nn.Sequential(
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1),  
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            View(32* 8 * 8),
            nn.CELU(), 
            nn.Linear(32 * 8 * 8, 128),
            nn.CELU(), 
            nn.Linear(128, 64),
            nn.CELU(), 
            nn.Linear(64, 32),
            nn.CELU(), 
            nn.Linear(32, 1),
            View()
        ).cuda()

    def forward(self, input): 
        output = self.first_linear(input) + self.first_squared(input) ** 2
        output = self.convex(output)
        return output
    
    def push(self, input):
        return autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True, grad_outputs=torch.ones(input.size()[0]).cuda().float()
        )[0]
    
    def convexify(self):
        for layer in self.convex:
            if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)
                
class ConvICNN16(nn.Module):
    '''
    ConvICNN for 1 x 16 x 16 images. 
    Several convolutional layers with input-quadratic convolutional skip connections are 
    followed by positive fully-connected layers.
    '''
    def __init__(self, strong_convexity=0.01, dropout=0.01, rank=1, unflatten=True):
        super(ConvICNN16, self).__init__()
        
        self.strong_convexity = strong_convexity
        self.dropout = dropout
        self.rank = rank
        self.unflatten = unflatten
        
        self.convex_layers = nn.ModuleList([
            nn.Conv2d(512, 512, 3, padding=1, stride=2), # bs x 256 x 8 x 8
            nn.Conv2d(512, 512, 3, padding=1, stride=2), # bs x 256 x 8 x 8
        ])
        
        self.quadratic_layers = nn.ModuleList([
            Conv2dConvexQuadratic(1, 512, 5, rank=self.rank, padding=2, stride=1, bias=False),  # bs x 128 x 16 x16
            Conv2dConvexQuadratic(1, 512, 7, rank=self.rank, padding=3, stride=2, bias=False),  # bs x 128 x 8 x 8
            Conv2dConvexQuadratic(1, 512, 9, rank=self.rank, padding=4, stride=4, bias=False),  # bs x 128 x 8 x 8
        ])
        
        self.pos_features = nn.Sequential(
            nn.Dropout2d(self.dropout),
            nn.Conv2d(512, 256, 4, padding=1, stride=2),
            nn.CELU(0.2, True),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(256, 1, 2, padding=0, stride=1), # bs x 1 x 1 x 1
            View(1),
        )
        
#         img = torch.randn(5, 1, 16, 16)
#         print(self(img).shape)
#         print('Input Quadratic Convolutions Output shapes')
#         for layer in self.quadratic_layers:
#             print(layer(img).shape)

#         print('Sequential Convolutions Output shapes\nEmpty')
#         img = self.quadratic_layers[0](img)
#         for layer in self.convex_layers:
#             img = layer(img)
#             print(img.shape)
#         print('Final Shape')
#         print(self.pos_features(img).shape)
                
    def forward(self, input):
        if self.unflatten:
            input = input.reshape(-1, 1, 16, 16)
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            output = torch.celu(output)
            if self.training:
                output = F.dropout2d(output, p=self.dropout)
        output = self.pos_features(output)
        
        return output + .5 * self.strong_convexity * (input ** 2).flatten(start_dim=1).sum(dim=1).reshape(-1, 1)
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output
    
    
    def convexify(self):
        for layer in self.convex_layers:
            if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)
        for layer in self.pos_features:
            if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)
import torch
from torch import nn

class Relation(nn.Module):
    def __init__(self, weight = None, size_average = None):
        super(Relation, self).__init__()
        
    def forward(self, inputs: torch.tensor) -> torch.float:
        return self.function(inputs)
    
    def function(self, inputs: torch.tensor, *args, **kwargs) -> torch.float:
        raise NotImplementedError()
        
class RelationalLoss(nn.Module):
    def __init__(self, loss, relation: Relation, alpha: float = None, weight = None, size_average = None):
        super(RelationalLoss, self).__init__()
        self.loss = loss
        self.relation = relation
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        alpha = self.alpha if self.alpha != None else 1
        reconstruction = self.loss(inputs, targets)
        relation = self.loss(self.relation(inputs), self.relation(targets))
        
        return (1 - alpha)*reconstruction + alpha*relation, reconstruction

def activation(act_function):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU(negative_slope = 0.01, inplace = True)],
        ['relu', nn.ReLU(inplace = True)],
        ['none', nn.Identity()],
        ['selu', nn.SELU(inplace = True)],
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()]
    ])
    return activations[act_function]

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.block = None
    
    def forward(self, inputs):
        return self.block(inputs)

class LinearBlock(Block):
    def __init__(self, in_size: int, out_size: int, act_function: str):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(in_size, out_size),
                                   activation(act_function))

class ConvBlock(Block):
    def __init__(self, in_size: int, out_size: int, kernel: int, act_function: str, stride: int = 1, pad_size: int = 0):
        super(ConvBlock, self).__init__()
        if pad_size == None: pad_size = kernel // 2 # Padding will be setted such that the maps dimensions are preserved, if padding dimension wasn't manually
                                                    # specified.
        self.block = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride = stride, padding = pad_size, bias = False),
                                   nn.BatchNorm2d(out_size),
                                   activation(act_function))
        
class ConvPoolBlock(Block):
    def __init__(self, in_size: int, out_size: int, kernel: int, pool_kernel: int, act_function: str, pool_stride: int = 2, stride: int = 1, pad_size: int = 0):
        super(ConvPoolBlock, self).__init__()
        if pad_size == None: pad_size = kernel // 2
        self.block = nn.Sequential(ConvBlock(in_size, out_size, kernel, act_function, stride, pad_size),
                                   nn.MaxPool2d(pool_kernel, pool_stride))
        
class ConvUpsampleBlock(Block):
    def __init__(self, in_size: int, out_size: int, scale_factor: int, act_function: str, kernel: int, stride: int = 1, pad_size: int = 0):
        super(ConvUpsampleBlock, self).__init__()
        if pad_size == None: pad_size = kernel // 2
        self.block = nn.Sequential(ConvBlock(in_size, out_size, kernel, act_function, stride, pad_size),
                                   nn.UpsamplingBilinear2d(scale_factor=scale_factor))


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz

class LorentzBatchNorm(nn.Module):
    """ 2D Lorentz Layer Normalization with Centroid and Fréchet variance """
    
    def __init__(self, manifold: CustomLorentz, num_channels: int):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold
        self.beta = ManifoldParameter(self.manifold.origin(num_channels), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))
        self.eps = 1e-5
        # No running statistics needed!

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)  # [bs, H*W, c]
        
        # Compute centroid per sample: [B, N, C] -> [B, C]
        mean = self.manifold.centroid(x)
        mean = mean.unsqueeze(1)  # [B, 1, C] for broadcasting
        
        # Transport each sample's points to origin using that sample's centroid
        x_T = self.manifold.logmap(mean, x)
        x_T = self.manifold.transp0back(mean, x_T)
        
        # Compute Fréchet variance per sample: [B, N, C] -> [B, 1]
        var = torch.mean(torch.norm(x_T, dim=-1), dim=1, keepdim=True)
        
        # Rescale (need to unsqueeze var for broadcasting over C dimension)
        x_T = x_T * (self.gamma / (var.unsqueeze(-1) + self.eps))
        
        # Transport to learned mean beta
        x_T = self.manifold.transp0(self.beta, x_T)
        output = self.manifold.expmap(self.beta, x_T)
        
        output = output.reshape(bs, h, w, c)
        return output

class LorentzBatchNorm1d(LorentzBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int):
        super(LorentzBatchNorm1d, self).__init__(manifold, num_features)

    def forward(self, x, momentum=0.75):
        return super(LorentzBatchNorm1d, self).forward(x)

class LorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int):
        super(LorentzBatchNorm2d, self).__init__(manifold, num_channels)

    def forward(self, x, momentum=0.75):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        # bs, h, w, c = x.shape
        # x = x.view(bs, -1, c)
        # print(x.shape)
        x = super(LorentzBatchNorm2d, self).forward(x)
        # print(x.shape)

        return x

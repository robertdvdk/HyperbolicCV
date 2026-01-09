import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz


class LorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold

        self.mean = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        # self.mean = torch.nn.Parameter(torch.zeros(num_features))
        self.var = torch.nn.Parameter(torch.ones((1,)))
        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum):
        # assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."
        input_mean = self.manifold.centroid(x).unsqueeze(0)
        xLmean = -(x.narrow(dim=-1, start=0, length=1) * input_mean[:, 0]) + (x[:, 1:] * input_mean[:, 1:]).sum(dim=-1, keepdim=True)
        d_sq = 2/self.manifold.k - 2*xLmean
        input_var = d_sq.mean()

        input_logm = self.manifold.transp(x=input_mean, y=self.mean, v=self.manifold.logmap(input_mean, x))
        
        input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

        output = self.manifold.expmap(self.mean.unsqueeze(-2), input_logm)
        return output


class LorentzBatchNorm1d(LorentzBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int):
        super(LorentzBatchNorm1d, self).__init__(manifold, num_features)

    def forward(self, x, momentum=0.1):
        return super(LorentzBatchNorm1d, self).forward(x, momentum)

class LorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int):
        super(LorentzBatchNorm2d, self).__init__(manifold, num_channels)

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(-1, c)
        x = super(LorentzBatchNorm2d, self).forward(x, momentum)
        x = x.reshape(bs, h, w, c)

        return x

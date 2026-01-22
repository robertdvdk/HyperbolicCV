import torch.nn as nn

from .LFC import LorentzFullyConnected
from .LConv import LorentzConv2d
from .LBnorm import LorentzBatchNorm1d as _ManifoldBatchNorm1d
from .LBnorm import LorentzBatchNorm2d as _ManifoldBatchNorm2d
from .LBnorm2 import LorentzBatchNorm2d as _EuclideanBatchNorm2d
from .LMLR import LorentzMLR
from .LModules import LorentzAct, LorentzReLU, LorentzGlobalAvgPool2d


class LorentzBatchNorm2d(nn.Module):
	"""Wrapper that selects batchnorm implementation based on batchnorm_impl."""
	def __new__(cls, *args, **kwargs):
		batchnorm_impl = kwargs.pop("batchnorm_impl", "manifold")
		if batchnorm_impl == "euclidean":
			if "num_channels" in kwargs and "num_features" not in kwargs:
				kwargs = dict(kwargs)
				kwargs["num_features"] = kwargs.pop("num_channels")
			return _EuclideanBatchNorm2d(*args, **kwargs)
		return _ManifoldBatchNorm2d(*args, **kwargs)


LorentzBatchNorm1d = _ManifoldBatchNorm1d

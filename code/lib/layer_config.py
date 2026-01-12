"""
Global configuration for layer behavior.
These settings can be modified from train.py to control layer implementations.
"""

# Controls which LorentzFullyConnected implementation to use
# Options: "ours" (new custom implementation) or "theirs" (Chen et al. 2022)
LINEAR_METHOD = "ours"

# Controls LorentzBatchNorm behavior
# Options: "default" (respect self.training) or "train" (always use training branch)
BATCHNORM_MODE = "default"

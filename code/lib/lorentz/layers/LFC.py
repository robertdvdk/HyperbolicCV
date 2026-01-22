import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz

class LorentzFullyConnectedNew(nn.Module):
    def __init__(
        self,
        manifold: CustomLorentz,
        in_features,
        out_features,
        init_method="eye",
        reset_params=None,
        a_default=0.0,
        activation=nn.Identity(),
        do_mlr = False,
        bias=False,
        normalize=False,
        mlr_std_mult=1.0,
    ):
        super().__init__()
        self.manifold = manifold
        in_features = in_features - 1
        out_features = out_features - 1
        self.init_method = init_method
        self.U = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features, out_features))
        resolved_reset = self._resolve_reset_params(init_method, reset_params, do_mlr)
        self.reset_parameters(reset_params=resolved_reset, a_default=a_default, mlr_std_mult=mlr_std_mult)
        self.activation = activation

        self.do_mlr = do_mlr
        

    def _resolve_reset_params(self, init_method, reset_params, do_mlr):
        if do_mlr:
            return "mlr"
        if reset_params is not None:
            return reset_params
        if init_method in [None, "old"]:
            return "eye"
        return init_method

    def reset_parameters(self, reset_params, a_default, mlr_std_mult=1.0):
        in_features, out_features = self.U.shape
            

        in_features, out_features = self.U.shape
        if reset_params == "eye":
            scale = 0.5
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(scale * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(scale * torch.eye(in_features, out_features))
        elif reset_params == "kaiming":
            with torch.no_grad():
                self.U.data.copy_(
                    torch.randn(in_features, out_features)
                    * (2 * in_features * out_features) ** -0.5
                )
            self.a.data.fill_(a_default)

        elif reset_params == "lorentz_kaiming":
            # For Lorentz models: divide std by 0.5 to account for time coordinate
            std = (1.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)
            self.a.data.fill_(a_default)

        elif reset_params == "mlr":
            std = (5.0 / in_features) ** 0.5 * mlr_std_mult
            with torch.no_grad():
                self.U.data.normal_(0, std)
            self.a.data.fill_(a_default)

        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

    def create_spacelike_vector(self):
        U_norm = self.U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k.sqrt() * U_norm * self.a
        time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * self.U
        return torch.cat([time, space], dim=0)

    def signed_dist2hyperplanes_scaled_angle(self, x):
        """Scale the distances by scaling the angle (implicitly)"""
        V = self.create_spacelike_vector()
        sqrt_k = self.manifold.k.sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)

    def signed_dist2hyperplanes_scaled_dist(self, x):
        """Scale the distances by scaling the total distance (explicitly)"""
        V = self.create_spacelike_vector()
        V_norm = self.manifold.normL(V.transpose(0, 1)).transpose(0, 1)
        sqrt_k = self.manifold.k.sqrt()
        return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V / V_norm))

    def compute_output_space(self, x):
        V = self.create_spacelike_vector()
        return self.activation(x @ V)

    def forward(self, x):
        if self.do_mlr:
            return self.mlr(x)
        output_space = self.compute_output_space(x)
        return self.manifold.projection_space_orthogonal(output_space)

    def forward_cache(self, x):
        output_space = self.activation(x @ self.V_auxiliary)
        return self.manifold.projection_space_orthogonal(output_space)

    def mlr(self, x):
        return self.signed_dist2hyperplanes_scaled_angle(x)

class LorentzFullyConnectedOld(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False,
            init_method=None,  # Added for compatibility
            reset_params=None,  # Added for compatibility
            a_default=None,  # Added for compatibility
            activation=None,  # Added for compatibility
            do_mlr=False  # Added for compatibility
        ):

        super(LorentzFullyConnectedOld, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x = self.weight(x)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzFullyConnected(nn.Module):
    """
    Wrapper class that delegates to the appropriate implementation
    based on the linear_method at instantiation time.
    """
    def __new__(cls, *args, **kwargs):
        # Check config at instantiation time, not import time
        linear_method = kwargs.pop("linear_method", "ours")
        if linear_method == "theirs":
            kwargs.pop("mlr_std_mult", None)
            return LorentzFullyConnectedOld(*args, **kwargs)
        else:  # "ours" or default
            return LorentzFullyConnectedNew(*args, **kwargs)

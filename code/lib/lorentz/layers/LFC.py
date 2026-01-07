import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz

# class LorentzFullyConnected(nn.Module):
#     def __init__(
#         self,
#         manifold: CustomLorentz,
#         in_features,
#         out_features,
#         reset_params="eye",
#         a_default=0.0,
#         activation=nn.Identity(),
#         do_mlr = False,
#         bias=False,
#         normalize=False
#     ):
#         super().__init__()
#         self.manifold = manifold
#         in_features = in_features - 1
#         out_features = out_features - 1
#         self.U = nn.Parameter(torch.randn(in_features, out_features))
#         self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
#         self.V_auxiliary = nn.Parameter(torch.randn(in_features, out_features))
#         self.reset_parameters(reset_params=reset_params, a_default=a_default)
#         self.activation = activation

#         self.do_mlr = do_mlr

#     def reset_parameters(self, reset_params, a_default):
#         in_features, out_features = self.U.shape
#         if reset_params == "eye":
#             if in_features <= out_features:
#                 with torch.no_grad():
#                     self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
#             else:
#                 print("not possible 'eye' initialization, defaulting to kaiming")
#                 with torch.no_grad():
#                     self.U.data.copy_(
#                         torch.randn(in_features, out_features)
#                         * (2 * in_features * out_features) ** -0.5
#                     )
#             self.a.data.fill_(a_default)
#         elif reset_params == "kaiming":
#             with torch.no_grad():
#                 self.U.data.copy_(
#                     torch.randn(in_features, out_features)
#                     * (2 * in_features * out_features) ** -0.5
#                 )
#             self.a.data.fill_(a_default)
#         else:
#             raise KeyError(f"Unknown reset_params value: {reset_params}")

#     def create_spacelike_vector(self):
#         U_norm = self.U.norm(dim=0, keepdim=True)
#         U_norm_sqrt_k_b = self.manifold.k.sqrt() * U_norm * self.a
#         time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
#         space = torch.cosh(U_norm_sqrt_k_b) * self.U
#         return torch.cat([time, space], dim=0)

#     def signed_dist2hyperplanes_scaled_angle(self, x):
#         """Scale the distances by scaling the angle (implicitly)"""
#         V = self.create_spacelike_vector()
#         sqrt_k = self.manifold.k.sqrt()
#         return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)

#     def signed_dist2hyperplanes_scaled_dist(self, x):
#         """Scale the distances by scaling the total distance (explicitly)"""
#         V = self.create_spacelike_vector()
#         V_norm = self.manifold.normL(V.transpose(0, 1)).transpose(0, 1)
#         sqrt_k = self.manifold.k.sqrt()
#         return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V / V_norm))

#     def compute_output_space(self, x):
#         V = self.create_spacelike_vector()
#         return self.activation(x @ V)

#     def forward(self, x):
#         if self.do_mlr:
#             return self.mlr(x)
#         output_space = self.compute_output_space(x)
#         return self.manifold.projection_space_orthogonal(output_space)

#     def forward_cache(self, x):
#         output_space = self.activation(x @ self.V_auxiliary)
#         return self.manifold.projection_space_orthogonal(output_space)

#     def mlr(self, x):
#         return self.signed_dist2hyperplanes_scaled_angle(x)

class LorentzFullyConnected(nn.Module):
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
            normalize=False
        ):
        super(LorentzFullyConnected, self).__init__()
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

"""
Custom kernels for ENGAGEGP.

These kernels are intended to support "pure GP" variants where the GP kernel
acts on the original input space (no deep feature extractor), while still
allowing learnable input reweighting / mixing mechanisms.
"""

from __future__ import annotations

import torch
import gpytorch


class AttentionWeightedRBFKernel(gpytorch.kernels.Kernel):
    """
    RBF kernel with learnable per-dimension attention weights.

    The kernel still receives the original inputs X, but applies a learnable
    positive weight per input dimension before computing distances:

        x' = x * w,   where w = softmax(raw_w)

    This is similar in spirit to ARD, but uses an explicit normalized weighting.
    """

    has_lengthscale = True

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.register_parameter(
            name="raw_attention_weights",
            parameter=torch.nn.Parameter(torch.zeros(self.input_dim)),
        )
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)

    def _weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_attention_weights, dim=0)

    def forward(self, x1, x2, diag: bool = False, **params):
        w = self._weights().to(device=x1.device, dtype=x1.dtype)
        x1w = x1 * w
        x2w = x2 * w
        return self.base_kernel(x1w, x2w, diag=diag, **params)


class InputMixingRBFKernel(gpytorch.kernels.Kernel):
    """
    RBF kernel with a learnable linear mixing of input dimensions.

    This keeps the kernel operating on the original input space (X is passed
    to the kernel), but the distance is computed after a learned linear map:

        x' = x @ W

    This is a lightweight way to capture cross-dimension interactions for small
    input_dim settings (e.g., 2D toy problems).
    """

    has_lengthscale = True

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        # Initialize near-identity for stability.
        eye = torch.eye(self.input_dim)
        self.register_parameter(
            name="mixing_matrix",
            parameter=torch.nn.Parameter(eye.clone()),
        )
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)

    def forward(self, x1, x2, diag: bool = False, **params):
        W = self.mixing_matrix.to(device=x1.device, dtype=x1.dtype)
        x1m = x1 @ W
        x2m = x2 @ W
        return self.base_kernel(x1m, x2m, diag=diag, **params)


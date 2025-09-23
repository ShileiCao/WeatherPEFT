import xarray as xr
import pandas as pd
import numpy as np
import os
import torch


def get_lat_weights(lat):
    w_lat = np.cos(np.deg2rad(lat))
    return w_lat / w_lat.mean()

def lat_weighted_rmse(pred, y, lat):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    if isinstance(lat, torch.Tensor):
        lat = lat.numpy()
    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, 1, H, 1)

    rmse = torch.sqrt((error * w_lat).mean(dim=(-1,-2))).mean(0)

    return rmse

def evaluate_rmse_downscale(pred_surface, pred_upper, target_surface, target_upper, lat):
    B,V,C,H,W = pred_upper.shape
    rmse_sur = lat_weighted_rmse(pred_surface, target_surface, lat)
    rmse_upp = lat_weighted_rmse(pred_upper.reshape(B,-1,H,W), target_upper.reshape(B,-1,H,W), lat).reshape(V,C)
    mean_bias_sur = pred_surface.mean(dim=(0,-1,-2)) - target_surface.mean(dim=(0,-1,-2))
    mean_bias_upp = pred_upper.mean(dim=(0,-1,-2)) - target_upper.mean(dim=(0,-1,-2))
    return rmse_sur, rmse_upp, mean_bias_sur, mean_bias_upp
    


# TODO:  WCRPS metric (Weighted CRPS with EFI)

import numpy as np
from scipy import special, integrate, stats
from torch import nn
from torch.autograd import Function
import torch
from typing import Union, Tuple, Callable


_normal_dist = torch.distributions.Normal(0., 1.)
_frac_sqrt_pi = 1 / np.sqrt(np.pi)


class CrpsGaussianLoss(nn.Module):
    """
      This is a CRPS loss function assuming a gaussian distribution. We use
      the following link:
      https://github.com/tobifinn/ensemble_transformer/blob/9b31f193048a31efd6aacb759e8a8b4a28734e6c/ens_transformer/measures.py

      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15):
        super(CrpsGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor):

        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)

        if self.mode == 'mean':
            return torch.mean(crps)
        return crps

class EECRPSGaussianLoss(nn.Module):
    """
      This is a EECRPS loss function assuming a gaussian distribution with EFI indeces.
      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15):
        super(EECRPSGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor,
                efi_tensor: torch.Tensor):

        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = torch.abs(efi_tensor) * pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)

        if self.mode == 'mean':
            return torch.mean(crps)
        return crps
    
if __name__ == '__main__':
    pred = torch.randn(2,3,128,256)
    y = torch.randn(2,3,128,256)
    pred1 = torch.randn(2,3,13,128,256)
    y1 = torch.randn(2,3,13,128,256)
    lat = torch.linspace(90-1.40625/2, -90+1.40625/2, 128)
    rmse = evaluate_rmse_downscale(pred,pred1, y, y1, lat)
    print(rmse)
    
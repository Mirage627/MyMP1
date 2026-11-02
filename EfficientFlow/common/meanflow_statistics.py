"""Training-only second-moment estimation for the MeanFlow H specification.

The predicted matrix is E[e e^T | features], including mean-prediction bias;
it is not automatically the covariance of the MeanFlow training label. The
diagonal head uses the explicitly permitted ``min_variance + softplus(raw)``
parameterization. The full head uses ``L @ L.T + min_variance * I``.

Features are detached at the head boundary, residuals are detached in its NLL,
and the bounded metric is detached in the velocity loss. These are two coupled
semi-gradient updates, not full gradients of one joint likelihood objective.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _matrix_precision(tensor):
    if not tensor.is_floating_point():
        raise ValueError("statistics require floating-point tensors")
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def _check_finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _check_covariance(covariance):
    if (not isinstance(covariance, torch.Tensor) or covariance.ndim != 3
            or covariance.shape[0] < 1 or covariance.shape[1] < 1
            or covariance.shape[1] != covariance.shape[2]):
        raise ValueError("covariance must have shape [B,D,D] with B,D positive")
    covariance = _matrix_precision(covariance)
    _check_finite(covariance, "covariance")
    tolerance = 1e-10 if covariance.dtype == torch.float64 else 1e-5
    if not torch.allclose(covariance, covariance.transpose(-1, -2),
                          rtol=tolerance, atol=tolerance):
        raise ValueError("covariance must be symmetric")
    return covariance


class ResidualSecondMomentHead(nn.Module):
    """Predict one matrix over the entire flattened [horizon, action_dim].

    No current expert action or residual is an input to this module. The
    decoder features must already be conditioned on observation and (t,r).
    All feature gradients stop here, even if the caller forgets to detach.
    """

    def __init__(self, feature_dim, horizon, action_dim, mode="diag",
                 hidden_dim=128, min_variance=1e-4, initial_variance=1.0):
        super().__init__()
        self.feature_dim = _positive_int(feature_dim, "feature_dim")
        self.horizon = _positive_int(horizon, "horizon")
        self.action_dim = _positive_int(action_dim, "action_dim")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if mode not in {"diag", "full"}:
            raise ValueError("mode must be 'diag' or 'full'")
        self.mode = mode
        self.min_variance = _positive_finite(min_variance, "min_variance")
        self.initial_variance = _positive_finite(initial_variance, "initial_variance")
        if self.initial_variance <= self.min_variance:
            raise ValueError("initial_variance must exceed min_variance")
        self.D = self.output_dim = self.horizon * self.action_dim
        outputs = self.D if mode == "diag" else self.D * (self.D + 1) // 2
        self.network = nn.Sequential(
            nn.Linear(self.horizon * self.feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, outputs),
        )
        indices = torch.tril_indices(self.D, self.D)
        self.register_buffer("tril_rows", indices[0], persistent=False)
        self.register_buffer("tril_cols", indices[1], persistent=False)
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        remaining_variance = self.initial_variance - self.min_variance
        with torch.no_grad():
            if mode == "diag":
                # Stable inverse softplus, also for large initial variance.
                raw = remaining_variance + math.log(-math.expm1(-remaining_variance))
                final.bias.fill_(raw)
            else:
                diagonal = self.tril_rows == self.tril_cols
                final.bias[diagonal] = math.sqrt(remaining_variance)

    def forward(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[0] < 1
                or features.shape[1:] != (self.horizon, self.feature_dim)):
            raise ValueError("features must have shape [B,horizon,feature_dim]")
        _check_finite(features, "statistics head features")
        raw = self.network(features.detach().flatten(1))
        raw = _matrix_precision(raw)
        _check_finite(raw, "statistics head output")
        # Matrix products must not silently run in half precision under AMP.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            if self.mode == "diag":
                covariance = torch.diag_embed(self.min_variance + F.softplus(raw))
            else:
                factor = raw.new_zeros((raw.shape[0], self.D, self.D))
                factor[:, self.tril_rows, self.tril_cols] = raw
                identity = torch.eye(self.D, device=raw.device, dtype=raw.dtype)
                covariance = factor @ factor.transpose(-1, -2) + self.min_variance * identity
        _check_finite(covariance, "statistics head covariance")
        return covariance


def metric_time_mask(t, r, endpoint_margin=1e-3):
    """Use H weights only for ``margin < r < t < 1-margin``.

    The diagonal r=t and endpoint neighborhoods retain identity weights in
    the velocity loss. They remain valid samples for the original MeanFlow
    objective and for training the statistics head. The margin is a declared
    engineering setting, not an error guarantee.
    """
    margin = float(endpoint_margin)
    if not math.isfinite(margin) or not 0 <= margin < 0.5:
        raise ValueError("endpoint_margin must lie in [0,0.5)")
    if (not isinstance(t, torch.Tensor) or not isinstance(r, torch.Tensor)
            or t.ndim != 1 or t.shape != r.shape or t.numel() < 1
            or t.device != r.device):
        raise ValueError("t and r must be matching nonempty tensors of shape [B]")
    _check_finite(t, "t")
    _check_finite(r, "r")
    if not bool(((r >= 0) & (r <= t) & (t <= 1)).all()):
        raise ValueError("flow intervals must satisfy 0 <= r <= t <= 1")
    return (r > margin) & (t < 1.0 - margin) & (r < t)


@torch.no_grad()
def bounded_metric(covariance, ridge=1e-3, rho=0.25, active_mask=None):
    """Implement H7/H8 exactly, without explicit matrix inversion.

    W = I + rho * (R - tr(R)/D I),
    R = (ridge + lambda_min(Sigma)) * (Sigma + ridge I)^(-1).

    Non-finite inputs, nonsymmetric matrices, materially negative eigenvalues
    and eigensolver failures are explicit errors. No sample is silently
    dropped or reweighted after inspecting its residual.
    """
    ridge = _positive_finite(ridge, "ridge")
    rho = float(rho)
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must lie in [0,1)")
    covariance = _check_covariance(covariance.detach())
    batch, dimension, _ = covariance.shape
    identity = torch.eye(dimension, dtype=covariance.dtype, device=covariance.device)
    identity = identity.expand(batch, -1, -1)
    if active_mask is not None:
        if (not isinstance(active_mask, torch.Tensor)
                or active_mask.shape != (batch,) or active_mask.dtype != torch.bool
                or active_mask.device != covariance.device):
            raise ValueError("active_mask must be a bool tensor of shape [B] on the same device")
    if rho == 0:
        return identity.clone()
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        _check_finite(eigenvalues, "covariance eigenvalues")
        tolerance = (32 * torch.finfo(covariance.dtype).eps * dimension
                     * eigenvalues.abs().amax(dim=-1).clamp_min(1))
        if bool((eigenvalues[:, 0] < -tolerance).any()):
            raise ValueError("covariance has materially negative eigenvalues")
        eigenvalues = eigenvalues.clamp_min(0)
        ratios = (ridge + eigenvalues[:, :1]) / (ridge + eigenvalues)
        corrections = rho * (ratios - ratios.mean(dim=-1, keepdim=True))
        # Reconstruct the correction, then add I. This gives exact I when all
        # spectral corrections vanish, without Q @ Q.T roundoff.
        weight = identity + (eigenvectors * corrections.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
        weight = 0.5 * (weight + weight.transpose(-1, -2))
        if active_mask is not None:
            weight = torch.where(active_mask[:, None, None], weight, identity)
    _check_finite(weight, "bounded metric")
    return weight


def _flatten_error(error, matrix):
    if (not isinstance(error, torch.Tensor) or error.ndim < 2
            or error.shape[0] != matrix.shape[0] or error.device != matrix.device):
        raise ValueError("error must have matching batch and device with the matrix")
    error = error.flatten(1).to(dtype=matrix.dtype)
    if error.shape[1] != matrix.shape[1]:
        raise ValueError("flattened error dimension does not match the matrix")
    _check_finite(error, "MeanFlow residual")
    return error


def weighted_meanflow_loss(error, weight):
    """H11a; update velocity parameters only, with exact W=I MSE reduction."""
    weight = _check_covariance(weight.detach())
    error = _flatten_error(error, weight)
    dimension = error.shape[1]
    with torch.autocast(device_type=error.device.type, enabled=False):
        identity = torch.eye(dimension, dtype=weight.dtype, device=weight.device)
        correction = torch.bmm(weight - identity, error.unsqueeze(-1)).squeeze(-1)
        loss = error.square().mean() + (error * correction).sum(dim=-1).mean() / dimension
    _check_finite(loss, "weighted MeanFlow loss")
    return loss


def gaussian_second_moment_nll(error, covariance):
    """H11b, omitting only the Gaussian constant and its common factor 1/2.

    Cholesky solve and logdet use float32 or float64. No residual gradients
    enter the mean/velocity network. The covariance graph is retained.
    """
    covariance = _check_covariance(covariance)
    error = _flatten_error(error.detach(), covariance)
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        cholesky = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(error.unsqueeze(-1), cholesky).squeeze(-1)
        quadratic = (error * solved).sum(dim=-1)
        logdet = 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        loss = ((logdet + quadratic) / error.shape[1]).mean()
    _check_finite(loss, "statistics head NLL")
    return loss

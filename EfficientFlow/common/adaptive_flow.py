"""Euler flow solvers with explicit, per-candidate computational accounting.

Times run from Gaussian noise (0) to action (1). ``log_variance`` denotes
the log *mean squared velocity residual per normalized action coordinate*.
Consequently eta is in normalized RMS action units, independent of horizon.
This is an empirical AdaFlow controller, not a certified error estimator.
"""
from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass
class FlowSample:
    samples: torch.Tensor
    nfe: torch.Tensor
    network_calls: int
    min_step_clamps: int = 0
    variance_fallbacks: int = 0


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_noise(noise):
    if noise.ndim < 2 or noise.shape[0] < 1 or not noise.is_floating_point():
        raise ValueError("initial_noise must be a nonempty floating point batch")
    if not torch.isfinite(noise).all():
        raise ValueError("initial_noise contains non-finite values")


def _check_velocity(velocity, sample):
    if velocity.shape != sample.shape:
        raise ValueError("velocity must have the same shape as the active samples")
    if not torch.isfinite(velocity).all():
        raise FloatingPointError("Flow velocity contains NaN or Inf")


@torch.no_grad()
def fixed_euler(field: Callable, initial_noise: torch.Tensor, *, steps=1):
    """field(z, times, original_batch_indices) -> velocity, with no extra head."""
    _positive_int(steps, "steps")
    _check_noise(initial_noise)
    z = initial_noise.clone()
    indices = torch.arange(z.shape[0], device=z.device)
    for i in range(steps):
        times = torch.full((z.shape[0],), i / steps, device=z.device, dtype=z.dtype)
        velocity = field(z, times, indices)
        _check_velocity(velocity, z)
        z = z + velocity / steps
    if not torch.isfinite(z).all():
        raise FloatingPointError("Flow integration produced non-finite samples")
    return FlowSample(z, torch.full_like(indices, steps), steps)


@torch.no_grad()
def adaptive_euler(field: Callable, initial_noise: torch.Tensor, *, eta=.1,
                   max_steps=8, variance_scale=1., log_variance_bounds=(-12., 8.)):
    """field(z, times, indices) -> (velocity, log_variance[active_batch]).

    Each candidate has its own time and NFE. Finished candidates are removed
    before the next backbone call. The lower step bound 1/max_steps limits
    computation; clipping to that bound can violate a requested tolerance.
    A bad variance takes the conservative minimum step and is counted.
    A bad velocity fails loudly rather than silently changing the policy.
    """
    _positive_int(max_steps, "max_steps")
    for value, name in ((eta, "eta"), (variance_scale, "variance_scale")):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    low, high = log_variance_bounds
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("log_variance_bounds must be finite and increasing")
    _check_noise(initial_noise)
    z = initial_noise.clone()
    # Keep the clock in float32 or better even under mixed precision.
    clock_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    clock_tolerance = 4 * torch.finfo(clock_dtype).eps
    times = torch.zeros(z.shape[0], device=z.device, dtype=clock_dtype)
    nfe = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
    calls, min_clamps, fallbacks = 0, 0, 0
    min_step = 1. / max_steps
    for iteration in range(max_steps):
        active = torch.nonzero(times < 1., as_tuple=True)[0]
        if active.numel() == 0:
            break
        current = z[active]
        velocity, log_variance = field(current, times[active].to(z.dtype), active)
        _check_velocity(velocity, current)
        if log_variance.shape != (active.numel(),):
            raise ValueError("log_variance must have shape [active_batch]")
        log_variance = log_variance.to(clock_dtype)
        finite = torch.isfinite(log_variance)
        std = torch.exp(.5 * log_variance.clamp(low, high)) * math.sqrt(variance_scale)
        proposed = eta / std
        proposed = torch.where(finite, proposed, torch.full_like(proposed, min_step))
        remaining = 1. - times[active]
        min_clamps += int(((proposed < min_step) & (remaining > min_step)).sum().item())
        fallbacks += int((~finite).sum().item())
        dt = torch.minimum(proposed.clamp_min(min_step), remaining)
        # Equivalent log-variance/threshold parameterizations can round a step
        # just below the endpoint. Finish within a few clock ULPs, using the
        # full remaining step, instead of spending another network evaluation
        # on a negligible floating-point tail.
        dt = torch.where(remaining - dt <= clock_tolerance, remaining, dt)
        # Avoid a spurious extra update caused by floating point clock drift.
        if iteration == max_steps - 1:
            dt = remaining
        shape = (-1,) + (1,) * (z.ndim - 1)
        z[active] = current + velocity * dt.to(z.dtype).reshape(shape)
        times[active] = torch.where(dt >= remaining, torch.ones_like(dt), times[active] + dt)
        nfe[active] += 1
        calls += 1
    if not torch.isfinite(z).all():
        raise FloatingPointError("Flow integration produced non-finite samples")
    if (times < 1.).any():
        raise RuntimeError("Adaptive integration did not reach time 1")
    return FlowSample(z, nfe, calls, min_clamps, fallbacks)


def residual_variance_loss(velocity, target_velocity, log_variance, *,
                           log_variance_bounds=(-12., 8.)):
    """Gaussian NLL for a detached RMS residual; no gradient reaches velocity.

    .5*(residual/variance + log variance) has optimum E[residual | z,t,o].
    It includes velocity estimation/regularization bias, not just aleatoric
    variance. The scalar reduction uses all horizon/action coordinates.
    """
    if velocity.shape != target_velocity.shape or velocity.ndim < 2:
        raise ValueError("velocity and target_velocity must have matching batch shapes")
    if log_variance.shape != (velocity.shape[0],):
        raise ValueError("log_variance must have shape [batch]")
    residual = (velocity.detach() - target_velocity.detach()).square().flatten(1).mean(1)
    low, high = log_variance_bounds
    log_var = log_variance.clamp(low, high)
    loss = .5 * (residual * torch.exp(-log_var) + log_var).mean()
    return loss, residual, log_var.exp()

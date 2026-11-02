"""EfficientFlow + invariant residual prediction + adaptive Euler integration.

The original policy remains available as the published baseline. This variant
uses the same encoder and velocity weights, a detached scalar residual head,
and a controller that integrates exactly over [0, 1].
"""
import math
from numbers import Real
from typing import Dict

import torch
import torch.nn.functional as F

from EfficientFlow.common.adaptive_flow import (
    adaptive_euler, fixed_euler, residual_variance_loss,
)
from EfficientFlow.common.pytorch_util import dict_apply
from EfficientFlow.policy.EfficientFlow import EfficientFlowPolicy


class EfficientAdaFlowPolicy(EfficientFlowPolicy):
    def __init__(self, *, sampler="adaptive", adaptive_eta=.1,
                 adaptive_max_steps=8, variance_scale=1., fixed_steps=1,
                 fabo_weight=1., flow_loss_weight=2., fabo_delta=.01,
                 variance_loss_weight=.01, variance_warmup_steps=1000,
                 variance_hidden_dim=128, variance_training="joint",
                 log_variance_min=-12., log_variance_max=8.,
                 boundary_time_probability=.1, candidate_reset_interval=10,
                 **kwargs):
        if sampler not in ("adaptive", "fixed"):
            raise ValueError("sampler must be 'adaptive' or 'fixed'")
        if variance_training not in ("joint", "head_only"):
            raise ValueError("variance_training must be 'joint' or 'head_only'")
        if kwargs.get("obs_as_global_cond", True) is not True:
            raise ValueError("EfficientAdaFlow currently requires obs_as_global_cond=True")
        super().__init__(**kwargs)
        if self.action_dim != 10:
            raise ValueError("This equivariant backbone supports single-arm 10D absolute actions")
        for value, name in ((adaptive_max_steps, "adaptive_max_steps"),
                            (fixed_steps, "fixed_steps"),
                            (self.num_of_trajectories, "num_of_trajectories"),
                            (self.horizon, "horizon"),
                            (self.n_obs_steps, "n_obs_steps"),
                            (self.n_action_steps, "n_action_steps"),
                            (candidate_reset_interval, "candidate_reset_interval")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(variance_warmup_steps, bool)
                or not isinstance(variance_warmup_steps, int) or variance_warmup_steps < 0):
            raise ValueError("variance_warmup_steps must be a nonnegative integer")
        for value, name in ((adaptive_eta, "adaptive_eta"), (variance_scale, "variance_scale"),
                            (fabo_delta, "fabo_delta"), (flow_loss_weight, "flow_loss_weight")):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for value, name in ((fabo_weight, "fabo_weight"),
                            (variance_loss_weight, "variance_loss_weight")):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (isinstance(boundary_time_probability, bool) or not isinstance(boundary_time_probability, Real)
                or not 0 <= boundary_time_probability <= 1 or not fabo_delta < 1):
            raise ValueError("Invalid boundary_time_probability or fabo_delta")
        if not (math.isfinite(log_variance_min) and math.isfinite(log_variance_max)
                and log_variance_min < log_variance_max):
            raise ValueError("Invalid log variance bounds")
        if self.n_obs_steps < 1 or self.n_action_steps < 1:
            raise ValueError("Observation and action step counts must be positive")
        if self.n_obs_steps - 1 + self.n_action_steps > self.horizon:
            raise ValueError("horizon is too short for the requested executed action chunk")
        if variance_training == "head_only" and variance_loss_weight <= 0:
            raise ValueError("head_only requires a positive variance_loss_weight")
        self.diff.enable_variance_prediction(variance_hidden_dim)
        self.sampler = sampler
        self.adaptive_eta = adaptive_eta
        self.adaptive_max_steps = adaptive_max_steps
        self.variance_scale = variance_scale
        self.fixed_steps = fixed_steps
        self.fabo_weight = fabo_weight
        self.flow_loss_weight = flow_loss_weight
        self.fabo_delta = fabo_delta
        self.variance_loss_weight = variance_loss_weight
        self.variance_warmup_steps = 0 if variance_training == "head_only" else variance_warmup_steps
        self.variance_training = variance_training
        self.log_variance_bounds = (log_variance_min, log_variance_max)
        self.boundary_time_probability = boundary_time_probability
        self.candidate_reset_interval = candidate_reset_interval
        self.register_buffer("_variance_updates", torch.zeros((), dtype=torch.long))
        self.last_loss_metrics = {}
        self._sampling_records = []
        if variance_training == "head_only":
            self.requires_grad_(False)
            self.diff.variance_head.requires_grad_(True)
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "variance_training", None) == "head_only":
            # Freezing parameters alone does not freeze BatchNorm/crop behavior.
            self.enc.eval()
            self.diff.eval()
            self.diff.variance_head.train(mode)
        return self

    def reset(self):
        # Metrics span all rollout chunks; clear explicitly through the getter.
        self.predict_action_times_tlag = 0

    def _observations(self, obs):
        return dict_apply(obs, lambda x: x[:, :self.n_obs_steps])

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(self._observations(batch["obs"]))
        target = self.normalizer["action"].normalize(batch["action"])
        if target.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("Training action shape does not match horizon/action_dim")
        if self.rot_aug:
            nobs, target = self.rot_randomizer(nobs, target)
        condition = self.enc(nobs).reshape(target.shape[0], -1)
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype) * (1. - self.fabo_delta)
        at_boundary = torch.rand_like(t) < self.boundary_time_probability
        t = torch.where(at_boundary, torch.zeros_like(t), t)
        t_view = t[:, None, None]
        xt = t_view * target + (1. - t_view) * noise
        velocity, log_variance = self.diff(
            xt, t * 99, global_cond=condition, return_variance=True,
            detach_variance_features=True)
        target_velocity = target - noise
        if not torch.isfinite(velocity).all() or not torch.isfinite(log_variance).all():
            raise FloatingPointError("Non-finite velocity or variance during training")
        nll, residual, variance = residual_variance_loss(
            velocity, target_velocity, log_variance,
            log_variance_bounds=self.log_variance_bounds)
        flow = F.mse_loss(velocity, target_velocity)
        fabo = flow.new_zeros(())
        if self.variance_training != "head_only" and self.fabo_weight > 0:
            r = t + self.fabo_delta
            xr = r[:, None, None] * target + (1. - r[:, None, None]) * noise
            vr = self.diff(xr, r * 99, global_cond=condition)
            if not torch.isfinite(vr).all():
                raise FloatingPointError("Non-finite FABO velocity during training")
            flow = F.mse_loss(vr, target_velocity)
            # Rescale the original loss by 1/delta^2; relative FABO/FM weights
            # agree with EfficientFlow away from its clipped endpoint interval.
            acceleration_proxy = (vr - velocity) / (r - t)[:, None, None]
            fabo = ((1. - t_view).square() * acceleration_proxy.square()).mean()
        ready = int(self._variance_updates.item()) >= self.variance_warmup_steps
        if self.variance_training == "head_only":
            loss = self.variance_loss_weight * nll
        else:
            loss = self.flow_loss_weight * flow + self.fabo_weight * fabo
            loss = loss + (self.variance_loss_weight if ready else 0.) * nll
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite EfficientAdaFlow loss")
        self.last_loss_metrics = {
            "flow_mse": float(flow.detach()), "fabo": float(fabo.detach()),
            "variance_nll": float(nll.detach()), "residual_mse": float(residual.mean()),
            "predicted_variance": float(variance.mean().detach()),
            "variance_ratio": float((residual / variance.detach()).mean()),
            "variance_clamp_fraction": float(((log_variance <= self.log_variance_bounds[0])
                | (log_variance >= self.log_variance_bounds[1])).float().mean().detach()),
            "variance_ready": float(ready),
        }
        if self.training and torch.is_grad_enabled():
            self._variance_updates.add_(1)
        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], preaction=None):
        nobs = self.normalizer.normalize(self._observations(obs_dict))
        features = self.enc(nobs)
        batch = features.shape[0]
        candidates_count = self.num_of_trajectories
        condition = features.reshape(batch, -1).repeat(candidates_count, 1)
        noise = torch.randn(candidates_count * batch, self.horizon, self.action_dim,
                            device=features.device, dtype=features.dtype)

        def field(z, times, indices):
            return self.diff(z, times * 99, global_cond=condition[indices],
                             return_variance=self.sampler == "adaptive",
                             detach_variance_features=True)

        if self.sampler == "adaptive":
            sampled = adaptive_euler(field, noise, eta=self.adaptive_eta,
                max_steps=self.adaptive_max_steps, variance_scale=self.variance_scale,
                log_variance_bounds=self.log_variance_bounds)
        else:
            sampled = fixed_euler(field, noise, steps=self.fixed_steps)
        normalized = sampled.samples.reshape(candidates_count, batch, self.horizon, self.action_dim)
        overlap = min(self.n_action_steps, self.horizon - self.n_action_steps)
        reset = self.predict_action_times_tlag % self.candidate_reset_interval == 0
        if preaction is not None and not reset and candidates_count > 1 and overlap > 0:
            previous = self.normalizer["action"].normalize(preaction)
            if previous.ndim != 3 or previous.shape[0] != batch or previous.shape[2] != self.action_dim:
                raise ValueError("preaction must have shape [batch, overlap, action_dim]")
            length = min(overlap, previous.shape[1])
            if length == 0:
                raise ValueError("preaction overlap is empty")
            distances = (normalized[:, :, :length] - previous[None, :, :length]).square().sum((2, 3))
            selected = distances.argmin(0)
        else:
            selected = torch.randint(candidates_count, (batch,), device=features.device)
        rows = torch.arange(batch, device=features.device)
        action_pred = self.normalizer["action"].unnormalize(normalized[selected, rows])
        start = self.n_obs_steps - 1
        self.predict_action_times_tlag += 1
        self._sampling_records.append({
            "nfe": sampled.nfe.cpu(), "network_calls": sampled.network_calls,
            "batch": batch, "min_clamps": sampled.min_step_clamps,
            "fallbacks": sampled.variance_fallbacks,
            "budget": self.adaptive_max_steps if self.sampler == "adaptive" else self.fixed_steps,
        })
        return {
            "action": action_pred[:, start:start + self.n_action_steps],
            "action_pred": action_pred,
            # The next observation window is shifted by n_action_steps.
            "action_to_compare": action_pred[:, self.n_action_steps:self.n_action_steps + overlap],
        }

    def get_sampling_metrics(self, reset=False):
        records = self._sampling_records
        metrics = {}
        if records:
            nfe = torch.cat([r["nfe"] for r in records]).float()
            evaluations = float(nfe.sum())
            metrics = {
                "nfe_mean": float(nfe.mean()), "nfe_p50": float(nfe.quantile(.5)),
                "nfe_p95": float(nfe.quantile(.95)), "nfe_max": float(nfe.max()),
                "one_step_fraction": float((nfe == 1).float().mean()),
                "max_step_fraction": sum(int((r["nfe"] == r["budget"]).sum()) for r in records) / nfe.numel(),
                "model_calls_per_prediction": sum(r["network_calls"] for r in records) / len(records),
                "candidate_evaluations_per_prediction": evaluations / sum(r["batch"] for r in records),
                "min_step_clamp_fraction": sum(r["min_clamps"] for r in records) / evaluations,
                "variance_fallback_fraction": sum(r["fallbacks"] for r in records) / evaluations,
                "sampled_candidates": float(nfe.numel()),
            }
        if reset:
            self._sampling_records = []
        return metrics

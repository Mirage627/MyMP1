"""MP1 with optional MGAD and H residual-second-moment training.

`theory` follows the supplied H2/H5/H7/H11 specification (w=1, no
dispersive term, quadratic regression). `mp1` retains MP1's w=2 and
dispersive loss; H uses the *actual* guided MeanFlow target in that profile.
The shared observation/action interface follows policy/mp1.py.
"""
import math
import torch
from EfficientFlow.policy.mp1 import Meanpolicy, adaptive_l2_loss
from EfficientFlow.common.pytorch_util import dict_apply
from EfficientFlow.model.meanflow_mgad import MGADConditionalUnet1D
from EfficientFlow.common.meanflow_statistics import (
    ResidualSecondMomentHead, metric_time_mask, bounded_metric,
    weighted_meanflow_loss, gaussian_second_moment_nll,
)


class MP1AblationPolicy(Meanpolicy):
    def __init__(self, *, use_mgad=False, statistics_mode="none", profile="theory",
                 baseline_loss=None, metric_rho=0.25, metric_ridge=1e-3,
                 metric_endpoint_margin=1e-3, statistics_hidden_dim=128,
                 statistics_min_variance=1e-4, statistics_initial_variance=1.0,
                 mgad_num_heads=4, mgad_mlp_ratio=2.0, **kwargs):
        if profile not in ("theory", "mp1"):
            raise ValueError("profile must be theory or mp1")
        if statistics_mode not in ("none", "diag", "full"):
            raise ValueError("statistics_mode must be none, diag or full")
        if not math.isfinite(metric_rho) or not 0 <= metric_rho < 1:
            raise ValueError("metric_rho must be in [0,1)")
        if not math.isfinite(metric_ridge) or metric_ridge <= 0:
            raise ValueError("metric_ridge must be positive")
        if not 0 <= metric_endpoint_margin < 0.5:
            raise ValueError("metric_endpoint_margin must be in [0,.5)")
        super().__init__(**kwargs)
        if not self.obs_as_global_cond or self.condition_type != "film":
            raise ValueError("These controlled ablations require global FiLM conditioning")
        self.profile = profile
        self.use_mgad = bool(use_mgad)
        self.statistics_mode = statistics_mode
        self.w = 1.0 if profile == "theory" else 2.0
        self.dispersive_weight = 0.0 if profile == "theory" else 0.5
        self.baseline_loss = baseline_loss or ("mse" if profile == "theory" else "adaptive")
        if self.baseline_loss not in ("mse", "adaptive"):
            raise ValueError("baseline_loss must be mse or adaptive")
        if profile == "theory" and self.baseline_loss != "mse":
            raise ValueError("theory profile requires plain MSE for the no-head controls")
        self.metric_rho, self.metric_ridge = metric_rho, metric_ridge
        self.metric_endpoint_margin = metric_endpoint_margin
        self.statistics_head = None

        # Preserve the baseline random stream and parameter initialization.
        # Auxiliary initialization must not change training noise/time draws.
        with torch.random.fork_rng(devices=[]):
            if use_mgad or statistics_mode != "none":
                replacement = MGADConditionalUnet1D(
                    input_dim=self.action_dim, global_cond_dim=self.obs_feature_dim*self.n_obs_steps,
                    diffusion_step_embed_dim=kwargs.get("diffusion_step_embed_dim", 256),
                    down_dims=kwargs.get("down_dims", (256, 512, 1024)),
                    kernel_size=kwargs.get("kernel_size", 5), n_groups=kwargs.get("n_groups", 8),
                    condition_type="film", use_down_condition=kwargs.get("use_down_condition", True),
                    use_mid_condition=kwargs.get("use_mid_condition", True),
                    use_up_condition=kwargs.get("use_up_condition", True),
                    mgad=use_mgad, horizon=self.horizon,
                    mgad_num_obs_steps=self.n_obs_steps, mgad_num_heads=mgad_num_heads,
                    mgad_mlp_ratio=mgad_mlp_ratio)
                incompatible = replacement.load_state_dict(self.model.state_dict(), strict=False)
                if incompatible.unexpected_keys or any(not k.startswith("mgad") for k in incompatible.missing_keys):
                    raise RuntimeError(f"Unexpected backbone mapping: {incompatible}")
                self.model = replacement
            if statistics_mode != "none":
                self.statistics_head = ResidualSecondMomentHead(
                    feature_dim=kwargs.get("down_dims", (256, 512, 1024))[0],
                    horizon=self.horizon, action_dim=self.action_dim, mode=statistics_mode,
                    hidden_dim=statistics_hidden_dim, min_variance=statistics_min_variance,
                    initial_variance=statistics_initial_variance)

    def parameter_groups(self):
        """Disjoint theta/phi groups, including the observation encoder in theta."""
        phi = [] if self.statistics_head is None else list(self.statistics_head.parameters())
        phi_ids = {id(p) for p in phi}
        theta = [p for p in self.parameters() if p.requires_grad and id(p) not in phi_ids]
        return theta, phi

    def _training_inputs(self, batch):
        actions = self.normalizer["action"].normalize(batch["action"])
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("Training action shape must match horizon and action_dim")
        nobs = self._normalize_observations(batch["obs"])
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        inputs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        condition = self.obs_encoder(inputs).reshape(actions.shape[0], -1)
        return actions, condition

    def loss_terms(self, batch):
        """All terms use pre-update theta/phi. No optimizer is stepped here.

        Teacher is a current-parameter snapshot in this deterministic forward:
        create_graph=False and detached condition freeze the target branch, but
        input derivatives remain enabled. Student is evaluated separately.
        """
        with torch.enable_grad():
            actions, condition = self._training_inputs(batch)
            t, r = self.sample_t_r(actions.shape[0], actions.device)
            t3, h3 = t[:, None, None], (t-r)[:, None, None]
            noise = torch.randn_like(actions)
            x = (1-t3)*actions + t3*noise
            direction = noise-actions
            frozen_condition = condition.detach()
            if self.w != 1:
                with torch.no_grad():
                    diagonal = self.model(sample=x, timestep=t, r=t,
                                          global_cond=frozen_condition, training=False)
                direction = self.w*direction + (1-self.w)*diagonal
            target = direction.detach().clone()
            off_diagonal = t > r
            if off_diagonal.any():
                c = frozen_condition[off_diagonal]
                _, derivative = torch.autograd.functional.jvp(
                    lambda xx, tt, rr: self.model(sample=xx, timestep=tt, r=rr,
                                                  global_cond=c, training=False),
                    (x[off_diagonal], t[off_diagonal], r[off_diagonal]),
                    (direction[off_diagonal], torch.ones_like(t[off_diagonal]),
                     torch.zeros_like(r[off_diagonal])), create_graph=False)
                target[off_diagonal] -= h3[off_diagonal]*derivative
            target = target.detach()
            if self.statistics_head is not None:
                prediction, features, decoded = self.model.forward_with_features(
                    sample=x, timestep=t, r=r, global_cond=condition)
            else:
                prediction, features = self.model(sample=x, timestep=t, r=r, global_cond=condition)
            error = prediction-target
            if not torch.isfinite(error).all():
                raise FloatingPointError("Non-finite MeanFlow target/prediction")
            head_loss = error.new_zeros(())
            diagnostics = {}
            if self.statistics_head is None:
                regression = adaptive_l2_loss(error) if self.baseline_loss == "adaptive" else error.square().mean()
            else:
                covariance = self.statistics_head(decoded.detach())
                active = metric_time_mask(t, r, endpoint_margin=self.metric_endpoint_margin)
                # H11a: theta sees a fixed metric; no velocity-loss gradient to phi.
                # The helpers also enforce these stops for standalone callers.
                weight = bounded_metric(covariance, ridge=self.metric_ridge,
                                        rho=self.metric_rho, active_mask=active).detach()
                regression = weighted_meanflow_loss(error, weight)
                # H11b: phi sees a fixed residual; keep covariance differentiable.
                head_loss = gaussian_second_moment_nll(error.detach(), covariance)
                diagnostics = {"metric_active_fraction": active.float().mean().item(),
                               "second_moment_mean": covariance.detach().diagonal(dim1=-2, dim2=-1).mean().item()}
            dispersive = error.new_zeros(())
            if self.dispersive_weight:
                for feature in features:
                    # MP1's pairwise dispersive loss; define the identical-feature
                    # limit explicitly instead of dividing 0/0 for batch size 1.
                    distances = torch.cdist(feature, feature, p=2).square()
                    maximum = distances.max()
                    if maximum.detach().item() > 0:
                        dispersive = dispersive + torch.exp(-distances/maximum).mean().log()
            velocity_loss = regression + self.dispersive_weight*dispersive
            logs = {"velocity_loss": velocity_loss.detach().item(),
                    "meanflow_loss": regression.detach().item(), "mse_val": error.detach().square().mean().item(),
                    "dis_loss": dispersive.detach().item(), "head_nll": head_loss.detach().item(), **diagnostics}
            return velocity_loss, head_loss, logs

    def compute_loss(self, batch):
        """Pack the two H11 updates into the scalar-loss training interface.

        Because loss_terms stops BOTH cross-gradients, backward on this sum
        gives d(velocity)/d(theta) and d(head)/d(phi), exactly as separate
        backward calls before either optimizer steps. This is a semi-gradient
        update, not full-gradient optimization of a joint Gaussian likelihood.
        Callers must also clip theta/phi separately (see the ablation workspace).
        bc_loss is the aggregate for compatibility; velocity_loss and head_nll
        report the two objectives separately.
        """
        velocity, head, logs = self.loss_terms(batch)
        total = velocity+head
        return total, {"bc_loss": total.detach().item(), **logs}

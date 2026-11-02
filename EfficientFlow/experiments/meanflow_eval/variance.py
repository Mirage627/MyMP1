"""Offline AdaFlow-style residual variance *proxy* for a frozen MeanFlow policy.

The residual includes the pretrained model's bias; it is not AdaFlow's oracle
conditional variance and carries no AdaFlow step-size/error guarantee.  Torch,
NumPy, and Hydra are imported only when a probe is trained or used.
"""

from contextlib import contextmanager
import math
from pathlib import Path
import random


FORMAT_VERSION = 1
LOGVAR_MIN = -16.0
LOGVAR_MAX = 16.0


def _torch():
    import torch
    return torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite(tensor, name):
    torch = _torch()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")


@contextmanager
def _isolated_rng(seed):
    import numpy as np
    torch = _torch()
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def _policy_eval(policy):
    """Keep every module's original mode; never change parameter gradients."""
    states = [(module, module.training) for module in policy.modules()]
    try:
        policy.eval()
        yield
    finally:
        for module, training in states:
            module.training = training


def _head(input_dim, hidden_dim):
    nn = _torch().nn
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, 1))


def _policy_device_dtype(policy):
    torch = _torch()
    parameter = next(policy.parameters(), None)
    if parameter is not None:
        return parameter.device, parameter.dtype
    return torch.device(policy.device), policy.dtype


def _move(value, device, dtype):
    torch = _torch()
    if isinstance(value, dict):
        return {key: _move(item, device, dtype) for key, item in value.items()}
    result = torch.as_tensor(value, device=device)
    if result.is_floating_point():
        result = result.to(dtype=dtype)
    _finite(result, "dataset tensor")
    return result


def _encode(policy, obs_dict):
    """Match the two supported policies' prediction-time conditioning."""
    torch = _torch()
    if not getattr(policy, "obs_as_global_cond", True):
        raise ValueError("variance_proxy currently requires obs_as_global_cond=True")
    if not isinstance(obs_dict, dict) or not obs_dict:
        raise ValueError("obs_dict must be a nonempty tensor dictionary")
    device, dtype = _policy_device_dtype(policy)
    obs_dict = _move(obs_dict, device, dtype)
    nobs = policy.normalizer.normalize(obs_dict)
    first = next(iter(nobs.values()))
    if first.ndim < 3:
        raise ValueError("observations must have shape (batch, observation_steps, ...)")
    batch_size = first.shape[0]
    to = int(policy.n_obs_steps)
    for name, tensor in nobs.items():
        if tensor.ndim < 3 or tensor.shape[0] != batch_size or tensor.shape[1] < to:
            raise ValueError(f"incompatible observation history for {name}")
        _finite(tensor, f"normalized observation {name}")
    nobs = {key: tensor[:, :to] for key, tensor in nobs.items()}
    if hasattr(policy, "enc") and hasattr(policy, "diff"):
        condition = policy.enc(nobs).reshape(batch_size, -1)
        kind = "equimeanflow"
    elif hasattr(policy, "obs_encoder") and hasattr(policy, "model"):
        if not getattr(policy, "use_pc_color", False):
            if "point_cloud" not in nobs:
                raise ValueError("MP1 requires point_cloud observations")
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        flattened = {key: tensor.reshape(-1, *tensor.shape[2:])
                     for key, tensor in nobs.items()}
        encoded = policy.obs_encoder(flattened)
        if "cross_attention" in policy.condition_type:
            condition = encoded.reshape(batch_size, to, -1)
        else:
            condition = encoded.reshape(batch_size, -1)
        kind = "mp1"
    else:
        raise ValueError("variance_proxy supports MP1 and EquiMeanFlow policies only")
    _finite(condition, "pretrained global conditioning")
    return condition, kind


def _features(condition, z, t):
    torch = _torch()
    if z.ndim != 3 or condition.shape[0] != z.shape[0] or t.shape != (z.shape[0],):
        raise ValueError("incompatible conditioning, latent, or time dimensions")
    result = torch.cat((condition.flatten(1), z.flatten(1), t[:, None]), dim=1).float()
    _finite(result, "variance_proxy features")
    return result


def fit_probe_from_arrays(feature, target, *, device="cpu", steps=200,
                          batch_size=64, lr=1e-3, seed=0, hidden_dim=128):
    """Fit from TRAINING features and per-dimension residual MSE targets.

    Returns a serializable probe payload.  No validation arrays are accepted or
    used to fit input normalization.  The Gaussian NLL is
    0.5 * (residual_mse / predicted_variance + log(predicted_variance)).
    """
    torch = _torch()
    for value, name in ((steps, "steps"), (batch_size, "batch_size"),
                        (hidden_dim, "hidden_dim")):
        _positive_int(value, name)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    x = torch.as_tensor(feature).detach().to(device=device, dtype=torch.float32)
    y = torch.as_tensor(target).detach().to(device=device, dtype=torch.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or y.shape != (x.shape[0],):
        raise ValueError("feature must be nonempty (N,D) and target must be (N,)")
    _finite(x, "training features")
    _finite(y, "residual MSE targets")
    if bool((y < 0).any()):
        raise ValueError("residual MSE targets must be nonnegative")
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-4)
    _finite(mean, "training feature mean")
    _finite(scale, "training feature scale")
    x = (x - mean) / scale
    _finite(x, "standardized training features")
    with _isolated_rng(seed):
        model = _head(x.shape[1], hidden_dim).to(device)
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.fill_(float(y.mean().clamp_min(1e-7).log().clamp(LOGVAR_MIN, LOGVAR_MAX)))
        # A tiny eager Adam update keeps this isolated probe independent of
        # torch.optim's optional Dynamo / ONNX / Transformers import chain.
        parameters = list(model.parameters())
        moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in parameters]
        losses = []
        for step in range(1, steps + 1):
            indexes = torch.randint(x.shape[0], (min(batch_size, x.shape[0]),), device=device)
            logvar = model(x[indexes]).squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)
            loss = .5 * (logvar + y[indexes] * torch.exp(-logvar)).mean()
            _finite(loss, "Gaussian NLL")
            for parameter in parameters:
                parameter.grad = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
            with torch.no_grad():
                for parameter, (first, second) in zip(parameters, moments):
                    first.mul_(.9).add_(parameter.grad, alpha=.1)
                    second.mul_(.999).addcmul_(parameter.grad, parameter.grad, value=.001)
                    denominator = second.sqrt() / math.sqrt(1. - .999 ** step) + 1e-8
                    parameter.addcdiv_(first, denominator, value=-lr / (1. - .9 ** step))
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logvar = model(x).squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)
            train_nll = float((.5 * (logvar + y * torch.exp(-logvar))).mean())
    return {
        "format_version": FORMAT_VERSION,
        "kind": "variance_proxy",
        "head_config": {"input_dim": int(x.shape[1]), "hidden_dim": hidden_dim},
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "feature_mean": mean.detach().cpu(),
        "feature_scale": scale.detach().cpu(),
        "metadata": {
            "kind": "variance_proxy", "n_training_samples": int(x.shape[0]),
            "optimization_steps": steps, "train_nll": train_nll,
            "initial_minibatch_nll": losses[0], "final_minibatch_nll": losses[-1],
            "target": "mean squared diagonal conditional-velocity residual per action dimension",
            "score": "sqrt(predicted per-dimension residual variance)",
            "input_order": ["flatten(global_cond)", "flatten(z)", "t"],
            "seed": seed,
            "caveat": "Residual variance includes pretrained model bias; AdaFlow error guarantees do not apply.",
        },
    }


def train_variance_probe(policy, cfg, output_path, *, device, batches=200,
                         batch_size=64, lr=1e-3, seed=0):
    """Train on cfg.task.dataset's existing training split without changing policy.

    The checkpoint normalizer is reused; dataset.get_normalizer() and validation
    datasets are deliberately never called.  Existing output files are refused.
    """
    torch = _torch()
    from hydra.utils import instantiate
    for value, name in ((batches, "batches"), (batch_size, "batch_size")):
        _positive_int(value, name)
    target_path = Path(output_path).expanduser().resolve()
    if target_path.exists():
        raise FileExistsError(f"Refusing to overwrite probe/checkpoint: {target_path}")
    policy_device, dtype = _policy_device_dtype(policy)
    requested_device = torch.device(device)
    if policy_device.type != requested_device.type or (
            requested_device.index is not None and policy_device.index != requested_device.index):
        raise ValueError("policy must already be on the requested training device")
    with _isolated_rng(seed), _policy_eval(policy):
        dataset_cfg = cfg["task"]["dataset"]
        dataset = instantiate(dataset_cfg)
        if len(dataset) == 0:
            raise ValueError("training dataset is empty")
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=0, drop_last=False,
                                            generator=torch.Generator().manual_seed(seed))
        iterator = iter(loader)
        all_features, all_targets = [], []
        endpoint_count, condition_dim, policy_kind = 0, None, None
        noise_shape = (int(policy.horizon), int(policy.action_dim))
        for _ in range(batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if "obs" not in batch or "action" not in batch:
                raise ValueError("dataset batches must contain obs and action")
            with torch.no_grad():
                condition, policy_kind = _encode(policy, batch["obs"])
                action = _move(batch["action"], policy_device, dtype)
                if action.ndim != 3 or tuple(action.shape[1:]) != noise_shape:
                    raise ValueError(f"normalized action shape must be (B,{noise_shape[0]},{noise_shape[1]})")
                action = policy.normalizer["action"].normalize(action)
                _finite(action, "normalized training action")
                n = action.shape[0]
                if condition.shape[0] != n:
                    raise ValueError("observation/action batch sizes differ")
                epsilon = torch.randn_like(action)
                t = torch.rand(n, device=policy_device, dtype=dtype)
                endpoint = torch.rand(n, device=policy_device) < .5
                t[endpoint] = 1.
                endpoint_count += int(endpoint.sum())
                z = (1. - t[:, None, None]) * action + t[:, None, None] * epsilon
                if policy_kind == "equimeanflow":
                    velocity = policy.diff(z=z, t=t, r=t, global_cond=condition)
                else:
                    velocity = policy.model(sample=z, timestep=t, r=t, local_cond=None,
                                            global_cond=condition, training=False)
                if velocity.shape != action.shape:
                    raise ValueError("diagonal velocity output has incompatible action shape")
                _finite(velocity, "diagonal velocity")
                residual_mse = (velocity.float() - (epsilon - action).float()).square().mean((1, 2))
                _finite(residual_mse, "diagonal residual MSE")
                all_features.append(_features(condition, z, t).cpu())
                all_targets.append(residual_mse.cpu())
                condition_dim = int(condition.flatten(1).shape[1])
        payload = fit_probe_from_arrays(torch.cat(all_features), torch.cat(all_targets),
                                        device=device, steps=batches, batch_size=batch_size,
                                        lr=lr, seed=seed)
    metadata = payload["metadata"]
    metadata.update({
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy_kind": policy_kind, "horizon": noise_shape[0], "action_dim": noise_shape[1],
        "condition_dim": condition_dim, "n_obs_steps": int(policy.n_obs_steps),
        "collection_batches": batches, "t_distribution": "0.5 * point_mass(1) + 0.5 * Uniform(0,1)",
        "endpoint_samples": endpoint_count, "dataset_split": "cfg.task.dataset (training only)",
        "dataset_target": str(dataset_cfg.get("_target_", "unknown")),
        "dataset_path": str(dataset_cfg.get("dataset_path", "unknown")),
        "inference_cost": "one additional observation encoding plus a small MLP; zero velocity evaluations",
    })
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("xb") as handle:
        torch.save(payload, handle)
    return dict(metadata)


class VarianceProbe:
    """Load a standalone variance_proxy head; score never evaluates velocity."""

    def __init__(self, path, device="cpu"):
        torch = _torch()
        self.device = torch.device(device)
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format_version") != FORMAT_VERSION or payload.get("kind") != "variance_proxy":
            raise ValueError("unsupported variance_proxy artifact")
        config = payload["head_config"]
        for name in ("input_dim", "hidden_dim"):
            _positive_int(config[name], name)
        self.input_dim = config["input_dim"]
        with _isolated_rng(0):
            self.model = _head(self.input_dim, config["hidden_dim"]).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=self.device)
        self.scale = torch.as_tensor(payload["feature_scale"], dtype=torch.float32, device=self.device)
        if self.mean.shape != (self.input_dim,) or self.scale.shape != (self.input_dim,):
            raise ValueError("invalid variance_proxy feature normalization dimensions")
        _finite(self.mean, "probe feature mean")
        _finite(self.scale, "probe feature scale")
        if bool((self.scale <= 0).any()):
            raise ValueError("probe feature scales must be positive")
        for tensor in self.model.parameters():
            _finite(tensor, "probe weights")
        self.metadata = dict(payload["metadata"])

    def predict_features(self, features):
        """Return per-sample standard deviations, for calibration and testing."""
        torch = _torch()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.input_dim:
            raise ValueError("variance_proxy feature dimensions do not match the saved head")
        _finite(x, "inference features")
        with torch.no_grad():
            logvar = self.model((x - self.mean) / self.scale).squeeze(-1)
            result = torch.exp(.5 * logvar.clamp(LOGVAR_MIN, LOGVAR_MAX))
        _finite(result, "variance_proxy standard deviation")
        return result

    def score(self, policy, obs_dict, initial_noise):
        torch = _torch()
        policy_device, dtype = _policy_device_dtype(policy)
        with _policy_eval(policy), torch.no_grad():
            condition, kind = _encode(policy, obs_dict)
            if condition.shape[0] != 1:
                raise ValueError("score requires a single anchor (batch size 1)")
            expected_shape = (1, int(policy.horizon), int(policy.action_dim))
            noise = torch.as_tensor(initial_noise, device=policy_device, dtype=dtype)
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"initial_noise must have shape {expected_shape}")
            _finite(noise, "initial noise")
            for key, value in (("horizon", policy.horizon), ("action_dim", policy.action_dim),
                               ("n_obs_steps", policy.n_obs_steps)):
                if key in self.metadata and self.metadata[key] != int(value):
                    raise ValueError(f"probe {key} does not match policy")
            if "policy_kind" in self.metadata and self.metadata["policy_kind"] != kind:
                raise ValueError("probe was trained for a different policy architecture")
            feature = _features(condition, noise, torch.ones(1, device=policy_device, dtype=dtype))
            return float(self.predict_features(feature)[0].cpu())

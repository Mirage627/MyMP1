"""Numpy smoke backend and a lazy-loaded robomimic/MeanFlow adapter."""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import random

import numpy as np


class ToyBackend:
    """Synthetic plumbing check, not evidence that adaptive MeanFlow works."""
    horizon, n_obs_steps, execute_steps, action_dim = 5, 2, 2, 2
    latent_shape = (5, 2)
    cfg = None

    def reset(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.position = np.zeros(2, dtype=np.float64)
        self.goal = np.array([1.0, np.random.uniform(-.4, .4)])
        self.difficulty = float(np.random.uniform(.2, 1.5))
        self.elapsed = 0
        self._success = False
        return self._obs()

    def _obs(self):
        return {"robot0_eef_pos": self.position.astype(np.float32),
                "goal": self.goal.astype(np.float32),
                "difficulty": np.array([self.difficulty], np.float32)}

    def state_vector(self):
        return np.r_[self.position, self.goal, self.difficulty, self.elapsed]

    def step(self, action):
        self.position += np.clip(action, -.35, .35)
        self.elapsed += 1
        self._success = bool(np.linalg.norm(self.position-self.goal) < .07)
        return self._obs(), float(self._success), self._success, {"success": self._success}

    def success(self, info):
        return bool(self._success)

    def predict(self, obs, noise, grid):
        target = np.broadcast_to(.4 * (self.goal-obs["robot0_eef_pos"][-1]), noise.shape)
        z = noise.astype(np.float64).copy()
        bias = self.difficulty * (.45 + .15 * np.tanh(noise))
        for t, r in zip(grid[:-1], grid[1:]):
            u = (z-target) / t + (t-r) * bias
            z -= (t-r) * u
        return z[1:1+self.execute_steps].copy(), z, len(grid)-1

    def to_env_action(self, action):
        return action

    def synchronize(self):
        pass

    def close(self):
        pass

    def describe(self):
        return {"backend": "toy", "synthetic": True,
                "horizon": self.horizon, "n_obs_steps": self.n_obs_steps,
                "execute_steps": self.execute_steps,
                "note": "Only validates experiment plumbing; not a trained policy or research result."}


def _register_resolvers():
    import ast
    import operator
    from omegaconf import OmegaConf
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                 ast.Pow: operator.pow, ast.Mod: operator.mod,
                 ast.USub: operator.neg, ast.UAdd: operator.pos}
    def arithmetic(expression):
        def visit(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                return operators[type(node.op)](visit(node.left), visit(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                return operators[type(node.op)](visit(node.operand))
            raise ValueError(f"Unsupported checkpoint arithmetic expression: {expression}")
        return visit(ast.parse(str(expression), mode="eval").body)
    limits = {"stack": 400, "stack_d0": 400, "stack_d1": 400, "stack_three_d1": 400,
              "square_d2": 400, "threading_d2": 400, "coffee_d2": 400,
              "three_piece_assembly_d2": 500, "hammer_cleanup_d1": 500,
              "mug_cleanup_d1": 500, "kitchen_d1": 800, "nut_assembly_d0": 500,
              "pick_place_d0": 1000, "coffee_preparation_d1": 800,
              "tool_hang": 700, "can": 400, "lift": 400, "square": 400}
    OmegaConf.register_new_resolver("eval", arithmetic, replace=True)
    OmegaConf.register_new_resolver("get_max_steps", lambda task: limits[task], replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center", lambda task:
        -.2 if task.startswith(("kitchen_", "hammer_cleanup_")) else 0., replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", lambda task: 0., replace=True)


def effective_execute_steps(horizon, n_obs_steps, n_action_steps, requested=None):
    """Use the actual length of the native policy's action_pred slice."""
    available = int(horizon)-int(n_obs_steps)+1
    steps = min(int(n_action_steps), available) if requested is None else int(requested)
    if steps <= 0 or steps > available:
        raise ValueError(f"horizon={horizon}, n_obs_steps={n_obs_steps}, execute_steps={steps}; "
                         f"--execute-steps must be between 1 and {available}")
    return steps


class RobomimicBackend:
    def __init__(self, checkpoint, *, device="cuda:0", dataset_path=None,
                 policy_target=None, weights="auto", execute_steps=None,
                 register_modules=(), overrides=()):
        # Import neither a workspace nor the old runner: they have unrelated
        # logging dependencies and the old runner silently re-registers Stack_D1.
        import torch
        import dill
        import hydra
        from omegaconf import OmegaConf, open_dict
        _register_resolvers()
        self.torch, self.device = torch, torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        load_kwargs = dict(map_location="cpu", pickle_module=dill)
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        payload = torch.load(path, **load_kwargs)
        if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
            raise ValueError("Expected workspace .ckpt with cfg and state_dicts (not a bare state_dict)")
        cfg = copy.deepcopy(payload["cfg"])
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        with open_dict(cfg):
            if dataset_path:
                resolved = str(Path(dataset_path).expanduser().resolve())
                cfg.dataset_path = resolved
                if OmegaConf.select(cfg, "task.env_runner") is not None:
                    cfg.task.env_runner.dataset_path = resolved
                if OmegaConf.select(cfg, "task.dataset") is not None:
                    cfg.task.dataset.dataset_path = resolved
            if policy_target:
                cfg.policy._target_ = policy_target
            elif cfg.policy._target_ == "EfficientFlow.policy.equi_meanflow.MeanFlowPolicy":
                cfg.policy._target_ = "EfficientFlow.policy.EquiMeanFlow.EquiMeanFlowPolicy"
        self.cfg, self.checkpoint_path = cfg, str(path)
        self.policy = hydra.utils.instantiate(cfg.policy)
        self.policy_source = inspect.getfile(type(self.policy))
        self.policy_source_sha256 = hashlib.sha256(Path(self.policy_source).read_bytes()).hexdigest()
        if not getattr(self.policy, "supports_meanflow_sampling", False):
            raise ValueError(
                f"{type(self.policy).__name__} does not expose interval MeanFlow sampling. "
                "Install the MP1 experiment bundle to update root policy/mp1.py. "
                "EfficientFlow/CFM checkpoints are not silently treated as MeanFlow."
            )
        self.sampler = self.policy
        state_dicts = payload["state_dicts"]
        use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
        self.weight_key = ("ema_model" if use_ema else "model") if weights == "auto" else weights
        if self.weight_key not in state_dicts:
            raise KeyError(f"Checkpoint lacks weights {self.weight_key}; available: {list(state_dicts)}")
        self.policy.load_state_dict(state_dicts[self.weight_key], strict=True)
        self.policy.to(self.device).eval()
        self.horizon, self.n_obs_steps = int(self.policy.horizon), int(self.policy.n_obs_steps)
        self.action_dim = int(self.policy.action_dim)
        self.checkpoint_n_action_steps = int(self.policy.n_action_steps)
        self.execute_steps = effective_execute_steps(
            self.horizon, self.n_obs_steps, self.checkpoint_n_action_steps, execute_steps)
        self.policy.n_action_steps = self.execute_steps
        self.latent_shape = (self.horizon, self.action_dim)
        if not bool(getattr(self.policy, "obs_as_global_cond", True)):
            raise ValueError("Evaluation backend currently supports global observation conditioning only")
        self.shape_meta = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
        self.abs_action = bool(OmegaConf.select(cfg, "task.env_runner.abs_action", default=False))
        self.dataset_path = str(Path(str(OmegaConf.select(
            cfg, "task.env_runner.dataset_path", default=OmegaConf.select(cfg, "dataset_path")))).expanduser().resolve())
        self.register_modules = tuple(register_modules)
        self.raw = None
        self.rotation_transformer = None
        self._success = False
        self.env_meta = None
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _create_env(self):
        import collections
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        # MimicGen is a registration import; absence is fine for native tasks.
        try:
            importlib.import_module("mimicgen")
        except ModuleNotFoundError as error:
            if error.name != "mimicgen":
                raise
        for name in self.register_modules:
            importlib.import_module(name)
        if not Path(self.dataset_path).is_file():
            raise FileNotFoundError(f"Dataset with environment metadata missing: {self.dataset_path}")
        meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        meta = copy.deepcopy(meta)
        meta["env_kwargs"]["use_object_obs"] = False
        if self.abs_action:
            controller = meta["env_kwargs"].get("controller_configs")
            if not isinstance(controller, dict) or not controller:
                raise ValueError("Absolute-action evaluation requires explicit controller_configs in metadata")
            if "body_parts" in controller:
                for item in controller["body_parts"].values():
                    if isinstance(item, dict):
                        item["control_delta"] = False
            else:
                controller["control_delta"] = False
            from EfficientFlow.model.common.rotation_transformer import RotationTransformer
            self.rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
        modalities = collections.defaultdict(list)
        for key, item in self.shape_meta["obs"].items():
            modalities[item.get("type", "low_dim")].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modalities)
        self.raw = EnvUtils.create_env_from_metadata(
            env_meta=meta, render=False, render_offscreen=True, use_image_obs=True)
        actual_name = type(self.raw.env).__name__
        if actual_name.lower() != str(meta["env_name"]).lower():
            self.raw = None
            raise RuntimeError(f"Environment mismatch: metadata={meta['env_name']}, actual={actual_name}. "
                               "Refusing a silently substituted task.")
        if hasattr(self.raw.env, "hard_reset"):
            self.raw.env.hard_reset = True
        self.env_meta = meta

    def _observation(self, raw):
        result = {}
        for key, meta in self.shape_meta["obs"].items():
            if key not in raw:
                raise KeyError(f"Environment did not produce {key}. Install the same robomimic "
                               "observation/point-cloud pipeline used during checkpoint training.")
            value = np.asarray(raw[key])
            expected = tuple(meta["shape"])
            if meta.get("type") == "rgb":
                if value.ndim == 3 and value.shape[-1] == expected[0] and value.shape[0] != expected[0]:
                    value = np.moveaxis(value, -1, 0)
                if np.issubdtype(value.dtype, np.integer):
                    value = value.astype(np.float32) / 255.
                if value.shape != expected:
                    import torch.nn.functional as F
                    value = F.interpolate(self.torch.as_tensor(value.copy()).float()[None],
                        size=expected[-2:], mode="bilinear", align_corners=False)[0].numpy()
            if value.shape != expected:
                raise ValueError(f"Observation {key}: expected {expected}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite observation {key}")
            result[key] = value.astype(np.float32, copy=True)
        return result

    def reset(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        if self.raw is None:
            self._create_env()
        # Re-seed after construction, so every reset follows the same RNG protocol.
        random.seed(seed)
        np.random.seed(seed)
        obs = self.raw.reset()
        self.policy.reset()
        self._success = bool(self.raw.is_success()["task"])
        return self._observation(obs)

    def state_vector(self):
        return np.asarray(self.raw.get_state()["states"]).copy()

    def step(self, action):
        obs, reward, done, info = self.raw.step(action)
        info = dict(info or {})
        self._success = bool(self.raw.is_success()["task"])
        info["success"] = self._success
        return self._observation(obs), reward, bool(done), info

    def success(self, info):
        return self._success

    def tensor_obs(self, obs):
        return {key: self.torch.as_tensor(value[None].copy(), device=self.device)
                for key, value in obs.items()}

    def predict(self, obs, noise, grid):
        torch = self.torch
        noise_tensor = torch.as_tensor(noise[None], device=self.device, dtype=self.policy.dtype)
        with torch.no_grad():
            result = self.sampler.predict_action(self.tensor_obs(obs),
                initial_noise=noise_tensor, time_grid=grid)
            action = result["action"][0].detach().cpu().numpy()
            normalized = self.policy.normalizer["action"].normalize(
                result["action_pred"])[0].detach().cpu().numpy()
        return action, normalized, self.sampler.last_sampling_nfe

    def variance_score(self, probe, obs, noise):
        tensor = self.torch.as_tensor(noise[None], device=self.device, dtype=self.policy.dtype)
        return probe.score(self.policy, self.tensor_obs(obs), tensor)

    def to_env_action(self, action):
        if not self.abs_action:
            return action
        shape = action.shape
        array = action.reshape(-1, 2, 10) if shape[-1] == 20 else action
        rotation_dim = array.shape[-1] - 4
        if rotation_dim != 6:
            raise ValueError("Absolute action conversion expects 10D or dual-arm 20D rotation_6d actions")
        rotation = self.rotation_transformer.inverse(array[..., 3:9])
        transformed = np.concatenate((array[..., :3], rotation, array[..., -1:]), axis=-1)
        return transformed.reshape(*shape[:-1], 14) if shape[-1] == 20 else transformed

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def close(self):
        if self.raw is not None:
            close = getattr(self.raw, "close", None) or getattr(self.raw.env, "close", None)
            if close:
                close()

    def describe(self):
        return {"backend": "robomimic", "synthetic": False,
                "checkpoint": self.checkpoint_path, "weights": self.weight_key,
                "policy_class": f"{type(self.policy).__module__}.{type(self.policy).__name__}",
                "policy_source": self.policy_source, "policy_source_sha256": self.policy_source_sha256,
                "device": str(self.device),
                "sampling_adapter": {"mode": "native_interval_sampling"},
                "dataset_path": self.dataset_path, "horizon": self.horizon,
                "n_obs_steps": self.n_obs_steps, "execute_steps": self.execute_steps,
                "checkpoint_n_action_steps": self.checkpoint_n_action_steps,
                "abs_action": self.abs_action,
                "environment": self.env_meta, "torch_version": self.torch.__version__}

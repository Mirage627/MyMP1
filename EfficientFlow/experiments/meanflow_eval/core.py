"""Serial, paired experiments with explicit noise and action-prefix replay.

Branches change exactly the CURRENT action chunk; all subsequent chunks use the
same one-step continuation. These labels are not adaptive-policy rollouts.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


def stable_seed(*parts):
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


@contextmanager
def isolated_policy_rng(seed):
    """Inference must not perturb the simulator's global random stream."""
    py_state, np_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get("torch")
    cpu_state = cuda_state = None
    if torch is not None:
        cpu_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if cpu_state is not None:
            torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def stack_history(history, count):
    frames = list(history)[-count:]
    frames = [frames[0]] * (count - len(frames)) + frames
    return {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}


def observation_hash(obs):
    digest = hashlib.sha256()
    for key in sorted(obs):
        arr = np.ascontiguousarray(obs[key])
        digest.update(key.encode())
        digest.update(str((arr.shape, arr.dtype.str)).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def routing_features(obs, noise):
    """Cheap PRE-sampling features. No expert actions or future labels."""
    values, names = [], []
    for key in sorted(obs):
        array = np.asarray(obs[key], dtype=np.float64)
        # Deliberately exclude image / point-cloud features from this cheap probe.
        if array.ndim != 2 or array.shape[-1] > 64 or key == "past_action":
            continue
        for idx, val in enumerate(array[-1].ravel()):
            names.append(f"{key}/last/{idx}")
            values.append(float(val))
        for idx, val in enumerate((array[-1] - array[0]).ravel()):
            names.append(f"{key}/delta/{idx}")
            values.append(float(val))
    for key, val in (("noise_mean", noise.mean()), ("noise_std", noise.std()),
                     ("noise_rms", np.sqrt(np.mean(noise ** 2)))):
        names.append(key)
        values.append(float(val))
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite routing features")
    return values, names


@dataclass
class Prediction:
    action: np.ndarray
    normalized_action: np.ndarray
    nfe: int
    latency_ms: float


def predict(backend, obs, noise, grid, seed):
    with isolated_policy_rng(seed):
        backend.synchronize()
        started = time.perf_counter()
        action, normalized, nfe = backend.predict(obs, noise.copy(), grid)
        backend.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
    action = np.asarray(action)
    if action.shape != (backend.execute_steps, backend.action_dim):
        raise ValueError(
            f"Expected fixed action chunk {(backend.execute_steps, backend.action_dim)}, "
            f"got {action.shape}. Check horizon >= n_obs_steps - 1 + execute_steps."
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("Policy produced NaN/Inf actions")
    if nfe != len(grid) - 1:
        raise ValueError(f"Requested {len(grid)-1} NFE but policy reported {nfe}")
    return Prediction(action.copy(), np.asarray(normalized).copy(), int(nfe), elapsed)


def make_noise(backend, episode_id, physical_step, base_seed):
    seed = stable_seed(base_seed, episode_id, physical_step, "latent")
    return np.random.default_rng(seed).standard_normal(backend.latent_shape).astype(np.float32)


def _diagnostics(actions):
    if not actions:
        return None, None
    array = np.asarray(actions, dtype=np.float64)
    delta = float(np.sqrt(np.mean(np.diff(array, axis=0) ** 2))) if len(array) > 1 else None
    jerk = float(np.sqrt(np.mean(np.diff(array, n=3, axis=0) ** 2))) if len(array) > 3 else None
    return delta, jerk


def _append(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def _execute(backend, action, history, remaining):
    raw_actions, rewards, infos = [], [], []
    done, success = False, False
    for command in backend.to_env_action(action)[:remaining]:
        observation, reward, done, info = backend.step(command)
        history.append(observation)
        raw_actions.append(np.asarray(command).copy())
        rewards.append(float(reward))
        infos.append(info)
        success = success or backend.success(info)
        if done or success:
            break
    return raw_actions, rewards, infos, done, success


def _collision(infos):
    # Contact is NOT a collision label. Only use explicitly supplied task signals.
    found = []
    for info in infos:
        for key in ("collision", "unsafe_collision"):
            if key in info:
                found.append(bool(np.any(info[key])))
    return any(found) if found else None


def fixed_rollout(backend, *, env_seed, trial, variant, grid, base_seed,
                  max_steps, anchor_every=5, max_anchors=3, capture_anchors=False,
                  chooser=None):
    episode_id, cluster_id = f"env{env_seed}/noise{trial}", f"env{env_seed}"
    obs = backend.reset(env_seed)
    history = deque([obs], maxlen=backend.n_obs_steps)
    prefix, rewards, infos, actions, predictions, anchors = [], [], [], [], [], []
    physical_step, decision = 0, 0
    success = backend.success({})
    router_times = []
    while physical_step < max_steps and not success:
        stacked = stack_history(history, backend.n_obs_steps)
        noise = make_noise(backend, episode_id, physical_step, stable_seed(base_seed, trial))
        if capture_anchors and decision % anchor_every == 0 and len(anchors) < max_anchors:
            anchors.append({
                "state_id": f"{episode_id}/step{physical_step}",
                "episode_id": episode_id, "cluster_id": cluster_id,
                "env_seed": env_seed, "trial": trial, "step": physical_step,
                "prefix_actions": [a.tolist() for a in prefix],
                "raw_action_dtype": np.asarray(prefix[0]).dtype.str if prefix else None,
                "sim_state": backend.state_vector().tolist(),
                "observation_hash": observation_hash(stacked),
                "prefix_reward": float(sum(rewards)),
            })
        selected_grid = grid
        started = time.perf_counter()
        if chooser is not None:
            features, feature_names = routing_features(stacked, noise)
            with isolated_policy_rng(stable_seed(base_seed, episode_id, physical_step, "router")):
                selected_grid = chooser(stacked, noise, features, feature_names)
        router_ms = (time.perf_counter() - started) * 1000 if chooser is not None else 0.0
        router_times.append(router_ms)
        pred = predict(backend, stacked, noise, selected_grid,
                       stable_seed(base_seed, episode_id, physical_step, "policy"))
        predictions.append(pred)
        raw, rew, info, done, achieved = _execute(
            backend, pred.action, history, max_steps - physical_step)
        prefix.extend(raw)
        actions.extend(raw)
        rewards.extend(rew)
        infos.extend(info)
        physical_step += len(raw)
        decision += 1
        success = success or achieved
        if done:
            break
    if not predictions:
        raise ValueError("Initial state was already successful; choose other evaluation seeds")
    delta, jerk = _diagnostics(actions)
    latencies = [p.latency_ms + r for p, r in zip(predictions, router_times)]
    row = {
        "episode_id": episode_id, "cluster_id": cluster_id,
        "env_seed": env_seed, "trial": trial, "variant": variant,
        "nfe": len(grid) - 1 if chooser is None else None,
        "mean_nfe": float(np.mean([p.nfe for p in predictions])),
        "total_nfe": sum(p.nfe for p in predictions),
        "nfe_counts": {str(k): sum(p.nfe == k for p in predictions)
                       for k in sorted({p.nfe for p in predictions})},
        "success": bool(success), "reward": float(sum(rewards)),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "router_latency_ms_mean": float(np.mean(router_times)),
        "decision_count": decision, "physical_steps": physical_step,
        "action_delta_rms": delta, "action_jerk_rms": jerk,
        "collision": _collision(infos),
        "action_hash": hashlib.sha256(np.asarray(actions).tobytes()).hexdigest(),
    }
    for anchor in anchors:
        suffix = actions[anchor["step"]:]
        anchor["reference_suffix_action_hash"] = hashlib.sha256(np.asarray(suffix).tobytes()).hexdigest()
        anchor["reference_success"] = bool(success)
        anchor["reference_remaining_reward"] = float(sum(rewards[anchor["step"]:]))
        anchor["reference_remaining_steps"] = len(suffix)
    return row, anchors


def restore_anchor(backend, anchor, state_atol=1e-7):
    """A reset and primitive action replay rebuild controller and task state."""
    first = backend.reset(anchor["env_seed"])
    history = deque([first], maxlen=backend.n_obs_steps)
    for command in anchor["prefix_actions"]:
        observation, _, done, info = backend.step(np.asarray(command, dtype=anchor.get("raw_action_dtype")))
        history.append(observation)
        if done or backend.success(info):
            raise RuntimeError(f"Replay terminated before anchor {anchor['state_id']}")
    actual = backend.state_vector()
    expected = np.asarray(anchor["sim_state"])
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=state_atol, rtol=0):
        error = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
        raise RuntimeError(f"Non-deterministic simulator replay at {anchor['state_id']}: max error {error}")
    stacked = stack_history(history, backend.n_obs_steps)
    if observation_hash(stacked) != anchor["observation_hash"]:
        raise RuntimeError(
            f"Observation replay mismatch at {anchor['state_id']}; refusing invalid oracle labels. "
            "Check stochastic sensors, observation preprocessing and simulator reset."
        )
    return history


def branch_rollout(backend, anchor, *, variant, grid, base_seed, max_steps,
                   variance_probe=None, state_atol=1e-7, continuation_grid=(1.0, 0.0)):
    history = restore_anchor(backend, anchor, state_atol)
    physical_step = anchor["step"]
    stacked = stack_history(history, backend.n_obs_steps)
    noise = make_noise(backend, anchor["episode_id"], physical_step,
                       stable_seed(base_seed, anchor["trial"]))
    features, feature_names = routing_features(stacked, noise)
    pred = predict(backend, stacked, noise, grid,
                   stable_seed(base_seed, anchor["episode_id"], physical_step, "policy"))
    signals = {}
    signal_ms = 0.0
    if variance_probe is not None:
        with isolated_policy_rng(stable_seed(base_seed, anchor["state_id"], "variance")):
            backend.synchronize()
            started = time.perf_counter()
            signals["variance_proxy"] = float(backend.variance_score(variance_probe, stacked, noise))
            backend.synchronize()
            signal_ms = (time.perf_counter() - started) * 1000
    raw, rewards, infos, done, success = _execute(
        backend, pred.action, history, max_steps - physical_step)
    physical_step += len(raw)
    first_chunk_steps = len(raw)
    future_nfe = 0
    continuation_hash = hashlib.sha256()
    while physical_step < max_steps and not done and not success:
        obs = stack_history(history, backend.n_obs_steps)
        future_noise = make_noise(backend, anchor["episode_id"], physical_step,
                                  stable_seed(base_seed, anchor["trial"]))
        follow = predict(backend, obs, future_noise, continuation_grid,
                         stable_seed(base_seed, anchor["episode_id"], physical_step, "policy"))
        future_nfe += follow.nfe
        following, rew, info, done, success = _execute(
            backend, follow.action, history, max_steps - physical_step)
        continuation_hash.update(np.asarray(following).tobytes())
        raw.extend(following)
        rewards.extend(rew)
        infos.extend(info)
        physical_step += len(following)
    row = {
        "state_id": anchor["state_id"], "episode_id": anchor["episode_id"],
        "cluster_id": anchor["cluster_id"], "variant": variant,
        "nfe": pred.nfe, "success": bool(success), "reward": float(sum(rewards)),
        "latency_ms": pred.latency_ms, "features": features, "feature_names": feature_names,
        "signals": signals, "variance_signal_latency_ms": signal_ms,
        "continuation_nfe": future_nfe, "remaining_physical_steps": max_steps-anchor["step"],
        "continuation_grid": list(continuation_grid),
        "actual_branch_steps": physical_step-anchor["step"],
        "first_chunk_steps": first_chunk_steps,
        "collision": _collision(infos),
        "first_action_hash": hashlib.sha256(pred.action.tobytes()).hexdigest(),
        "continuation_action_hash": continuation_hash.hexdigest(),
        "raw_action_hash": hashlib.sha256(np.asarray(raw).tobytes()).hexdigest(),
    }
    return row, pred


def collect_branches(backend, anchors, variants, output_path, *, base_seed,
                     max_steps, verify_repeats=1, variance_probe=None, state_atol=1e-7,
                     continuation_grid=(1.0, 0.0)):
    if verify_repeats < 1:
        raise ValueError("At least one repeated branch check is required")
    for anchor_index, anchor in enumerate(anchors):
        rows, predictions = {}, {}
        # Alternating order also exposes unintentional cross-candidate state.
        ordered = list(variants.items())
        if anchor_index % 2:
            ordered.reverse()
        for name, grid in ordered:
            row, pred = branch_rollout(backend, anchor, variant=name, grid=grid,
                base_seed=base_seed, max_steps=max_steps, variance_probe=variance_probe,
                state_atol=state_atol, continuation_grid=continuation_grid)
            rows[name], predictions[name] = row, pred
        # A repeated but systematically wrong restore is insufficient. Match the
        # original baseline suffix, including every executed primitive command.
        for name, grid in variants.items():
            if tuple(grid) == (1.0, 0.0) and "reference_suffix_action_hash" in anchor:
                checked = rows[name]
                if tuple(continuation_grid) != (1.0, 0.0):
                    checked, _ = branch_rollout(backend, anchor, variant=name, grid=grid,
                        base_seed=base_seed, max_steps=max_steps, state_atol=state_atol)
                for key, expected in (("raw_action_hash", anchor["reference_suffix_action_hash"]),
                        ("success", anchor["reference_success"]),
                        ("actual_branch_steps", anchor["reference_remaining_steps"])):
                    if checked[key] != expected:
                        raise RuntimeError(f"Baseline continuation mismatch: {anchor['state_id']} {key}")
                if not np.isclose(checked["reward"], anchor["reference_remaining_reward"], atol=1e-9, rtol=0):
                    raise RuntimeError(f"Baseline continuation reward mismatch: {anchor['state_id']}")
        for name in list(variants)[:verify_repeats]:
            repeat, _ = branch_rollout(backend, anchor, variant=name, grid=variants[name],
                base_seed=base_seed, max_steps=max_steps, state_atol=state_atol,
                continuation_grid=continuation_grid)
            for key in ("success", "reward", "first_action_hash", "continuation_action_hash",
                        "actual_branch_steps"):
                if repeat[key] != rows[name][key]:
                    raise RuntimeError(f"Repeated branch mismatch: {anchor['state_id']} {name} {key}")
        # Compute signals from already-collected candidates; these are OFFLINE
        # diagnostics. Online evaluation must pay 1+K calls to obtain them.
        for name, row in rows.items():
            family = name.rsplit("_", 1)[0]
            low = predictions.get(f"{family}_1")
            high = predictions.get(f"{family}_2")
            if low is not None and high is not None:
                row["signals"]["composition_defect"] = float(np.sqrt(np.mean(
                    (low.normalized_action-high.normalized_action)**2)))
                row["signals"]["action_disagreement"] = float(np.sqrt(np.mean(
                    (low.action-high.action)**2)))
                row["composition_signal_nfe"] = 3
                row["composition_signal_latency_ms"] = low.latency_ms + high.latency_ms
            row["replay_verified"] = True
            row["branch_repeat_verified"] = name in list(variants)[:verify_repeats]
            _append(output_path, row)
        print(f"branches {anchor_index+1}/{len(anchors)}: {anchor['state_id']}", flush=True)


def run_fixed(backend, variants, output_dir, *, env_seeds, trials, base_seed,
              max_steps, anchor_every, max_anchors, capture_anchors=True, chooser=None):
    output_dir = Path(output_dir)
    anchor_path = output_dir / "anchors.jsonl"
    fixed_path = output_dir / ("router_rollouts.jsonl" if chooser else "fixed_rollouts.jsonl")
    initial_states = {}
    for seed in env_seeds:
        for trial in trials:
            for name, grid in variants.items():
                # Check reset reproducibility independently of episode outcome.
                backend.reset(seed)
                state = backend.state_vector()
                key = (seed, trial)
                if key in initial_states and not np.allclose(state, initial_states[key], atol=1e-7, rtol=0):
                    raise RuntimeError(f"Reset is not reproducible for seed {seed}")
                initial_states[key] = state.copy()
                row, anchors = fixed_rollout(backend, env_seed=seed, trial=trial,
                    variant=name, grid=grid, base_seed=base_seed, max_steps=max_steps,
                    anchor_every=anchor_every, max_anchors=max_anchors,
                    capture_anchors=capture_anchors and name == "uniform_1", chooser=chooser)
                _append(fixed_path, row)
                for anchor in anchors:
                    _append(anchor_path, anchor)
                print(f"fixed {row['episode_id']} {name}: success={row['success']} "
                      f"latency={row['latency_ms_mean']:.2f}ms nfe={row['mean_nfe']:.2f}", flush=True)


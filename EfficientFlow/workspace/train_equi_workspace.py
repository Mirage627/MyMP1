"""Training and evaluation for equivariant image policies."""
import collections
import copy
import math
import os
from pathlib import Path
import random

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import tqdm
import wandb

from EfficientFlow.workspace.base_workspace import BaseWorkspace
from EfficientFlow.dataset.base_dataset import BaseImageDataset
from EfficientFlow.common.checkpoint_util import TopKCheckpointManager
from EfficientFlow.common.equivariant_checkpoint import (
    equivariant_cache_keys, load_equivariant_state_dict, refresh_equivariant_caches,
)
from EfficientFlow.common.json_logger import JsonLogger
from EfficientFlow.common.pytorch_util import dict_apply, optimizer_to
from EfficientFlow.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _epoch_count(value, name, allow_zero=False):
    """Hydra's dataset-scaled schedules can evaluate to noninteger floats."""
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return int(math.ceil(value))


def _loss_and_metrics(policy, batch):
    result = policy.compute_loss(batch)
    if isinstance(result, tuple):
        loss, metrics = result
    else:
        loss = result
        metrics = getattr(policy, "last_loss_metrics", {})
    return loss, {str(key): float(value) for key, value in metrics.items()}


def _instantiate_image_dataset(dataset_cfg):
    """Validate the dataset family before an expensive image/cache load."""
    target = dataset_cfg.get("_target_")
    if not isinstance(target, str) or not target:
        raise ValueError("Image training requires task.dataset._target_")
    dataset_class = hydra.utils.get_class(target)
    if not isinstance(dataset_class, type) or not issubclass(dataset_class, BaseImageDataset):
        raise TypeError(
            f"TrainEquiWorkspace requires BaseImageDataset, but {target} is incompatible. "
            "Check the workspace and dataset configuration before loading demonstrations.")
    dataset = hydra.utils.instantiate(dataset_cfg)
    if not isinstance(dataset, BaseImageDataset):
        raise TypeError(f"Expected BaseImageDataset, got {type(dataset)}")
    return dataset


def _initialize_normalizer(model, ema_model, dataset, *, loaded_weights, refit=False):
    """Keep a restored velocity field in its original normalized coordinates."""
    if refit and getattr(model, "variance_training", None) == "head_only":
        raise ValueError("head_only training cannot refit a frozen policy's normalizer")
    existing = getattr(model, "normalizer", None)
    parameters = getattr(existing, "params_dict", None)
    preserve = loaded_weights and parameters is not None and len(parameters) > 0 and not refit
    if preserve:
        if ema_model is not None:
            ema_model.set_normalizer(existing)
        print("Preserved checkpoint action/observation normalizer")
    else:
        normalizer = dataset.get_normalizer()
        model.set_normalizer(normalizer)
        if ema_model is not None:
            ema_model.set_normalizer(normalizer)
        if loaded_weights:
            reason = "explicit refit_normalizer=true" if refit else "checkpoint normalizer was empty"
            print(f"Initialized dataset normalizer because {reason}")
    return preserve


class TrainEquiWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch", "optimizer_step"]

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.checkpoint_tag = "latest"
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("The policy has no trainable parameters")
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0

    def _load_initial_checkpoint(self, path, use_ema=True):
        """Warm start weights without inheriting optimizer, epoch, or output dir."""
        path = Path(hydra.utils.to_absolute_path(str(path)))
        payload = torch.load(str(path), pickle_module=dill, map_location="cpu")
        states = payload["state_dicts"]
        source_key = "ema_model" if use_ema and "ema_model" in states else "model"
        source = states[source_key]
        for model in (self.model, self.ema_model):
            if model is None:
                continue
            load_equivariant_state_dict(
                model, source, allowed_missing=("diff.variance_head.", "_variance_updates"))
        print(f"Initialized policy weights from {path} ({source_key}); optimizer starts fresh")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        excluded = set(exclude_keys or ())
        for key in ("model", "ema_model"):
            if key not in excluded and key in payload["state_dicts"]:
                model = getattr(self, key, None)
                if model is not None:
                    load_equivariant_state_dict(model, payload["state_dicts"][key])
            excluded.add(key)
        super().load_payload(payload, exclude_keys=tuple(excluded),
                             include_keys=include_keys, **kwargs)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        training = cfg.training
        num_epochs = _epoch_count(training.num_epochs, "training.num_epochs")
        frequencies = {
            name: _epoch_count(training.get(name, 0), f"training.{name}", allow_zero=True)
            for name in ("rollout_every", "checkpoint_every", "val_every", "sample_every")
        }
        if training.debug:
            num_epochs = 2
            training.max_train_steps = 3
            training.max_val_steps = 3
            frequencies = {key: (1 if value else 0) for key, value in frequencies.items()}
        accumulate = _epoch_count(training.gradient_accumulate_every, "gradient_accumulate_every")
        if accumulate != float(training.gradient_accumulate_every):
            raise ValueError("gradient_accumulate_every must be an integer")

        resumed = False
        if training.resume:
            latest = self.get_checkpoint_path(self.checkpoint_tag)
            if latest.is_file():
                payload = self.load_checkpoint(path=latest)
                self.epoch += 1
                if "optimizer_step" not in payload.get("pickles", {}):
                    self.optimizer_step = self.global_step // accumulate
                resumed = True
                print(f"Resumed training from {latest}")
        if not resumed and training.get("init_checkpoint"):
            self._load_initial_checkpoint(training.init_checkpoint, training.get("init_use_ema", True))

        dataset = _instantiate_image_dataset(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataloader = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)
        _initialize_normalizer(
            self.model, self.ema_model, dataset,
            loaded_weights=resumed or bool(training.get("init_checkpoint")),
            refit=bool(training.get("refit_normalizer", False)))

        batches_per_epoch = len(train_dataloader)
        if training.max_train_steps is not None:
            batches_per_epoch = min(batches_per_epoch, int(training.max_train_steps))
        if batches_per_epoch < 1:
            raise ValueError("Training loader is empty; reduce batch size or disable drop_last")
        updates_per_epoch = math.ceil(batches_per_epoch / accumulate)
        lr_scheduler = get_scheduler(
            training.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=int(training.lr_warmup_steps),
            num_training_steps=updates_per_epoch * num_epochs,
            last_epoch=self.optimizer_step - 1)

        ema = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.optimizer_step
        env_runner = None
        if frequencies["rollout_every"] and cfg.task.get("env_runner") is not None:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        wandb_run.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)
        device = torch.device(training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device).eval()
        optimizer_to(self.optimizer, device)
        sampling_batch = None
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        def due(name):
            period = frequencies[name]
            return period > 0 and self.epoch % period == 0

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            while self.epoch < num_epochs:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                train_losses = []
                train_metrics = collections.defaultdict(list)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=training.tqdm_interval_sec) as progress:
                    for batch_idx, batch in enumerate(progress):
                        if batch_idx >= batches_per_epoch:
                            break
                        batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                        if sampling_batch is None:
                            sampling_batch = batch
                        raw_loss, metrics = _loss_and_metrics(self.model, batch)
                        if not torch.isfinite(raw_loss):
                            raise FloatingPointError(f"Nonfinite loss at epoch {self.epoch}, batch {batch_idx}")
                        # Scale the final partial accumulation group by its actual size.
                        group_start = (batch_idx // accumulate) * accumulate
                        group_size = min(accumulate, batches_per_epoch - group_start)
                        (raw_loss / group_size).backward()
                        end_group = (batch_idx + 1) % accumulate == 0 or batch_idx + 1 == batches_per_epoch
                        if end_group:
                            max_norm = training.get("max_grad_norm")
                            if max_norm is not None:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    trainable, float(max_norm), error_if_nonfinite=True)
                                train_metrics["grad_norm"].append(float(grad_norm))
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()
                            self.optimizer_step += 1
                            if ema is not None:
                                ema.step(self.model)
                                # EMAModel averages parameters only. Counters and normalization
                                # buffers must follow the training model for reliable resume/eval.
                                with torch.no_grad():
                                    source_buffers = dict(self.model.named_buffers())
                                    derived_caches = equivariant_cache_keys(self.ema_model)
                                    for name, buffer in self.ema_model.named_buffers():
                                        if name in source_buffers and name not in derived_caches:
                                            buffer.copy_(source_buffers[name])
                        loss_value = float(raw_loss.detach())
                        train_losses.append(loss_value)
                        for key, value in metrics.items():
                            train_metrics[key].append(value)
                        progress.set_postfix(loss=loss_value, refresh=False)
                        self.global_step += 1

                step_log = {
                    "train_loss": float(np.mean(train_losses)),
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    "epoch": self.epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                step_log.update({f"train/{key}": float(np.mean(values))
                                 for key, values in train_metrics.items()})
                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()
                # EMA changes coefficients while the model remains in eval mode;
                # escnn's expanded filter/matrix buffers must follow those weights.
                refresh_equivariant_caches(policy)
                if due("rollout_every") and env_runner is not None:
                    step_log.update(env_runner.run(policy))
                if due("val_every"):
                    val_losses = []
                    val_metrics = collections.defaultdict(list)
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_dataloader):
                            if training.max_val_steps is not None and batch_idx >= int(training.max_val_steps):
                                break
                            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                            loss, metrics = _loss_and_metrics(policy, batch)
                            val_losses.append(float(loss))
                            for key, value in metrics.items():
                                val_metrics[key].append(value)
                    if val_losses:
                        step_log["val_loss"] = float(np.mean(val_losses))
                        step_log.update({f"val/{key}": float(np.mean(values))
                                         for key, values in val_metrics.items()})
                if due("sample_every") and sampling_batch is not None:
                    policy.reset()
                    with torch.no_grad():
                        result = policy.predict_action(sampling_batch["obs"])
                        step_log["train_action_mse_error"] = float(torch.nn.functional.mse_loss(
                            result["action_pred"], sampling_batch["action"]))
                if due("checkpoint_every"):
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(use_thread=False)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    if topk_manager.monitor_key in metric_dict:
                        path = topk_manager.get_ckpt_path(metric_dict)
                        if path is not None:
                            self.save_checkpoint(path=path, use_thread=False)
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1
        wandb_run.finish()

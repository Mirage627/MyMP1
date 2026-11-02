"""One training loop for every MP1/MGAD/H ablation; separate theta/phi AdamW."""
import copy
import json
import math
import random
import inspect
from pathlib import Path
import numpy as np
import torch
import hydra
import dill
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from EfficientFlow.workspace.base_workspace import BaseWorkspace
from EfficientFlow.common.pytorch_util import dict_apply, optimizer_to
from EfficientFlow.common.swanlab_logging import SwanLabLogger, scalar_metrics, validate_rollout_videos
from EfficientFlow.model.common.lr_scheduler import get_scheduler
from EfficientFlow.model.diffusion.ema_model import EMAModel


class TrainMP1AblationWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch", "rng_state", "scheduler_states")
    # Schedulers need dataset length to construct; plain state dictionaries can
    # also be restored for evaluation before run() constructs a dataloader.
    exclude_keys = ("lr_scheduler", "head_lr_scheduler")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir)
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        theta, phi = self.model.parameter_groups()
        theta_ids, phi_ids = {id(p) for p in theta}, {id(p) for p in phi}
        expected = {id(p) for p in self.model.parameters() if p.requires_grad}
        if theta_ids & phi_ids or theta_ids | phi_ids != expected:
            raise RuntimeError("Optimizer groups must be disjoint and cover all trainable parameters")
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=theta)
        self.head_optimizer = hydra.utils.instantiate(cfg.head_optimizer, params=phi) if phi else None
        self.global_step, self.epoch = 0, 0
        self.rng_state = None
        self.scheduler_states = {}
        self.lr_scheduler = self.head_lr_scheduler = None
        self._checkpoint_loaded = False

    @staticmethod
    def _resume_signature(cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        keys = ("policy", "optimizer", "head_optimizer", "ema", "task.dataset",
                "dataloader", "training.seed", "training.use_ema",
                "training.gradient_accumulate_every", "training.max_train_steps",
                "training.lr_scheduler", "training.lr_warmup_steps",
                "training.num_epochs", "training.max_grad_norm")
        return {key: OmegaConf.to_container(OmegaConf.create({"value": OmegaConf.select(cfg, key)}),
                                           resolve=True)["value"] for key in keys}

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        current, saved = self._resume_signature(self.cfg), self._resume_signature(payload["cfg"])
        differences = [key for key in current if current[key] != saved[key]]
        if differences:
            raise ValueError("Incompatible ablation checkpoint configuration: " + ", ".join(differences)
                             + ". Resume the same experiment; use explicit model-only loading for warm starts.")
        # A relocated run must keep the caller's output destination.
        try:
            output_dir = self.output_dir
        except ValueError:  # standalone create_from_checkpoint, outside Hydra
            output_dir = None
        super().load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        if output_dir is not None:
            self._output_dir = output_dir
        self._checkpoint_loaded = True

    def save_checkpoint(self, *args, **kwargs):
        for name in ("lr_scheduler", "head_lr_scheduler"):
            scheduler = getattr(self, name)
            if scheduler is not None:
                self.scheduler_states[name] = scheduler.state_dict()
        return super().save_checkpoint(*args, **kwargs)

    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None,
                               map_location="cpu", **kwargs):
        with open(path, "rb") as stream:
            payload = torch.load(stream, pickle_module=dill, map_location=map_location,
                                 weights_only=False)
        instance = cls(payload["cfg"])
        instance.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        return instance

    def _step_batch(self, batches):
        """Accumulate both disjoint gradients before either optimizer steps."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.head_optimizer is not None:
            self.head_optimizer.zero_grad(set_to_none=True)
        totals = {}
        total_samples = sum(batch["action"].shape[0] for batch in batches)
        for batch in batches:
            velocity, head, logs = self.model.loss_terms(batch)
            # Equivalent disjoint backward in one call: both losses have explicit
            # stop-gradients. Clip theta and phi separately to avoid hidden coupling.
            loss = velocity + head
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            fraction = batch["action"].shape[0]/total_samples
            (loss*fraction).backward()
            for name, value in {"train_loss": loss.detach().item(), **logs}.items():
                totals[name] = totals.get(name, 0.) + fraction*float(value)
        theta, phi = self.model.parameter_groups()
        limit = self.cfg.training.max_grad_norm
        for parameters in (theta, phi):
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, float("inf") if limit is None else limit,
                                               error_if_nonfinite=True)
        self.optimizer.step()
        if self.head_optimizer is not None:
            self.head_optimizer.step()
        return totals

    def _capture_rng(self):
        self.rng_state = {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def run(self):
        cfg = self.cfg
        device = torch.device(cfg.training.device)
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        print(f"OUTPUT_DIR={output.resolve()}", flush=True)
        print(f"WORKSPACE_FILE={inspect.getfile(type(self))}", flush=True)
        print(f"POLICY_FILE={inspect.getfile(type(self.model))}", flush=True)
        print(f"NORMALIZER_FILE={inspect.getfile(type(self.model.normalizer))}", flush=True)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        loader = DataLoader(dataset, **cfg.dataloader)
        if not len(loader):
            raise ValueError("Empty training dataloader")
        validation = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()
        if not self._checkpoint_loaded:
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)
        max_batches = min(len(loader), cfg.training.max_train_steps or len(loader))
        accumulation = int(cfg.training.gradient_accumulate_every)
        if accumulation < 1:
            raise ValueError("gradient_accumulate_every must be positive")
        steps = math.ceil(max_batches/accumulation)*cfg.training.num_epochs
        def scheduler(optimizer):
            return get_scheduler(cfg.training.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=cfg.training.lr_warmup_steps,
                                 num_training_steps=steps)
        self.lr_scheduler = scheduler(self.optimizer)
        if self.head_optimizer is not None:
            self.head_lr_scheduler = scheduler(self.head_optimizer)
        if cfg.training.resume and self.get_checkpoint_path().is_file():
            self.load_checkpoint(map_location="cpu", weights_only=False)
        for name, state in self.scheduler_states.items():
            scheduler = getattr(self, name)
            if scheduler is not None:
                scheduler.load_state_dict(state)
                # Loading an optimizer before scheduler construction (e.g.
                # create_from_checkpoint followed by run) resets the current LR.
                for group, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr()):
                    group["lr"] = lr
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        if self.head_optimizer is not None:
            optimizer_to(self.head_optimizer, device)
        ema = None
        if self.ema_model is not None:
            self.ema_model.to(device)
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.global_step
        runner = None
        logger = SwanLabLogger(cfg, output, resume=self._checkpoint_loaded)
        try:
            runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output)) if cfg.training.enable_rollouts else None
            if runner is not None and hasattr(runner, "video_backend"):
                runner.video_backend = "path"
            if self.rng_state is not None:
                random.setstate(self.rng_state["python"])
                np.random.set_state(self.rng_state["numpy"])
                torch.set_rng_state(self.rng_state["torch"])
                if self.rng_state["cuda"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(self.rng_state["cuda"])
            while self.epoch < cfg.training.num_epochs:
                pending, history = [], []
                for index, batch in enumerate(loader):
                    pending.append(dict_apply(batch, lambda x: x.to(device, non_blocking=True)))
                    last = index+1 >= max_batches
                    if len(pending) == accumulation or last:
                        history.append(self._step_batch(pending))
                        pending.clear()
                        self.lr_scheduler.step()
                        if self.head_optimizer is not None:
                            self.head_lr_scheduler.step()
                        if ema is not None:
                            ema.step(self.model)
                        self.global_step += 1
                    if last:
                        break
                report = {k: float(np.mean([row[k] for row in history])) for k in history[0]}
                report.update(epoch=self.epoch, global_step=self.global_step, profile=self.model.profile,
                              use_mgad=self.model.use_mgad, statistics_mode=self.model.statistics_mode)
                policy = self.ema_model if self.ema_model is not None else self.model
                results = {}
                if cfg.training.enable_validation and (self.epoch+1) % cfg.training.val_every == 0:
                    policy.eval()
                    loss_sum, sample_count = 0., 0
                    # MeanFlow labels require input derivatives, even during validation.
                    for index, batch in enumerate(validation):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        _, _, logs = policy.loss_terms(batch)
                        count = batch["action"].shape[0]
                        loss_sum += count*logs["mse_val"]
                        sample_count += count
                        if cfg.training.max_val_steps and index+1 >= cfg.training.max_val_steps:
                            break
                    if sample_count:
                        report["validation_mse"] = loss_sum/sample_count
                if runner is not None and (self.epoch+1) % cfg.training.rollout_every == 0:
                    policy.eval()
                    with torch.no_grad():
                        results = runner.run(policy)
                    report.update(scalar_metrics(results))
                    runner_cfg = cfg.task.env_runner
                    expected_videos = None
                    if all(key in runner_cfg for key in ("n_train", "n_train_vis", "n_test", "n_test_vis")):
                        expected_videos = (min(runner_cfg.n_train, runner_cfg.n_train_vis)
                                           + min(runner_cfg.n_test, runner_cfg.n_test_vis))
                    report["rollout/video_count"] = validate_rollout_videos(results, expected_videos)
                logger.log(report, step=self.global_step, rollout=results)
                with (output/"metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, ensure_ascii=False)+"\n")
                print(json.dumps(report, ensure_ascii=False), flush=True)
                self.epoch += 1
                if cfg.checkpoint.save_ckpt and (self.epoch % cfg.training.checkpoint_every == 0 or self.epoch == cfg.training.num_epochs):
                    self._capture_rng()
                    self.save_checkpoint(use_thread=False)
        finally:
            try:
                if runner is not None:
                    closer = getattr(runner, "close", None)
                    if closer is not None:
                        closer()
            finally:
                logger.finish()

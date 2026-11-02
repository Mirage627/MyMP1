"""Exercise the actual training loop without loading robot simulation packages."""
import ast
import collections
import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    import dill
except ImportError:
    torch = None
    dill = None


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HELPERS = {}
if torch is not None:
    helper_spec = importlib.util.spec_from_file_location(
        "equivariant_checkpoint_under_test", ROOT / "common/equivariant_checkpoint.py")
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    CHECKPOINT_HELPERS = {name: getattr(helper_module, name) for name in (
        "equivariant_cache_keys", "load_equivariant_state_dict", "refresh_equivariant_caches")}


class Options(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value


def load_workspace_definitions(namespace):
    """Keep the real methods while avoiding their optional simulator imports."""
    tree = ast.parse((ROOT / "workspace/train_equi_workspace.py").read_text(encoding="utf-8-sig"))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "train_equi_workspace.py", "exec"), namespace)
    return namespace


class Progress:
    def __init__(self, values, **kwargs):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, **kwargs):
        pass


class NullLogger:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log(self, *args, **kwargs):
        pass


@unittest.skipIf(torch is None or dill is None, "torch and dill are required")
class TrainingIntegrationTests(unittest.TestCase):
    def base_namespace(self):
        class WorkspaceBase:
            @property
            def output_dir(self):
                return self._output_dir

        return {
            "BaseWorkspace": WorkspaceBase, "torch": torch, "dill": dill,
            "Path": Path, "copy": copy, "collections": collections,
            "math": math, "os": os,
            **CHECKPOINT_HELPERS,
        }

    def cached_layer_types(self):
        class CachedLayer(torch.nn.Module):
            cache_name = "matrix"

            def __init__(self, value=2.):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.tensor(value))
                self.register_buffer("basis_expansion", torch.tensor(3.))

            def expand_parameters(self):
                return self.weights * self.basis_expansion

            def forward(self, value):
                matrix = self.expand_parameters() if self.training else getattr(self, self.cache_name)
                return value * matrix

            def train(self, mode=True):
                if mode:
                    if hasattr(self, self.cache_name):
                        delattr(self, self.cache_name)
                elif self.training:
                    self.register_buffer(self.cache_name, self.expand_parameters().detach().clone())
                return super().train(mode)

        class Linear(CachedLayer):
            __module__ = "escnn.nn.modules.linear"

        class _RdConv(CachedLayer):
            __module__ = "escnn.nn.modules.conv.rd_convolution"
            cache_name = "filter"

        return Linear, _RdConv

    def test_refresh_rebuilds_ema_caches_without_toggling_head_only_policy(self):
        linear, convolution = self.cached_layer_types()

        class HeadOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = torch.nn.Sequential(linear(), convolution())
                self.other = torch.nn.Dropout()
                self.mode_calls = 0
                self.train(True)

            def train(self, mode=True):
                self.mode_calls += 1
                super().train(mode)
                self.enc.eval()
                return self

        model = HeadOnly()
        original_mode_calls = model.mode_calls
        for layer in model.enc:
            with torch.no_grad():
                layer.weights.fill_(5.)
            self.assertEqual(float(layer(torch.tensor(1.))), 6.)
        CHECKPOINT_HELPERS["refresh_equivariant_caches"](model)
        self.assertEqual(model.mode_calls, original_mode_calls)
        self.assertTrue(model.training)
        self.assertTrue(model.other.training)
        self.assertFalse(model.enc.training)
        for layer in model.enc:
            self.assertFalse(layer.training)
            self.assertEqual(float(layer(torch.tensor(1.))), 15.)

    def test_cache_state_cross_mode_loads_real_weights_strictly(self):
        linear, convolution = self.cached_layer_types()
        source = torch.nn.Sequential(linear(7.), convolution(11.)).eval()
        target = torch.nn.Sequential(linear(), convolution())
        target[0].eval()  # Mixed modes must survive a load.
        state = source.state_dict()
        state["0.matrix"].fill_(-100.)
        state["1.filter"].fill_(-100.)
        loader = CHECKPOINT_HELPERS["load_equivariant_state_dict"]
        loader(target, state)
        self.assertFalse(target[0].training)
        self.assertTrue(target[1].training)
        self.assertEqual(float(target[0](torch.tensor(1.))), 21.)
        self.assertEqual(float(target[1](torch.tensor(1.))), 33.)
        self.assertFalse(hasattr(target[1], "filter"))
        bad = dict(state)
        del bad["0.basis_expansion"]
        with self.assertRaisesRegex(RuntimeError, "basis_expansion"):
            loader(target, bad)
        bad = dict(state)
        bad["unknown.matrix"] = torch.tensor(1.)
        with self.assertRaisesRegex(RuntimeError, "unknown.matrix"):
            loader(target, bad)
        ordinary = torch.nn.Module()
        ordinary.register_buffer("matrix", torch.tensor(1.))
        with self.assertRaisesRegex(RuntimeError, "matrix"):
            loader(ordinary, {})

    def test_loaded_normalizer_is_preserved_and_refit_is_explicit(self):
        namespace = load_workspace_definitions(self.base_namespace())
        initialize = namespace["_initialize_normalizer"]
        old = SimpleNamespace(params_dict={"action": 1}, tag="checkpoint")
        new = SimpleNamespace(params_dict={"action": 2}, tag="dataset")

        class Policy:
            variance_training = "joint"

            def __init__(self):
                self.normalizer = old

            def set_normalizer(self, normalizer):
                self.normalizer = normalizer

        calls = []
        dataset = SimpleNamespace(get_normalizer=lambda: calls.append(True) or new)
        model, ema = Policy(), Policy()
        self.assertTrue(initialize(model, ema, dataset, loaded_weights=True))
        self.assertIs(model.normalizer, old)
        self.assertIs(ema.normalizer, old)
        self.assertEqual(calls, [])
        self.assertFalse(initialize(model, ema, dataset, loaded_weights=True, refit=True))
        self.assertIs(model.normalizer, new)
        self.assertIs(ema.normalizer, new)
        model.variance_training = "head_only"
        with self.assertRaisesRegex(ValueError, "head_only"):
            initialize(model, ema, dataset, loaded_weights=True, refit=True)
        self.assertEqual(calls, [True])

    @unittest.skipUnless(importlib.util.find_spec("escnn"), "escnn is not installed")
    def test_real_escnn_linear_and_convolution_cache_refresh(self):
        from escnn import gspaces, nn
        from escnn.group import CyclicGroup

        group = gspaces.no_base_space(CyclicGroup(4))
        linear_type = nn.FieldType(group, [group.regular_repr])
        linear = nn.Linear(linear_type, linear_type).eval()
        spatial_group = gspaces.rot2dOnR2(N=4)
        conv_type = nn.FieldType(spatial_group, [spatial_group.trivial_repr])
        convolution = nn.R2Conv(conv_type, conv_type, kernel_size=3).eval()
        for layer, field in ((linear, "matrix"), (convolution, "filter")):
            with torch.no_grad():
                layer.weights.add_(.25)
                expected, _ = layer.expand_parameters()
            CHECKPOINT_HELPERS["refresh_equivariant_caches"](layer)
            torch.testing.assert_close(getattr(layer, field), expected)

    def test_warmstart_accepts_only_new_variance_state(self):
        namespace = self.base_namespace()
        namespace["hydra"] = SimpleNamespace(utils=SimpleNamespace(
            to_absolute_path=lambda path: str(Path(path).resolve())))
        cls = load_workspace_definitions(namespace)["TrainEquiWorkspace"]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diff = torch.nn.Module()
                self.diff.velocity = torch.nn.Linear(2, 2)
                self.diff.variance_head = torch.nn.Linear(2, 1)
                self.register_buffer("_variance_updates", torch.tensor(0))

        workspace = cls.__new__(cls)
        workspace.model, workspace.ema_model = Model(), Model()
        state = {key: value.clone().fill_(3.) for key, value in workspace.model.state_dict().items()
                 if not key.startswith("diff.variance_head.") and key != "_variance_updates"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.ckpt"
            torch.save({"state_dicts": {"model": state}}, str(path), pickle_module=dill)
            workspace._load_initial_checkpoint(path)
            self.assertTrue(torch.all(workspace.model.diff.velocity.weight == 3.))
            self.assertTrue(torch.all(workspace.ema_model.diff.velocity.weight == 3.))
            state.pop("diff.velocity.bias")
            torch.save({"state_dicts": {"model": state}}, str(path), pickle_module=dill)
            with self.assertRaisesRegex(RuntimeError, "diff.velocity.bias"):
                workspace._load_initial_checkpoint(path)

    def test_actual_loop_accumulates_final_partial_group_and_skips_disabled_rollouts(self):
        class Dataset:
            def __len__(self):
                return 5

            def __iter__(self):
                return iter([{"value": torch.tensor(float(value))} for value in range(1, 6)])

            def get_normalizer(self):
                return None

            def get_validation_dataset(self):
                return []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def set_normalizer(self, normalizer):
                pass

            def compute_loss(self, batch):
                return self.weight * batch["value"]

        class Scheduler:
            def step(self):
                pass

            def get_last_lr(self):
                return [.1]

        class PlainSGD:
            # Keep this test about the workspace's gradient scheduling. Optional
            # torch.onnx/transformers imports in torch.optim are unnecessary here.
            def __init__(self, parameters):
                self.parameters = list(parameters)

            def step(self):
                with torch.no_grad():
                    for parameter in self.parameters:
                        parameter.add_(parameter.grad, alpha=-.1)

            def zero_grad(self, set_to_none=True):
                for parameter in self.parameters:
                    parameter.grad = None

        dataset = Dataset()
        dataset_cfg = Options(_target_="test.ImageDataset")
        created = []

        def instantiate(cfg, **kwargs):
            created.append(cfg)
            if cfg == dataset_cfg:
                return dataset
            raise AssertionError(f"Unexpected component created: {cfg}")

        namespace = self.base_namespace()
        namespace.update({
            "hydra": SimpleNamespace(utils=SimpleNamespace(
                instantiate=instantiate, get_class=lambda target: Dataset)),
            "BaseImageDataset": Dataset, "DataLoader": lambda data, **kwargs: data,
            "get_scheduler": lambda *args, **kwargs: Scheduler(),
            "TopKCheckpointManager": lambda **kwargs: SimpleNamespace(monitor_key="score"),
            "dict_apply": lambda batch, function: {key: function(value) for key, value in batch.items()},
            "optimizer_to": lambda *args: None,
            "tqdm": SimpleNamespace(tqdm=Progress), "JsonLogger": NullLogger,
            "np": SimpleNamespace(mean=lambda values: sum(values) / len(values)),
            "OmegaConf": SimpleNamespace(to_container=lambda *args, **kwargs: {}),
            "wandb": SimpleNamespace(init=lambda **kwargs: SimpleNamespace(
                config={}, log=lambda *args, **kwargs: None, finish=lambda: None)),
        })
        cls = load_workspace_definitions(namespace)["TrainEquiWorkspace"]
        workspace = cls.__new__(cls)
        workspace.model = Model()
        workspace.ema_model = None
        workspace.optimizer = PlainSGD(workspace.model.parameters())
        workspace.global_step = workspace.optimizer_step = workspace.epoch = 0
        workspace.cfg = Options(
            training=Options(num_epochs=1, debug=False, gradient_accumulate_every=3,
                             resume=False, max_train_steps=None, max_val_steps=None,
                             lr_scheduler="constant", lr_warmup_steps=0, device="cpu",
                             tqdm_interval_sec=1., rollout_every=0, checkpoint_every=0,
                             val_every=0, sample_every=0),
            task=Options(dataset=dataset_cfg, env_runner="must not instantiate simulator"),
            dataloader={}, val_dataloader={}, logging={}, checkpoint=Options(topk={}),
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace._output_dir = directory
            workspace.run()
        self.assertAlmostEqual(float(workspace.model.weight), .35, places=6)
        self.assertEqual(workspace.optimizer_step, 2)
        self.assertEqual(workspace.global_step, 5)
        self.assertEqual(created, [dataset_cfg])

    def test_incompatible_dataset_fails_before_constructor_or_cache_load(self):
        class ImageBase:
            pass

        class PointCloudBase:
            pass

        class PointCloudDataset(PointCloudBase):
            pass

        calls = []
        namespace = self.base_namespace()
        namespace.update({
            "BaseImageDataset": ImageBase,
            "hydra": SimpleNamespace(utils=SimpleNamespace(
                get_class=lambda target: PointCloudDataset,
                instantiate=lambda cfg: calls.append("loaded images"))),
        })
        instantiate_dataset = load_workspace_definitions(namespace)["_instantiate_image_dataset"]
        with self.assertRaisesRegex(TypeError, "requires BaseImageDataset"):
            instantiate_dataset(Options(_target_="test.PointCloudDataset"))
        self.assertEqual(calls, [])

    def test_image_dataset_uses_its_image_base_contract(self):
        class ImageBase:
            pass

        class ImageDataset(ImageBase):
            pass

        dataset = ImageDataset()
        namespace = self.base_namespace()
        namespace.update({
            "BaseImageDataset": ImageBase,
            "hydra": SimpleNamespace(utils=SimpleNamespace(
                get_class=lambda target: ImageDataset,
                instantiate=lambda cfg: dataset)),
        })
        instantiate_dataset = load_workspace_definitions(namespace)["_instantiate_image_dataset"]
        self.assertIs(instantiate_dataset(Options(_target_="test.ImageDataset")), dataset)


class ConfigCompositionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra is not installed")
    def test_cli_composes_inherited_config_and_resolvers(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--config-name=EfficientAdaFlow",
             "--cfg", "job", "--resolve", "n_demo=100", "policy.sampler=fixed",
             "policy.fixed_steps=4", "training.rollout_every=0"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("EfficientFlow.policy.EfficientAdaFlow.EfficientAdaFlowPolicy", result.stdout)
        self.assertIn("num_epochs: 500", result.stdout)
        self.assertIn("fixed_steps: 4", result.stdout)
        self.assertIn("rollout_every: 0", result.stdout)


if __name__ == "__main__":
    unittest.main()

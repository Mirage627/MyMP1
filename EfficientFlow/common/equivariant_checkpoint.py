"""Load learned escnn coefficients strictly and rebuild their derived caches.

escnn Linear and steerable convolution layers register expanded matrices or
filters in eval mode. These buffers depend on the current learned coefficients;
an EMA update or state-dict load must not leave a previous expansion in use.
"""
from collections import OrderedDict

import torch


def _cache_layers(model):
    for name, layer in model.named_modules():
        if not type(layer).__module__.startswith("escnn."):
            continue
        bases = [base for base in type(layer).__mro__
                 if base.__module__.startswith("escnn.")]
        if any(base.__name__ == "Linear" for base in bases):
            fields = ("matrix", "expanded_bias")
        elif any(base.__name__ in ("_RdConv", "_RdConvTransposed") for base in bases):
            fields = ("filter", "expanded_bias")
        else:
            continue
        if callable(getattr(layer, "expand_parameters", None)):
            yield name, layer, fields


def equivariant_cache_keys(model):
    """Return only known mode-derived buffers on actual escnn cache layers."""
    return frozenset(
        f"{name}.{field}" if name else field
        for name, _, fields in _cache_layers(model) for field in fields)


@torch.no_grad()
def refresh_equivariant_caches(model):
    """Refresh eval-mode escnn layers without toggling policy/BatchNorm modes."""
    for _, layer, _ in list(_cache_layers(model)):
        if not layer.training:
            layer.train(True)
            layer.train(False)


def load_equivariant_state_dict(model, state_dict, *, allowed_missing=()):
    """Strictly load real state while ignoring known derived escnn caches.

``allowed_missing`` accepts complete key names, or prefixes ending in a dot.
Only a new variance head and its counter should normally need this option when
initializing from the published EfficientFlow baseline. All other learned state,
including escnn basis buffers, remains mandatory. Each cache layer's original
training mode is restored, rebuilding eval caches from the loaded coefficients.
"""
    if any(not isinstance(key, str) or not key for key in allowed_missing):
        raise ValueError("allowed_missing must contain nonempty key names or prefixes")
    layers = [(layer, layer.training) for _, layer, _ in _cache_layers(model)]
    cache_keys = equivariant_cache_keys(model)
    cleaned = OrderedDict((key, value) for key, value in state_dict.items() if key not in cache_keys)
    if hasattr(state_dict, "_metadata"):
        cleaned._metadata = state_dict._metadata

    def allowed(key):
        return any(key.startswith(pattern) if pattern.endswith(".") else key == pattern
                   for pattern in allowed_missing)

    try:
        # Operate on the concrete cache layers: head_only policy.train(True)
        # intentionally keeps its frozen encoder and velocity network in eval.
        for layer, _ in layers:
            layer.train(True)
        incompatible = model.load_state_dict(cleaned, strict=False)
        missing = [key for key in incompatible.missing_keys if not allowed(key)]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible policy state: "
                f"missing={missing}, unexpected={incompatible.unexpected_keys}")
        return incompatible
    finally:
        with torch.no_grad():
            for layer, training in layers:
                layer.train(training)

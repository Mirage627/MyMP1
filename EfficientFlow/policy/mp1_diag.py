"""Explicit diag ablation; shared implementation avoids copy drift."""
from EfficientFlow.policy.mp1_ablation import MP1AblationPolicy


class Policy(MP1AblationPolicy):
    def __init__(self, **kwargs):
        super().__init__(use_mgad=False, statistics_mode="diag", **kwargs)


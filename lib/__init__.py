"""MuseFuse library entry point.

This module aggregates the core classes and functions required by the MuseFuse
training script, providing a concise API for the Memo2496 public release.
"""

from .config import MuseFuseConfig, get_default_config, resolve_dataset_paths
from .data import MusicEmotionDataset, build_dataloaders, get_dataset_stats
from .models import MuseFuseModel
from .losses import (
    gdd_loss,
    lsd_loss,
    reliability_weighted_ce,
    cosine_prototype_loss,
    distill_kl,
)
from .train_utils import (
    set_seed,
    MetricsTracker,
    GradientScaler,
    compute_dynamic_tradeoff,
)

__all__ = [
    "MuseFuseConfig",
    "get_default_config",
    "resolve_dataset_paths",
    "MusicEmotionDataset",
    "build_dataloaders",
    "get_dataset_stats",
    "MuseFuseModel",
    "gdd_loss",
    "lsd_loss",
    "reliability_weighted_ce",
    "cosine_prototype_loss",
    "distill_kl",
    "set_seed",
    "MetricsTracker",
    "GradientScaler",
    "compute_dynamic_tradeoff",
]


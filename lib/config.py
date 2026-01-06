import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class MuseFuseConfig:
    """Configuration for the MuseFuse framework (Memo2496 dataset).

    This public release only supports the Memo2496 dataset, with hyperparameters
    aligned with the Experimental Implementation section of the paper.
    """

    # Dataset (fixed)
    dataset: str = "memo"
    label_mode: str = "a"  # 'a' for Arousal, 'v' for Valence
    data_root: str = "/data/qilin.li/dataset"
    num_classes: int = 2

    # Architecture
    mel_hidden_dim: int = 512
    coch_hidden_dim: int = 512
    fusion_hidden_dim: int = 256  # D = 256 in paper
    feat_dim: int = 128

    # Training
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    num_workers: int = 4
    val_ratio: float = 0.3
    seed: int = 42

    # ProtoAlign (enabled)
    use_proto_align: bool = True
    proto_weight: float = 0.15
    proto_temperature: float = 0.5
    proto_momentum: float = 0.9
    proto_on_fusion: bool = True
    proto_on_mel: bool = False
    proto_on_coch: bool = False
    proto_use_ema: bool = True
    proto_use_pseudo: bool = False
    proto_pseudo_weight: float = 0.0
    proto_pseudo_conf_soft: bool = False

    # ReliaPseudo (enabled)
    use_relia_pseudo: bool = True
    relia_reliability: str = "entropy"  # or "confidence"
    relia_soft_mask: bool = True
    pseudo_threshold: float = 0.6
    pseudo_threshold_min: float = 0.3
    gamma_pseudo: float = 0.5

    # TriDistill (enabled)
    distill_weight: float = 0.2
    distill_temperature: float = 2.0
    distill_warmup_epochs: int = 0

    # Domain alignment & consistency
    beta_tradeoff: float = 0.3
    consistency_weight: float = 0.1

    # Computational efficiency
    cosine_scheduler: bool = True
    mixed_precision: bool = True
    max_grad_norm: float = 5.0

    # Logging
    log_path: str = "."
    log_file: str = "musefuse.log"
    out_dir: str = "checkpoints_musefuse"
    log_interval: int = 10

    # Extra unused fields (from argparse)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a plain dict for logging/serialization."""
        return dataclasses.asdict(self)


def get_default_config(**overrides: Any) -> MuseFuseConfig:
    """Create default MuseFuseConfig and apply argparse-style overrides."""
    cfg = MuseFuseConfig()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value
    return cfg


# Memo2496-only dataset mapping
DATASET_FILES: Dict[str, Dict[str, str]] = {
    "memo": {
        "mel": "memo/mel_spec.npy",
        "coch": "memo/cochlegram.npy",
        "label_a": "memo/labels/label_a.npy",
        "label_v": "memo/labels/label_v.npy",
    }
}


def resolve_dataset_paths(data_root: str, dataset: str) -> Dict[str, str]:
    """Resolve Memo2496 file paths from root directory.

    Raises:
        ValueError: if dataset is not ``'memo'`` in this public release.
    """
    name = dataset.lower()
    if name not in DATASET_FILES:
        raise ValueError(
            f"Only 'memo' dataset is supported in this public release, got '{dataset}'"
        )
    mapping = DATASET_FILES[name]
    return {key: f"{data_root.rstrip('/')}/{value}" for key, value in mapping.items()}


def dataset_input_shapes(dataset: str) -> Tuple[int, int, int, int]:
    """Return (mel_nodes, coch_nodes, feature_dim, num_classes) for Memo2496."""
    name = dataset.lower()
    if name != "memo":
        raise ValueError(
            f"Only 'memo' dataset is supported in this public release, got '{dataset}'"
        )
    # Memo2496: 1-second clips, 22050 Hz, hop_length=256 -> ~87 frames
    return 128, 84, 87, 2


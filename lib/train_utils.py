import random
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class MetricsTracker:
    history: Dict[str, list] = field(default_factory=lambda: {})

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.history.setdefault(key, []).append(float(value))

    def summary(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.history.items()}


class GradientScaler:
    """Mixed-precision wrapper compatible with training with or without AMP."""
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        return torch.cuda.amp.autocast(enabled=self.enabled)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        if self.enabled:
            self.scaler.update()


def compute_dynamic_tradeoff(
    global_gap: torch.Tensor,
    local_gap: torch.Tensor,
    momentum: float = 0.9,
) -> float:
    """Dynamically adjust the trade-off between GDD and LSD losses."""
    gap_ratio = (local_gap + 1e-6) / (global_gap + 1e-6)
    ratio = torch.tanh(gap_ratio).item()
    return float(momentum * ratio + (1 - momentum) * 0.5)


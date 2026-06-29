"""Configuration classes for research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "placeholder_dataset"
    data_dir: Path = Path("data")
    target_column: str = "target"
    sensitive_column: str = "sensitive"
    test_size: float = 0.2
    random_state: int = 42


@dataclass(slots=True)
class SelectorConfig:
    selector_name: str = "fa_cmim_conditional"
    n_features_to_select: int = 10
    relevance_estimator: str = "mutual_information"
    leakage_estimator: str = "placeholder_leakage"
    conditional_mode: bool = True
    subset_aware: bool = True


@dataclass(slots=True)
class LeakageConfig:
    max_allowed_leakage: float = 0.05
    fairness_tradeoff: float = 0.5
    conditioning_strategy: Literal["none", "pairwise", "subset"] = "pairwise"


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str = "facmim_scaffold"
    output_dir: Path = Path("results")
    n_runs: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)

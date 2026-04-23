from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    dataset_name: str = "census"
    classifier_name: str = "svm"
    random_state: int = 42
    alpha: float = 0.99
    ga_pop_size: int = 20
    ga_generations: int = 10
    pso_swarmsize: int = 15
    pso_maxiter: int = 10
    pso_omega: float = 0.5
    pso_phip: float = 1.5
    pso_phig: float = 1.5
    pso_threshold: float = 0.5
    sample_rows: Optional[int] = None
    test_size: float = 0.2

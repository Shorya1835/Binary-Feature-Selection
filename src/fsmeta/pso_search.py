from __future__ import annotations

import time
from typing import List

import numpy as np
from pyswarm import pso

from .config import Config
from .evaluator import FeatureSelectionEvaluator


def to_binary_mask(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    mask = (np.asarray(x).ravel() >= threshold).astype(int)
    if mask.sum() == 0:
        mask[int(np.argmax(x))] = 1
    return mask


def run_pso(x_train, x_val, y_train, y_val, feature_names: List[str], config: Config):
    evaluator = FeatureSelectionEvaluator(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        classifier_name=config.classifier_name,
        alpha=config.alpha,
        random_state=config.random_state,
    )

    n_features = x_train.shape[1]
    lb = np.zeros(n_features)
    ub = np.ones(n_features)

    def objective(x):
        mask = to_binary_mask(x, threshold=config.pso_threshold)
        return evaluator.score_mask(mask)

    start = time.time()
    xopt, fopt = pso(
        objective,
        lb,
        ub,
        swarmsize=config.pso_swarmsize,
        maxiter=config.pso_maxiter,
        omega=config.pso_omega,
        phip=config.pso_phip,
        phig=config.pso_phig,
        debug=False,
    )
    elapsed = time.time() - start

    best_mask = to_binary_mask(xopt, threshold=config.pso_threshold)
    best_acc = evaluator.accuracy_of_mask(best_mask)
    selected = [feature_names[i] for i, flag in enumerate(best_mask) if flag == 1]

    return {
        "dataset": config.dataset_name,
        "classifier": config.classifier_name,
        "method": "Binary PSO",
        "n_total_features": len(feature_names),
        "n_selected": int(best_mask.sum()),
        "selected_ratio": float(best_mask.sum() / len(feature_names)),
        "objective": float(fopt),
        "accuracy": best_acc,
        "runtime_sec": elapsed,
        "selected_features": selected,
        "best_mask": best_mask,
    }

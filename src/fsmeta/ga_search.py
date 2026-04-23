from __future__ import annotations

import time
from typing import List

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from .config import Config
from .evaluator import FeatureSelectionEvaluator


class GAFeatureProblem(ElementwiseProblem):
    def __init__(self, evaluator: FeatureSelectionEvaluator):
        super().__init__(
            n_var=evaluator.n_features,
            n_obj=1,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator.score_mask(np.array(x, dtype=int))


def run_ga(x_train, x_val, y_train, y_val, feature_names: List[str], config: Config):
    evaluator = FeatureSelectionEvaluator(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        classifier_name=config.classifier_name,
        alpha=config.alpha,
        random_state=config.random_state,
    )

    problem = GAFeatureProblem(evaluator)
    algorithm = GA(
        pop_size=config.ga_pop_size,
        sampling=BinaryRandomSampling(),
        crossover=HUX(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    start = time.time()
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", config.ga_generations),
        seed=config.random_state,
        verbose=False,
    )
    elapsed = time.time() - start

    best_mask = np.array(result.X).astype(int).ravel()
    best_obj = float(result.F[0]) if np.ndim(result.F) > 0 else float(result.F)
    best_acc = evaluator.accuracy_of_mask(best_mask)
    selected = [feature_names[i] for i, flag in enumerate(best_mask) if flag == 1]

    return {
        "dataset": config.dataset_name,
        "classifier": config.classifier_name,
        "method": "GA",
        "n_total_features": len(feature_names),
        "n_selected": int(best_mask.sum()),
        "selected_ratio": float(best_mask.sum() / len(feature_names)),
        "objective": best_obj,
        "accuracy": best_acc,
        "runtime_sec": elapsed,
        "selected_features": selected,
        "best_mask": best_mask,
    }

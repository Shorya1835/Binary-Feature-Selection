from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import Config
from .data import load_dataset, make_split
from .evaluator import FeatureSelectionEvaluator
from .ga_search import run_ga
from .pso_search import run_pso


def run_baseline(x_train, x_val, y_train, y_val, feature_names, config: Config):
    evaluator = FeatureSelectionEvaluator(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        classifier_name=config.classifier_name,
        alpha=config.alpha,
        random_state=config.random_state,
    )

    mask = np.ones(x_train.shape[1], dtype=int)
    start = time.time()
    acc = evaluator.accuracy_of_mask(mask)
    obj = evaluator.score_mask(mask)
    elapsed = time.time() - start

    return {
        "dataset": config.dataset_name,
        "classifier": config.classifier_name,
        "method": "Baseline",
        "n_total_features": len(feature_names),
        "n_selected": int(mask.sum()),
        "selected_ratio": 1.0,
        "objective": float(obj),
        "accuracy": acc,
        "runtime_sec": elapsed,
        "selected_features": feature_names,
        "best_mask": mask,
    }


def run_experiment(dataset_name: str, classifier_name: str, base_config: Config):
    config = Config(
        dataset_name=dataset_name,
        classifier_name=classifier_name,
        random_state=base_config.random_state,
        alpha=base_config.alpha,
        ga_pop_size=base_config.ga_pop_size,
        ga_generations=base_config.ga_generations,
        pso_swarmsize=base_config.pso_swarmsize,
        pso_maxiter=base_config.pso_maxiter,
        pso_omega=base_config.pso_omega,
        pso_phip=base_config.pso_phip,
        pso_phig=base_config.pso_phig,
        pso_threshold=base_config.pso_threshold,
        sample_rows=base_config.sample_rows,
        test_size=base_config.test_size,
    )

    x, y, feature_names, raw_df = load_dataset(config)
    x_train, x_val, y_train, y_val = make_split(x, y, config)

    print("=" * 90)
    print(f"DATASET: {dataset_name.upper()} | CLASSIFIER: {classifier_name.upper()}")
    print(f"Raw shape: {raw_df.shape} | Model shape: {x.shape} | Features: {len(feature_names)}")

    baseline = run_baseline(x_train, x_val, y_train, y_val, feature_names, config)
    print("ran baseline")
    ga = run_ga(x_train, x_val, y_train, y_val, feature_names, config)
    print("ran ga")
    pso = run_pso(x_train, x_val, y_train, y_val, feature_names, config)
    print("ran binary pso")

    results = [baseline, ga, pso]

    for row in results:
        print(
            f"{row['method']:10s} | "
            f"Acc={row['accuracy']:.6f} | "
            f"Selected={row['n_selected']}/{row['n_total_features']} | "
            f"Obj={row['objective']:.6f} | "
            f"Time={row['runtime_sec']:.2f}s"
        )

    return results


def run_all(base_config: Config | None = None):
    cfg = base_config or Config()
    datasets = ["santander", "census"]
    classifiers = ["svm", "knn", "dt"]
    all_results = []

    for dataset_name in datasets:
        for classifier_name in classifiers:
            try:
                results = run_experiment(dataset_name, classifier_name, cfg)
                all_results.extend(results)
            except Exception as exc:
                print("=" * 90)
                print(f"FAILED: dataset={dataset_name}, classifier={classifier_name}")
                print("Reason:", str(exc))
                print()

    df = pd.DataFrame(all_results)
    if df.empty:
        return None, None, None, None

    cols = [
        "dataset",
        "classifier",
        "method",
        "accuracy",
        "n_selected",
        "n_total_features",
        "selected_ratio",
        "objective",
        "runtime_sec",
    ]

    summary = df[cols].sort_values(
        by=["dataset", "classifier", "accuracy"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    best_idx = df.groupby(["dataset", "classifier"])["accuracy"].idxmax()
    best = df.loc[best_idx, cols].sort_values(by=["dataset", "classifier"]).reset_index(drop=True)

    pivot = summary.pivot_table(
        index=["dataset", "classifier"],
        columns="method",
        values="accuracy",
    ).reset_index()

    return df, summary, best, pivot

from __future__ import annotations

import warnings

from .config import Config
from .experiment import run_all

warnings.filterwarnings("ignore")


def main():
    config = Config(
        random_state=42,
        alpha=0.99,
        ga_pop_size=20,
        ga_generations=10,
        pso_swarmsize=15,
        pso_maxiter=10,
        sample_rows=None,
    )

    _, summary, best, pivot = run_all(config)

    if summary is None:
        print("No results generated.")
        return

    print("\n" + "=" * 100)
    print("FINAL SUMMARY TABLE")
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("BEST METHOD FOR EACH DATASET + CLASSIFIER")
    print(best.to_string(index=False))

    print("\n" + "=" * 100)
    print("ACCURACY PIVOT TABLE")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()

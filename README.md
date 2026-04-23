# Feature Selection with GA and Binary PSO

A small repo for feature selection on tabular classification datasets using:

- Genetic Algorithm with `pymoo`
- Binary-style PSO with `pyswarm`
- classifiers: SVM, KNN, Decision Tree

## Layout

```text
feature-selection-metaheuristics/
├── requirements.txt
└── src/
    └── fsmeta/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── evaluator.py
        ├── experiment.py
        ├── ga_search.py
        ├── main.py
        ├── models.py
        └── pso_search.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
cd src
python -m fsmeta.main
```

## Notes

- `census` expects the Census Income CSV at the Kaggle path used in `data.py`
- `santander` expects the Santander training CSV at the Kaggle path used in `data.py`
- PSO is handled as a binary mask by thresholding particle positions

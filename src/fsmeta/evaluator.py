from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from .models import get_classifier


class FeatureSelectionEvaluator:
    def __init__(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        classifier_name: str,
        alpha: float = 0.99,
        random_state: int = 42,
    ):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.n_features = x_train.shape[1]
        self.alpha = alpha
        self.model = get_classifier(classifier_name, random_state=random_state)
        self.cache: Dict[Tuple[int, ...], float] = {}

    def score_mask(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask).astype(int).ravel()

        if mask.sum() == 0:
            return 1.0

        key = tuple(mask.tolist())
        if key in self.cache:
            return self.cache[key]

        x_tr = self.x_train[:, mask == 1]
        x_va = self.x_val[:, mask == 1]

        self.model.fit(x_tr, self.y_train)
        y_pred = self.model.predict(x_va)

        acc = accuracy_score(self.y_val, y_pred)
        ratio = mask.sum() / self.n_features
        value = self.alpha * (1.0 - acc) + (1.0 - self.alpha) * ratio

        self.cache[key] = value
        return value

    def accuracy_of_mask(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask).astype(int).ravel()

        if mask.sum() == 0:
            return 0.0

        x_tr = self.x_train[:, mask == 1]
        x_va = self.x_val[:, mask == 1]

        self.model.fit(x_tr, self.y_train)
        y_pred = self.model.predict(x_va)
        return accuracy_score(self.y_val, y_pred)

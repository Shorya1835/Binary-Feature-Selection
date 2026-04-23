from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classifier(name: str, random_state: int = 42):
    name = name.lower()

    if name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", max_iter=500)),
            ]
        )

    if name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        )

    if name == "dt":
        return Pipeline(
            [
                ("clf", DecisionTreeClassifier(random_state=random_state)),
            ]
        )

    raise ValueError("classifier_name must be one of: 'svm', 'knn', 'dt'")

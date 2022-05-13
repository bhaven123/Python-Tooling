from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

MAT_FILE = Path(__file__).resolve().parent / "NumberRecognitionBigger.mat"
WINE_DATA = Path(__file__).resolve().parent / "winequality-red.csv"


# Helper function to load custom dataset for question 2 and question 3
def load_csv(csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv, delimiter=";")
    X = df.iloc[:, : len(df.columns) - 1]  # data
    y = df.iloc[:, -1]  # target
    return X, y


# Helper function to generate error rates for question1 and question3
def validation_scores(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    names = ["rf", "knn1", "knn5", "knn10", "svm_rbf", "svm_linear"]
    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=0),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
        SVC(kernel="rbf", random_state=0),
        SVC(kernel="linear", random_state=0),
    ]

    kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    error_rate = []
    for name, clf in zip(names, classifiers):
        scores = cross_validate(clf, X, y, scoring="accuracy", cv=kfold)
        errors = 1 - scores["test_score"]
        print(f"{name}:", errors)
        error_rate.append(np.mean(errors))
    kfold_scores = pd.DataFrame([error_rate], columns=names, index=["err"])
    print("\nK Fold validated Error Rates:\n", kfold_scores)
    return kfold_scores


# Helper function to save kfold error values for question 2 and question 3
def save_kfold_json(data: str, df: pd.DataFrame) -> None:
    outfile = Path(__file__).resolve().parent / f"kfold_{data}.json"
    df.to_json(outfile)
    print(f"\nK-Fold error rates for {data} successfully saved to {outfile}")


def question1(file: Path) -> pd.DataFrame:
    dataset = loadmat(file)
    X, y = dataset["X"], dataset["y"]
    X, y = np.transpose(X), np.transpose(y)
    img_samples = X.shape[0]
    img_features = np.prod(X.shape[1:])
    X = X.reshape(img_samples, img_features)
    eights, nines = X[y.squeeze() == 8, :], X[y.squeeze() == 9, :]
    data = np.concatenate([eights, nines], axis=0)
    label_8, label_9 = y[(y == 8)], y[(y == 9)]
    target = np.concatenate([label_8, label_9])

    dfq1 = validation_scores(data, target)
    save_kfold_json("mnist", dfq1)


def question2(csv: Path) -> None:
    X, y = load_csv(csv)
    isBest = y > np.median(y)
    group_1 = X[isBest]
    group_2 = X[~isBest]
    labels, rounded_values = [], []
    for i in range(X.shape[1]):
        g_1 = group_1.iloc[:, i]
        g_2 = group_2.iloc[:, i]
        y_score = np.concatenate([g_1, g_2])
        y_true = np.array([0 for _ in g_1] + [1 for _ in g_2])
        auc_value = roc_auc_score(y_true, y_score)
        if np.median(g_2) > np.median(g_1):
            auc_value = 1 - auc_value
        labels.append(X.columns[i])
        rounded_values.append(np.around(auc_value, decimals=3))

    leading_10 = sorted(
        zip(labels, rounded_values),
        key=lambda value: value[1],
    )
    dfq2 = pd.DataFrame(leading_10, columns=["Feature", "AUC"], index=None)
    md_table = dfq2.to_markdown(tablefmt="grid")
    print(md_table)
    dfq2.to_json(Path(__file__).resolve().parent / "aucs.json")


def question3(csv: Path) -> None:
    X, y = load_csv(csv)
    dfq3 = validation_scores(X, y)
    save_kfold_json("data", dfq3)


if __name__ == "__main__":
    print("Question 1\n")
    question1(MAT_FILE)
    print("Question 2\n")
    question2(WINE_DATA)
    print("Question 3\n")
    question3(WINE_DATA)

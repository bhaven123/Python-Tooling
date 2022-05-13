from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

MAT_FILE = Path(__file__).resolve().parent / "NumberRecognition.mat"
WINE_DATA = Path(__file__).resolve().parent / "winequality-red.csv"


class KNN:
    def __init__(self) -> None:
        pass

    # Helper function to create X_train, X_test,
    #  y_train, y_test for Number Recognition data
    def data(self, eights: ndarray, nines: ndarray) -> Tuple[ndarray, ndarray]:
        X = np.concatenate([eights, nines], axis=2).transpose([2, 0, 1])
        img_samples = X.shape[0]
        img_features = np.prod(X.shape[1:])
        X = X.reshape([img_samples, img_features])

        y_labels_8 = np.zeros(eights.shape[-1])
        y_labels_9 = np.ones(nines.shape[-1])

        y = np.concatenate([y_labels_8, y_labels_9], axis=0)

        return X, y

    # Helper function to load custom dataset for question 2 and question 3
    def load_csv(self, csv: Path) -> pd.DataFrame:
        df = pd.read_csv(csv, delimiter=";")
        X = df.iloc[:, : len(df.columns) - 1]  # data
        y = df.iloc[:, -1]  # target
        return X, y

    # Helper function to generate error rates for question1 and question3
    def error_scores(self, X_train, y_train, X_test, y_test) -> list:
        error_rate = []
        for k in range(1, 21):
            clf = KNeighborsClassifier(n_neighbors=k)
            model = clf.fit(X_train, y_train)
            model.predict(X_test)
            classification_error = 100 * (1 - model.score(X_test, y_test))
            error_rate.append(classification_error)
        return error_rate

    # Helper function for plotting erros in question1 and question3
    def plot(self, errors: list, q: int) -> None:
        k = np.arange(1, 21)
        sbn.set_style("ticks")
        plt.plot(k, errors)
        plt.savefig(Path(__file__).resolve().parent / f"knn_q{q}.png")
        plt.show()


def question1(mat: Path) -> None:
    data = loadmat(mat)

    train_8, test_8 = data["imageArrayTraining8"], data["imageArrayTesting8"]
    train_9, test_9 = data["imageArrayTraining9"], data["imageArrayTesting9"]

    X_train, y_train = myobj.data(train_8, train_9)
    X_test, y_test = myobj.data(test_8, test_9)

    errors_q1 = myobj.error_scores(X_train, y_train, X_test, y_test)
    myobj.plot(errors_q1, 1)

    outfile = Path(__file__).resolve().parent / "errors.npy"
    arr = np.array(errors_q1)
    np.save(outfile, arr, allow_pickle=False)
    print(f"Error rates succesfully saved to {outfile}")


def question2(csv: Path) -> None:
    X, y = myobj.load_csv(csv)
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
        labels.append(X.columns[i])
        rounded_values.append(np.around(auc_value, decimals=3))

    leading_10 = sorted(
        zip(labels, rounded_values), key=lambda value: value[1], reverse=True
    )
    md_table = pd.DataFrame(
        leading_10, columns=["Feature", "AUC"], index=None
    ).to_markdown(tablefmt="grid")
    print(md_table)


def question3(csv: Path) -> None:
    X, y = myobj.load_csv(csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    errors_q3 = myobj.error_scores(X_train, y_train, X_test, y_test)
    myobj.plot(errors_q3, 3)


if __name__ == "__main__":
    myobj = KNN()
    print("Question 1")
    question1(MAT_FILE)
    print("Question 2")
    question2(WINE_DATA)
    print("Question 3")
    question3(WINE_DATA)

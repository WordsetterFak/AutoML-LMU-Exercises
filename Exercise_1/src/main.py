from pathlib import Path
from typing import Tuple
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

HERE = Path(__file__).parent.resolve()

# Paths to the test data
paths = {
    "MC": HERE / "MCTestData.csv",
    "TMSt": HERE / "TMStTestData.csv",
    "F": HERE / "FTestData.csv",
}


def load_data_MNTest() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data stored in McNemarTest.csv
    :param fl: path to the csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(paths["MC"], header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data stored in fl
    :param fl: path to the csv file
    :return: y1, y2
    """
    data = np.loadtxt(paths["TMSt"], delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest() -> np.ndarray:
    """
    Loads data stored in fl
    :param fl: path to the csv file
    :return: evaluations
    """
    errors = np.loadtxt(paths["F"], delimiter=",")
    return errors


def McNemar_test(
    labels: np.ndarray,
    prediction_1: np.ndarray,
    prediction_2: np.ndarray,
) -> float:
    """
    TODO
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    chi2_Mc = np.random.uniform(0, 1)
    return chi2_Mc


def TwoMatchedSamplest_test(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    TODO
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    t_value = np.random.uniform(0, 1)
    return t_value


def Friedman_test(errors: np.ndarray) -> Tuple[float, dict]:
    """
    TODO
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, containing anything
    you need to solve the `Nemenyi_test` and `box_plot` functions.
    """
    chi2_F = np.random.uniform(0, 1)

    FData_stats = {
        "errors": errors,
        "hello": "world",
    }
    return chi2_F, FData_stats


def Nemenyi_test(fdata_stats: dict) -> np.ndarray:
    """
    TODO
    :param fdata_stats np.ndarray: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    Q_value = np.empty_like([1])
    return Q_value


def box_plot(fdata_stats: dict) -> None:
    """
    TODO
    :param fdata_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    pass


def main() -> None:
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)
    print("chi2_Mc ", chi2_Mc)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)
    print("t_value ", t_value)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)
    print("chi2_F ", chi2_F)

    # (d)
    Q_value = Nemenyi_test(FData_stats)
    print("Q value ", Q_value)

    # (e)
    box_plot(FData_stats)


if __name__ == "__main__":
    main()

from typing import Tuple

import numpy
from scipy.stats import beta as Beta
import scipy.stats as st
import numpy as np
import pandas as pd

# https://github.com/Quinta13/AI_assignments/blob/346b9ec300fd2b4945ab01df0ca16d99220c13c8/assignment_2/digits_classifiers/classifiers.py#L324
# https://github.com/Lotto98/Discriminative-and-Generative-Classifiers/blob/69816c6a95ca51ad093118d245728ea1f064578d/naive_bayes.py#L44

def _get_alpha_beta(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Given a data frame which represent a single class compute
        alphas and betas for the beta-distribution
    :param df: dataframe of a single class
    :return: alphas and betas for the beta distribution
    """

    # exploit of element-wise numpy operation

    mean = np.mean(df, axis=0)  # E[X]
    var = np.var(df, axis=0)  # Var[X]
    k = mean * (1 - mean) / var - 1  # K = ( E[X] * (1 - E[X]) / Var[X] ) - 1
    alpha = k * mean  # alpha = K E[X] + 1
    beta = k * (1 - mean)  # beta  = K (1 - E[X]) + 1

    return alpha, beta


alphaBeta: list[tuple[float, float]] = []

labelProb: list[float] = []


# TODO put it in a class and save the parameter alphaBeta and labelProb
def fit(X: pd.DataFrame, y: np.ndarray):
    """
    Save the alphas and betas for each class (as for each pixel)
    Save the relative frequency of each class
    :param X: feature space
    :param y: labels
    """

    for i in range(0, 10):
        # pick just the vectors with label i
        tempdf: pd.DataFrame = X.loc[y == i]

        # compute mean, variance, k, alpha and beta
        mean = np.mean(tempdf, axis=0)  # E[X]
        var = np.var(tempdf, axis=0)  # Var[X]
        k = (mean * (1 - mean) / var) - 1  # K = ( E[X] * (1 - E[X]) / Var[X] ) - 1
        alpha = k * mean  # alpha = K E[X] + 1
        beta = k * (1 - mean)  # beta  = K (1 - E[X]) + 1

        # add them inside the list
        alphaBeta.append((alpha, beta))

        # save the probability for each label
        labelProb.append(len(tempdf) / len(X))


def predict(X: pd.DataFrame) -> np.ndarray:
    """
    It predict the label for all instances in the test set
    :param X: Test set
    :return: predicted labels
    """

    X = X.to_numpy()  # cast to array to enforce performance
    predictions = []

    for i in range(X.shape[0]):
        prediction: Tuple[np.ndarray, np.ndarray] = st.mode(a=findProb(X[i, :]),
                                                            axis=None, keepdims=False)
        print("found one")
        predictions.append(prediction.mode)

    return np.array(predictions)


def findProb(vector: numpy.ndarray):
    newArr: list[float] = []

    for i in range(0, 10):
        alpha, beta = alphaBeta[i]
        epsilon = 0.1  # length of neighborhood
        leftBound = vector - epsilon
        rightBound = vector + epsilon

        # cumulative density function in the neighbor
        probs = Beta.cdf(a=alpha, b=beta, x=rightBound) - \
                Beta.cdf(a=alpha, b=beta, x=leftBound)

        # where the probability dist doesn't exist (variance less or equal to zero)
        #   we assign one in order to not affect the multiplication
        np.nan_to_num(probs, nan=1., copy=False)
        newArr.append(np.product(probs) * labelProb[i])

    maximum = 0
    idx = -1

    for j in range(0, len(newArr)):
        if newArr[j] > maximum:
            idx = j

    return idx

from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    data_plus = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
    z = np.dot(data_plus, weights)
    y = sigmoid(z)
    # y = y[:, [0]]
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################

    ce_1 = np.dot(np.transpose(targets), np.log(y))
    ce_2 = np.dot(np.transpose((1 - targets)), np.log(1 - y))
    ce = - np.sum((ce_1 + ce_2)) / y.shape[0]

    frac_correct = np.sum((targets == 0) == (y < 0.5)) / y.shape[0]

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    N, M = np.shape(data)

    #####################################################################
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################

    f_1 = np.dot(np.transpose(targets), np.log(y))
    f_2 = np.dot(np.transpose((1 - targets)), np.log(1 - y))
    f = - np.sum((f_1 + f_2)) / N

    df = np.dot(np.transpose(data), (y - targets))
    other = np.array([[np.sum(y - targets)]])
    df = np.concatenate((df, other), axis=0)
    df = df / N

    return f, df, y

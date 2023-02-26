from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    k = [1, 3, 5, 7, 9]
    percentage_by_k = []

    for i in range(len(k)):
        results = knn(k[i], train_inputs, train_targets, valid_inputs)
        total_number = len(valid_targets)
        so_far = sum(valid_targets == results)
        percentage_by_k.append(so_far/total_number)

    plt.title("Classification Accuracy as a Function of k")
    plt.scatter(k, percentage_by_k)
    plt.xlabel("k")
    plt.ylabel("Classification Accuracy")
    plt.show()

    # i have chosen the integer 5 to by my optimal value of k
    # denote this value as k_star
    # the reason i have chosen this integer is not only since it is optimal,
    # but also because the neighbouring values of k are also optimal
    # all the values of k to be tested have the highest validation accuracies
    k_stars = [3, 5, 7]
    validation_percentage_by_k = []
    test_percentage_by_k = []

    for i in range(len(k_stars)):
        valid_results = knn(k_stars[i], train_inputs, train_targets, valid_inputs)
        validation_total_number = len(valid_targets)
        validation_so_far = sum(valid_targets == valid_results)
        validation_percentage_by_k.append(validation_so_far/validation_total_number)
        print("Validation Accuracy for", k_stars[i], "is", validation_so_far/validation_total_number)

        test_results = knn(k_stars[i], train_inputs, train_targets, test_inputs)
        test_total_number = len(test_targets)
        test_so_far = sum(test_targets == test_results)
        test_percentage_by_k.append(test_so_far/test_total_number)
        print("Test Accuracy for", k_stars[i], "is", test_so_far/test_total_number)

    # plt.title("Classification Accuracy for k-2, k, k+2")
    # plt.scatter(k_stars, validation_percentage_by_k)
    # plt.xlabel("k-2, k, k+2")
    # plt.ylabel("Classification Accuracy")
    # plt.show()

    plt.title("Test Accuracy for k-2, k, k+2")
    plt.scatter(k_stars, test_percentage_by_k)
    plt.xlabel("k-2, k, k+2")
    plt.ylabel("Classification Accuracy")
    plt.show()

    # we see that the validation accuracies for all values of k-2, k, k+2 are the same
    # they remain constant at a value of 86% accuracy
    # on the other hand, the test accuracy is slightly higher
    # k-2 has 92% test accuracy, and k and k+2 have 94% accuracy
    # thus we can see that the values of the test accuracy slightly increase as we increase k


if __name__ == "__main__":
    run_knn()

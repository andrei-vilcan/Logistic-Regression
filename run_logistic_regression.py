from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    small_train_inputs, small_train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    N, M = train_inputs.shape

    test_inputs, test_targets = load_test()

    #####################################################################
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.005,
        "weight_regularization": 0.1,
        "num_iterations": 1000
    }
    weights = np.ones((M + 1, 1))
    weights = weights * 0.05

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################

    # for plots
    # For large training set

    train_ce_s = []
    valid_ce_s = []

    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = weights - df * hyperparameters["learning_rate"]

        train_ce, train_frac_correct = evaluate(train_targets, y)
        train_ce_s.append(train_ce)

        valid_prediction = logistic_predict(weights, valid_inputs)
        valid_ce, valid_frac_correct = evaluate(valid_targets, valid_prediction)
        valid_ce_s.append(valid_ce)

    test_prediction = logistic_predict(weights, test_inputs)
    test_ce, test_frac_correct = evaluate(test_targets, test_prediction)
    print("valid_frac_correct:", valid_frac_correct)
    print("test_frac_correct:", test_frac_correct)

    iterations = [i for i in range(hyperparameters["num_iterations"])]

    plt.ylabel("Cross Entropy")
    plt.xlabel("Iteration Number")
    plt.title("Cross Entropy vs. Iteration for large training set")
    plt.plot(iterations, train_ce_s, label="training")
    plt.plot(iterations, valid_ce_s, label="validation")
    plt.legend()
    plt.show()

    # For small training set
    # new weights
    small_weights = np.ones((M + 1, 1))
    small_weights = small_weights * 0.05

    small_train_ce_s = []
    small_valid_ce_s = []

    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(small_weights, small_train_inputs, small_train_targets, hyperparameters)
        small_weights = small_weights - df * hyperparameters["learning_rate"]

        small_train_ce, small_train_frac_correct = evaluate(small_train_targets, y)
        small_train_ce_s.append(small_train_ce)

        small_valid_prediction = logistic_predict(small_weights, valid_inputs)
        small_valid_ce, small_valid_frac_correct = evaluate(valid_targets, small_valid_prediction)
        small_valid_ce_s.append(small_valid_ce)
    plt.ylabel("Cross Entropy")
    plt.xlabel("Iteration Number")
    plt.title("Cross Entropy vs. Iteration for small training set")
    plt.plot(iterations, small_train_ce_s, label="training")
    plt.plot(iterations, small_valid_ce_s, label="validation")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()

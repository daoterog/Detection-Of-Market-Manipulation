"""
Model evaluation module.
"""

import numpy as np

import operator as op
from functools import reduce


def n_combined_r(n: int, r: int):

    """Returns the number of combinations of n and r."""

    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def number_of_samples(vc_dim: int, gen_error: float, train_error: float):

    """Give the probably approximated correct learning guarantee for a vc dimension (of hypothesis
    family), a generalization error, and a train error."""

    return (np.log(vc_dim) + np.log(1 / train_error)) / gen_error


def number_of_samples_decision_tree(
    depth: int, gen_error: float, train_error: float, n_features: int
):

    """Give the probably approximated correct learning guarantee for a given depth, a generalization
    error, a train error, and a number of features."""

    return (
        np.log(2)
        * (
            (np.power(2, depth) - 1) * (1 + np.log2(n_features))
            + 1
            + np.log(1 / train_error)
        )
        / (2 * (gen_error**2))
    )


def find_vc_dim(hypthesis_family: str, n_features: int, pol_degree: int = None):

    """Returns vc dimension of a given hypothesis family and number of features."""

    if hypthesis_family == "linear":
        return n_features + 1
    elif hypthesis_family == "polynomial":
        if pol_degree is None:
            raise ValueError("Polynomial degree not specified.")
        return n_combined_r(n_features + pol_degree - 1, pol_degree)
    elif hypthesis_family == "rbf":
        return np.inf
    else:
        raise ValueError(
            "Unknown hypothesis family. Available hypothesis families are decision_tree, linear, polynomial, and rbf."
        )


def get_number_of_samples(
    hypthesis_family: str,
    n_features: int,
    gen_error: float,
    train_error: float,
    pol_degree: int = None,
    depth: int = None,
):

    """Returns the number of samples needed to guarantee correct learning for a given hypothesis family,
    number of features, generalization error, and train error."""

    if hypthesis_family == "decision_tree":
        if depth is None:
            raise ValueError("Depth not specified.")
        return number_of_samples_decision_tree(
            depth, gen_error, train_error, n_features
        )
    return number_of_samples(
        find_vc_dim(hypthesis_family, n_features, pol_degree), gen_error, train_error
    )


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:

    """Obtains train and test error for a given model."""

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    target_labels = np.unique(y_test)

    labels_errors = {}

    for label in target_labels:
        train_label_indexes = np.where(y_train == label)
        val_label_indexes = np.where(y_val == label)
        test_label_indexes = np.where(y_test == label)

        train_label_error = 1 - np.mean(y_pred_train[train_label_indexes] == label)
        val_label_error = 1 - np.mean(y_pred_val[val_label_indexes] == label)
        test_label_error = 1 - np.mean(y_pred_test[test_label_indexes] == label)

        labels_errors[label] = (train_label_error, val_label_error, test_label_error)

    return labels_errors


def evaluate_classifiers(
    classifiers_dict: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:

    """Evaluates all classifiers in a dictionary."""

    classifiers_errors = {}

    for classifier_name, classifier in classifiers_dict.items():
        classifiers_errors[classifier_name] = evaluate_model(
            classifier, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Print results
        label_one_errors = classifiers_errors[classifier_name][1]
        label_zero_errors = classifiers_errors[classifier_name][0]
        print(f"{classifier_name}")
        print(
            f"Label One: train error: {label_one_errors[0]}, val error: {label_one_errors[1]}, test error: {label_one_errors[2]}"
        )
        print(
            f"Label Zero: train error: {label_zero_errors[0]}, val error: {label_zero_errors[1]}, test error: {label_zero_errors[2]}"
        )

    return classifiers_errors

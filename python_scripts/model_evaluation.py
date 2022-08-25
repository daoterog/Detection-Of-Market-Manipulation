"""
Model evaluation module.
"""

import numpy as np



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


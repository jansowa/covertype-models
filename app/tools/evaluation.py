from sklearn.model_selection import train_test_split
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, accuracy_score
import numpy as np


def print_roc_curves_return_accuracy(classifier, X, y) -> float:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    accuracy = accuracy_score(y_test, np.argmax(y_score, axis=1) + 1)
    fig, ax = plt.subplots(figsize=(6, 6))

    fpr, tpr, roc_auc = dict(), dict(), dict()

    n_classes = 7  # TODO: take from y or as parameters

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    target_names = ['Cover type 1', 'Cover type 2', 'Cover type 3', 'Cover type 4', 'Cover type 5', 'Cover type 6',
                    'Cover type 7']  # implement smarter
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "black", "rosybrown", "beige", "cyan"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()
    return accuracy


def plot_models_accuracy(names_accuracies: dict):
    names = list(names_accuracies.keys())
    accuracies = list(names_accuracies.values())
    plt.figure(figsize=(7, 5))
    plt.bar(names, accuracies, width=0.3, color='maroon')
    plt.xlabel("Model name")
    plt.ylabel("Model accuracy")
    plt.title("Comparison of model accuracy")
    plt.show()

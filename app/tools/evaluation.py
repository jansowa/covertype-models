from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
import numpy as np
from numpy.typing import ArrayLike


def print_training_curves(history, metric: str) -> None:
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('Neural network ' + metric + ' training curve')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def print_roc_curves(y_score, y_train: ArrayLike, y_test: ArrayLike, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    n_classes = 7
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    target_names = ['cover type ' + str(i) for i in range(1, 8)]
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
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass for " + model_name)
    plt.legend()
    plt.show()


def plot_models_accuracy(names_accuracies: dict) -> None:
    names = list(names_accuracies.keys())
    accuracies = list(names_accuracies.values())
    plt.figure(figsize=(7, 5))
    plt.bar(names, accuracies, width=0.3, color='maroon')
    plt.xlabel("Model name")
    plt.ylabel("Model accuracy")
    plt.title("Comparison of model accuracy")
    plt.show()


def predict_proba_to_class(y_score) -> ArrayLike:
    return np.argmax(y_score, axis=1) + 1

from app.models.generating_scripts.untrained_model import untrained_logistic_regression
from app.models.generating_scripts.untrained_model import untrained_random_forest
from app.models.heuristic import Heuristic
from app.tools.evaluation import print_roc_curves, plot_models_accuracy, predict_proba_to_class
from app.tools.load_data import LoadData
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = LoadData().load_X_y_splitted()
# X_train, X_test, y_train, y_test = X_train.iloc[:2000, :], X_test.iloc[:1000, :], y_train[:2000], y_test[:1000]

accuracies = dict()
y_score = dict()

logistic_regression = untrained_logistic_regression()
y_score["logistic regression"] = logistic_regression.fit(X_train, y_train).predict_proba(X_test)
accuracies["logistic regression"] = accuracy_score(y_test, predict_proba_to_class(y_score["logistic regression"]))

random_forest = untrained_random_forest()
y_score["random forest"] = random_forest.fit(X_train, y_train).predict_proba(X_test)
accuracies["random forest"] = accuracy_score(y_test, predict_proba_to_class(y_score["random forest"]))

y_score["heuristic"] = accuracy_score(y_test, Heuristic().predict(X_test))

plot_models_accuracy(accuracies)
print_roc_curves(y_score["logistic regression"], y_train, y_test, "logistic regression")
print_roc_curves(y_score["random forest"], y_train, y_test, "random forest")

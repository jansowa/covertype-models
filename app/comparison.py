from app.models.untrained_model import untrained_logistic_regression, untrained_random_forest, untrained_neural_network
from app.models.heuristic import Heuristic
from app.tools.evaluation import print_roc_curves, plot_models_accuracy, predict_proba_to_class, print_training_curves
from app.tools.load_data import LoadData
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

HEURISTIC = "heuristic"
RANDOM_FOREST = "random forest"
LOGISTIC_REGRESSION = "logistic regression"
NEURAL_NETWORK = "neural network"

X_train, X_test, y_train, y_test = LoadData().load_X_y_splitted()

accuracies = dict()
y_score = dict()

logistic_regression = untrained_logistic_regression()
y_score[LOGISTIC_REGRESSION] = logistic_regression.fit(X_train, y_train).predict_proba(X_test)
accuracies[LOGISTIC_REGRESSION] = accuracy_score(y_test, predict_proba_to_class(y_score[LOGISTIC_REGRESSION]))

random_forest = untrained_random_forest()
y_score[RANDOM_FOREST] = random_forest.fit(X_train, y_train).predict_proba(X_test)
accuracies[RANDOM_FOREST] = accuracy_score(y_test, predict_proba_to_class(y_score[RANDOM_FOREST]))

accuracies[HEURISTIC] = accuracy_score(y_test, Heuristic().predict(X_test))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_dummies = pd.get_dummies(y_train)
y_test_dummies = pd.get_dummies(y_test)
neural_network = untrained_neural_network()

history = neural_network.fit(X_train_scaled, y_train_dummies, epochs=35, validation_data=(X_test_scaled, y_test_dummies))
y_score[NEURAL_NETWORK] = neural_network.predict(X_test_scaled)
accuracies[NEURAL_NETWORK] = accuracy_score(y_test, predict_proba_to_class(y_score[NEURAL_NETWORK]))


print_training_curves(history)
plot_models_accuracy(accuracies)
for model in [LOGISTIC_REGRESSION, RANDOM_FOREST, NEURAL_NETWORK]:
    print_roc_curves(y_score[model], y_train, y_test, model)

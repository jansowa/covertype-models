from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import tensorflow as tf
from app.file_paths import ROOT_DIR


def untrained_logistic_regression():
    k_best = 35
    selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    scaler = StandardScaler()
    clf = LogisticRegression(random_state=0, C=0.01)
    return make_pipeline(scaler, selector, clf)


def untrained_random_forest():
    k_best = 37
    n_estimators = 50

    selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    clf = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=n_estimators, max_depth=30)
    return make_pipeline(selector, clf)


def untrained_neural_network():
    params_df = pd.read_csv(ROOT_DIR + "/models/saved/best_nn_params.csv")
    params = pd.Series(params_df.iloc[:, 1].values, index=params_df.iloc[:, 0].values)

    activation_function = params["activation_function"]
    if activation_function == "leaky_relu":
        activation_function = tf.keras.layers.LeakyReLU()

    first_dense_units = 2 ** int(params["first_dense_power"])
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(first_dense_units, activation=activation_function),
        tf.keras.layers.Dense(first_dense_units * 2, activation=activation_function),
        tf.keras.layers.Dense(first_dense_units * 4, activation=activation_function),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    sgd = tf.keras.optimizers.legacy.SGD(
        learning_rate=float(params["learning_rate"]), momentum=float(params["momentum"]))

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

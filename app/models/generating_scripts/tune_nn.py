import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from app.tools.load_data import LoadData

X_train, X_test, y_train, y_test = LoadData().load_X_y_splitted()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

y_train_dummies = pd.get_dummies(y_train)
y_test_dummies = pd.get_dummies(y_test)

import tensorflow as tf
import optuna


def objective(trial):
    first_dense_power = trial.suggest_int("first_dense_power", 6, 9)
    first_dense_units = 2 ** first_dense_power
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    momentum = trial.suggest_uniform("momentum", 0.0, 1.0)
    activation_function = trial.suggest_categorical("activation_function", ["relu", "elu", "leaky_relu"])
    if activation_function == "leaky_relu":
        activation_function = tf.keras.layers.LeakyReLU()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(first_dense_units, activation=activation_function),
        tf.keras.layers.Dense(first_dense_units * 2, activation=activation_function),
        tf.keras.layers.Dense(first_dense_units * 4, activation=activation_function),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    sgd = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=momentum)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                     mode="max", patience=5,
                                                     restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train_dummies, epochs=100, validation_data=(X_test_scaled, y_test_dummies),
                        callbacks=[earlystopping])
    return max(history.history['val_accuracy'])


study_name = 'network-study'
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage_name, load_if_exists=True)
study.enqueue_trial({
    "first_dense_power": 7,
    "learning_rate": 0.01,
    "activation_function": "relu",
    "momentum": 0.0
})
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

pd.Series(trial.params).to_csv("../saved/best_nn_params.csv")
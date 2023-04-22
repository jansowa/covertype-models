from app.tools.load_data import LoadData
from app.models.untrained_model import untrained_neural_network
from sklearn.preprocessing import MinMaxScaler
from app.tools.file_helper import save_model
from app.file_paths import NEURAL_NETWORK_PATH, MIN_MAX_SCALER_PATH
import pandas as pd
import tensorflow as tf

X, y = LoadData().load_X_y()
scaler = MinMaxScaler()
scaler.fit(X)
save_model(scaler, MIN_MAX_SCALER_PATH)
X_scaled = scaler.transform(X)
y_dummies = pd.get_dummies(y)

network = untrained_neural_network()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=NEURAL_NETWORK_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)


network.fit(X_scaled, y_dummies, epochs=35, callbacks=[cp_callback])

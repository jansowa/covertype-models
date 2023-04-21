from app.tools.load_data import LoadData
from app.models.untrained_model import untrained_neural_network
from sklearn.preprocessing import MinMaxScaler
from app.tools.file_connector import NEURAL_NETWORK_PATH
import pandas as pd
import tensorflow as tf

X, y = LoadData().load_X_y()
X_scaled = MinMaxScaler().fit_transform(X)
y_dummies = pd.get_dummies(y)

network = untrained_neural_network()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=NEURAL_NETWORK_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)


network.fit(X_scaled, y_dummies, epochs=35, callbacks=[cp_callback])

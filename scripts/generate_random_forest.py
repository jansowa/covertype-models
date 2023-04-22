from app.tools.load_data import LoadData
from app.tools.file_helper import save_model
from app.file_paths import RANDOM_FOREST_PATH
from app.models.untrained_model import untrained_random_forest


pipe = untrained_random_forest()

X, y = LoadData().load_X_y()
pipe.fit(X, y)

save_model(pipe, RANDOM_FOREST_PATH)

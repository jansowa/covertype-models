from app.tools.load_data import LoadData
from app.tools.file_helper import save_model
from app.file_paths import LOGISTIC_REGRESSION_PATH
from app.models.untrained_model import untrained_logistic_regression

pipe = untrained_logistic_regression()
X, y = LoadData().load_X_y()
pipe.fit(X, y)

save_model(pipe, LOGISTIC_REGRESSION_PATH)

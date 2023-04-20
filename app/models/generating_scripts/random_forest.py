from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from app.tools.load_data import *
from app.tools.file_connector import save_model, RANDOM_FOREST_PATH

from sklearn.ensemble import RandomForestClassifier

k_best = 37
n_estimators = 50

selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
clf = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=n_estimators, max_depth=30)
pipe = make_pipeline(selector, clf)

X, y = LoadData().load_X_y()
pipe.fit(X, y)

save_model(pipe, RANDOM_FOREST_PATH)

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from app.tools.load_data import LoadData
from app.tools.file_connector import save_model, LOGISTIC_REGRESSION_PATH

k_best = 35
selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
scaler = StandardScaler()
clf = LogisticRegression(random_state=0, C=0.01)
pipe = make_pipeline(scaler, selector, clf)

X, y = LoadData().load_X_y()
pipe.fit(X, y)

save_model(pipe, LOGISTIC_REGRESSION_PATH)

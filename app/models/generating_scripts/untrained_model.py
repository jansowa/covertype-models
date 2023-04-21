from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


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
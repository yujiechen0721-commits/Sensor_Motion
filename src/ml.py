"""模型與評估的共用函式。"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from common import *

CLASSES = ["MRT", "bus", "car"]

class SubsampledSVM(BaseEstimator, ClassifierMixin):
    """RBF-SVM 在視窗數很多時很慢，訓練時隨機抽最多 max_n 個視窗（固定種子）。"""
    def __init__(self, max_n=15000, C=3.0):
        self.max_n, self.C = max_n, C
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if len(X) > self.max_n:
            idx = np.random.RandomState(0).choice(len(X), self.max_n, replace=False); X, y = X[idx], y[idx]
        self.m_ = make_pipeline(StandardScaler(), SVC(C=self.C, kernel="rbf", class_weight="balanced", probability=True, random_state=0)).fit(X, y)
        self.classes_ = self.m_.classes_; return self
    def predict_proba(self, X): return self.m_.predict_proba(np.asarray(X))
    def predict(self, X): return self.m_.predict(np.asarray(X))

def make_model(name):
    if name == "RandomForest": return RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    if name == "ExtraTrees": return ExtraTreesClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    if name == "HistGradientBoosting": return HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    if name == "SVM": return SubsampledSVM()
    if name == "kNN": return make_pipeline(StandardScaler(), KNeighborsClassifier(15, n_jobs=-1))
    raise ValueError(name)

def prep(X, cols, normalize):
    return normalize_per_trip(X, cols) if normalize else X

def metrics(y, pred, trips=None, proba=None):
    """視窗層級指標；若給 trips 與 proba，另外算『整趟錄製』層級（每趟平均機率取最大）。"""
    r = dict(視窗數=len(y), 視窗準確率=accuracy_score(y, pred), MacroF1=f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0))
    for c in CLASSES:
        m = (np.asarray(y) == c); r[f"{CLASS_NAMES[c]}召回"] = float((np.asarray(pred)[m] == c).mean()) if m.any() else np.nan
    if trips is not None and proba is not None:
        P = pd.DataFrame(proba, columns=CLASSES); P["trip"] = np.asarray(trips); P["y"] = np.asarray(y)
        g = P.groupby("trip").agg({**{c: "mean" for c in CLASSES}, "y": "first"})
        tp = g[CLASSES].idxmax(axis=1); r["整趟錄製數"] = len(g); r["整趟準確率"] = float((tp == g["y"]).mean())
    return r

def smooth_proba(proba, trips, k):
    """時間平滑：同一趟錄製內，把相鄰 k 個視窗的預測機率取平均（置中）。k=1 代表不平滑。"""
    if k <= 1: return np.asarray(proba)
    P = pd.DataFrame(np.asarray(proba)); P["trip"] = np.asarray(trips)
    return P.groupby("trip")[[0, 1, 2]].transform(lambda s: s.rolling(k, center=True, min_periods=1).mean()).values

def predict_full(model, X):
    """predict_proba 並對齊成 [MRT, bus, car] 三欄。"""
    P = model.predict_proba(X); full = np.zeros((len(X), 3))
    for j, c in enumerate(model.classes_): full[:, CLASSES.index(c)] = P[:, j]
    return full

"""共用設定與函式。所有步驟與網站都讀這個檔案。"""
import os, glob
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 專案根目錄
DATA_DIR = os.path.join(BASE, "data", "recordings")
SPLIT_CSV = os.path.join(BASE, "data", "split.csv")
OUT_DIR = os.path.join(BASE, "outputs")

FS = 50            # 重新取樣頻率 (Hz)
OVERLAP = 0.5      # 視窗重疊比例
MAX_GAP = 0.5      # 原始資料中斷超過這個秒數的視窗丟掉
TRIM_SEC = 30      # 每趟頭尾各裁掉幾秒（走路、等車、上下車）
SESSION_GAP_MIN = 30   # 兩筆錄製間隔超過 30 分鐘 = 不同「場次」

CLASS_NAMES = {"MRT": "捷運", "bus": "公車", "car": "汽車"}
POS_NAMES = {"hand": "手持", "pocket": "口袋", "bag": "包包"}

# ---- 訓練 / 測試切分規則 ----
# 以「整天」為單位把某幾天整批留作測試，這幾天的資料完全不參與訓練、調參或選模型。
# 原因：同一天、同一段路同時錄的手持／口袋／包包是同一趟車，若拆到兩邊會高估準確率。
TEST_DAYS = {
    "MRT": ["2026-10-22", "2026-11-14", "2026-11-18"],
    "bus": ["2026-11-18", "2026-11-19"],
    "car": ["2026-11-14"],
}

def read_sensor(folder, name):
    """讀 Accelerometer / Gyroscope（接受 .csv 或 .csv.gz）。"""
    for ext in (".csv.gz", ".csv"):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return pd.read_csv(p, usecols=["seconds_elapsed", "x", "y", "z"])
    raise FileNotFoundError(f"{folder} 找不到 {name}.csv(.gz)")

def parse_name(name):
    low = name.lower()
    cls = "bus" if "bus" in low else "car" if "car" in low else "MRT" if "mrt" in low else None
    pos = "hand" if "hand" in low else "pocket" if ("pocket" in low or "pocker" in low) else "bag" if "bag" in low else None
    return cls, pos

def normalize_per_trip(X, cols):
    """每趟各自標準化：減該趟中位數、除以該趟四分位距（IQR）。"""
    g = X.groupby("trip")[cols]
    med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
    Z = X.copy(); t = X["trip"].values
    Z[cols] = (X[cols].values - med.loc[t].values) / ((q3 - q1).loc[t].values + 1e-9)
    return Z

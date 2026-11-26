"""前處理與特徵擷取（訓練與網站共用）。
流程：裁掉頭尾 -> 重新取樣 50 Hz -> 取 Magnitude（與手機朝向無關）-> 視窗切分 -> 丟掉含資料中斷的視窗 -> 52 個特徵（加速度與陀螺儀各 26 個）"""
import numpy as np, pandas as pd
from scipy import signal
from common import FS, MAX_GAP, TRIM_SEC, OVERLAP

def mag_resampled(df, trim=TRIM_SEC, min_len=100):
    df = df.sort_values("seconds_elapsed")
    t = df["seconds_elapsed"].values
    m = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2).values
    grid = np.arange(t[0] + trim, t[-1] - trim, 1 / FS)
    if len(grid) < min_len: return None, None
    out = np.interp(grid, t, m)
    idx = np.searchsorted(t, grid).clip(1, len(t) - 1)
    valid = (t[idx] - t[idx - 1]) <= MAX_GAP
    return out, valid

def feats(x, prefix):
    mean = x.mean(); xc = x - mean
    f, P = signal.welch(xc, FS, nperseg=len(xc)); Pn = P / (P.sum() + 1e-12)
    d = dict(mean=mean, std=xc.std(), max=x.max(), min=x.min(), rms=np.sqrt((xc**2).mean()),
             skew=pd.Series(xc).skew(), kurt=pd.Series(xc).kurt(), zcr=((xc[:-1] * xc[1:]) < 0).mean(),
             energy=(xc**2).mean(), domf=f[Pn.argmax()], spec_ent=-(Pn * np.log(Pn + 1e-12)).sum(),
             p_low=Pn[f <= 3].sum(), p_mid=Pn[(f > 3) & (f <= 10)].sum(), p_high=Pn[f > 10].sum())
    # 較細的頻帶能量比例（引擎／軌道振動的差異常出現在特定頻率）、頻譜重心、變化率（jerk）、自相關峰值
    edges = [0, 1, 2, 3, 5, 8, 12, 18, 25.1]
    for i in range(len(edges) - 1):
        d[f"b{i}"] = Pn[(f >= edges[i]) & (f < edges[i + 1])].sum()
    d["centroid"] = (f * Pn).sum()
    dx = np.diff(x) * FS; d["jerk_std"] = dx.std(); d["jerk_absmean"] = np.abs(dx).mean()
    ac = np.correlate(xc, xc, "full")[len(xc) - 1:]; ac = ac / (ac[0] + 1e-12); lo, hi = int(FS * 0.04), int(FS * 0.5)
    d["acorr_peak"] = ac[lo:hi].max() if len(ac) > hi else 0.0
    return {f"{prefix}_{k}": v for k, v in d.items()}

def window_features(acc_df, gyr_df, win_sec=2.0, trim=TRIM_SEC):
    """回傳 (DataFrame[每列一個視窗: t + 28 特徵], 總視窗數)。資料太短回傳 (None, 0)。"""
    ma, va = mag_resampled(acc_df, trim, int(win_sec * FS)); mg, vg = mag_resampled(gyr_df, trim, int(win_sec * FS))
    if ma is None or mg is None: return None, 0
    win = int(win_sec * FS); step = int(win * (1 - OVERLAP))
    n = min(len(ma), len(mg)); ma, mg, v = ma[:n], mg[:n], va[:n] & vg[:n]
    rows, total = [], 0
    for s in range(0, n - win + 1, step):
        total += 1
        if not v[s:s + win].all(): continue
        r = {"t": s / FS}; r.update(feats(ma[s:s + win], "acc")); r.update(feats(mg[s:s + win], "gyr")); rows.append(r)
    return pd.DataFrame(rows), total

def feature_columns(df):
    return [c for c in df.columns if c.startswith("acc_") or c.startswith("gyr_")]

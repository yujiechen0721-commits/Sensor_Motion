"""交通工具辨識網站（後端）。
啟動：在專案根目錄執行  python web/app.py   再開 http://127.0.0.1:5000
上傳 Sensor Logger 匯出的 zip（或 Accelerometer.csv + Gyroscope.csv）即可辨識；
"""
import os, sys, shutil, tempfile, zipfile, traceback
import numpy as np, pandas as pd, joblib
from flask import Flask, request, jsonify, render_template

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import *
from features import window_features, mag_resampled
from ml import predict_full, smooth_proba, CLASSES

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(OUT_DIR, "final_model.joblib"))
MAX_MB = 300
import live as _live
app = Flask(__name__); app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
APP_VERSION = "v2（訓練/測試分開版）"
_b = None
def bundle():
    """載入模型。若這台電腦的 scikit-learn 版本和存模型時不同（模型檔可能讀不對），就用訓練集特徵在本機重新訓練一次。"""
    global _b
    if _b is not None: return _b
    import sklearn, json
    b = joblib.load(MODEL_PATH)
    feat_ok = os.path.exists(os.path.join(OUT_DIR, f"features_w{b['win_sec']}.csv"))
    if b.get("sklearn_version") != sklearn.__version__ and feat_ok:
        print(f"[提醒] 模型是用 scikit-learn {b.get('sklearn_version')} 存的，這台電腦是 {sklearn.__version__}，改用本機重新訓練（約 1 分鐘）…")
        from ml import make_model, prep
        sel = json.load(open(os.path.join(OUT_DIR, "selection.json")))
        D = pd.read_csv(os.path.join(OUT_DIR, f"features_w{sel['win_sec']}.csv")); cols = b["cols"]
        D = prep(D, cols, sel["normalize"]); D = D[D["split"] == "train"]
        b = dict(b, model=make_model(sel["model"]).fit(D[cols].values, D["cls"].values), sklearn_version=sklearn.__version__)
    if b.get("sklearn_version") != sklearn.__version__: print(f"[警告] 模型存檔版本 {b.get('sklearn_version')} 與目前 scikit-learn {sklearn.__version__} 不同，結果可能不正確。")
    _b = b; return _b

def safe_extract(zf, dest):
    base = os.path.realpath(dest)
    for m in zf.infolist():
        t = os.path.realpath(os.path.join(dest, m.filename))
        if not (t == base or t.startswith(base + os.sep)): raise ValueError("壓縮檔內含不安全的路徑")
    zf.extractall(dest)

def find_trips(root):
    out = []
    for dp, _, fs in os.walk(root):
        low = {f.lower(): f for f in fs}
        if "accelerometer.csv" in low and "gyroscope.csv" in low: out.append((dp, low["accelerometer.csv"], low["gyroscope.csv"]))
    return out

MODEL_LABEL = {"HistGradientBoosting": "HGB 梯度提升", "RandomForest": "隨機森林 RF", "SVM": "支援向量機 SVM", "kNN": "k 近鄰 kNN", "ExtraTrees": "ExtraTrees"}

def downsample_sensor(df, n=700):
    t = df["seconds_elapsed"].values - df["seconds_elapsed"].values[0]; step = max(1, len(df) // n); d = df.iloc[::step]
    mag = np.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
    return dict(t=[round(float(v), 2) for v in t[::step]], x=[round(float(v), 3) for v in d["x"]], y=[round(float(v), 3) for v in d["y"]], z=[round(float(v), 3) for v in d["z"]], m=[round(float(v), 3) for v in mag])

def location_summary(loc, n=400):
    if loc is None or not {"latitude", "longitude"} <= set(loc.columns): return None
    loc = loc.dropna(subset=["latitude", "longitude"])
    if len(loc) < 2: return None
    step = max(1, len(loc) // n); d = loc.iloc[::step]
    out = dict(lat=[round(float(v), 6) for v in d["latitude"]], lon=[round(float(v), 6) for v in d["longitude"]])
    if "speed" in d.columns: out["speed_kmh"] = [round(float(v) * 3.6, 1) if v == v and v >= 0 else None for v in d["speed"]]
    if "seconds_elapsed" in d.columns: out["t"] = [round(float(v), 1) for v in d["seconds_elapsed"] - loc["seconds_elapsed"].iloc[0]]
    return out

def analyze(acc, gyr, loc=None):
    b = bundle(); model, cols, win = b["model"], b["cols"], b["win_sec"]
    W, _ = window_features(acc, gyr, win)
    min_w = max(10, int(20 / win))
    if W is None or len(W) < min_w:
        return {"error": f"有效資料不足（頭尾各會裁掉 {TRIM_SEC} 秒，需要至少約 {TRIM_SEC*2+40} 秒的錄製，且不能有太多資料中斷）"}
    t = W["t"].values; X = W[cols].copy()
    if b["normalize"]:
        X["trip"] = "t"; X = normalize_per_trip(X, cols)
    K = b.get("smooth_k", 1); classes = CLASSES
    P = smooth_proba(predict_full(model, X[cols].values), np.zeros(len(X)), K)
    mean = P.mean(0); best = classes[int(mean.argmax())]; per_win = [classes[i] for i in P.argmax(1)]
    compare = []
    for name, m in (b.get("models") or {}).items():            # 多模型比較：同一份資料，各模型各自判斷
        pm = smooth_proba(predict_full(m, X[cols].values), np.zeros(len(X)), K); mm = pm.mean(0)
        compare.append(dict(name=MODEL_LABEL.get(name, name), main=(name == b.get("model_name")), pred=classes[int(mm.argmax())], pred_name=CLASS_NAMES[classes[int(mm.argmax())]],
                            proba={c: round(float(p), 3) for c, p in zip(classes, mm)}))
    bins = {}
    for tt, c in zip(t, per_win): bins.setdefault(int(tt // 10), []).append(c)
    timeline = [{"t": k * 10, "cls": max(set(v), key=v.count)} for k, v in sorted(bins.items())]
    ma, _ = mag_resampled(acc); step = max(1, len(ma) // 400)
    minutes = round((acc["seconds_elapsed"].max() - acc["seconds_elapsed"].min()) / 60, 1)
    return {"pred": best, "pred_name": CLASS_NAMES[best], "proba": {c: round(float(p), 4) for c, p in zip(classes, mean)},
            "names": CLASS_NAMES, "windows": len(W), "minutes": minutes, "confidence": round(float(mean.max()), 3),
            "warning": ("錄製偏短（少於約 3 分鐘），結果較不穩定" if minutes < 3 else None),
            "timeline": timeline, "signal": {"acc": [round(float(x), 3) for x in ma[::step]]},
            "models": compare, "sensor": {"acc": downsample_sensor(acc.sort_values("seconds_elapsed")), "gyro": downsample_sensor(gyr.sort_values("seconds_elapsed"))},
            "location": location_summary(loc), "window_sec": b["win_sec"], "smooth_sec": b.get("smooth_sec")}

def read_csv_any(folder, name):
    return pd.read_csv(os.path.join(folder, name), usecols=["seconds_elapsed", "x", "y", "z"])

def read_location(folder):
    for f in os.listdir(folder):
        if f.lower() == "location.csv":
            try: return pd.read_csv(os.path.join(folder, f))
            except Exception: return None
    return None

_live.init(bundle); app.register_blueprint(_live.bp)

@app.route("/about")
def about():
    import csv
    def rd(n):
        p = os.path.join(OUT_DIR, n)
        return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if os.path.exists(p) else []
    return render_template("about.html", test=(rd("最終測試結果.csv") or [{}])[0], rt=rd("即時辨識評估.csv"), cv=rd("交叉驗證_平滑比較.csv"))

@app.route("/outputs/<path:name>")
def outputs_file(name):
    from flask import send_from_directory
    return send_from_directory(OUT_DIR, name)

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/health")
def health():
    import sklearn
    return jsonify(ok=os.path.exists(MODEL_PATH), version=APP_VERSION, model=os.path.abspath(MODEL_PATH), sklearn=sklearn.__version__)

@app.route("/api/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    if not files: return jsonify(error="沒有收到檔案"), 400
    if not os.path.exists(MODEL_PATH): return jsonify(error="找不到模型檔 outputs/final_model.joblib，請先執行 python run_all.py"), 500
    tmp = tempfile.mkdtemp(prefix="vehicle_")
    try:
        for i, f in enumerate(files):
            name = os.path.basename(f.filename or f"f{i}")
            if name.lower().endswith(".zip"):
                z = os.path.join(tmp, f"up{i}.zip"); f.save(z); d = os.path.join(tmp, f"z{i}"); os.makedirs(d)
                with zipfile.ZipFile(z) as zf: safe_extract(zf, d)
            elif name.lower().endswith(".csv"):
                d = os.path.join(tmp, "csv"); os.makedirs(d, exist_ok=True); f.save(os.path.join(d, name))
            else: return jsonify(error=f"不支援的檔案類型：{name}（請上傳 .zip 或 .csv）"), 400
        trips = find_trips(tmp)
        if not trips: return jsonify(error="找不到 Accelerometer.csv 和 Gyroscope.csv，請確認匯出時有勾選這兩個感測器"), 400
        results = []
        for folder, a, g in trips:
            try: r = analyze(read_csv_any(folder, a), read_csv_any(folder, g), read_location(folder))
            except Exception as e: r = {"error": f"處理失敗：{e}"}
            r["name"] = os.path.basename(folder) if folder != tmp else "錄製"; results.append(r)
        return jsonify(results=results)
    except zipfile.BadZipFile: return jsonify(error="zip 檔無法解壓縮"), 400
    except Exception as e:
        traceback.print_exc(); return jsonify(error=f"伺服器錯誤：{e}"), 500
    finally: shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)); host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n網站已啟動：http://127.0.0.1:{port}  （按 Ctrl+C 結束）")
    if host == "0.0.0.0":
        print("區域網路模式：手機可連到 " + "、".join(f"http://{ip}:{port}" for ip in _live.lan_ips()))
    print()
    if os.environ.get("HTTPS") == "1":     # 手機瀏覽器讀感測器需要 HTTPS；用自簽憑證（手機會跳安全警告，選「繼續前往」）
        print("HTTPS 模式：手機請開 " + "、".join(f"https://{ip}:{port}/phone" for ip in (_live.lan_ips() or ["電腦IP"])) + "\n")
        app.run(host=host, port=port, debug=False, threaded=True, ssl_context="adhoc")
    else:
        app.run(host=host, port=port, debug=False, threaded=True)

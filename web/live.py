"""即時辨識：手機上的 Sensor Logger 用「HTTP Push」把感測資料持續送到這個網站，網站邊收邊辨識。
另外提供「示範回放」：把一筆錄製當成即時資料餵進來（沒有手機也能展示）。"""
import time, socket, threading
import numpy as np, pandas as pd
from flask import Blueprint, request, jsonify, render_template
from common import *
from features import mag_resampled, feats
from ml import predict_full, CLASSES

bp = Blueprint("live", __name__)
LOCK = threading.RLock()
SESS = {}                 # session id -> 狀態
HEAD_SKIP = TRIM_SEC      # 開始後前 30 秒不辨識（和訓練時裁掉頭尾一致，避開上下車動作）
MAX_HIST = 600
_get_bundle = None
def init(bundle_fn):
    global _get_bundle; _get_bundle = bundle_fn

def _sec(t):
    t = float(t)
    if t > 1e17: return t / 1e9      # 奈秒
    if t > 1e14: return t / 1e6      # 微秒
    if t > 1e11: return t / 1e3      # 毫秒
    return t

def new_session(name, source):
    return dict(name=name, source=source, acc=[], gyr=[], t0=None, next_t=None, probas=[], hist=[], last_rx=time.time(),
                n_acc=0, n_gyr=0, acc_mag=[], device=None)

def ingest(sess, name, t, x, y, z):
    row = (_sec(t), float(x), float(y), float(z))
    if sess["t0"] is None: sess["t0"] = row[0]
    if name == "accelerometer": sess["acc"].append(row); sess["n_acc"] += 1
    elif name == "gyroscope": sess["gyr"].append(row); sess["n_gyr"] += 1
    sess["last_rx"] = time.time()

def _df(rows, t0):
    a = np.array(rows, dtype=float); a = a[np.argsort(a[:, 0])]
    return pd.DataFrame(dict(seconds_elapsed=a[:, 0] - t0, x=a[:, 1], y=a[:, 2], z=a[:, 3]))

def advance(s):
    """把已收到、還沒處理的資料切成視窗 -> 特徵 -> 模型 -> 因果式時間平滑（只看過去的視窗）。"""
    b = _get_bundle(); win = b["win_sec"]; step_s = win * (1 - OVERLAP); K = b.get("smooth_k", 1)
    if len(s["acc"]) < 100 or len(s["gyr"]) < 100: return
    t_end = min(s["acc"][-1][0], s["gyr"][-1][0])
    if s["next_t"] is None: s["next_t"] = s["t0"] + HEAD_SKIP
    if s["next_t"] + win > t_end: return
    lo = s["next_t"] - 0.2
    A = [r for r in s["acc"] if r[0] >= lo]; G = [r for r in s["gyr"] if r[0] >= lo]
    if len(A) < 50 or len(G) < 50: return
    dA, dG = _df(A, s["t0"]), _df(G, s["t0"])
    ma, va = mag_resampled(dA, 0, int(win * FS)); mg, vg = mag_resampled(dG, 0, int(win * FS))
    if ma is None or mg is None: return
    g0 = max(dA["seconds_elapsed"].iloc[0], dG["seconds_elapsed"].iloc[0])
    n = min(len(ma), len(mg)); ma, mg, v = ma[:n], mg[:n], va[:n] & vg[:n]
    W, step = int(win * FS), int(win * FS * (1 - OVERLAP)); starts = list(range(0, n - W + 1, step))
    if not starts: return
    rows, ends = [], []
    for st in starts:
        if v[st:st + W].all():
            r = {}; r.update(feats(ma[st:st + W], "acc")); r.update(feats(mg[st:st + W], "gyr")); rows.append(r); ends.append(g0 + (st + W) / FS)
    if rows:
        X = pd.DataFrame(rows)[b["cols"]]
        P = predict_full(b["model"], X.values)
        for p, te in zip(P, ends):
            s["probas"].append(p); sm = np.mean(s["probas"][-K:], axis=0)
            s["hist"].append(dict(t=round(te, 1), cls=CLASSES[int(sm.argmax())], sm=[round(float(q), 4) for q in sm]))
        s["probas"] = s["probas"][-K:]; s["hist"] = s["hist"][-MAX_HIST:]
        s["acc_mag"] = [round(float(np.sqrt((np.array(A)[:, 1:] ** 2).sum(1)).mean()), 3)]
    s["next_t"] = s["t0"] + dA["seconds_elapsed"].iloc[0] + len(starts) * step_s
    cut = s["next_t"] - 1.0                                   # 已處理的舊資料丟掉，記憶體不會一直長
    s["acc"] = [r for r in s["acc"] if r[0] >= cut]; s["gyr"] = [r for r in s["gyr"] if r[0] >= cut]

def lan_ips():
    ips = set()
    try:
        for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET): ips.add(ai[4][0])
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ips.add(s.getsockname()[0]); s.close()
    except Exception: pass
    return sorted(i for i in ips if not i.startswith("127."))

@bp.route("/live")
def live_page(): return render_template("live.html")

@bp.route("/data", methods=["POST"])
def receive():
    """Sensor Logger 的 HTTP Push 會 POST 到這裡。"""
    j = request.get_json(force=True, silent=True)
    if not isinstance(j, dict): return jsonify(error="bad json"), 400
    sid = str(j.get("sessionId") or j.get("deviceId") or "phone")
    with LOCK:
        s = SESS.get(sid)
        if s is None: s = SESS[sid] = new_session(sid, "phone")
        for it in j.get("payload", []):
            name = str(it.get("name", "")).lower(); v = it.get("values", {})
            if name in ("accelerometer", "gyroscope") and all(k in v for k in "xyz"): ingest(s, name, it.get("time"), v["x"], v["y"], v["z"])
        if j.get("deviceId"): s["device"] = str(j["deviceId"])[:12]
    return "ok"

@bp.route("/phone")
def phone_page(): return render_template("phone.html")

@bp.route("/api/live/push", methods=["POST"])
def push():
    """手機瀏覽器（/phone 頁面）每秒送來新的感測資料：{sid, acc:[[秒,x,y,z]...], gyro:[...]}，單位 m/s² 與 rad/s。"""
    j = request.get_json(silent=True)
    if not isinstance(j, dict): return jsonify(error="bad json"), 400
    sid = str(j.get("sid") or "")[:40]
    if not sid: return jsonify(error="missing sid"), 400
    acc, gyr = j.get("acc") or [], j.get("gyro") or []
    if not isinstance(acc, list) or not isinstance(gyr, list) or len(acc) > 6000 or len(gyr) > 6000: return jsonify(error="bad payload"), 400
    with LOCK:
        s = SESS.get(sid)
        if s is None: s = SESS[sid] = new_session(sid, "browser")
        try:
            for name, rows in (("accelerometer", acc), ("gyroscope", gyr)):
                for r in rows:
                    if len(r) == 4 and all(isinstance(v, (int, float)) for v in r): ingest(s, name, r[0], r[1], r[2], r[3])
        except Exception as e: return jsonify(error=str(e)), 400
        s["device"] = str(j.get("device") or "手機瀏覽器")[:30]
    return jsonify(ok=True)

@bp.route("/api/live/info")
def info():
    return jsonify(ips=lan_ips(), port=request.host.split(":")[-1] if ":" in request.host else "80")

@bp.route("/api/live/state")
def state():
    with LOCK:
        sid = request.args.get("sid")
        if sid:
            if sid not in SESS: return jsonify(status="waiting")
            s = SESS[sid]
        else:
            if not SESS: return jsonify(status="waiting")
            s = max(SESS.values(), key=lambda x: x["last_rx"])
        try: advance(s)
        except Exception as e: return jsonify(status="error", error=str(e))
        age = time.time() - s["last_rx"]
        t_rec = (max(s["acc"][-1][0] if s["acc"] else 0, s["gyr"][-1][0] if s["gyr"] else 0) - s["t0"]) if s["t0"] else 0
        out = dict(source=s["source"], name=s["name"], rx_age=round(age, 1), seconds=round(t_rec, 1), n_acc=s["n_acc"], n_gyr=s["n_gyr"], device=s["device"],
                   acc_mag=(s["acc_mag"] or [None])[0], names=CLASS_NAMES, head_skip=HEAD_SKIP)
        if len(s["hist"]) < 2:
            out["status"] = "warming" if t_rec < HEAD_SKIP + 15 else "collecting"
            return jsonify(out)
        last = s["hist"][-1]
        out.update(status="ok", pred=last["cls"], proba=dict(zip(CLASSES, last["sm"])), timeline=[h["cls"] for h in s["hist"][-80:]], windows=len(s["hist"]))
        return jsonify(out)

@bp.route("/api/live/reset", methods=["POST"])
def reset():
    sid = (request.get_json(silent=True) or {}).get("sid")
    with LOCK:
        if sid: SESS.pop(sid, None)
        else: SESS.clear(); _replay_stop.set()
    return jsonify(ok=True)

# ---------- 示範回放 ----------
_replay_stop = threading.Event()
@bp.route("/api/live/demo/list")
def demo_list():
    s = pd.read_csv(SPLIT_CSV); s = s[s["split"] == "test"]
    s = s[[os.path.isdir(os.path.join(DATA_DIR, t)) for t in s["trip"]]]     # 部署版沒有原始資料，就沒有示範回放
    return jsonify(items=[dict(id=r.trip, label=f"{r.trip}（{CLASS_NAMES[r.cls]}・{POS_NAMES[r.position]}）") for r in s.itertuples()])

@bp.route("/api/live/demo/start", methods=["POST"])
def demo_start():
    uid = (request.get_json(silent=True) or {}).get("id", ""); speed = float((request.get_json(silent=True) or {}).get("speed", 10))
    s = pd.read_csv(SPLIT_CSV)
    if s[(s["trip"] == uid) & (s["split"] == "test")].empty: return jsonify(error="找不到這筆示範資料"), 404
    speed = min(max(speed, 1), 30)
    d = os.path.join(DATA_DIR, uid); A = read_sensor(d, "Accelerometer").values; G = read_sensor(d, "Gyroscope").values
    _replay_stop.set(); time.sleep(0.3); _replay_stop.clear()
    with LOCK:
        SESS.clear(); sess = SESS["demo"] = new_session("demo", "demo"); sess["device"] = "示範回放"
    def run():
        base = 1.7e9; ia = ig = 0; t_start = time.time()
        while not _replay_stop.is_set() and (ia < len(A) or ig < len(G)):
            now = (time.time() - t_start) * speed
            with LOCK:
                while ia < len(A) and A[ia, 0] <= now: ingest(sess, "accelerometer", base + A[ia, 0], *A[ia, 1:4]); ia += 1
                while ig < len(G) and G[ig, 0] <= now: ingest(sess, "gyroscope", base + G[ig, 0], *G[ig, 1:4]); ig += 1
            time.sleep(0.1)
    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True)

@bp.route("/api/live/demo/stop", methods=["POST"])
def demo_stop():
    _replay_stop.set(); return jsonify(ok=True)

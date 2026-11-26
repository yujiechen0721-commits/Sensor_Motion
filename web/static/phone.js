const $ = (id) => document.getElementById(id);
const COLORS = { MRT: "#5c4033", bus: "#8a5a3c", car: "#a8582a" }, NAMES = { MRT: "捷運", bus: "公車", car: "汽車" };
const RAD = Math.PI / 180, LIMIT = 200000;
const sid = "web-" + Math.random().toString(36).slice(2, 10);
let running = false, pending = { acc: [], gyro: [] }, rec = [], timer = null, poll = null, wake = null, nAcc = 0, nGyr = 0, lastEv = 0, recent = [], accSource = "none";
const el = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };
const num = (v) => typeof v === "number" && Number.isFinite(v);
function envCheck() {
  const secure = window.isSecureContext, ok = "DeviceMotionEvent" in window;
  $("env").textContent = (secure ? "安全連線 ✓" : "非安全連線：需要 HTTPS（請用 https:// 開啟）") + "　|　" + (ok ? "支援動作感測器" : "此瀏覽器不支援 DeviceMotionEvent");
  $("startBtn").disabled = !(secure && ok);
}
function onMotion(e) {
  if (!running) return;
  const t = (performance.timeOrigin + e.timeStamp) / 1000, now = performance.now(); lastEv = now;
  recent.push(now); while (recent.length && now - recent[0] > 5000) recent.shift();
  const a = e.acceleration, r = e.rotationRate; let acc = null, gyr = null;
  if (a && num(a.x) && num(a.y) && num(a.z)) { acc = [a.x, a.y, a.z]; accSource = "linear"; }
  else if (e.accelerationIncludingGravity) accSource = "with_gravity";      // 含重力的資料和訓練資料不同，不送進模型
  if (r && num(r.alpha) && num(r.beta) && num(r.gamma)) gyr = [r.beta * RAD, r.gamma * RAD, r.alpha * RAD];
  if (acc) { pending.acc.push([t, ...acc]); nAcc++; }
  if (gyr) { pending.gyro.push([t, ...gyr]); nGyr++; }
  if (rec.length < LIMIT) rec.push([t, ...(acc || ["", "", ""]), ...(gyr || ["", "", ""])]);
  else stop("已達記錄上限，自動停止。");
}
async function flush() {
  if (!pending.acc.length && !pending.gyro.length) return;
  const body = { sid, device: navigator.userAgent.slice(0, 30), acc: pending.acc.splice(0, 6000), gyro: pending.gyro.splice(0, 6000) };
  try { await fetch("/api/live/push", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
  catch (e) { $("msg").className = "status err"; $("msg").textContent = "無法連到電腦上的網站：" + e.message; }
}
async function refresh() {
  try {
    const d = await (await fetch("/api/live/state?sid=" + sid)).json(); const hz = recent.length > 1 ? ((recent.length - 1) * 1000 / (recent.at(-1) - recent[0])).toFixed(0) : "—";
    $("liveStatus").textContent = `已錄 ${d.seconds ? Math.round(d.seconds) : 0} 秒　取樣約 ${hz} Hz　加速度來源：${accSource === "linear" ? "線性（正確）" : accSource === "with_gravity" ? "含重力（不可用）" : "尚未收到"}`;
    if (d.status === "warming") $("liveMeta").textContent = `準備中：前 ${d.head_skip} 秒不辨識`;
    else if (d.status === "collecting") $("liveMeta").textContent = "收集資料中，快好了…";
    else if (d.status === "ok") {
      const b = $("liveBadge"); b.textContent = NAMES[d.pred]; b.style.background = COLORS[d.pred];
      $("liveMeta").textContent = `目前判斷：${NAMES[d.pred]}（信心 ${(d.proba[d.pred] * 100).toFixed(0)}%）`;
      const bars = $("liveBars"); bars.replaceChildren();
      Object.entries(d.proba).sort((a, b) => b[1] - a[1]).forEach(([c, p]) => { const row = el("div", "bar"); row.appendChild(el("span", null, NAMES[c]));
        const tr = el("div", "track"), f = el("div", "fill"); f.style.width = (p * 100).toFixed(1) + "%"; f.style.background = COLORS[c]; tr.appendChild(f); row.appendChild(tr); row.appendChild(el("span", null, (p * 100).toFixed(0) + "%")); bars.appendChild(row); });
      const tl = $("liveTl"); tl.replaceChildren(); d.timeline.forEach((c) => { const i = el("i"); i.style.background = COLORS[c]; tl.appendChild(i); });
    }
  } catch (e) {}
}
async function start() {
  try {
    if (typeof DeviceMotionEvent.requestPermission === "function") { const p = await DeviceMotionEvent.requestPermission(); if (p !== "granted") { $("msg").textContent = "未取得動作感測器權限（iPhone：設定 > Safari > 動作與方向存取）。"; return; } }
    try { if (navigator.wakeLock) wake = await navigator.wakeLock.request("screen"); } catch (_) {}
    await fetch("/api/live/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sid }) });
    rec = []; pending = { acc: [], gyro: [] }; nAcc = nGyr = 0; recent = []; accSource = "none"; running = true;
    window.addEventListener("devicemotion", onMotion);
    timer = setInterval(flush, 1000); poll = setInterval(refresh, 2000);
    $("startBtn").disabled = true; $("stopBtn").disabled = false; $("csvBtn").disabled = true; $("msg").className = "status"; $("msg").textContent = "讀取中。請保持螢幕開啟、頁面留在前景。";
    setTimeout(() => { if (running && !lastEv) { $("msg").className = "status err"; $("msg").textContent = "3 秒內沒有收到感測器事件，請確認權限與瀏覽器支援。"; } }, 3000);
  } catch (e) { $("msg").className = "status err"; $("msg").textContent = "無法啟動：" + e.message; }
}
function stop(text) {
  running = false; window.removeEventListener("devicemotion", onMotion); clearInterval(timer); clearInterval(poll); flush(); refresh();
  try { wake && wake.release(); } catch (_) {} wake = null;
  $("startBtn").disabled = false; $("stopBtn").disabled = true; $("csvBtn").disabled = !rec.length; $("msg").className = "status"; $("msg").textContent = text || `已停止，共 ${rec.length} 筆。`;
}
function csv() {
  const lines = ["epoch_s,acc_x_m_s2,acc_y_m_s2,acc_z_m_s2,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s"].concat(rec.map((r) => r.join(",")));
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" }));
  a.download = "phone_live_" + new Date().toISOString().replace(/[:.]/g, "-") + ".csv"; document.body.appendChild(a); a.click(); a.remove();
}
$("startBtn").addEventListener("click", start); $("stopBtn").addEventListener("click", () => stop()); $("csvBtn").addEventListener("click", csv);
document.addEventListener("visibilitychange", () => { if (document.hidden && running) stop("頁面進入背景，已自動停止（避免資料不連續）。"); });
envCheck();

const $ = (id) => document.getElementById(id);
const COLORS = { MRT: "#5c4033", bus: "#8a5a3c", car: "#a8582a" };
const NAMES = { MRT: "捷運", bus: "公車", car: "汽車" };
let files = [];

function setFiles(list) {
  files = Array.from(list);
  $("picked").textContent = files.length ? "已選擇：" + files.map((f) => f.name).join("、") : "";
  $("go").disabled = !files.length;
}
$("file").addEventListener("change", (e) => setFiles(e.target.files));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFiles(e.dataTransfer.files));

function status(msg, err) { const s = $("status"); s.textContent = msg; s.className = "status" + (err ? " err" : ""); }
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

$("go").addEventListener("click", async () => {
  const fd = new FormData(); files.forEach((f) => fd.append("files", f));
  $("go").disabled = true; $("results").replaceChildren(); status("上傳並分析中，較長的錄製可能需要一分鐘…");
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { status(data.error || "發生錯誤", true); return; }
    status(`完成，共 ${data.results.length} 筆錄製`);
    data.results.forEach((r) => $("results").appendChild(render(r)));
  } catch (e) { status("無法連線到伺服器：" + e.message, true); }
  finally { $("go").disabled = !files.length; }
});

function render(r) {
  const card = el("section", "card result");
  card.appendChild(el("h2", null, r.name));
  if (r.error) { card.appendChild(el("p", "warn", r.error)); return card; }
  const v = el("div", "verdict");
  const b = el("div", "badge", r.pred_name); b.style.background = COLORS[r.pred]; v.appendChild(b);
  const m = el("div", "meta");
  m.appendChild(el("div", null, `判定：${r.pred_name}（平均信心 ${(r.confidence * 100).toFixed(0)}%）`));
  m.appendChild(el("div", null, `錄製長度約 ${r.minutes} 分鐘，${r.windows} 個有效視窗`));
  v.appendChild(m); card.appendChild(v);
  if (r.warning) card.appendChild(el("p", "warn", r.warning));

  card.appendChild(el("h3", null, "各類別機率"));
  Object.entries(r.proba).sort((a, b) => b[1] - a[1]).forEach(([c, p]) => {
    const row = el("div", "bar"); row.appendChild(el("span", null, NAMES[c]));
    const t = el("div", "track"), f = el("div", "fill"); f.style.width = (p * 100).toFixed(1) + "%"; f.style.background = COLORS[c];
    t.appendChild(f); row.appendChild(t); row.appendChild(el("span", null, (p * 100).toFixed(0) + "%")); card.appendChild(row);
  });

  card.appendChild(el("h3", null, "時間軸（每格 10 秒，顏色為該段判定）"));
  const tl = el("div", "timeline");
  r.timeline.forEach((s) => { const i = el("i"); i.style.background = COLORS[s.cls]; i.title = `${s.t} 秒起：${NAMES[s.cls]}`; tl.appendChild(i); });
  card.appendChild(tl);
  const lg = el("div", "legend");
  Object.keys(NAMES).forEach((c) => { const s = el("span"); const sw = el("b"); sw.style.background = COLORS[c]; s.appendChild(sw); s.appendChild(document.createTextNode(NAMES[c])); lg.appendChild(s); });
  card.appendChild(lg);

  if (r.models && r.models.length) {
    card.appendChild(el("h3", null, "多模型比較（同一筆資料，各模型各自判斷）"));
    const tb = el("table", "tbl"), hd = el("tr"); ["模型", "判定", "捷運", "公車", "汽車"].forEach((x) => hd.appendChild(el("th", null, x))); tb.appendChild(hd);
    r.models.forEach((m) => { const tr = el("tr"); tr.appendChild(el("td", null, m.name + (m.main ? "（主模型）" : ""))); const p = el("td", null, m.pred_name); p.style.fontWeight = "700"; p.style.color = COLORS[m.pred]; tr.appendChild(p);
      ["MRT", "bus", "car"].forEach((c) => tr.appendChild(el("td", null, (m.proba[c] * 100).toFixed(0) + "%"))); tb.appendChild(tr); });
    card.appendChild(tb);
  }
  const mk = (title) => { card.appendChild(el("h3", null, title)); const d = el("div", "plot"); card.appendChild(d); return d; };
  if (r.sensor && typeof Plotly !== "undefined") {
    const a = mk("加速度（X／Y／Z／Magnitude，錄製時間為秒）"), g = mk("陀螺儀（X／Y／Z／Magnitude）");
    setTimeout(() => { plot(a, r.sensor.acc, "m/s²"); plot(g, r.sensor.gyro, "rad/s"); }, 0);
  } else { card.appendChild(el("h3", null, "加速度強度（Magnitude）")); const cv = el("canvas"); card.appendChild(cv); requestAnimationFrame(() => drawSignal(cv, r.signal)); }
  if (r.location) {
    const mp = mk("行進軌跡（GPS）"); mp.classList.add("map"); setTimeout(() => drawMap(mp, r.location), 0);
    if (r.location.speed_kmh && typeof Plotly !== "undefined") { const sp = mk("GPS 速度（km/h）"); setTimeout(() => plotSpeed(sp, r.location), 0); }
  } else if (r.location === null) { card.appendChild(el("p", "meta", "這筆資料沒有 GPS（Location.csv），所以不顯示地圖。")); }
  return card;
}

function drawSignal(cv, s) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const ys = s.acc, lo = Math.min(...ys), hi = Math.max(...ys), pad = 8;
  g.strokeStyle = "#8a5a3c"; g.lineWidth = 1; g.beginPath();
  ys.forEach((y, i) => { const px = (i / (ys.length - 1)) * (w - 2 * pad) + pad, py = h - pad - ((y - lo) / (hi - lo || 1)) * (h - 2 * pad); i ? g.lineTo(px, py) : g.moveTo(px, py); });
  g.stroke();
}

const PLOT_FONT = { family: "Noto Serif TC, serif", color: "#3e2c20", size: 12 };
const LAYOUT = (unit) => ({ margin: { l: 50, r: 10, t: 10, b: 40 }, height: 260, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#fffaf0", font: PLOT_FONT, legend: { orientation: "h", y: 1.18 }, xaxis: { title: "秒", gridcolor: "#eadfc8" }, yaxis: { title: unit, gridcolor: "#eadfc8" } });
function plot(div, s, unit) {
  const T = [["X", s.x, "#b58863"], ["Y", s.y, "#7a6a4f"], ["Z", s.z, "#a8582a"], ["Magnitude", s.m, "#3e2c20"]].map(([n, y, c]) => ({ x: s.t, y, name: n, mode: "lines", line: { width: n === "Magnitude" ? 1.6 : 1, color: c } }));
  Plotly.newPlot(div, T, LAYOUT(unit), { responsive: true, displaylogo: false });
}
function plotSpeed(div, l) {
  Plotly.newPlot(div, [{ x: l.t || l.speed_kmh.map((_, i) => i), y: l.speed_kmh, mode: "lines", line: { color: "#8a5a3c", width: 1.8 }, name: "速度" }], LAYOUT("km/h"), { responsive: true, displaylogo: false });
}
function drawMap(div, l) {
  if (typeof L === "undefined") { div.textContent = "地圖需要網路載入（Leaflet／OpenStreetMap），目前無法顯示。"; return; }
  const pts = l.lat.map((la, i) => [la, l.lon[i]]), map = L.map(div); L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(map);
  const line = L.polyline(pts, { color: "#a8582a", weight: 4 }).addTo(map); L.circleMarker(pts[0], { radius: 7, color: "#3e2c20", fillColor: "#fff", fillOpacity: 1 }).bindTooltip("起點").addTo(map);
  L.circleMarker(pts[pts.length - 1], { radius: 7, color: "#3e2c20", fillColor: "#3e2c20", fillOpacity: 1 }).bindTooltip("終點").addTo(map); map.fitBounds(line.getBounds(), { padding: [20, 20] });
}

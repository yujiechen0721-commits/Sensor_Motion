const $ = (id) => document.getElementById(id);
const COLORS = { MRT: "#5c4033", bus: "#8a5a3c", car: "#a8582a" }, NAMES = { MRT: "捷運", bus: "公車", car: "汽車" };
fetch("/api/live/info").then((r) => r.json()).then((d) => { $("pushUrl").textContent = (d.ips.length ? d.ips : ["（電腦的 IP）"]).map((ip) => `http://${ip}:${d.port}/data`).join("　或　"); });
fetch("/api/live/demo/list").then((r) => r.json()).then((d) => { const s = $("demoSel"); s.replaceChildren(); d.items.forEach((x) => s.appendChild(new Option(x.label, x.id))); if (!d.items.length) { const sec = $("demoSel").closest("section"); if (sec) sec.style.display = "none"; } });
$("demoBtn").addEventListener("click", async () => {
  await fetch("/api/live/demo/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: $("demoSel").value, speed: $("demoSpeed").value }) });
});
$("resetBtn").addEventListener("click", async () => { await fetch("/api/live/reset", { method: "POST" }); });
const el = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };
async function tick() {
  try {
    const d = await (await fetch("/api/live/state")).json(); const st = $("liveStatus"); st.className = "status";
    if (d.status === "waiting") { st.textContent = "等待手機傳資料…（或使用下方示範回放）"; }
    else if (d.status === "error") { st.className = "status err"; st.textContent = "處理錯誤：" + d.error; }
    else {
      const src = d.source === "demo" ? "示範回放" : "手機連線中";
      const stale = d.source === "phone" && d.rx_age > 5 ? `（已 ${Math.round(d.rx_age)} 秒沒收到資料）` : "";
      st.textContent = `${src}${stale}　已錄 ${Math.round(d.seconds)} 秒　加速度 ${d.n_acc} 筆／陀螺儀 ${d.n_gyr} 筆`;
      if (d.status === "warming") $("liveMeta").textContent = `準備中：前 ${d.head_skip} 秒不辨識`;
      else if (d.status === "collecting") $("liveMeta").textContent = "收集資料中，快好了…";
      else {
        const b = $("liveBadge"); b.textContent = NAMES[d.pred]; b.style.background = COLORS[d.pred];
        $("liveMeta").textContent = `目前判斷：${NAMES[d.pred]}（信心 ${(d.proba[d.pred] * 100).toFixed(0)}%），已累積 ${d.windows} 個視窗`;
        const bars = $("liveBars"); bars.replaceChildren();
        Object.entries(d.proba).sort((a, b) => b[1] - a[1]).forEach(([c, p]) => {
          const row = el("div", "bar"); row.appendChild(el("span", null, NAMES[c]));
          const t = el("div", "track"), f = el("div", "fill"); f.style.width = (p * 100).toFixed(1) + "%"; f.style.background = COLORS[c]; t.appendChild(f);
          row.appendChild(t); row.appendChild(el("span", null, (p * 100).toFixed(0) + "%")); bars.appendChild(row); });
        const tl = $("liveTl"); tl.replaceChildren(); d.timeline.forEach((c) => { const i = el("i"); i.style.background = COLORS[c]; i.title = NAMES[c]; tl.appendChild(i); });
      }
    }
  } catch (e) { $("liveStatus").className = "status err"; $("liveStatus").textContent = "無法連線到網站：" + e.message; }
  setTimeout(tick, 2000);
}
tick();

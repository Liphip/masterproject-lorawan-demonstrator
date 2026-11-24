"use strict";
// Plain JS, no external libraries: the dashboard must work without internet.
const POLL_MS = 1500;
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
let STATE = null, CLASS_NAMES = {}, activeTab = "replay", recordingsLoaded = false;

const UNITS = {
  lse01_conductivity: ["conductivity", "µS/cm", 0], lse01_moisture: ["moisture", "%", 1],
  lse01_temperature: ["temp", "°C", 1], ltcfs_temperature: ["temp", "°C", 1],
  ws202_light_status: ["light", "", -1], ws202_pir_status: ["motion", "", -1],
};
const MODEL_KIND = {
  logreg: "linear · interpretable baseline",
  rf: "tree ensemble · nonlinear",
  mlp: "small neural network",
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
function ago(iso) {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return `${Math.max(0, Math.round(s))} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}
function phaseColor(label) {
  const l = String(label || "").toLowerCase();
  if (l.includes("umroet") || l.includes("curing")) return "var(--p0)";
  if (l.includes("1")) return "var(--p1)";
  if (l.includes("2")) return "var(--p2)";
  return "var(--none)";
}
const phaseName = (l) => (l == null ? "–" : CLASS_NAMES[l] || l);
function fmtVal(k, v) {
  const u = UNITS[k] || [k, "", 2];
  if (u[2] < 0) {
    const txt = k === "ws202_light_status" ? (v ? "light" : "dark") : (v ? "triggered" : "idle");
    return `<span class="val">${u[0]} <b>${txt}</b></span>`;
  }
  return `<span class="val">${u[0]} <b>${Number(v).toFixed(u[2])}</b> ${u[1]}</span>`;
}

// ── header + data path ─────────────────────────────────────────────────────
function renderChips(st) {
  const chips = [];
  const lu = st.live.last_uplink_at;
  const age = lu ? (Date.now() - new Date(lu).getTime()) / 1000 : Infinity;
  chips.push(`<span class="chip ${age < 600 ? "ok" : "warn"}">Gateway: ${lu ? "last uplink " + ago(lu) : "no uplink yet"}</span>`);
  const m = st.models;
  chips.push(m.ok ? `<span class="chip ok">${m.models.length} models loaded${m.quick ? " (QUICK TEST MODELS)" : ""}</span>`
                  : `<span class="chip err">Models: ${esc(m.error || "not loaded")}</span>`);
  if (m.version_warning) chips.push(`<span class="chip warn" title="${esc(m.version_warning)}">⚠ sklearn version mismatch</span>`);
  const mq = st.ingest.mqtt;
  chips.push(`<span class="chip">Ingest: HTTP ${esc(st.ingest.http_path)}${mq.enabled ? ` + MQTT ${mq.connected ? "✓" : "✗"}` : ""}</span>`);
  const r = st.replay.status;
  chips.push(r.running ? `<span class="chip warn">REPLAY running</span>` : `<span class="chip">Replay idle</span>`);
  $("#chips").innerHTML = chips.join("");
  const active = age < 120;
  document.querySelectorAll(".node").forEach((n) => n.classList.toggle("active", active));
}

// ── 1 · live sensors ───────────────────────────────────────────────────────
function renderDevices(live) {
  const rows = live.devices.map((d) => {
    const fresh = d.last_seen && (Date.now() - new Date(d.last_seen)) / 1000 < 30;
    const vals = d.configured
      ? Object.entries(d.values || {}).map(([k, v]) => fmtVal(k, v)).join("")
      : `<span class="muted">${esc(JSON.stringify(d.values || {})).slice(0, 120)}</span>`;
    const radio = d.rssi != null ? `<span class="val">${d.rssi} dBm</span>${d.snr != null ? `<br><span class="muted">SNR ${d.snr} dB</span>` : ""}` : "–";
    return `<tr class="${fresh ? "fresh" : ""} ${d.configured ? "" : "unconfigured"}">
      <td>${esc(d.name)}<br><span class="eui">${esc(d.type)} · ${esc(d.dev_eui)}</span>
        ${d.configured ? "" : `<br><span class="err-text">not in config</span>`}</td>
      <td>${esc(d.slot)}</td>
      <td>${vals || '<span class="muted">waiting…</span>'}
        ${d.last_error ? `<br><span class="err-text">${esc(d.last_error)}</span>` : ""}
        ${d.decode_method ? `<br><span class="muted">decoded by ${d.decode_method === "codec" ? "gateway codec" : "built-in decoder"}</span>` : ""}</td>
      <td>${radio}</td><td>${ago(d.last_seen)}</td><td>${d.uplinks}</td></tr>`;
  });
  $("#devices").innerHTML = rows.join("") ||
    `<tr><td colspan="6" class="muted">No devices configured. Copy config/demo.example.toml to config/demo.toml.</td></tr>`;
  $("#live-summary").textContent = `${live.uplinks} uplinks since start`;
  $("#recent-count").textContent = `(${live.recent.length})`;
  $("#recent").innerHTML = live.recent.map((e) =>
    `<li>${new Date(e.time).toLocaleTimeString()} · <b>${esc(e.name)}</b> ${e.source !== "live" ? `<span class="badge replay">${esc(e.source)}</span>` : ""}
     ${Object.entries(e.values).map(([k, v]) => fmtVal(k, v)).join("")}
     ${e.rssi != null ? `<span class="muted">RSSI ${e.rssi}</span>` : ""}
     ${e.error ? `<span class="err-text">${esc(e.error)}</span>` : ""}</li>`).join("");
  $("#live-errors").innerHTML = live.errors.slice(0, 3).map((e) =>
    `<div class="err-text">${new Date(e.time).toLocaleTimeString()} ${esc(e.error)}</div>`).join("");
}

// ── 2 · model cards ────────────────────────────────────────────────────────
function modelCards(pred, models, opts = {}) {
  if (!pred) return `<p class="muted">${esc(opts.empty || "No prediction yet.")}</p>`;
  if (!pred.ready) return `<p class="muted">Waiting for data: ${esc(pred.reason)}</p>`;
  const truth = opts.truth;
  const compare = opts.compareTruth !== false;
  const cards = models.map((mi) => {
    const m = pred.models[mi.key];
    if (!m) return "";
    const bars = Object.entries(m.proba).sort((a, b) => b[1] - a[1]).map(([c, p]) =>
      `<div class="bar"><span>${esc(phaseName(c))}</span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(0)}%;background:${phaseColor(c)}"></span></span><span class="num">${(p * 100).toFixed(0)}%</span></div>`).join("");
    const match = truth == null || !compare ? "" :
      (m.label === truth ? `<span class="match ok">✓ matches annotation</span>` : `<span class="match no">✗ differs from annotation</span>`);
    return `<div class="model"><h3>${esc(mi.name)}</h3><div class="kind">${MODEL_KIND[mi.key] || ""}</div>
      <div class="label"><span class="dot" style="background:${phaseColor(m.label)}"></span>${esc(phaseName(m.label))}</div>
      <div class="bars">${bars}</div>
      <div class="foot"><span>${m.ms.toFixed(1)} ms on CPU</span>${match}</div></div>`;
  });
  if (opts.showTruth) {
    cards.unshift(`<div class="model truth"><h3>Annotation</h3><div class="kind">${esc(opts.truthNote || "human label from the recording")}</div>
      <div class="label"><span class="dot" style="background:${phaseColor(truth)}"></span>${esc(truth == null ? "none (unlabeled run)" : phaseName(truth))}</div>
      ${opts.score && opts.score.n ? `<div class="bars">${models.map((mi) =>
        `<div class="bar"><span>${esc(mi.name)}</span><span class="track"><span class="fill" style="display:block;width:${(opts.score[mi.key] * 100).toFixed(0)}%;background:var(--accent)"></span></span><span class="num">${(opts.score[mi.key] * 100).toFixed(0)}%</span></div>`).join("")}
        <div class="muted" style="margin-top:.3rem">agreement with annotation so far (${opts.score.n} windows)</div></div>` : ""}</div>`);
  }
  const agreeTxt = {all: "All three models agree.", partial: "The models disagree: two against one.", none: "All three models disagree."}[pred.agreement];
  let warn = "";
  if (pred.compatible === false)
    warn += `<div class="banner warn">${(pred.ood_fraction * 100).toFixed(0)} % of the input features are outside the training range. Treat these predictions as unreliable.</div>`;
  if (pred.any_model_saw_data && !opts.simulated)
    warn += `<div class="banner warn">These models were trained on this recording. They are reproducing training data, not predicting.</div>`;
  const btn = opts.explain ? `<div class="explain-row"><button class="explain-btn" data-src="${opts.explain}">🔍 How was this decided?</button></div>` : "";
  return `<div class="agree">${agreeTxt}</div>${warn}<div class="models">${cards.join("")}</div>${btn}`;
}

// ── timeline ───────────────────────────────────────────────────────────────
function renderTimeline(rep, models) {
  const svg = $("#timeline");
  const W = 1000, H = 260, L = 92, R = 12;
  const st = rep.status;
  if (!st.start) { svg.innerHTML = `<text x="500" y="130" text-anchor="middle">Start a replay to see the run unfold.</text>`; return; }
  const t0 = new Date(st.start).getTime(), t1 = new Date(st.end).getTime();
  const x = (t) => L + ((new Date(t).getTime() - t0) / Math.max(1, t1 - t0)) * (W - L - R);
  const SHORT = {logreg: "Log. Regr.", rf: "Rand. Forest", mlp: "MLP"};
  const lanes = [{key: "truth", name: "Annotation"}, ...models.map((m) => ({key: m.key, name: SHORT[m.key] || m.key}))];
  const laneH = 18, gap = 4;
  let out = "";
  lanes.forEach((ln, li) => {
    const y = 8 + li * (laneH + gap);
    out += `<text x="${L - 6}" y="${y + 13}" text-anchor="end">${esc(ln.name)}</text>`;
    out += `<rect x="${L}" y="${y}" width="${W - L - R}" height="${laneH}" fill="#f1eee7"/>`;
    const h = rep.history;
    for (let i = 0; i < h.length; i++) {
      const xa = x(h[i].t), xb = i + 1 < h.length ? x(h[i + 1].t) : xa + 2;
      const lab = h[i][ln.key];
      if (lab == null) continue;
      out += `<rect x="${xa.toFixed(1)}" y="${y}" width="${Math.max(1, xb - xa + 0.6).toFixed(1)}" height="${laneH}" fill="${phaseColor(lab)}" shape-rendering="crispEdges"/>`;
    }
  });
  // signal plot
  const sig = $("#signal").value;
  const pts = rep.series.filter((p) => p[sig] != null);
  const top = 8 + lanes.length * (laneH + gap) + 12, bottom = H - 22;
  out += `<line x1="${L}" x2="${W - R}" y1="${bottom}" y2="${bottom}" stroke="#d8d2c6"/>`;
  if (pts.length > 1) {
    const vs = pts.map((p) => p[sig]);
    let lo = Math.min(...vs), hi = Math.max(...vs);
    if (hi === lo) { hi += 1; lo -= 1; }
    const y = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
    out += `<polyline fill="none" stroke="var(--accent)" stroke-width="1.4" points="${pts.map((p) => `${x(p.t).toFixed(1)},${y(p[sig]).toFixed(1)}`).join(" ")}"/>`;
    const u = UNITS[sig];
    out += `<text x="${L - 6}" y="${top + 8}" text-anchor="end">${hi.toFixed(u[2])} ${u[1]}</text>`;
    out += `<text x="${L - 6}" y="${bottom}" text-anchor="end">${lo.toFixed(u[2])} ${u[1]}</text>`;
  }
  if (st.data_time) {
    const xc = x(st.data_time);
    out += `<line x1="${xc}" x2="${xc}" y1="4" y2="${bottom}" stroke="var(--replay)" stroke-width="2"/>`;
  }
  const days = (t1 - t0) / 86400000;
  for (let d = 0; d <= days; d += Math.max(1, Math.ceil(days / 10))) {
    const xd = x(t0 + d * 86400000);
    out += `<text x="${xd}" y="${H - 6}" text-anchor="middle">day ${d}</text>`;
  }
  svg.innerHTML = out;
  $("#legend").innerHTML = (STATE.models.classes || []).map((c) =>
    `<span><span class="dot" style="background:${phaseColor(c)}"></span>${esc(phaseName(c))}</span>`).join("");
}

function renderReplay(st) {
  const rep = st.replay, s = rep.status, models = st.models.models || [];
  const banner = $("#replay-banner");
  if (s.running && s.group && $("#rec").value !== s.group && document.activeElement !== $("#rec")) $("#rec").value = s.group;
  if (s.group) {
    banner.hidden = false;
    banner.innerHTML = `<span class="badge replay">REPLAY</span> Recorded data, not live: <b>${esc(s.group)}</b>
      (${esc(s.kind || "")}, recorded ${new Date(s.start).toLocaleDateString()} – ${new Date(s.end).toLocaleDateString()}).
      Replay time <b>${s.data_time ? new Date(s.data_time).toLocaleString() : "–"}</b>
      ${s.paused ? "· <b>PAUSED</b>" : s.running ? `at ${Math.round(s.speed / 60)} min of data per second` : s.finished ? "· finished" : "· stopped"}
      ${s.error ? `<br><span class="err-text">${esc(s.error)}</span>` : ""}`;
  } else banner.hidden = true;
  $("#progress-bar").style.width = `${Math.min(100, (s.progress || 0) * 100).toFixed(1)}%`;
  $("#btn-start").disabled = !st.models.ok;
  $("#btn-pause").disabled = !s.running;
  $("#btn-pause").textContent = s.paused ? "▶ Resume" : "❚❚ Pause";
  $("#replay-models").innerHTML = modelCards(rep.latest, models, {
    truth: rep.latest ? rep.latest.truth : null, showTruth: !!rep.latest, score: rep.running_score, explain: "replay",
    empty: st.models.ok ? "Select a recording and start the replay." : "Train the models first (see README).",
  });
  renderTimeline(rep, models);
}

function renderLive(st) {
  const sel = $("#slot");
  const slots = st.live.slots;
  if (sel.options.length !== slots.length) sel.innerHTML = slots.map((s) => `<option>${esc(s)}</option>`).join("");
  const p = st.live.predictions[sel.value];
  $("#live-models").innerHTML = modelCards(p, st.models.models || [], {empty: "No live uplinks for this slot yet.", explain: "live"});
}

// ── 3 · evaluation ─────────────────────────────────────────────────────────
async function loadEval() {
  let m;
  try { m = await api("/api/metrics"); } catch (e) {
    $("#eval-body").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return;
  }
  const keys = Object.keys(m.models);
  const best = (f) => keys.reduce((a, b) => (m.models[b][f].mean > m.models[a][f].mean ? b : a));
  const pm = (o) => `${(o.mean * 100).toFixed(1)} <span class="muted">± ${(o.std * 100).toFixed(1)}</span>`;
  const bb = best("balanced_accuracy"), bf = best("macro_f1");
  const rows = keys.map((k) => {
    const r = m.models[k];
    return `<tr><td><b>${esc(r.name)}</b><br><span class="muted">${MODEL_KIND[k] || ""}</span></td>
      <td class="num ${k === bb ? "best" : ""}">${pm(r.balanced_accuracy)}</td>
      <td class="num ${k === bf ? "best" : ""}">${pm(r.macro_f1)}</td>
      <td class="num">${pm(r.accuracy)}</td>
      <td class="num">${r.single_prediction_ms.toFixed(1)} ms</td>
      <td class="num">${(r.model_file_bytes / 1e6).toFixed(r.model_file_bytes < 1e6 ? 2 : 1)} MB</td>
      <td class="num">${r.fit_seconds_all_data.toFixed(1)} s</td></tr>`;
  }).join("");
  const agree = Object.entries(m.agreement_on_held_out).map(([pair, v]) => {
    const [a, b] = pair.split("|");
    return `${esc(m.models[a].name.split(" (")[0])} ↔ ${esc(m.models[b].name.split(" (")[0])}: <b>${(v * 100).toFixed(0)} %</b>`;
  }).join(" · ");
  const cls = m.classes;
  const cms = keys.map((k) => {
    const cm = m.models[k].confusion_matrix;
    return `<div><h4>${esc(m.models[k].name)}</h4><table><tr><th>annotated ↓ / predicted →</th>${cls.map((c) => `<th>${esc(phaseName(c))}</th>`).join("")}</tr>
      ${cm.map((row, i) => { const s = row.reduce((a, b) => a + b, 0) || 1;
        return `<tr><th>${esc(phaseName(cls[i]))}</th>${row.map((v, j) => `<td style="background:${i === j ? `rgba(31,138,76,${(v / s * 0.5).toFixed(2)})` : `rgba(179,38,30,${(v / s * 0.5).toFixed(2)})`}">${(v / s * 100).toFixed(0)}%</td>`).join("")}</tr>`; }).join("")}
      </table></div>`;
  }).join("");
  const d = m.dataset;
  $("#eval-meta").textContent = `${d.labeled_groups.length} sausages · ${d.n_windows.toLocaleString()} windows · trained ${new Date(m.created_at).toLocaleDateString()}`;
  $("#eval-body").innerHTML = `
    ${m.quick ? `<div class="banner warn">QUICK TEST MODELS: these numbers come from a smoke test, not the real evaluation.</div>` : ""}
    <p class="muted">Leave-one-sausage-out: each model is scored on a sausage it never saw during training (mean ± std over ${d.labeled_groups.length} folds, in %).
      Balanced accuracy and macro-F1 weight all phases equally, even though the phases have different lengths.</p>
    <table class="eval-table"><thead><tr><th>Model</th><th class="num">Balanced acc.</th><th class="num">Macro-F1</th><th class="num">Accuracy</th>
      <th class="num">1 prediction</th><th class="num">Model size</th><th class="num">Training time</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="muted">How often two models give the same label on held-out windows: ${agree}</p>
    <details><summary>Confusion matrices (row-normalised)</summary><div class="cm">${cms}</div></details>
    <p class="muted">${esc(d.note)}</p>`;
}

// ── controls / loop ────────────────────────────────────────────────────────
async function loadRecordings() {
  const recs = await api("/api/recordings");
  $("#rec").innerHTML = recs.map((r) => `<option value="${esc(r.group)}" ${r.lfs_pointer ? "disabled" : ""}>${esc(r.group)}${r.group.startsWith("series_2") ? " (unlabeled, independent run)" : ""}${r.kind === "live recording" ? " (live recording)" : ""}${r.lfs_pointer ? " (LFS pointer – git lfs pull)" : ""}</option>`).join("");
  const first = recs.find((r) => !r.lfs_pointer && r.group.startsWith("series_1"));
  if (first) $("#rec").value = first.group;
  $("#sim-rec").innerHTML = $("#rec").innerHTML;
  if (first) $("#sim-rec").value = first.group;
  recordingsLoaded = true;
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  activeTab = b.dataset.tab;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $("#tab-replay").hidden = activeTab !== "replay";
  $("#tab-live").hidden = activeTab !== "live";
  $("#tab-sim").hidden = activeTab !== "sim";
  if (activeTab === "sim") runSim();
  else if (STATE) render(STATE);
}));
$("#startpos").addEventListener("input", (e) => ($("#startpos-v").textContent = `${e.target.value} %`));
$("#btn-start").addEventListener("click", async () => {
  try {
    await api("/api/replay/start", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({group: $("#rec").value, speed: Number($("#speed").value), start_fraction: Number($("#startpos").value) / 100})});
  } catch (e) { alert(e.message); }
});
$("#btn-pause").addEventListener("click", () =>
  api("/api/replay/pause", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({paused: !(STATE && STATE.replay.status.paused)})}));
$("#btn-stop").addEventListener("click", () => api("/api/replay/stop", {method: "POST"}));
$("#signal").addEventListener("change", () => STATE && renderTimeline(STATE.replay, STATE.models.models || []));
$("#slot").addEventListener("change", () => STATE && renderLive(STATE));

function render(st) {
  STATE = st;
  CLASS_NAMES = st.models.class_names || {};
  renderChips(st);
  renderDevices(st.live);
  if ($("#explain").hidden) {
    if (activeTab === "replay") renderReplay(st);
    else if (activeTab === "live") renderLive(st);
  }
  $("#footer").textContent = `Server up ${Math.round(st.uptime_s / 60)} min · config: ${st.ingest.config_file || "none"} · gateway URL: http://<laptop-ip>:<port>${st.ingest.http_path}${st.ingest.token_required ? "?token=…" : ""}`;
}

async function tick() {
  try {
    render(await api("/api/state"));
    if (!recordingsLoaded) await loadRecordings();
  } catch (e) {
    $("#chips").innerHTML = `<span class="chip err">Server unreachable: ${esc(e.message)}</span>`;
  }
  setTimeout(tick, POLL_MS);
}
tick();
loadEval();


// ── explanation overlay ────────────────────────────────────────────────────
let pausedByExplain = false;
const MODEL_TITLE = {logreg: "Logistic Regression", rf: "Random Forest", mlp: "MLP (neural net)"};
const pct = (p) => `${(p * 100).toFixed(p > 0.995 || p < 0.005 ? 1 : 0)} %`;
function fmtFeat(f) {
  if (f.value == null) return "–";
  if (f.signal === "timestamp") {
    if (f.stat === "uplinks") return f.value.toFixed(0);
    return f.value >= 120 ? `${(f.value / 60).toFixed(1)} min` : `${f.value.toFixed(0)} s`;
  }
  const unitful = ["minimum", "maximum", "mean", "spread (std)"].includes(f.stat);
  const digits = f.signal.includes("conductivity") ? 0 : 2;
  return `${f.value.toFixed(f.stat === "trend (slope)" ? 3 : digits)}${unitful ? " " + f.unit : f.stat === "trend (slope)" ? " /uplink" : ""}`;
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest(".explain-btn");
  if (!b) return;
  const src = b.dataset.src;
  pausedByExplain = false;
  if (src === "replay" && STATE && STATE.replay.status.running && !STATE.replay.status.paused) {
    await api("/api/replay/pause", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({paused: true})});
    pausedByExplain = true;
  }
  $("#ex-resume").hidden = !pausedByExplain;
  $("#ex-body").innerHTML = `<p class="muted">Computing explanation…</p>`;
  $("#explain").hidden = false;
  try {
    const q = src === "live" ? `source=live&slot=${encodeURIComponent($("#slot").value)}` : `source=${src}`;
    renderExplain(await api(`/api/explain?${q}`));
  } catch (err) { $("#ex-body").innerHTML = `<p class="err-text">${esc(err.message)}</p>`; }
});
function closeExplain(resume) {
  $("#explain").hidden = true;
  if (resume) api("/api/replay/pause", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({paused: false})});
}
$("#ex-close").addEventListener("click", () => closeExplain(false));
$("#ex-resume").addEventListener("click", () => closeExplain(true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#explain").hidden) closeExplain(false); });
$("#explain").addEventListener("click", (e) => { if (e.target.id === "explain") closeExplain(false); });
document.querySelectorAll(".stepper a").forEach((a) => a.addEventListener("click", (e) => {
  e.preventDefault(); document.querySelector(a.getAttribute("href")).scrollIntoView({behavior: "smooth"});
}));

function renderExplain(ex) {
  if (!ex.ready) { $("#ex-body").innerHTML = `<p class="muted">${esc(ex.reason)}</p>`; return; }
  Object.assign(CLASS_NAMES, ex.class_names || {});
  const keys = Object.keys(ex.models);
  const badge = {replay: '<span class="badge replay">REPLAY</span>', live: '<span class="badge live">LIVE</span>',
                 simulate: '<span class="badge sim">SIMULATED</span>'}[ex.source] || "";
  $("#ex-meta").innerHTML = `${badge}
    ${esc(ex.group || "")} · snapshot at ${ex.time ? new Date(ex.time).toLocaleString() : "–"}
    ${ex.truth != null ? ` · annotation${ex.source === "simulate" ? " of the unmodified recording" : ""}: <b>${esc(phaseName(ex.truth))}</b>` : ""}
    ${ex.adjust ? ` · changes: moisture ${signed(ex.adjust.moisture_delta)} %pt, conductivity ×${ex.adjust.conductivity_factor.toFixed(2)}, temperature ${signed(ex.adjust.temperature_delta)} °C` : ""}`;

  // step 1 · uplinks
  const ups = ex.uplinks.map((u) => `<tr><td>${new Date(u.time).toLocaleTimeString()}</td><td>${esc(u.sensor)}</td>
    <td>${Object.entries(u.values).map(([k, v]) => fmtVal(k, v)).join("") || '<span class="muted">environment (not a model input)</span>'}</td></tr>`).join("");
  const s1 = `<section class="ex-step" id="ex-s1"><h3><b>1</b>Uplinks received</h3>
    <p>The models never see a single reading. They see the last ${ex.uplinks.length} uplinks from this sausage's sensors,
    ${(ex.window_seconds / 60).toFixed(0)} minutes of data, in the order they arrived through gateway and router.
    ${ex.source === "simulate" ? "<b>Simulated:</b> these are recorded uplinks with the levels changed by the sliders." : ex.source === "replay" ? "These are recorded uplinks from the historical run." : ""}</p>
    <table class="uplinks"><thead><tr><th>Received</th><th>Sensor</th><th>Payload (decoded)</th></tr></thead><tbody>${ups}</tbody></table></section>`;

  // step 2 · features
  const bySig = {};
  ex.features.forEach((f) => (bySig[f.signal_label] = bySig[f.signal_label] || []).push(f));
  const sigs = Object.entries(bySig).map(([label, fs]) => `<div class="sig"><h4>${esc(label)}</h4>${fs.map((f) => {
    const out = f.position != null && (f.position < 0 || f.position > 1);
    const mark = f.position == null ? "" : `<i style="left:calc(${Math.min(1, Math.max(0, f.position)) * 100}% - 1px)"></i>`;
    return `<div class="frow"><span>${esc(f.stat)}</span><span class="v">${fmtFeat(f)}</span><span class="range ${out ? "out" : ""}" title="training range ${f.range[0]?.toFixed?.(2)} … ${f.range[1]?.toFixed?.(2)}">${mark}</span></div>`;
  }).join("")}</div>`).join("");
  const s2 = `<section class="ex-step" id="ex-s2"><h3><b>2</b>The ${ex.features.length} numbers the models see</h3>
    <p>For every sensor signal the window is summarised into a few statistics: level (min, mean, max), spread,
    trend and the timing of the uplinks. The bar shows where each number sits within the training data. A red marker means it lies outside anything seen in training.</p>
    <div class="ex-cols">${sigs}</div></section>`;

  // step 3 · occlusion, same test for every model
  const signals = ex.models[keys[0]].occlusion.map((o) => o.signal);
  const maxEv = Object.fromEntries(keys.map((k) => [k, Math.max(1e-9, ...ex.models[k].occlusion.map((o) => Math.abs(o.evidence)))]));
  const occRows = signals.map((sig) => `<tr><td><b>${esc(ex.models[keys[0]].occlusion.find((o) => o.signal === sig).label)}</b></td>${keys.map((k) => {
    const o = ex.models[k].occlusion.find((x) => x.signal === sig);
    const w = Math.abs(o.evidence) / maxEv[k] * 100;
    const flip = o.would_predict !== ex.models[k].label;
    return `<td><div class="cell"><span class="track"><span class="fill ${o.evidence < 0 ? "neg" : ""}" style="display:block;width:${w.toFixed(0)}%"></span></span>
      <small>${pct(o.p_before)} → ${pct(o.p_after)}</small></div>${flip ? `<small class="flip">would switch to ${esc(phaseName(o.would_predict))}</small>` : ""}</td>`;
  }).join("")}</tr>`).join("");
  const s3 = `<section class="ex-step" id="ex-s3"><h3><b>3</b>Which signals mattered? (same test for all three models)</h3>
    <p>We hide one sensor signal at a time, replacing its numbers with a typical training value, and watch how sure
    each model stays of its answer. A long bar means the model relied on that signal. Grey bars: hiding the signal made the model <i>more</i> sure.
    Signals that move together, such as two temperature probes, can stand in for each other, so this is a guide, not a proof.</p>
    <table class="occ"><thead><tr><th>Hidden signal</th>${keys.map((k) => `<th>${esc(MODEL_TITLE[k] || k)}: <span style="color:${phaseColor(ex.models[k].label)}">${esc(phaseName(ex.models[k].label))}</span></th>`).join("")}</tr></thead>
    <tbody>${occRows}</tbody></table></section>`;

  // step 4 · model internals
  const cols = keys.map((k) => {
    const m = ex.models[k], d = m.detail;
    let body = "";
    if (d.kind === "weighted_sum") body = exLogreg(d, m);
    else if (d.kind === "votes") body = exForest(d, m);
    else if (d.kind === "network") body = exNet(d, m);
    return `<div class="model-ex sig"><h4>${esc(MODEL_TITLE[k] || k)} → <span style="color:${phaseColor(m.label)}">${esc(phaseName(m.label))}</span> (${pct(m.proba[m.label])})</h4>
      <div class="principle">${esc(d.text || "")}</div>${body}
      ${m.model_saw_this_recording ? '<div class="err-text">This model was trained on this recording.</div>' : ""}</div>`;
  }).join("");
  const s4 = `<section class="ex-step" id="ex-s4"><h3><b>4</b>Inside each model: three different principles</h3>
    <p>The same numbers go into three very different kinds of model. Each column shows what that model actually computed for this snapshot.</p>
    <div class="ex-cols">${cols}</div></section>`;
  $("#ex-body").innerHTML = s1 + s2 + s3 + s4;
}

function contribRows(items, max, fmt) {
  return items.map((t) => {
    const w = Math.abs(t.contribution) / max * 50;
    const pos = t.contribution >= 0;
    return `<div class="contrib"><span>${esc(t.label)}</span><span class="axis"><span class="${pos ? "pos" : "neg"}" style="${pos ? "left:50%" : `left:${50 - w}%`};width:${w}%"></span></span><span class="num">${fmt(t.contribution)}</span></div>`;
  }).join("");
}
function exLogreg(d, m) {
  const max = Math.max(1e-9, ...d.top.map((t) => Math.abs(t.contribution)));
  const f = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
  return `<div class="muted" style="font-size:.75rem">weight × input = evidence for "${esc(phaseName(m.label))}"</div>
    ${contribRows(d.top, max, f)}
    <div class="sum-line">+ ${d.n_features - d.top.length} smaller inputs: ${f(d.rest)} · + bias: ${f(d.intercept)}<br>
    = score <b>${f(d.score)}</b> → probability <b>${pct(m.proba[m.label])}</b> after normalising across the three phases</div>`;
}
function exForest(d, m) {
  const k = d.classes.indexOf(m.label);
  const grid = d.tree_votes.map((v) => `<i style="background:${phaseColor(d.classes[v])}" title="${esc(phaseName(d.classes[v]))}"></i>`).join("");
  const max = Math.max(1e-9, ...d.top.map((t) => Math.abs(t.contribution)));
  return `<div><b>${d.votes[k]} of ${d.n_trees}</b> trees vote "${esc(phaseName(m.label))}"
    ${d.classes.map((c, i) => (i !== k && d.votes[i] ? ` · ${d.votes[i]} vote ${esc(phaseName(c))}` : "")).join("")}</div>
    <div class="votes">${grid}</div>
    <div class="muted" style="font-size:.75rem">Which questions moved the trees towards this answer (average over all trees, up to ${d.max_depth} questions per tree)</div>
    ${contribRows(d.top, max, (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "pp")}`;
}
function exNet(d, m) {
  const W = 420, H = 300, layers = d.layers;
  const xs = layers.map((_, i) => 30 + i * ((W - 60) / (layers.length - 1)));
  const ys = (n, j) => (n === 1 ? H / 2 : 14 + j * ((H - 28) / (n - 1)));
  let svg = "";
  // connections: strongest neurons of the last hidden layer → predicted output
  const L = layers.length - 1, outIdx = d.layers[L].act.indexOf(Math.max(...d.layers[L].act));
  d.strongest_last_hidden.forEach((j) => {
    svg += `<line x1="${xs[L - 1]}" y1="${ys(layers[L - 1].size, j)}" x2="${xs[L]}" y2="${ys(layers[L].size, outIdx)}" stroke="var(--accent)" stroke-opacity=".5" stroke-width="1.5"/>`;
  });
  layers.forEach((ly, i) => {
    const r = Math.max(2, Math.min(9, (H - 28) / ly.size / 2.4));
    ly.act.forEach((a, j) => {
      const isOut = i === L;
      const fill = isOut ? phaseColor(STATE.models.classes[j]) : "var(--accent)";
      svg += `<circle cx="${xs[i]}" cy="${ys(ly.size, j)}" r="${isOut ? 9 : r}" fill="${fill}" fill-opacity="${(0.08 + 0.92 * Math.abs(a)).toFixed(2)}" stroke="#bdb6a8" stroke-width=".4"/>`;
    });
    const lab = i === 0 ? `${ly.size} inputs` : i === L ? "output" : `${ly.active}/${ly.size} active`;
    svg += `<text x="${xs[i]}" y="${H - 1}" text-anchor="middle">${lab}</text>`;
  });
  return `<svg class="net" viewBox="0 0 ${W} ${H + 4}">${svg}</svg>
    <div class="muted" style="font-size:.75rem">Lines: the most influential neurons of the last hidden layer for "${esc(phaseName(m.label))}". Output circles are coloured by phase, darker means more likely.</div>`;
}


// ── virtual sausage (what-if simulation) ───────────────────────────────────
const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
let simTimer = null, simSeq = 0, simRunDays = null;
const PRESETS = {
  fresh: {pos: 20, dm: 0, cf: 100, dt: 0},
  dry: {dm: -20, cf: 60},
  warm: {dt: 6},
  reset: {dm: 0, cf: 100, dt: 0},
};
function simLabels() {
  const pos = Number($("#sim-pos").value) / 1000;
  $("#sim-pos-v").textContent = simRunDays != null ? `day ${(pos * simRunDays).toFixed(1)}` : `${(pos * 100).toFixed(0)} %`;
  $("#sim-dm-v").textContent = `${signed(Number($("#sim-dm").value), 0)} %pt`;
  $("#sim-cf-v").textContent = `× ${(Number($("#sim-cf").value) / 100).toFixed(2)}`;
  $("#sim-dt-v").textContent = `${signed(Number($("#sim-dt").value))} °C`;
}
function scheduleSim() { simLabels(); clearTimeout(simTimer); simTimer = setTimeout(runSim, 120); }
async function runSim() {
  if (!$("#sim-rec").value) return;
  simLabels();
  const seq = ++simSeq;
  let res;
  try {
    res = await api("/api/simulate", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({group: $("#sim-rec").value, position: Number($("#sim-pos").value) / 1000,
        adjust: {moisture_delta: Number($("#sim-dm").value), conductivity_factor: Number($("#sim-cf").value) / 100,
                 temperature_delta: Number($("#sim-dt").value)}})});
  } catch (e) { $("#sim-models").innerHTML = `<p class="err-text">${esc(e.message)}</p>`; return; }
  if (seq !== simSeq) return;  // a newer slider position is already on its way
  simRunDays = res.run_days;
  simLabels();
  const fmt = {lse01_moisture: ["moisture", "%", 1], lse01_conductivity: ["conductivity", "µS/cm", 0],
               lse01_temperature: ["core temp", "°C", 1], ltcfs_temperature: ["probe temp", "°C", 1]};
  $("#sim-values").innerHTML = "Window average, recorded → simulated: " + Object.entries(fmt).map(([k, [n, u, d]]) => {
    const a = res.recorded_means[k], b = res.simulated_means[k];
    if (a == null) return "";
    const changed = Math.abs(a - b) > 1e-9;
    return `<span class="val">${n} ${a.toFixed(d)}${changed ? ` → <span class="chg">${b.toFixed(d)}</span>` : ""} ${u}</span>`;
  }).join("");
  $("#sim-models").innerHTML = modelCards(res, (STATE && STATE.models.models) || [], {
    truth: res.truth, showTruth: true, compareTruth: !res.truth_is_for_unmodified, simulated: true, explain: "simulate",
    truthNote: res.truth_is_for_unmodified ? "label of the UNMODIFIED recording at this day" : "human label from the recording",
  });
}
["#sim-pos", "#sim-dm", "#sim-cf", "#sim-dt"].forEach((id) => $(id).addEventListener("input", scheduleSim));
$("#sim-rec").addEventListener("change", runSim);
document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
  const p = PRESETS[b.dataset.preset];
  if (p.pos != null) $("#sim-pos").value = p.pos;
  if (p.dm != null) $("#sim-dm").value = p.dm;
  if (p.cf != null) $("#sim-cf").value = p.cf;
  if (p.dt != null) $("#sim-dt").value = p.dt;
  runSim();
}));
simLabels();

"use strict";
// Plain JS, no external libraries: the dashboard must work without internet.
// All user-visible text goes through t() from i18n.js; shared helpers are in common.js.
let STATE = null, METRICS = null, RECORDINGS = null, LAST_EX = null, LAST_SIM = null;
let activeTab = "replay";

// ── header + data path ─────────────────────────────────────────────────────
function renderChips(st) {
  const chips = [];
  const lu = st.live.last_uplink_at;
  const age = lu ? (Date.now() - new Date(lu).getTime()) / 1000 : Infinity;
  chips.push(`<span class="chip ${age < 600 ? "ok" : "warn"}">${lu ? t("chip.gateway_last", {ago: ago(lu)}) : t("chip.gateway_none")}</span>`);
  const m = st.models;
  chips.push(m.ok ? `<span class="chip ok">${t("chip.models", {n: m.models.length})}${m.quick ? t("chip.quick") : ""}</span>`
                  : `<span class="chip err">${t("chip.models_err", {err: esc(m.error || t("chip.not_loaded"))})}</span>`);
  if (m.version_warning) chips.push(`<span class="chip warn" title="${esc(m.version_warning)}">${t("chip.sklearn")}</span>`);
  chips.push(`<span class="chip">Ingest: HTTP ${esc(st.ingest.http_path)}</span>`);
  const r = st.replay.status;
  chips.push(r.running ? `<span class="chip warn">${t("chip.replay_running")}</span>` : `<span class="chip">${t("chip.replay_idle")}</span>`);
  $("#chips").innerHTML = chips.join("");
  document.querySelectorAll(".node").forEach((n) => n.classList.toggle("active", age < 120));
}

// ── 1 · live sensors ───────────────────────────────────────────────────────
function renderDevices(live) {
  const rows = live.devices.map((d) => {
    const fresh = d.last_seen && (Date.now() - new Date(d.last_seen)) / 1000 < 30;
    const vals = d.configured
      ? Object.entries(d.values || {}).map(([k, v]) => fmtVal(k, v)).join("")
      : `<span class="muted">${esc(JSON.stringify(d.values || {})).slice(0, 120)}</span>`;
    const radio = d.rssi != null ? `<span class="val">${d.rssi} dBm</span>${d.snr != null ? `<br><span class="muted">SNR ${fmtNum(d.snr, 1)} dB</span>` : ""}` : "–";
    return `<tr class="${fresh ? "fresh" : ""} ${d.configured ? "" : "unconfigured"}">
      <td>${esc(d.name)}<br><span class="eui">${esc(d.type)} · ${esc(d.dev_eui)}</span>
        ${d.configured ? "" : `<br><span class="err-text">${t("live.not_in_config")}</span>`}</td>
      <td>${esc(d.slot)}</td>
      <td>${vals || `<span class="muted">${t("live.waiting")}</span>`}
        ${d.last_error ? `<br><span class="err-text">${esc(d.last_error)}</span>` : ""}
        ${d.decode_method ? `<br><span class="muted">${t(d.decode_method === "codec" ? "live.decoded_codec" : "live.decoded_builtin")}</span>` : ""}</td>
      <td>${radio}</td><td>${ago(d.last_seen)}</td><td>${d.uplinks}</td></tr>`;
  });
  $("#devices").innerHTML = rows.join("") || `<tr><td colspan="6" class="muted">${t("live.no_devices")}</td></tr>`;
  $("#live-summary").textContent = t("live.summary", {n: live.uplinks});
  $("#recent-count").textContent = `(${live.recent.length})`;
  $("#recent").innerHTML = live.recent.map((e) =>
    `<li>${fmtTime(e.time)} · <b>${esc(e.name)}</b> ${e.source !== "live" ? `<span class="badge replay">${esc(e.source)}</span>` : ""}
     ${Object.entries(e.values).map(([k, v]) => fmtVal(k, v)).join("")}
     ${e.rssi != null ? `<span class="muted">RSSI ${e.rssi}</span>` : ""}
     ${e.error ? `<span class="err-text">${esc(e.error)}</span>` : ""}</li>`).join("");
  $("#live-errors").innerHTML = live.errors.slice(0, 3).map((e) =>
    `<div class="err-text">${fmtTime(e.time)} ${esc(e.error)}</div>`).join("");
}

// ── 2 · model cards ────────────────────────────────────────────────────────
function modelCards(pred, models, opts = {}) {
  if (!pred) return `<p class="muted">${esc(opts.empty || t("card.no_prediction"))}</p>`;
  if (!pred.ready) return `<p class="muted">${t("card.waiting", {reason: esc(pred.reason)})}</p>`;
  const truth = opts.truth;
  const compare = opts.compareTruth !== false;
  const cards = models.map((mi) => {
    const m = pred.models[mi.key];
    if (!m) return "";
    const bars = Object.entries(m.proba).sort((a, b) => b[1] - a[1]).map(([c, p]) =>
      `<div class="bar"><span>${esc(phaseName(c))}</span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(0)}%;background:${phaseColor(c)}"></span></span><span class="num">${fmtNum(p * 100)}%</span></div>`).join("");
    const match = truth == null || !compare ? "" :
      (m.label === truth ? `<span class="match ok">${t("card.matches")}</span>` : `<span class="match no">${t("card.differs")}</span>`);
    return `<div class="model"><h3>${esc(modelName(mi.key, mi.name))}</h3><div class="kind">${t("kind." + mi.key)}</div>
      <div class="label"><span class="dot" style="background:${phaseColor(m.label)}"></span>${esc(phaseName(m.label))}</div>
      <div class="bars">${bars}</div>
      <div class="foot"><span>${t("card.ms", {ms: fmtNum(m.ms, 1)})}</span>${match}</div></div>`;
  });
  if (opts.showTruth) {
    cards.unshift(`<div class="model truth"><h3>${t("card.annotation")}</h3><div class="kind">${esc(opts.truthNote || t("card.annotation_note"))}</div>
      <div class="label"><span class="dot" style="background:${phaseColor(truth)}"></span>${esc(truth == null ? t("card.none_unlabeled") : phaseName(truth))}</div>
      ${opts.score && opts.score.n ? `<div class="bars">${models.map((mi) =>
        `<div class="bar"><span>${esc(modelName(mi.key, mi.name))}</span><span class="track"><span class="fill" style="display:block;width:${(opts.score[mi.key] * 100).toFixed(0)}%;background:var(--accent)"></span></span><span class="num">${fmtNum(opts.score[mi.key] * 100)}%</span></div>`).join("")}
        <div class="muted" style="margin-top:.3rem">${t("card.score_note", {n: opts.score.n})}</div></div>` : ""}</div>`);
  }
  let warn = "";
  if (pred.compatible === false) warn += `<div class="banner warn">${t("warn.ood", {pct: fmtNum(pred.ood_fraction * 100)})}</div>`;
  if (pred.any_model_saw_data && !opts.simulated) warn += `<div class="banner warn">${t("warn.seen")}</div>`;
  const btn = opts.explain ? `<div class="explain-row"><button class="explain-btn" data-src="${opts.explain}">${t("btn.explain")}</button></div>` : "";
  return `<div class="agree">${t("agree." + pred.agreement)}</div>${warn}<div class="models">${cards.join("")}</div>${btn}`;
}

// ── timeline ───────────────────────────────────────────────────────────────
function renderTimeline(rep, models) {
  const svg = $("#timeline");
  const W = 1000, H = 260, L = 92, R = 12;
  const st = rep.status;
  if (!st.start) { svg.innerHTML = `<text x="500" y="130" text-anchor="middle">${t("replay.timeline_empty")}</text>`; return; }
  const t0 = new Date(st.start).getTime(), t1 = new Date(st.end).getTime();
  const x = (ts) => L + ((new Date(ts).getTime() - t0) / Math.max(1, t1 - t0)) * (W - L - R);
  const lanes = [{key: "truth", name: t("card.annotation")}, ...models.map((m) => ({key: m.key, name: t("short." + m.key)}))];
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
    out += `<text x="${L - 6}" y="${top + 8}" text-anchor="end">${fmtNum(hi, u[2])} ${u[1]}</text>`;
    out += `<text x="${L - 6}" y="${bottom}" text-anchor="end">${fmtNum(lo, u[2])} ${u[1]}</text>`;
  }
  if (st.data_time) {
    const xc = x(st.data_time);
    out += `<line x1="${xc}" x2="${xc}" y1="4" y2="${bottom}" stroke="var(--replay)" stroke-width="2"/>`;
  }
  const days = (t1 - t0) / 86400000;
  for (let d = 0; d <= days; d += Math.max(1, Math.ceil(days / 10))) {
    out += `<text x="${x(t0 + d * 86400000)}" y="${H - 6}" text-anchor="middle">${t("replay.day", {n: d})}</text>`;
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
    const state = s.paused ? t("replay.state_paused_html") : s.running ? t("replay.state_running", {min: Math.round(s.speed / 60)})
      : s.finished ? t("replay.state_finished") : t("replay.state_stopped");
    banner.innerHTML = t("replay.banner_html", {group: esc(s.group), kind: esc(t("kind." + (s.kind || ""))),
      from: fmtDate(s.start), to: fmtDate(s.end), now: s.data_time ? fmtDateTime(s.data_time) : "–", state})
      + (s.error ? `<br><span class="err-text">${esc(s.error)}</span>` : "");
  } else banner.hidden = true;
  $("#progress-bar").style.width = `${Math.min(100, (s.progress || 0) * 100).toFixed(1)}%`;
  $("#btn-start").disabled = !st.models.ok;
  $("#btn-pause").disabled = !s.running;
  $("#btn-pause").textContent = t(s.paused ? "replay.resume" : "replay.pause");
  $("#replay-models").innerHTML = modelCards(rep.latest, models, {
    truth: rep.latest ? rep.latest.truth : null, showTruth: !!rep.latest, score: rep.running_score, explain: "replay",
    empty: st.models.ok ? t("replay.empty") : t("replay.train_first"),
  });
  renderTimeline(rep, models);
}

function renderLive(st) {
  const amb = $("#ambient");
  if (document.activeElement !== amb) amb.value = st.live.ambient_temperature;
  const signals = st.models.signals;
  $("#ambient-note").hidden = !(signals && signals.some((s) => s.endsWith("_diff")));
  const sel = $("#slot");
  const slots = st.live.slots;
  if (sel.options.length !== slots.length) sel.innerHTML = slots.map((s) => `<option>${esc(s)}</option>`).join("");
  $("#live-models").innerHTML = modelCards(st.live.predictions[sel.value], st.models.models || [],
    {empty: t("livetab.empty"), explain: "live"});
}

// ── 3 · evaluation ─────────────────────────────────────────────────────────
async function loadEval() {
  try { METRICS = await api("/api/metrics"); } catch (e) {
    $("#eval-body").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return;
  }
  renderEval();
}
function renderEval() {
  const m = METRICS;
  if (!m) return;
  const keys = Object.keys(m.models);
  const best = (f) => keys.reduce((a, b) => (m.models[b][f].mean > m.models[a][f].mean ? b : a));
  const pm = (o) => `${fmtNum(o.mean * 100, 1)} <span class="muted">± ${fmtNum(o.std * 100, 1)}</span>`;
  const bb = best("balanced_accuracy"), bf = best("macro_f1");
  const hasOrder = keys.some((k) => m.models[k].unannotated_progression);
  const order = (r) => {
    const p = r.unannotated_progression;
    if (!p || p.spearman_mean == null) return "–";
    const v = p.spearman_mean, cls = v >= 0.5 ? "good" : v < 0 ? "bad" : "";
    return `<span class="rho ${cls}">${v >= 0 ? "+" : "−"}${fmtNum(Math.abs(v), 2)}</span>`;
  };
  const rows = keys.map((k) => {
    const r = m.models[k];
    return `<tr><td><b>${esc(modelName(k, r.name))}</b><br><span class="muted">${t("kind." + k)}</span></td>
      <td class="num ${k === bb ? "best" : ""}">${pm(r.balanced_accuracy)}</td>
      <td class="num ${k === bf ? "best" : ""}">${pm(r.macro_f1)}</td>
      <td class="num">${fmtNum(r.single_prediction_ms, 1)} ms</td>
      <td class="num">${fmtNum(r.model_file_bytes / 1e6, r.model_file_bytes < 1e6 ? 2 : 1)} MB</td>
${hasOrder ? `<td class="num">${order(r)}</td>` : ""}</tr>`;
  }).join("");
  const agree = Object.entries(m.agreement_on_held_out).map(([pair, v]) => {
    const [a, b] = pair.split("|");
    return `${esc(t("short." + a))} ↔ ${esc(t("short." + b))}: <b>${fmtNum(v * 100)} %</b>`;
  }).join(" · ");
  const cls = m.classes;
  const cms = keys.map((k) => {
    const cm = m.models[k].confusion_matrix;
    return `<div><h4>${esc(modelName(k, m.models[k].name))}</h4><table><tr><th>${t("eval.cm_corner")}</th>${cls.map((c) => `<th>${esc(phaseName(c))}</th>`).join("")}</tr>
      ${cm.map((row, i) => { const s = row.reduce((a, b) => a + b, 0) || 1;
        return `<tr><th>${esc(phaseName(cls[i]))}</th>${row.map((v, j) => `<td style="background:${i === j ? `rgba(31,138,76,${(v / s * 0.5).toFixed(2)})` : `rgba(179,38,30,${(v / s * 0.5).toFixed(2)})`}">${fmtNum(v / s * 100)}%</td>`).join("")}</tr>`; }).join("")}
      </table></div>`;
  }).join("");
  const d = m.dataset;
  $("#eval-meta").textContent = t("eval.meta", {s: d.labeled_groups.length, w: d.n_windows.toLocaleString(LOCALE()), date: fmtDate(m.created_at)})
    + " · " + t("eval.features", {fs: t("fs." + (m.feature_set || "all"))});
  $("#eval-body").innerHTML = `
    ${m.quick ? `<div class="banner warn">${t("eval.quick")}</div>` : ""}
    <p class="muted">${t("eval.intro", {n: d.labeled_groups.length})}</p>
    <table class="eval-table"><thead><tr><th>${t("eval.th_model")}</th><th class="num">${t("eval.th_bal")}</th><th class="num">${t("eval.th_f1")}</th>
      <th class="num">${t("eval.th_pred")}</th><th class="num">${t("eval.th_size")}</th>
      ${hasOrder ? `<th class="num">${t("eval.th_order")}</th>` : ""}</tr></thead><tbody>${rows}</tbody></table>
    ${hasOrder ? `<p class="muted">${t("eval.order_note_html")}</p>` : ""}
    <p class="muted">${t("eval.agree")}${agree}</p>
    <details><summary>${t("eval.cm")}</summary><div class="cm">${cms}</div></details>
    <p class="muted">${t("eval.note")}</p>`;
}

// ── controls / loop ────────────────────────────────────────────────────────
async function loadRecordings() {
  RECORDINGS = await api("/api/recordings");
  renderRecordings(true);
}
function renderRecordings(initial) {
  if (!RECORDINGS) return;
  const opts = RECORDINGS.map((r) => `<option value="${esc(r.group)}" ${r.lfs_pointer ? "disabled" : ""}>${esc(r.group)}${r.group.startsWith("series_2") ? t("rec.unlabeled") : ""}${r.kind === "live recording" ? t("rec.live") : ""}${r.lfs_pointer ? t("rec.lfs") : ""}</option>`).join("");
  for (const id of ["#rec", "#sim-rec"]) {
    const prev = $(id).value;
    $(id).innerHTML = opts;
    if (prev && !initial) $(id).value = prev;
  }
  if (initial) {
    const first = RECORDINGS.find((r) => !r.lfs_pointer && r.group.startsWith("series_1"));
    if (first) { $("#rec").value = first.group; $("#sim-rec").value = first.group; }
  }
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
    await post("/api/replay/start", {group: $("#rec").value, speed: Number($("#speed").value), start_fraction: Number($("#startpos").value) / 100});
  } catch (e) { alert(e.message); }
});
$("#btn-pause").addEventListener("click", () => post("/api/replay/pause", {paused: !(STATE && STATE.replay.status.paused)}));
$("#btn-stop").addEventListener("click", () => post("/api/replay/stop"));
$("#signal").addEventListener("change", () => STATE && renderTimeline(STATE.replay, STATE.models.models || []));
$("#slot").addEventListener("change", () => STATE && renderLive(STATE));
$("#ambient").addEventListener("change", async (e) => {
  try { await post("/api/ambient", {value: Number(e.target.value)}); } catch (err) { alert(err.message); }
});

function render(st) {
  STATE = st;
  renderChips(st);
  renderDevices(st.live);
  if ($("#explain").hidden) {
    if (activeTab === "replay") renderReplay(st);
    else if (activeTab === "live") renderLive(st);
  }
  $("#footer").textContent = t("footer", {min: Math.round(st.uptime_s / 60), cfg: st.ingest.config_file || t("footer.none"),
    path: st.ingest.http_path + (st.ingest.token_required ? "?token=…" : "")});
}

async function tick() {
  try {
    render(await api("/api/state"));
    if (!RECORDINGS) await loadRecordings();
  } catch (e) {
    $("#chips").innerHTML = `<span class="chip err">${t("chip.unreachable", {err: esc(e.message)})}</span>`;
  }
  setTimeout(tick, POLL_MS);
}

// ── language switch ────────────────────────────────────────────────────────
function setLang(lang) {
  if (!I18N[lang]) return;
  LANG = lang;
  try { localStorage.setItem("lang", lang); } catch (e) { /* storage unavailable */ }
  applyStaticI18n();
  renderRecordings(false);
  renderEval();
  simLabels();
  if (STATE) render(STATE);
  if (activeTab === "sim" && LAST_SIM) renderSim(LAST_SIM);
  if (!$("#explain").hidden && LAST_EX) renderExplain(LAST_EX);
}
document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));

// ── explanation overlay ────────────────────────────────────────────────────
let pausedByExplain = false;
function fmtFeat(f) {
  if (f.value == null) return "–";
  if (f.signal === "timestamp") {
    if (f.stat_key === "count") return fmtNum(f.value);
    return f.value >= 120 ? `${fmtNum(f.value / 60, 1)} min` : `${fmtNum(f.value)} s`;
  }
  const unitful = ["min", "max", "mean", "std"].includes(f.stat_key);
  const digits = f.signal.includes("conductivity") ? 0 : 2;
  if (f.stat_key === "slope") return `${fmtNum(f.value, 3)} ${t("unit.per_uplink")}`;
  return `${fmtNum(f.value, digits)}${unitful ? " " + f.unit : ""}`;
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest(".explain-btn");
  if (!b) return;
  const src = b.dataset.src;
  pausedByExplain = false;
  if (src === "replay" && STATE && STATE.replay.status.running && !STATE.replay.status.paused) {
    await post("/api/replay/pause", {paused: true});
    pausedByExplain = true;
  }
  $("#ex-resume").hidden = !pausedByExplain;
  $("#ex-body").innerHTML = `<p class="muted">${t("ex.computing")}</p>`;
  $("#explain").hidden = false;
  try {
    const q = src === "live" ? `source=live&slot=${encodeURIComponent($("#slot").value)}` : `source=${src}`;
    LAST_EX = await api(`/api/explain?${q}`);
    renderExplain(LAST_EX);
  } catch (err) { $("#ex-body").innerHTML = `<p class="err-text">${esc(err.message)}</p>`; }
});
function closeExplain(resume) {
  $("#explain").hidden = true;
  if (resume) post("/api/replay/pause", {paused: false});
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
  const keys = Object.keys(ex.models);
  const badge = {replay: `<span class="badge replay">${LANG === "de" ? "WIEDERGABE" : "REPLAY"}</span>`,
                 live: '<span class="badge live">LIVE</span>',
                 simulate: `<span class="badge sim">${LANG === "de" ? "SIMULIERT" : "SIMULATED"}</span>`}[ex.source] || "";
  const truthTxt = ex.truth != null
    ? ` · ${t(ex.source === "simulate" ? "ex.annotation_unmod" : "ex.annotation")}: <b>${esc(phaseName(ex.truth))}</b>` : "";
  const adj = ex.adjust ? ` · ${t("ex.changes", {dm: signed(ex.adjust.moisture_delta), cf: fmtNum(ex.adjust.conductivity_factor, 2),
                                                  dt: signed(ex.adjust.temperature_delta)})}` : "";
  $("#ex-meta").innerHTML = `${badge} ${esc(ex.group || "")} · ${t("ex.snapshot", {time: ex.time ? fmtDateTime(ex.time) : "–"})}${truthTxt}${adj}`;

  // step 1 · uplinks
  const ups = ex.uplinks.map((u) => `<tr><td>${fmtTime(u.time)}</td><td>${esc(u.sensor)}</td>
    <td>${Object.entries(u.values).map(([k, v]) => fmtVal(k, v)).join("") || `<span class="muted">${t("ex.env")}</span>`}</td></tr>`).join("");
  const srcNote = ex.source === "simulate" ? t("ex.s1_sim_html") : ex.source === "replay" ? t("ex.s1_replay") : "";
  const s1 = `<section class="ex-step" id="ex-s1"><h3><b>1</b>${t("ex.s1_title")}</h3>
    <p>${t("ex.s1_text", {n: ex.uplinks.length, min: fmtNum(ex.window_seconds / 60)})} ${srcNote}</p>
    <table class="uplinks"><thead><tr><th>${t("ex.th_received")}</th><th>${t("ex.th_sensor")}</th><th>${t("ex.th_payload")}</th></tr></thead><tbody>${ups}</tbody></table></section>`;

  // step 2 · features
  const bySig = {};
  ex.features.forEach((f) => (bySig[f.signal] = bySig[f.signal] || []).push(f));
  const sigs = Object.entries(bySig).map(([sig, fs]) => `<div class="sig"><h4>${esc(t("sig." + sig))}</h4>${fs.map((f) => {
    const out = f.position != null && (f.position < 0 || f.position > 1);
    const mark = f.position == null ? "" : `<i style="left:calc(${Math.min(1, Math.max(0, f.position)) * 100}% - 1px)"></i>`;
    const title = f.range[0] != null ? t("ex.range", {lo: fmtNum(f.range[0], 2), hi: fmtNum(f.range[1], 2)}) : "";
    return `<div class="frow"><span>${esc(t("stat." + f.stat_key))}</span><span class="v">${fmtFeat(f)}</span><span class="range ${out ? "out" : ""}" title="${esc(title)}">${mark}</span></div>`;
  }).join("")}</div>`).join("");
  const s2 = `<section class="ex-step" id="ex-s2"><h3><b>2</b>${t("ex.s2_title", {n: ex.features.length})}</h3>
    <p>${t("ex.s2_text")}</p><div class="ex-cols">${sigs}</div></section>`;

  // step 3 · occlusion, same test for every model
  const signals = ex.models[keys[0]].occlusion.map((o) => o.signal);
  const maxEv = Object.fromEntries(keys.map((k) => [k, Math.max(1e-9, ...ex.models[k].occlusion.map((o) => Math.abs(o.evidence)))]));
  const occRows = signals.map((sig) => `<tr><td><b>${esc(t("sig." + sig))}</b></td>${keys.map((k) => {
    const o = ex.models[k].occlusion.find((x) => x.signal === sig);
    const w = Math.abs(o.evidence) / maxEv[k] * 100;
    const flip = o.would_predict !== ex.models[k].label;
    return `<td><div class="cell"><span class="track"><span class="fill ${o.evidence < 0 ? "neg" : ""}" style="display:block;width:${w.toFixed(0)}%"></span></span>
      <small>${pct(o.p_before)} → ${pct(o.p_after)}</small></div>${flip ? `<small class="flip">${t("ex.would_switch", {phase: esc(phaseName(o.would_predict))})}</small>` : ""}</td>`;
  }).join("")}</tr>`).join("");
  const s3 = `<section class="ex-step" id="ex-s3"><h3><b>3</b>${t("ex.s3_title")}</h3>
    <p>${t("ex.s3_text_html")}</p>
    <table class="occ"><thead><tr><th>${t("ex.th_hidden")}</th>${keys.map((k) => `<th>${esc(modelName(k))}: <span style="color:${phaseColor(ex.models[k].label)}">${esc(phaseName(ex.models[k].label))}</span></th>`).join("")}</tr></thead>
    <tbody>${occRows}</tbody></table></section>`;

  // step 4 · model internals
  const cols = keys.map((k) => {
    const m = ex.models[k], d = m.detail;
    let body = "", principle = "";
    if (d.kind === "weighted_sum") { body = exLogreg(d, m); principle = t("ex.lr_text"); }
    else if (d.kind === "votes") { body = exForest(d, m); principle = t("ex.rf_text", {n: d.n_trees}); }
    else if (d.kind === "network") {
      body = exNet(d, m);
      principle = t("ex.mlp_text", {l: d.layers.length - 2, p: d.n_params.toLocaleString(LOCALE())});
    }
    return `<div class="model-ex sig"><h4>${esc(modelName(k))} → <span style="color:${phaseColor(m.label)}">${esc(phaseName(m.label))}</span> (${pct(m.proba[m.label])})</h4>
      <div class="principle">${esc(principle)}</div>${body}
      ${m.model_saw_this_recording ? `<div class="err-text">${t("ex.seen")}</div>` : ""}</div>`;
  }).join("");
  const s4 = `<section class="ex-step" id="ex-s4"><h3><b>4</b>${t("ex.s4_title")}</h3>
    <p>${t("ex.s4_text")}</p><div class="ex-cols">${cols}</div></section>`;
  $("#ex-body").innerHTML = s1 + s2 + s3 + s4;
}

function contribRows(items, max, fmt) {
  return items.map((it) => {
    const w = Math.abs(it.contribution) / max * 50;
    const pos = it.contribution >= 0;
    return `<div class="contrib"><span>${esc(featLabel(it.feature))}</span><span class="axis"><span class="${pos ? "pos" : "neg"}" style="${pos ? "left:50%" : `left:${50 - w}%`};width:${w}%"></span></span><span class="num">${fmt(it.contribution)}</span></div>`;
  }).join("");
}
function exLogreg(d, m) {
  const max = Math.max(1e-9, ...d.top.map((x) => Math.abs(x.contribution)));
  const f = (v) => (v >= 0 ? "+" : "−") + fmtNum(Math.abs(v), 2);
  return `<div class="muted" style="font-size:.75rem">${t("ex.lr_head", {phase: esc(phaseName(m.label))})}</div>
    ${contribRows(d.top, max, f)}
    <div class="sum-line">${t("ex.lr_sum_html", {n: d.n_features - d.top.length, rest: f(d.rest), bias: f(d.intercept), score: f(d.score), p: pct(m.proba[m.label])})}</div>`;
}
function exForest(d, m) {
  const k = d.classes.indexOf(m.label);
  const grid = d.tree_votes.map((v) => `<i style="background:${phaseColor(d.classes[v])}" title="${esc(phaseName(d.classes[v]))}"></i>`).join("");
  const max = Math.max(1e-9, ...d.top.map((x) => Math.abs(x.contribution)));
  const others = d.classes.map((c, i) => (i !== k && d.votes[i] ? t("ex.rf_other", {k: d.votes[i], phase: esc(phaseName(c))}) : "")).join("");
  return `<div>${t("ex.rf_votes_html", {k: d.votes[k], n: d.n_trees, phase: esc(phaseName(m.label))})}${others}</div>
    <div class="votes">${grid}</div>
    <div class="muted" style="font-size:.75rem">${t("ex.rf_head", {d: d.max_depth})}</div>
    ${contribRows(d.top, max, (v) => (v >= 0 ? "+" : "−") + fmtNum(Math.abs(v) * 100, 1))}`;
}
function exNet(d, m) {
  const W = 420, H = 300, layers = d.layers;
  const xs = layers.map((_, i) => 30 + i * ((W - 60) / (layers.length - 1)));
  const ys = (n, j) => (n === 1 ? H / 2 : 14 + j * ((H - 28) / (n - 1)));
  let svg = "";
  const L = layers.length - 1, outIdx = layers[L].act.indexOf(Math.max(...layers[L].act));
  d.strongest_last_hidden.forEach((j) => {
    svg += `<line x1="${xs[L - 1]}" y1="${ys(layers[L - 1].size, j)}" x2="${xs[L]}" y2="${ys(layers[L].size, outIdx)}" stroke="var(--accent)" stroke-opacity=".5" stroke-width="1.5"/>`;
  });
  layers.forEach((ly, i) => {
    const r = Math.max(2, Math.min(9, (H - 28) / ly.size / 2.4));
    ly.act.forEach((a, j) => {
      const isOut = i === L;
      const fill = isOut ? phaseColor(d.classes[j]) : "var(--accent)";
      svg += `<circle cx="${xs[i]}" cy="${ys(ly.size, j)}" r="${isOut ? 9 : r}" fill="${fill}" fill-opacity="${(0.08 + 0.92 * Math.abs(a)).toFixed(2)}" stroke="#bdb6a8" stroke-width=".4"/>`;
    });
    const lab = i === 0 ? t("ex.mlp_inputs", {n: ly.size}) : i === L ? t("ex.mlp_output") : t("ex.mlp_active", {a: ly.active, n: ly.size});
    svg += `<text x="${xs[i]}" y="${H - 1}" text-anchor="middle">${lab}</text>`;
  });
  return `<svg class="net" viewBox="0 0 ${W} ${H + 4}">${svg}</svg>
    <div class="muted" style="font-size:.75rem">${t("ex.mlp_note", {phase: esc(phaseName(m.label))})}</div>`;
}

// ── virtual sausage (what-if simulation) ───────────────────────────────────
let simTimer = null, simSeq = 0, simRunDays = null;
const PRESETS = {
  fresh: {pos: 20, dm: 0, cf: 100, dt: 0},
  dry: {dm: -20, cf: 60},
  warm: {dt: 6},
  reset: {dm: 0, cf: 100, dt: 0},
};
function simLabels() {
  const pos = Number($("#sim-pos").value) / 1000;
  $("#sim-pos-v").textContent = simRunDays != null ? t("sim.day_v", {n: fmtNum(pos * simRunDays, 1)}) : `${(pos * 100).toFixed(0)} %`;
  $("#sim-dm-v").textContent = `${signed(Number($("#sim-dm").value), 0)} ${t("sim.pt")}`;
  $("#sim-cf-v").textContent = `× ${fmtNum(Number($("#sim-cf").value) / 100, 2)}`;
  $("#sim-dt-v").textContent = `${signed(Number($("#sim-dt").value))} °C`;
}
function scheduleSim() { simLabels(); clearTimeout(simTimer); simTimer = setTimeout(runSim, 120); }
async function runSim() {
  if (!$("#sim-rec").value) return;
  simLabels();
  const seq = ++simSeq;
  let res;
  try {
    res = await post("/api/simulate", {group: $("#sim-rec").value, position: Number($("#sim-pos").value) / 1000,
      adjust: {moisture_delta: Number($("#sim-dm").value), conductivity_factor: Number($("#sim-cf").value) / 100,
               temperature_delta: Number($("#sim-dt").value)}});
  } catch (e) { $("#sim-models").innerHTML = `<p class="err-text">${esc(e.message)}</p>`; return; }
  if (seq !== simSeq) return;  // a newer slider position is already on its way
  LAST_SIM = res;
  renderSim(res);
}
function updateSimInputs() {
  const signals = STATE && STATE.models.signals;
  const usesTemp = !signals || signals.some((s) => s.includes("temperature"));
  $("#sim-dt").disabled = !usesTemp;
  document.querySelector("[data-preset=warm]").disabled = !usesTemp;
  $("#sim-dt-note").hidden = usesTemp;
  $("#sim-dt-rel").hidden = !(signals && signals.some((s) => s.endsWith("_diff")));
  if (!usesTemp && Number($("#sim-dt").value) !== 0) $("#sim-dt").value = 0;
}

function renderSim(res) {
  updateSimInputs();
  simRunDays = res.run_days;
  simLabels();
  const fmt = {lse01_moisture: ["%", 1], lse01_conductivity: ["µS/cm", 0], lse01_temperature: ["°C", 1], ltcfs_temperature: ["°C", 1],
               ambient_temperature: ["°C", 1], ltcfs_temperature_diff: ["°C", 1]};
  $("#sim-values").innerHTML = t("sim.values") + Object.entries(fmt).map(([k, [u, d]]) => {
    const a = res.recorded_means[k], b = res.simulated_means[k];
    if (a == null) return "";
    const changed = Math.abs(a - b) > 1e-9;
    return `<span class="val">${t("simv." + k)} ${fmtNum(a, d)}${changed ? ` → <span class="chg">${fmtNum(b, d)}</span>` : ""} ${u}</span>`;
  }).join("");
  $("#sim-models").innerHTML = modelCards(res, (STATE && STATE.models.models) || [], {
    truth: res.truth, showTruth: true, compareTruth: !res.truth_is_for_unmodified, simulated: true, explain: "simulate",
    truthNote: t(res.truth_is_for_unmodified ? "card.annotation_unmod" : "card.annotation_note"),
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

// ── start ──────────────────────────────────────────────────────────────────
applyStaticI18n();
simLabels();
tick();
loadEval();

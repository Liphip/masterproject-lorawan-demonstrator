"use strict";
// Presentation view: one source at a time, one consensus phase, three models, four measurements.
let mode = "replay", ST = null, SIM = null, recs = null, simSeq = 0, simTimer = null;
const MODES = ["replay", "sim", "live"];
const VALUES = [["lse01_moisture", "val.moisture", "%", 1], ["lse01_conductivity", "val.conductivity", "µS/cm", 0],
                ["ltcfs_temperature", "s.core", "°C", 1], ["ambient_temperature", "val.air", "°C", 1]];

// consensus: phase with the highest mean probability over the models
function consensus(pred) {
  const sum = [0, 0, 0];
  const ms = Object.values(pred.models);
  for (const m of ms) for (const [c, p] of Object.entries(m.proba)) {
    const i = phaseIndex(c);
    if (i != null) sum[i] += p / ms.length;
  }
  const i = sum.indexOf(Math.max(...sum));
  return {i, p: sum[i]};
}

function renderPrediction(pred, truth, when) {
  $("#s-when").innerHTML = when || "";
  const steps = document.querySelectorAll("#s-steps li");
  const ok = pred && pred.ready;
  const c = ok ? consensus(pred) : null;
  steps.forEach((li) => {
    const p = Number(li.dataset.p);
    li.querySelector(".s-step-name").textContent = t("phase." + p);
    li.classList.toggle("on", !!c && c.i === p);
    li.classList.toggle("truth", truth != null && phaseIndex(truth) === p);
  });
  const models = (ST && ST.models.models) || [];
  if (!ok) {
    $("#s-models").innerHTML = `<p class="s-wait">${esc(pred && pred.reason ? t("s.waiting") : emptyText())}</p>`;
    $("#s-values").innerHTML = "";
    return;
  }
  $("#s-models").innerHTML = models.map((mi) => {
    const m = pred.models[mi.key];
    if (!m) return "";
    return `<div class="s-model"><span class="s-mname">${esc(modelName(mi.key, mi.name))}</span>
      <span class="s-mlabel"><span class="dot" style="background:${phaseColor(m.label)}"></span>${esc(phaseName(m.label))}</span>
      <span class="track"><span class="fill" style="display:block;width:${(m.confidence * 100).toFixed(0)}%;background:${phaseColor(m.label)}"></span></span>
      <span class="num">${fmtNum(m.confidence * 100)} %</span></div>`;
  }).join("") + `<div class="s-agree">${t("agree." + pred.agreement)}</div>`;
  const v = pred.values || {};
  $("#s-values").innerHTML = VALUES.filter(([k]) => v[k] != null).map(([k, label, unit, d]) =>
    `<div class="s-tile"><span>${t(label)}</span><b>${fmtNum(v[k], d)}</b><small>${unit}</small></div>`).join("");
}

function emptyText() {
  if (ST && !ST.models.ok) return t("replay.train_first");
  return {replay: t("s.press_play"), sim: "", live: t("s.no_live")}[mode];
}

function render() {
  if (!ST) return;
  const lu = ST.live.last_uplink_at;
  const fresh = lu && (Date.now() - new Date(lu).getTime()) / 1000 < 600;
  $("#s-status").innerHTML = `<span class="dot" style="background:${fresh ? "var(--live)" : "var(--none)"}"></span>${lu ? t("s.gateway", {ago: ago(lu)}) : t("chip.gateway_none")}`;
  if (mode === "replay") {
    const s = ST.replay.status;
    $("#s-play").textContent = s.running && !s.paused ? "❚❚" : "▶";
    $("#s-play").disabled = !ST.models.ok;
    $("#s-bar").style.width = `${Math.min(100, (s.progress || 0) * 100).toFixed(1)}%`;
    if (s.group && $("#s-rec").value !== s.group && document.activeElement !== $("#s-rec")) $("#s-rec").value = s.group;
    const day = s.data_time && s.start ? (new Date(s.data_time) - new Date(s.start)) / 864e5 : null;
    const when = s.group ? `<span class="s-badge replay">${t("s.recorded")}</span> ${t("sim.day_v", {n: fmtNum(day ?? 0, 0)})} · ${fmtDate(s.data_time || s.start)}` : "";
    const latest = s.group ? ST.replay.latest : null;
    renderPrediction(latest, latest && latest.truth, when);
  } else if (mode === "live") {
    const amb = $("#s-ambient");
    if (document.activeElement !== amb) amb.value = ST.live.ambient_temperature;
    const slot = ST.live.slots[0];
    renderPrediction(slot ? ST.live.predictions[slot] : null, null, `<span class="s-badge live">LIVE</span>`);
  }
}

// ── replay ──
$("#s-play").addEventListener("click", async () => {
  const s = ST && ST.replay.status;
  try {
    if (s && s.running) await post("/api/replay/pause", {paused: !s.paused});
    else await post("/api/replay/start", {group: $("#s-rec").value, speed: Number($("#s-speed").value), start_fraction: 0});
  } catch (e) { alert(e.message); }
});
$("#s-rec").addEventListener("change", () => post("/api/replay/stop"));

// ── virtual sausage ──
function simLabels() {
  const days = SIM && SIM.run_days;
  const pos = Number($("#s-pos").value) / 1000;
  $("#s-pos-v").textContent = days != null ? t("sim.day_v", {n: fmtNum(pos * days, 0)}) : `${Math.round(pos * 100)} %`;
  $("#s-dm-v").textContent = `${signed(Number($("#s-dm").value), 0)} ${t("sim.pt")}`;
}
async function runSim() {
  const group = $("#s-rec").value;
  if (!group || !ST || !ST.models.ok) { renderPrediction(null, null, ""); return; }
  simLabels();
  const seq = ++simSeq;
  try {
    const res = await post("/api/simulate", {group, position: Number($("#s-pos").value) / 1000,
      adjust: {moisture_delta: Number($("#s-dm").value), conductivity_factor: 1, temperature_delta: 0}});
    if (seq !== simSeq) return;
    SIM = res;
    simLabels();
    const unchanged = Number($("#s-dm").value) === 0;
    renderPrediction(res, unchanged ? res.truth : null, `<span class="s-badge sim">${t("s.simulated")}</span>`);
  } catch (e) { $("#s-models").innerHTML = `<p class="err-text">${esc(e.message)}</p>`; }
}
const scheduleSim = () => { simLabels(); clearTimeout(simTimer); simTimer = setTimeout(runSim, 120); };
["#s-pos", "#s-dm"].forEach((id) => $(id).addEventListener("input", scheduleSim));
$("#s-reset").addEventListener("click", () => { $("#s-dm").value = 0; runSim(); });

// ── live ──
$("#s-ambient").addEventListener("change", async (e) => {
  try { await post("/api/ambient", {value: Number(e.target.value)}); } catch (err) { alert(err.message); }
});

// ── modes, language, polling ──
function setMode(m) {
  mode = m;
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  for (const x of MODES) $("#c-" + x).hidden = x !== m;
  $("#s-rec").hidden = false;
  if (m === "sim") { $("#c-sim").prepend($("#s-rec")); runSim(); }
  else { $("#c-replay").insertBefore($("#s-rec"), $("#s-speed")); render(); }
  try { history.replaceState(null, "", `?mode=${m}`); } catch (e) { /* ignore */ }
}
document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => {
  LANG = b.dataset.lang;
  try { localStorage.setItem("lang", LANG); } catch (e) { /* storage unavailable */ }
  applyStaticI18n();
  if (mode === "sim") runSim(); else render();
}));

async function loadRecordings() {
  recs = (await api("/api/recordings")).filter((r) => !r.lfs_pointer && r.group.startsWith("series_"));
  $("#s-rec").innerHTML = recs.map((r) => `<option value="${esc(r.group)}">${esc(r.group.replace(/_/g, " "))}</option>`).join("");
  const s = ST && ST.replay.status;
  if (s && s.group) $("#s-rec").value = s.group;
  if (mode === "sim") runSim();
}
async function tick() {
  try {
    ST = await api("/api/state");
    if (!recs) await loadRecordings();
    render();
  } catch (e) {
    $("#s-status").innerHTML = `<span class="err-text">${esc(t("chip.unreachable", {err: e.message}))}</span>`;
  }
  setTimeout(tick, POLL_MS);
}

applyStaticI18n();
const startMode = new URLSearchParams(location.search).get("mode");
setMode(MODES.includes(startMode) ? startMode : "replay");
tick();

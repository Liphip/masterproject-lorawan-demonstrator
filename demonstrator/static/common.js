"use strict";
// Helpers shared by the presentation page (simple.js) and the detail dashboard (app.js). Needs i18n.js.
const POLL_MS = 1500;
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

// [label key, unit, decimals]; decimals < 0 = on/off state
const UNITS = {
  lse01_conductivity: ["val.conductivity", "µS/cm", 0], lse01_moisture: ["val.moisture", "%", 1],
  lse01_temperature_diff: ["val.temp_diff", "°C", 1], ltcfs_temperature_diff: ["val.temp_diff", "°C", 1],
  ambient_temperature: ["val.air", "°C", 1],
  lse01_temperature: ["val.temp", "°C", 1], ltcfs_temperature: ["val.temp", "°C", 1],
  ws202_light_status: ["val.light", "", -1], ws202_pir_status: ["val.motion", "", -1],
};
const SIGNAL_KEYS = ["lse01_temperature_diff", "ltcfs_temperature_diff", "lse01_conductivity", "lse01_moisture",
                     "lse01_temperature", "ltcfs_temperature", "timestamp"];
const modelName = (k, fallback) => (I18N.en["model." + k] ? t("model." + k) : fallback || k);

// GET/POST JSON; throws with the server's error detail.
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
const post = (path, body) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
// '12 s ago' / '3 min ago' in the current language.
function ago(iso) {
  if (!iso) return t("ago.never");
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return t("ago.s", {n: Math.max(0, Math.round(s))});
  if (s < 3600) return t("ago.min", {n: Math.round(s / 60)});
  return t("ago.h", {n: Math.round(s / 3600)});
}
const fmtTime = (iso) => new Date(iso).toLocaleTimeString(LOCALE());
const fmtDate = (iso) => new Date(iso).toLocaleDateString(LOCALE());
const fmtDateTime = (iso) => new Date(iso).toLocaleString(LOCALE());

// Position of a phase label in the curing order (0..2), or null for unknown labels.
function phaseIndex(label) {
  const l = String(label || "").toLowerCase();
  if (l.includes("umroet") || l.includes("umröt") || l.includes("curing")) return 0;
  if (l.includes("1")) return 1;
  if (l.includes("2")) return 2;
  return null;
}
const phaseColor = (label) => ["var(--p0)", "var(--p1)", "var(--p2)"][phaseIndex(label)] || "var(--none)";
// Translated phase name (unknown labels are shown as they are).
function phaseName(label) {
  if (label == null) return "–";
  const i = phaseIndex(label);
  return i == null ? String(label) : t("phase." + i);
}
// A measurement with its unit and decimals as configured in UNITS.
function fmtVal(k, v) {
  const u = UNITS[k];
  if (!u) return `<span class="val">${esc(k)} <b>${esc(typeof v === "number" ? fmtNum(v, 2) : v)}</b></span>`;
  if (u[2] < 0) {
    const txt = k === "ws202_light_status" ? t(v ? "val.light_on" : "val.light_off") : t(v ? "val.motion_on" : "val.motion_off");
    return `<span class="val">${t(u[0])} <b>${txt}</b></span>`;
  }
  return `<span class="val">${t(u[0])} <b>${fmtNum(v, u[2])}</b> ${u[1]}</span>`;
}
const pct = (p) => `${fmtNum(p * 100, p > 0.995 || p < 0.005 ? 1 : 0)} %`;
const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + fmtNum(Math.abs(v), d);
// Readable name of a window feature, e.g. 'Moisture (LSE01) · mean'.
function featLabel(name) {
  for (const sig of SIGNAL_KEYS) {
    if (name.startsWith(sig + "_")) return `${t("sig." + sig)} · ${statLabel(name.slice(sig.length + 1))}`;
  }
  return name;
}
// "mean" -> "Mittelwert"; context statistics "ctxtrend_7d" -> "Trend der letzten 7 Tage"
function statLabel(key) {
  const m = /^(ctx(?:mean|std|trend))_(\d+)([hd])$/.exec(key || "");
  if (!m) return t("stat." + key);
  const n = Number(m[2]);
  const words = {de: {h: ["Stunde", "Stunden"], d: ["Tag", "Tage"]}, en: {h: ["hour", "hours"], d: ["day", "days"]}};
  const w = (words[LANG] || words.en)[m[3]][n === 1 ? 0 : 1];
  return t("stat." + m[1], {h: `${n} ${w}`});
}

// ── readable names for recordings and slots ────────────────────────────────
let REC_META = {};  // group -> {start, end} from /api/recordings (set by the pages)
function setRecordings(list) { REC_META = Object.fromEntries((list || []).map((r) => [r.group, r])); }
// "sausage_1" -> "Wurst 1" / "sausage 1"; "environment" -> "Umgebung"
function slotName(slot) {
  const m = /^sausage_(\d+)$/.exec(slot || "");
  if (m) return t("slot.sausage", {n: m[1]});
  return slot === "environment" ? t("slot.environment") : (slot || "–");
}
// "Mär–Aug 2023" (one year) or "Jul 2023 – Apr 2024"
function monthSpan(start, end) {
  if (!start || !end) return "";
  const a = new Date(start), b = new Date(end);
  const mo = (d) => d.toLocaleDateString(LOCALE(), {month: "short"});
  if (a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth()) return `${mo(b)} ${b.getFullYear()}`;
  return a.getFullYear() === b.getFullYear() ? `${mo(a)}–${mo(b)} ${b.getFullYear()}`
    : `${mo(a)} ${a.getFullYear()} – ${mo(b)} ${b.getFullYear()}`;
}
// series_1_sausage_3 -> "Durchgang 1 · Wurst 3 (Mär–Aug 2023)"; live_20261125_sausage_1 -> "Koffer-Aufnahme 25.11.2026 · Wurst 1"
function recName(group, withSpan = true) {
  if (!group) return "";
  const meta = REC_META[group] || {};
  let m = /^series_(\d+)_sausage_(\d+)$/.exec(group);
  if (m) {
    const span = withSpan ? monthSpan(meta.start, meta.end) : "";
    return t("rec.series", {run: m[1], n: m[2]}) + (span ? ` (${span})` : "");
  }
  m = /^live_(\d{4})(\d{2})(\d{2})_(.+)$/.exec(group);
  if (m) return t("rec.live_name", {date: fmtDate(`${m[1]}-${m[2]}-${m[3]}T12:00:00`), slot: slotName(m[4])});
  return group.replace(/_/g, " ");
}
// Picker groups: annotated runs, unannotated runs, case recordings, others.
function recCategory(r) {
  if (r.kind === "live recording") return "live";
  if (/^series_1_/.test(r.group)) return "annotated";
  if (/^series_\d+_/.test(r.group)) return "unannotated";
  return "other";
}
// <option>s grouped by category; disabled for Git LFS pointers.
function recOptions(list, only) {
  const order = ["annotated", "unannotated", "live", "other"];
  return order.filter((c) => !only || only.includes(c)).map((c) => {
    const items = list.filter((r) => recCategory(r) === c);
    if (!items.length) return "";
    return `<optgroup label="${esc(t("rec.group_" + c))}">${items.map((r) =>
      `<option value="${esc(r.group)}" ${r.lfs_pointer ? "disabled" : ""}>${esc(recName(r.group))}${r.lfs_pointer ? esc(t("rec.lfs")) : ""}</option>`).join("")}</optgroup>`;
  }).join("");
}
// Agreement sentence for a prediction (uses the number of models and the size of the largest group).
function agreeText(pred) {
  return t("agree." + pred.agreement, {k: pred.agreement_count, n: Object.keys(pred.models).length});
}

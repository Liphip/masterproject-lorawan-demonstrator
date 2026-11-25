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
    if (name.startsWith(sig + "_")) return `${t("sig." + sig)} · ${t("stat." + name.slice(sig.length + 1))}`;
  }
  return name;
}

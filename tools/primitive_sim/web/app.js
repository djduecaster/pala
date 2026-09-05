const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");
const pageParams = new URLSearchParams(window.location.search);
if (pageParams.get("shell") === "1") {
  document.body.classList.add("embedded-shell");
  document.addEventListener("click", (ev) => {
    const link = ev.target?.closest?.("a.nav-btn[href*='/tools/primitive_sim/web/lamp_sim.html']");
    if (!link || !(window.top && window.top !== window)) {
      return;
    }
    ev.preventDefault();
    window.top.location.href = link.href;
  });
}

const playPauseBtn = document.getElementById("playPauseBtn");
const resetBtn = document.getElementById("resetBtn");
const speedSelect = document.getElementById("speedSelect");
const timeline = document.getElementById("timeline");
const timeLabel = document.getElementById("timeLabel");
const statusBox = document.getElementById("statusBox");
const jointTableBody = document.querySelector("#jointTable tbody");
const traceSource = document.getElementById("traceSource");
const fileInput = document.getElementById("fileInput");
const modeLabel = document.getElementById("modeLabel");

const primitiveSelect = document.getElementById("primitiveSelect");
const styleSelect = document.getElementById("styleSelect");
const durationInput = document.getElementById("durationInput");
const paramForm = document.getElementById("paramForm");
const baselineStatus = document.getElementById("baselineStatus");
const runBtn = document.getElementById("runBtn");
const runSaveBtn = document.getElementById("runSaveBtn");
const resetBaselineBtn = document.getElementById("resetBaselineBtn");
const saveBaselineBtn = document.getElementById("saveBaselineBtn");
const saveAllBaselineBtn = document.getElementById("saveAllBaselineBtn");
const runSuiteBtn = document.getElementById("runSuiteBtn");
const suiteStatus = document.getElementById("suiteStatus");
const autoRunToggle = document.getElementById("autoRunToggle");
const autoRunMsInput = document.getElementById("autoRunMsInput");
const compareToggle = document.getElementById("compareToggle");
const compareModeSelect = document.getElementById("compareModeSelect");
const runStateBadge = document.getElementById("runStateBadge");
const paramFilterInput = document.getElementById("paramFilterInput");
const zoomOutBtn = document.getElementById("zoomOutBtn");
const zoomInBtn = document.getElementById("zoomInBtn");
const orbitLeftBtn = document.getElementById("orbitLeftBtn");
const orbitRightBtn = document.getElementById("orbitRightBtn");
const resetViewBtn = document.getElementById("resetViewBtn");

const state = {
  samples: [],
  jointNames: [],
  jointLimits: [],
  lampGeom: null,
  jointRows: [],
  idx: 0,
  playhead: 0,
  playing: false,
  speed: 1.0,
  dtS: 1 / 60,
  lastFrameMs: null,
  cameraOrbit: 0.82,
  cameraDistance: 2.85,
  drag: {
    active: false,
    lastX: 0,
  },
  mapping: {
    yaw: -1,
    roll: -1,
    pitch1: -1,
    pitch2: -1,
    pitch3: -1,
  },
  studio: {
    enabled: false,
    specs: [],
    styles: [],
    baseline: null,
    selectedPrimitive: "",
    commandDraft: {},
    dirty: false,
    autoRunEnabled: true,
    autoRunDebounceMs: 200,
    autoRunTimer: null,
    runInFlight: false,
    rerunQueued: false,
    requestToken: 0,
    compareEnabled: false,
    compareMode: "overlay",
    baselineTrace: null,
    baselineSourceLabel: "",
    draftSourceLabel: "",
    paramFilter: "",
  },
};

const RAD_TO_DEG = 180 / Math.PI;
const STUDIO_PREFS_KEY = "lamp_sim_studio_prefs_v1";

const LAMP_GEOM = Object.freeze({
  baseRadius: 0.18,
  baseThickness: 0.028,
  mastHeight: 1.28,
  hubRise: 0.02,
  upperArmLen: 0.52,
  foreArmLen: 0.42,
  wristStubLen: 0.08,
  shadeNeckLen: 0.08,
  shadeLen: 0.18,
  shadeRearRadius: 0.068,
  shadeFrontRadius: 0.046,
  pitch1ZeroOffsetRad: -Math.PI / 2,
  pitch2ZeroOffsetRad: Math.PI / 2,
  pitch3ZeroOffsetRad: 0.0,
});

const LAMP_GEOM_KEYS = Object.keys(LAMP_GEOM);
const LAMP_GEOM_POSITIVE_KEYS = new Set([
  "baseRadius",
  "baseThickness",
  "mastHeight",
  "hubRise",
  "upperArmLen",
  "foreArmLen",
  "wristStubLen",
  "shadeNeckLen",
  "shadeLen",
  "shadeRearRadius",
  "shadeFrontRadius",
]);

const LAMP_THEME_DRAFT = Object.freeze({
  baseOuter: "rgba(240, 225, 194, 0.92)",
  baseInner: "rgba(198, 170, 126, 0.86)",
  mast: "rgba(227, 232, 237, 0.96)",
  hubRing: "rgba(178, 198, 212, 0.84)",
  hubTick: "rgba(182, 198, 212, 0.76)",
  arm1: "#d8e2e6",
  arm2: "#dbe6eb",
  arm3: "#cfd8de",
  neck: "#becbd4",
  shadeRear: "rgba(253, 232, 185, 0.95)",
  shadeFront: "rgba(255, 214, 150, 0.95)",
  shadeStrut: "rgba(255, 206, 128, 0.85)",
  rollRing: "rgba(255, 154, 61, 0.9)",
  rollTick: "rgba(255, 154, 61, 0.92)",
  yawArrow: "rgba(255, 154, 61, 0.95)",
  marker1: "#a8c5d6",
  marker2: "#8bb8d7",
  marker3: "#94d4d0",
  marker4: "#f7ce86",
});

const LAMP_THEME_BASELINE = Object.freeze({
  baseOuter: "rgba(136, 199, 206, 0.55)",
  baseInner: "rgba(96, 161, 170, 0.55)",
  mast: "rgba(149, 206, 216, 0.62)",
  hubRing: "rgba(132, 191, 209, 0.62)",
  hubTick: "rgba(132, 191, 209, 0.5)",
  arm1: "rgba(146, 214, 220, 0.74)",
  arm2: "rgba(146, 214, 220, 0.74)",
  arm3: "rgba(146, 214, 220, 0.74)",
  neck: "rgba(146, 214, 220, 0.74)",
  shadeRear: "rgba(111, 186, 197, 0.78)",
  shadeFront: "rgba(111, 186, 197, 0.78)",
  shadeStrut: "rgba(111, 186, 197, 0.72)",
  rollRing: "rgba(118, 215, 211, 0.85)",
  rollTick: "rgba(118, 215, 211, 0.85)",
  yawArrow: "rgba(118, 215, 211, 0.88)",
  marker1: "rgba(118, 215, 211, 0.88)",
  marker2: "rgba(118, 215, 211, 0.88)",
  marker3: "rgba(118, 215, 211, 0.88)",
  marker4: "rgba(118, 215, 211, 0.88)",
});

function inEmbeddedShell() {
  return pageParams.get("shell") === "1" && window.top && window.top !== window;
}

function navigateViewerUrl(url) {
  const target = String(url || "").trim();
  if (!target) {
    return false;
  }
  if (inEmbeddedShell()) {
    window.top.location.href = target;
    return true;
  }
  window.location.href = target;
  return true;
}

function notifyShell(kind, message, details = "") {
  if (window.parent === window) {
    return;
  }
  try {
    window.parent.postMessage(
      {
        source: "lamp-mode",
        type: "status",
        mode: "studio",
        kind: String(kind || "ok"),
        message: String(message || ""),
        details: String(details || ""),
      },
      window.location.origin,
    );
  } catch (_err) {
    // ignore postMessage failures
  }
}

function deepClone(v) {
  return JSON.parse(JSON.stringify(v));
}

function stableJson(value) {
  return JSON.stringify(value ?? null);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function safeBool(value, fallback = false) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  if (typeof value === "string") {
    const token = value.trim().toLowerCase();
    if (token === "true" || token === "1" || token === "yes" || token === "on") {
      return true;
    }
    if (token === "false" || token === "0" || token === "no" || token === "off") {
      return false;
    }
  }
  return fallback;
}

function loadStudioPrefs() {
  try {
    const raw = window.localStorage.getItem(STUDIO_PREFS_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed;
  } catch (_err) {
    return {};
  }
}

function saveStudioPrefs() {
  if (!state.studio.enabled) {
    return;
  }
  const payload = {
    primitive: String(state.studio.selectedPrimitive || ""),
    style: String(styleSelect.value || ""),
    duration_s: Number(durationInput.value),
    auto_run_enabled: Boolean(state.studio.autoRunEnabled),
    auto_run_debounce_ms: Number(state.studio.autoRunDebounceMs),
    compare_enabled: Boolean(state.studio.compareEnabled),
    compare_mode: String(state.studio.compareMode || "overlay"),
    param_filter: String(state.studio.paramFilter || ""),
    speed: Number(state.speed || 1),
  };
  try {
    window.localStorage.setItem(STUDIO_PREFS_KEY, JSON.stringify(payload));
  } catch (_err) {
    // ignore storage errors
  }
}

function applyStudioPrefs(meta) {
  const prefs = loadStudioPrefs();
  const styles = Array.isArray(meta?.styles) ? meta.styles.map((v) => String(v)) : [];
  const primitives = Array.isArray(meta?.primitives) ? meta.primitives.map((v) => String(v.id || "")) : [];
  const preferredPrimitive = String(prefs.primitive || "");
  const selectedPrimitive = primitives.includes(preferredPrimitive) ? preferredPrimitive : String(meta.default_primitive);

  const preferredStyle = String(prefs.style || "");
  const selectedStyle = styles.includes(preferredStyle)
    ? preferredStyle
    : (styles.includes("calm") ? "calm" : styles[0]);
  styleSelect.value = selectedStyle;

  const preferredDuration = Number(prefs.duration_s);
  const validPreferredDuration = Number.isFinite(preferredDuration) && preferredDuration > 0
    ? preferredDuration
    : null;

  state.studio.autoRunEnabled = safeBool(prefs.auto_run_enabled, Boolean(autoRunToggle.checked));
  state.studio.autoRunDebounceMs = clamp(
    Math.round(Number.isFinite(Number(prefs.auto_run_debounce_ms)) ? Number(prefs.auto_run_debounce_ms) : Number(autoRunMsInput.value)),
    50,
    1000,
  );
  autoRunToggle.checked = state.studio.autoRunEnabled;
  autoRunMsInput.value = String(state.studio.autoRunDebounceMs);

  state.studio.compareEnabled = safeBool(prefs.compare_enabled, Boolean(compareToggle.checked));
  state.studio.compareMode = String(prefs.compare_mode || compareModeSelect.value || "overlay");
  if (!["overlay", "split"].includes(state.studio.compareMode)) {
    state.studio.compareMode = "overlay";
  }
  compareToggle.checked = state.studio.compareEnabled;
  compareModeSelect.value = state.studio.compareMode;
  state.studio.paramFilter = String(prefs.param_filter || "").trim();
  if (paramFilterInput) {
    paramFilterInput.value = state.studio.paramFilter;
  }

  const speed = Number(prefs.speed);
  state.speed = Number.isFinite(speed) && speed > 0 ? speed : Number(speedSelect.value || 1);
  const speedOpt = Array.from(speedSelect.options).find((opt) => Number(opt.value) === state.speed);
  speedSelect.value = speedOpt ? speedOpt.value : "1";
  state.speed = Number(speedSelect.value || 1);

  return {
    primitive: selectedPrimitive,
    durationS: validPreferredDuration,
  };
}

function stepScaleFromEvent(ev) {
  if (ev.altKey) {
    return 0.2;
  }
  if (ev.shiftKey) {
    return 5.0;
  }
  return 1.0;
}

function scaledStep(baseStep, ev) {
  const base = Number.isFinite(Number(baseStep)) ? Math.abs(Number(baseStep)) : 0.01;
  return base * stepScaleFromEvent(ev);
}

function bindNumericNudge(input, { getValue, setValue, baseStep }) {
  if (!input) {
    return;
  }
  input.addEventListener("wheel", (ev) => {
    if (document.activeElement !== input) {
      return;
    }
    const current = Number(getValue());
    if (!Number.isFinite(current)) {
      return;
    }
    const delta = scaledStep(baseStep, ev);
    const sign = ev.deltaY < 0 ? 1 : -1;
    setValue(current + sign * delta);
    ev.preventDefault();
  }, { passive: false });
}

function isEditableTarget(target) {
  if (!(target instanceof Element)) {
    return false;
  }
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
    return true;
  }
  return Boolean(target.closest("[contenteditable='true']"));
}

function vecAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vecScale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n < 1e-8) {
    return [0, 0, 0];
  }
  return [v[0] / n, v[1] / n, v[2] / n];
}

function matMul(a, b) {
  return [
    [
      a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
      a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
      a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
    ],
    [
      a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
      a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
      a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
    ],
    [
      a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
      a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
      a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
    ],
  ];
}

function matVec(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

function rotX(a) {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotY(a) {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

function rotZ(a) {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

// Pitch frames are defined with local -Z axis so +pitch moves forward by construction.
function rotPitch(a) {
  return rotZ(-a);
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const targetW = Math.max(1, Math.floor(w * dpr));
  const targetH = Math.max(1, Math.floor(h * dpr));
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
}

function makeCamera() {
  const orbit = state.cameraOrbit;
  const dist = clamp(Number(state.cameraDistance), 1.8, 6.0);
  const camPos = [dist * Math.cos(orbit), 1.58, dist * Math.sin(orbit)];
  const target = [0, 0.72, 0];

  const forward = norm(vecSub(target, camPos));
  const right = norm(cross(forward, [0, 1, 0]));
  const up = norm(cross(right, forward));
  return { camPos, forward, right, up };
}

function adjustCameraDistance(delta) {
  state.cameraDistance = clamp(Number(state.cameraDistance) + Number(delta || 0), 1.8, 6.0);
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
}

function adjustCameraOrbit(delta) {
  state.cameraOrbit += Number(delta || 0);
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
}

function resetCameraView() {
  state.cameraOrbit = 0.82;
  state.cameraDistance = 2.85;
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
}

function projectPoint(p, camera) {
  const v = vecSub(p, camera.camPos);
  const x = dot(v, camera.right);
  const y = dot(v, camera.up);
  const z = dot(v, camera.forward);
  if (z <= 0.05) {
    return null;
  }

  const f = Math.min(canvas.width, canvas.height) * 0.9;
  const sx = canvas.width * 0.5 + (x / z) * f;
  const sy = canvas.height * 0.62 - (y / z) * f;
  return { x: sx, y: sy, z };
}

function drawLine3D(a, b, color, width, camera) {
  const pa = projectPoint(a, camera);
  const pb = projectPoint(b, camera);
  if (!pa || !pb) {
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
}

function drawGrid(camera) {
  ctx.save();
  ctx.lineWidth = 1;
  for (let i = -6; i <= 6; i += 1) {
    const t = i * 0.22;
    const alpha = i === 0 ? 0.28 : 0.14;
    const color = `rgba(99, 160, 178, ${alpha})`;
    drawLine3D([-1.4, 0, t], [1.4, 0, t], color, 1, camera);
    drawLine3D([t, 0, -1.4], [t, 0, 1.4], color, 1, camera);
  }
  ctx.restore();
}

function orthonormalBasis(axis) {
  const n = norm(axis);
  const ref = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = norm(cross(n, ref));
  const v = norm(cross(n, u));
  return { n, u, v };
}

function drawCircle3D(center, axis, radius, color, width, camera, segments = 24) {
  const basis = orthonormalBasis(axis);
  const pts = [];
  for (let i = 0; i < segments; i += 1) {
    const a = (i / segments) * Math.PI * 2;
    const p = vecAdd(
      center,
      vecAdd(
        vecScale(basis.u, Math.cos(a) * radius),
        vecScale(basis.v, Math.sin(a) * radius),
      ),
    );
    pts.push(p);
  }
  for (let i = 0; i < pts.length; i += 1) {
    drawLine3D(pts[i], pts[(i + 1) % pts.length], color, width, camera);
  }
}

function drawJointMarker(point, color, radiusPx, camera) {
  const p = projectPoint(point, camera);
  if (!p) {
    return;
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radiusPx, 0, Math.PI * 2);
  ctx.fill();
}

function drawShadeFrustum(rearCenter, frontCenter, axis, rearR, frontR, camera, theme) {
  const basis = orthonormalBasis(axis);
  const rearPts = [];
  const frontPts = [];
  const segs = 16;
  for (let i = 0; i < segs; i += 1) {
    const a = (i / segs) * Math.PI * 2;
    const c = Math.cos(a);
    const s = Math.sin(a);
    const radial = vecAdd(vecScale(basis.u, c), vecScale(basis.v, s));
    rearPts.push(vecAdd(rearCenter, vecScale(radial, rearR)));
    frontPts.push(vecAdd(frontCenter, vecScale(radial, frontR)));
  }
  for (let i = 0; i < segs; i += 1) {
    drawLine3D(rearPts[i], rearPts[(i + 1) % segs], theme.shadeRear, 1.8, camera);
    drawLine3D(frontPts[i], frontPts[(i + 1) % segs], theme.shadeFront, 1.4, camera);
  }
  const struts = [0, 4, 8, 12];
  struts.forEach((i) => {
    drawLine3D(rearPts[i], frontPts[i], theme.shadeStrut, 1.5, camera);
  });
}

function resolveLampGeometry(raw) {
  const out = { ...LAMP_GEOM };
  if (!raw || typeof raw !== "object") {
    return out;
  }
  LAMP_GEOM_KEYS.forEach((key) => {
    const val = Number(raw[key]);
    if (!Number.isFinite(val)) {
      return;
    }
    if (LAMP_GEOM_POSITIVE_KEYS.has(key)) {
      if (val > 0) {
        out[key] = val;
      }
      return;
    }
    out[key] = val;
  });
  return out;
}

function resolveMapping(names) {
  const lower = names.map((n) => String(n).toLowerCase());
  const mapping = {
    yaw: lower.indexOf("yaw"),
    roll: lower.indexOf("roll"),
    pitch1: lower.indexOf("pitch1"),
    pitch2: lower.indexOf("pitch2"),
    pitch3: lower.indexOf("pitch3"),
  };

  const missing = Object.entries(mapping)
    .filter(([, idx]) => idx < 0)
    .map(([name]) => name);
  if (missing.length) {
    throw new Error(
      `Missing required joint names: ${missing.join(", ")}. expected yaw,pitch1,pitch2,roll,pitch3`,
    );
  }
  return mapping;
}

function sampleAngles(sample) {
  const out = new Array(state.jointNames.length).fill(0);
  const raw = Array.isArray(sample.joint_angles_rad) ? sample.joint_angles_rad : [];
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number(raw[i] || 0);
  }
  return out;
}

function angleAt(angles, idx) {
  if (idx < 0 || idx >= angles.length) {
    return 0;
  }
  return Number(angles[idx] || 0);
}

function forwardKinematics(sample) {
  const angles = sampleAngles(sample);
  const map = state.mapping;
  const g = state.lampGeom || LAMP_GEOM;

  const yaw = angleAt(angles, map.yaw);
  const roll = angleAt(angles, map.roll);
  const pitch1 = angleAt(angles, map.pitch1) + Number(g.pitch1ZeroOffsetRad || 0);
  const pitch2 = angleAt(angles, map.pitch2) + Number(g.pitch2ZeroOffsetRad || 0);
  const pitch3 = angleAt(angles, map.pitch3) + Number(g.pitch3ZeroOffsetRad || 0);

  const baseCenter = [0, 0, 0];
  const mastBottom = [0, g.baseThickness, 0];
  const mastTop = [0, g.mastHeight, 0];
  const yawHub = vecAdd(mastTop, [0, g.hubRise, 0]);

  const rotYaw = rotY(yaw);
  const rotShoulder = matMul(rotYaw, rotPitch(pitch1));
  const elbow = vecAdd(yawHub, matVec(rotShoulder, [g.upperArmLen, 0, 0]));

  const rotElbow = matMul(rotShoulder, rotPitch(pitch2));
  const wrist = vecAdd(elbow, matVec(rotElbow, [g.foreArmLen, 0, 0]));

  const rotRoll = matMul(rotElbow, rotX(roll));
  const headPivot = vecAdd(wrist, matVec(rotRoll, [g.wristStubLen, 0, 0]));

  const rotHead = matMul(rotRoll, rotPitch(pitch3));
  const shadeRear = vecAdd(headPivot, matVec(rotHead, [g.shadeNeckLen, 0, 0]));
  const shadeFront = vecAdd(shadeRear, matVec(rotHead, [0, -g.shadeLen, 0]));

  return {
    angles,
    baseCenter,
    mastBottom,
    mastTop,
    yawHub,
    elbow,
    wrist,
    headPivot,
    shadeRear,
    shadeFront,
    rotYaw,
    forearmAxis: norm(matVec(rotElbow, [1, 0, 0])),
    rollCueDir: matVec(rotRoll, [0, 0, 1]),
  };
}

function drawLamp(sample, camera, theme = LAMP_THEME_DRAFT) {
  const k = forwardKinematics(sample);
  const g = state.lampGeom || LAMP_GEOM;

  drawCircle3D(k.baseCenter, [0, 1, 0], g.baseRadius, theme.baseOuter, 3.6, camera, 36);
  drawCircle3D(k.baseCenter, [0, 1, 0], g.baseRadius * 0.58, theme.baseInner, 1.8, camera, 28);
  drawLine3D(k.mastBottom, k.mastTop, theme.mast, 8, camera);

  drawCircle3D(k.yawHub, [0, 1, 0], 0.055, theme.hubRing, 1.9, camera, 24);
  drawLine3D(k.yawHub, vecAdd(k.yawHub, [0, 0.04, 0]), theme.hubTick, 3, camera);

  drawLine3D(k.yawHub, k.elbow, theme.arm1, 6.5, camera);
  drawLine3D(k.elbow, k.wrist, theme.arm2, 6.2, camera);
  drawLine3D(k.wrist, k.headPivot, theme.arm3, 5.2, camera);
  drawLine3D(k.headPivot, k.shadeRear, theme.neck, 4.2, camera);

  drawShadeFrustum(
    k.shadeRear,
    k.shadeFront,
    vecSub(k.shadeFront, k.shadeRear),
    g.shadeRearRadius,
    g.shadeFrontRadius,
    camera,
    theme,
  );

  drawCircle3D(k.headPivot, k.forearmAxis, 0.04, theme.rollRing, 1.4, camera, 18);
  drawLine3D(
    vecAdd(k.headPivot, vecScale(k.rollCueDir, 0.05)),
    vecAdd(k.headPivot, vecScale(k.rollCueDir, -0.05)),
    theme.rollTick,
    2.2,
    camera,
  );

  drawJointMarker(k.yawHub, theme.marker1, 4.3, camera);
  drawJointMarker(k.elbow, theme.marker2, 4.0, camera);
  drawJointMarker(k.wrist, theme.marker3, 3.8, camera);
  drawJointMarker(k.headPivot, theme.marker4, 3.8, camera);

  const yawArrow = matVec(k.rotYaw, [0.2, 0, 0]);
  drawLine3D(
    vecAdd(k.yawHub, [0, 0.02, 0]),
    vecAdd(vecAdd(k.yawHub, [0, 0.02, 0]), yawArrow),
    theme.yawArrow,
    2.5,
    camera,
  );
}

function drawBackdrop() {
  const bg = ctx.createLinearGradient(0, 0, 0, canvas.height);
  bg.addColorStop(0.0, "#0d2f3c");
  bg.addColorStop(0.65, "#0a1d28");
  bg.addColorStop(1.0, "#071219");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function sampleFromTrace(trace, idx) {
  const samples = Array.isArray(trace?.samples) ? trace.samples : [];
  if (!samples.length) {
    return null;
  }
  const clamped = clamp(Math.round(idx), 0, samples.length - 1);
  return samples[clamped];
}

function drawScene(sample) {
  resizeCanvas();
  drawBackdrop();

  const camera = makeCamera();
  drawGrid(camera);

  const compareEnabled =
    state.studio.compareEnabled &&
    state.studio.baselineTrace &&
    Array.isArray(state.studio.baselineTrace.samples) &&
    state.studio.baselineTrace.samples.length > 0;
  if (compareEnabled) {
    const baselineSample = sampleFromTrace(state.studio.baselineTrace, state.idx);
    if (state.studio.compareMode === "split" && baselineSample) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, canvas.width * 0.5, canvas.height);
      ctx.clip();
      drawGrid(camera);
      drawLamp(baselineSample, camera, LAMP_THEME_BASELINE);
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.rect(canvas.width * 0.5, 0, canvas.width * 0.5, canvas.height);
      ctx.clip();
      drawGrid(camera);
      drawLamp(sample, camera, LAMP_THEME_DRAFT);
      ctx.restore();

      ctx.strokeStyle = "rgba(179, 212, 222, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(canvas.width * 0.5, 0);
      ctx.lineTo(canvas.width * 0.5, canvas.height);
      ctx.stroke();
    } else {
      if (baselineSample) {
        drawLamp(baselineSample, camera, LAMP_THEME_BASELINE);
      }
      drawLamp(sample, camera, LAMP_THEME_DRAFT);
    }
  } else {
    drawLamp(sample, camera, LAMP_THEME_DRAFT);
  }

  ctx.fillStyle = "rgba(225, 245, 241, 0.82)";
  ctx.font = `${Math.max(14, Math.floor(canvas.height * 0.025))}px Space Grotesk, sans-serif`;
  ctx.fillText("Drag canvas to orbit | Chain: yaw -> pitch1 -> pitch2 -> roll -> pitch3", 16, 28);
  if (compareEnabled) {
    const modeTag = state.studio.compareMode === "split" ? "split" : "overlay";
    ctx.fillStyle = "rgba(193, 235, 228, 0.84)";
    ctx.fillText(`Compare: baseline(cyan) vs draft(orange) [${modeTag}]`, 16, 54);
  }
}

function buildJointRows() {
  jointTableBody.innerHTML = "";
  state.jointRows = [];

  for (let i = 0; i < state.jointNames.length; i += 1) {
    const tr = document.createElement("tr");

    const nameTd = document.createElement("td");
    nameTd.textContent = state.jointNames[i];

    const angleTd = document.createElement("td");
    angleTd.textContent = "0.000";

    const degTd = document.createElement("td");
    degTd.textContent = "0.00";

    const normTd = document.createElement("td");
    const normSpan = document.createElement("span");
    normSpan.className = "norm";
    normSpan.textContent = "0.000";
    normTd.appendChild(normSpan);

    tr.appendChild(nameTd);
    tr.appendChild(angleTd);
    tr.appendChild(degTd);
    tr.appendChild(normTd);
    jointTableBody.appendChild(tr);

    state.jointRows.push({ angleTd, degTd, normSpan });
  }
}

function updateInfo(sample) {
  const t = Number(sample.t_s || 0);
  timeLabel.textContent = `t=${t.toFixed(2)}s`;

  const lines = [
    `segment: ${sample.segment || "n/a"}`,
    `request: ${sample.request_primitive || "n/a"}`,
    `active: ${sample.active_primitive || "none"}`,
    `status: ${sample.status || "n/a"}`,
    `reason: ${sample.reason || ""}`,
    `action_id: ${sample.action_id || ""}`,
  ];
  statusBox.textContent = lines.join("\n");

  const angles = sampleAngles(sample);
  for (let i = 0; i < state.jointRows.length; i += 1) {
    const row = state.jointRows[i];
    const angle = Number(angles[i] || 0);
    row.angleTd.textContent = angle.toFixed(3);
    row.degTd.textContent = (angle * RAD_TO_DEG).toFixed(2);

    const lim = state.jointLimits[i] || [-1.57, 1.57];
    const lo = Number(lim[0]);
    const hi = Number(lim[1]);
    const margin = Math.min(angle - lo, hi - angle);
    row.normSpan.textContent = margin.toFixed(3);
    row.normSpan.className = `norm ${margin < 0 ? "bad" : "ok"}`;
  }
}

function setIndex(nextIdx) {
  if (!state.samples.length) {
    return;
  }
  const idx = clamp(Math.round(nextIdx), 0, state.samples.length - 1);
  state.idx = idx;
  state.playhead = idx;
  timeline.value = String(idx);

  const sample = state.samples[idx];
  updateInfo(sample);
  drawScene(sample);
}

function updateTraceSourceLabel() {
  if (state.studio.compareEnabled && state.studio.baselineSourceLabel && state.studio.draftSourceLabel) {
    traceSource.textContent = `Trace: draft=${state.studio.draftSourceLabel} | baseline=${state.studio.baselineSourceLabel}`;
    return;
  }
  if (state.studio.draftSourceLabel) {
    traceSource.textContent = `Trace: ${state.studio.draftSourceLabel}`;
    return;
  }
  if (state.studio.baselineSourceLabel) {
    traceSource.textContent = `Trace: ${state.studio.baselineSourceLabel}`;
    return;
  }
  traceSource.textContent = "Trace: loading...";
}

function loadTrace(trace, sourceLabel, { asBaseline = false } = {}) {
  if (!trace || !Array.isArray(trace.samples) || trace.samples.length === 0) {
    throw new Error("Trace must contain a non-empty samples array");
  }

  if (asBaseline) {
    state.studio.baselineTrace = trace;
    state.studio.baselineSourceLabel = String(sourceLabel);
    updateTraceSourceLabel();
    if (state.samples.length) {
      drawScene(state.samples[state.idx]);
    }
    return;
  }

  state.samples = trace.samples;

  const namesRaw = trace?.metadata?.joint_names;
  const limitsRaw = trace?.metadata?.joint_limits_rad;
  const dtRaw = trace?.metadata?.dt_s;
  if (!Array.isArray(namesRaw) || !namesRaw.length) {
    throw new Error("Trace metadata.joint_names is required");
  }
  if (!Array.isArray(limitsRaw) || limitsRaw.length !== namesRaw.length) {
    throw new Error("Trace metadata.joint_limits_rad is required and must match joint_names");
  }
  if (!Number.isFinite(Number(dtRaw)) || Number(dtRaw) <= 0) {
    throw new Error("Trace metadata.dt_s must be a positive number");
  }

  const names = namesRaw.map((v) => String(v));
  const limits = limitsRaw;

  state.jointNames = names;
  state.jointLimits = limits;
  state.lampGeom = resolveLampGeometry(trace?.metadata?.lamp_geometry);
  state.dtS = Number(dtRaw);
  state.mapping = resolveMapping(names);

  timeline.min = "0";
  timeline.max = String(state.samples.length - 1);
  timeline.step = "1";

  buildJointRows();

  state.idx = 0;
  state.playhead = 0;
  state.playing = false;
  state.lastFrameMs = null;
  playPauseBtn.textContent = "Play";

  state.studio.draftSourceLabel = String(sourceLabel);
  updateTraceSourceLabel();
  setIndex(0);
}

async function loadFromUrl(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  const json = await res.json();
  loadTrace(json, path);
}

function playPause() {
  state.playing = !state.playing;
  state.lastFrameMs = null;
  playPauseBtn.textContent = state.playing ? "Pause" : "Play";
}

function resetPlayback() {
  state.playing = false;
  state.lastFrameMs = null;
  playPauseBtn.textContent = "Play";
  setIndex(0);
}

function animate(ts) {
  if (state.playing && state.samples.length > 1) {
    if (state.lastFrameMs == null) {
      state.lastFrameMs = ts;
    }

    const dt = (ts - state.lastFrameMs) / 1000;
    state.lastFrameMs = ts;

    const frames = (dt * state.speed) / Math.max(1e-6, state.dtS);
    const next = state.playhead + frames;

    if (next >= state.samples.length - 1) {
      state.playing = false;
      playPauseBtn.textContent = "Play";
      setIndex(state.samples.length - 1);
    } else {
      setIndex(next);
    }
  }

  requestAnimationFrame(animate);
}

async function apiGet(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.text();
  let data = null;
  try {
    data = body ? JSON.parse(body) : {};
  } catch (err) {
    throw new Error(`${res.status} ${res.statusText}: invalid JSON response`);
  }
  if (!res.ok) {
    const code = data?.code ? `${data.code}: ` : "";
    throw new Error(`${code}${data?.error || `${res.status} ${res.statusText}`}`);
  }
  return data;
}

function defaultDurationForPrimitive(primitive) {
  if (primitive === "breath") {
    return 5.0;
  }
  if (primitive === "hold") {
    return 3.0;
  }
  if (primitive === "home") {
    return 2.0;
  }
  if (primitive === "move_to") {
    return 2.4;
  }
  if (primitive === "gaze_to") {
    return 2.0;
  }
  if (primitive === "glance") {
    return 1.2;
  }
  if (primitive === "nod") {
    return 1.5;
  }
  if (primitive === "orient_to_zone") {
    return 1.8;
  }
  return 2.0;
}

function setMode(text) {
  document.getElementById("studioGroup").hidden = text !== "studio";
  modeLabel.textContent = `Mode: ${text}`;
}

function setRunState(kind, detail = "") {
  const next = String(kind || "idle");
  runStateBadge.className = `run-state ${next}`;
  if (detail) {
    runStateBadge.textContent = detail;
    return;
  }
  if (next === "dirty") {
    runStateBadge.textContent = "Unsaved";
    return;
  }
  if (next === "running") {
    runStateBadge.textContent = "Running";
    return;
  }
  if (next === "synced") {
    runStateBadge.textContent = "Synced";
    return;
  }
  if (next === "error") {
    runStateBadge.textContent = "Run Error";
    return;
  }
  runStateBadge.textContent = "Idle";
}

function setBaselineStatus(text, isError = false) {
  baselineStatus.textContent = text;
  baselineStatus.className = isError ? "bad" : "ok";
}

function setSuiteStatus(text, isError = false) {
  suiteStatus.textContent = text;
  suiteStatus.className = isError ? "muted bad" : "muted";
}

function getSelectedSpec() {
  const id = state.studio.selectedPrimitive;
  return state.studio.specs.find((s) => s.id === id) || null;
}

function baselineCommandFor(primitiveId) {
  const src = state.studio.baseline?.primitives?.[primitiveId];
  if (!src || typeof src !== "object") {
    throw new Error(`Missing baseline command for primitive: ${primitiveId}`);
  }
  return deepClone(src);
}

function syncDirtyFromBaseline() {
  const baselineCmd = baselineCommandFor(state.studio.selectedPrimitive);
  state.studio.dirty = stableJson(baselineCmd) !== stableJson(state.studio.commandDraft);
  const dirtyTag = state.studio.dirty ? " (unsaved)" : "";
  const stamp = String(state.studio.baseline?.updated_at_utc || "").trim();
  const stampTag = stamp ? ` | updated ${stamp}` : "";
  setBaselineStatus(`Baseline: ${state.studio.selectedPrimitive}${dirtyTag}${stampTag}`);
}

function clearAutoRunTimer() {
  if (state.studio.autoRunTimer != null) {
    window.clearTimeout(state.studio.autoRunTimer);
    state.studio.autoRunTimer = null;
  }
}

function scheduleStudioRun(reason = "edit") {
  if (!state.studio.enabled) {
    return;
  }
  clearAutoRunTimer();
  if (!state.studio.autoRunEnabled) {
    setRunState("dirty", `Dirty (${reason})`);
    return;
  }
  const waitMs = clamp(Math.round(Number(state.studio.autoRunDebounceMs || 200)), 50, 1000);
  state.studio.autoRunTimer = window.setTimeout(() => {
    state.studio.autoRunTimer = null;
    runStudioPreview({ manual: false, reason: `auto:${reason}` });
  }, waitMs);
}

function markDirty(isDirty, { schedule = true, reason = "edit" } = {}) {
  state.studio.dirty = Boolean(isDirty);
  syncDirtyFromBaseline();
  if (state.studio.dirty && !state.studio.runInFlight) {
    setRunState("dirty");
  }
  if (schedule) {
    scheduleStudioRun(reason);
  }
}

function createNumberInput(value, spec) {
  const input = document.createElement("input");
  input.type = "number";
  input.value = Number.isFinite(Number(value)) ? String(value) : "0";
  if (Number.isFinite(Number(spec.min))) {
    input.min = String(spec.min);
  }
  if (Number.isFinite(Number(spec.max))) {
    input.max = String(spec.max);
  }
  if (Number.isFinite(Number(spec.step))) {
    input.step = String(spec.step);
  }
  return input;
}

function isAngularParam(name) {
  return String(name).toLowerCase().endsWith("_rad");
}

function setNumberInputVisualState(input, isError) {
  if (isError) {
    input.classList.add("field-error");
  } else {
    input.classList.remove("field-error");
  }
}

function clampBySpec(numeric, spec, type) {
  let next = numeric;
  if (Number.isFinite(Number(spec.min))) {
    next = Math.max(next, Number(spec.min));
  }
  if (Number.isFinite(Number(spec.max))) {
    next = Math.min(next, Number(spec.max));
  }
  if (type === "int") {
    next = Math.round(next);
  }
  return next;
}

function updateVectorDraft(name, idx, numeric, spec) {
  const next = clampBySpec(numeric, spec, "float");
  state.studio.commandDraft[name][idx] = next;
}

function makeNudgeButtons() {
  const minusCoarse = document.createElement("button");
  minusCoarse.type = "button";
  minusCoarse.className = "step-btn";
  minusCoarse.textContent = "--";

  const minusFine = document.createElement("button");
  minusFine.type = "button";
  minusFine.className = "step-btn";
  minusFine.textContent = "-";

  const plusFine = document.createElement("button");
  plusFine.type = "button";
  plusFine.className = "step-btn";
  plusFine.textContent = "+";

  const plusCoarse = document.createElement("button");
  plusCoarse.type = "button";
  plusCoarse.className = "step-btn";
  plusCoarse.textContent = "++";

  return { minusCoarse, minusFine, plusFine, plusCoarse };
}

function renderParamForm() {
  paramForm.innerHTML = "";
  const spec = getSelectedSpec();
  if (!spec) {
    return;
  }
  const filterToken = String(state.studio.paramFilter || "").trim().toLowerCase();

  if (!Array.isArray(spec.params) || spec.params.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No command parameters for this primitive.";
    paramForm.appendChild(empty);
    return;
  }

  spec.params.forEach((paramSpec) => {
    const labelText = String(paramSpec.label || paramSpec.name || "");
    const nameText = String(paramSpec.name || "");
    if (filterToken && !labelText.toLowerCase().includes(filterToken) && !nameText.toLowerCase().includes(filterToken)) {
      return;
    }

    const item = document.createElement("div");
    item.className = "param-item";

    const label = document.createElement("label");
    label.className = "param-label";
    label.textContent = paramSpec.label || paramSpec.name;
    item.appendChild(label);

    const name = paramSpec.name;
    const type = paramSpec.type;
    const currentValue = state.studio.commandDraft[name];

    if (type === "bool") {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(currentValue);
      input.addEventListener("change", () => {
        state.studio.commandDraft[name] = input.checked;
        markDirty(true, { reason: name });
      });
      item.appendChild(input);
      paramForm.appendChild(item);
      return;
    }

    if (type === "enum") {
      const select = document.createElement("select");
      const options = Array.isArray(paramSpec.options) ? paramSpec.options : [];
      options.forEach((opt) => {
        const el = document.createElement("option");
        el.value = String(opt);
        el.textContent = String(opt);
        select.appendChild(el);
      });
      const fallback = options.length > 0 ? String(options[0]) : "";
      select.value = currentValue == null ? fallback : String(currentValue);
      select.addEventListener("change", () => {
        state.studio.commandDraft[name] = select.value;
        markDirty(true, { reason: name });
      });
      item.appendChild(select);
      paramForm.appendChild(item);
      return;
    }

    if (type === "vector") {
      const vecWrap = document.createElement("div");
      vecWrap.className = "param-vector";
      const labels = Array.isArray(paramSpec.labels) ? paramSpec.labels : [];
      const mins = Array.isArray(paramSpec.mins) ? paramSpec.mins : [];
      const maxs = Array.isArray(paramSpec.maxs) ? paramSpec.maxs : [];
      const raw = Array.isArray(currentValue) ? currentValue : [];
      const size = Math.max(labels.length, raw.length);
      const values = [];
      for (let i = 0; i < size; i += 1) {
        values.push(Number(raw[i] ?? 0));
      }
      state.studio.commandDraft[name] = values;

      for (let i = 0; i < size; i += 1) {
        const row = document.createElement("div");
        row.className = "param-vector-row";

        const compLabel = document.createElement("span");
        compLabel.className = "muted";
        compLabel.textContent = labels[i] || `joint_${i}`;
        row.appendChild(compLabel);

        const vectorSpec = {
          min: Number.isFinite(Number(mins[i])) ? Number(mins[i]) : undefined,
          max: Number.isFinite(Number(maxs[i])) ? Number(maxs[i]) : undefined,
          step: Number.isFinite(Number(paramSpec.step)) ? Number(paramSpec.step) : 0.01,
        };
        const input = createNumberInput(values[i], vectorSpec);
        const control = document.createElement("div");
        control.className = "param-control-row";
        const nudges = makeNudgeButtons();
        const note = document.createElement("span");
        note.className = "param-value-note";

        const applyValue = (rawValue) => {
          const numeric = Number(rawValue);
          if (!Number.isFinite(numeric)) {
            setNumberInputVisualState(input, true);
            return;
          }
          setNumberInputVisualState(input, false);
          updateVectorDraft(name, i, numeric, vectorSpec);
          input.value = String(state.studio.commandDraft[name][i]);
          note.textContent = `${(Number(state.studio.commandDraft[name][i]) * RAD_TO_DEG).toFixed(1)} deg`;
          markDirty(true, { reason: `${name}[${i}]` });
        };

        input.addEventListener("input", () => {
          applyValue(input.value);
        });

        const stepValue = Number.isFinite(Number(vectorSpec.step)) ? Number(vectorSpec.step) : 0.01;
        bindNumericNudge(input, {
          getValue: () => input.value,
          setValue: (next) => applyValue(next),
          baseStep: stepValue,
        });
        nudges.minusFine.addEventListener("click", (ev) => {
          const baseValue = Number(input.value);
          if (!Number.isFinite(baseValue)) {
            return;
          }
          applyValue(baseValue - scaledStep(stepValue, ev));
        });
        nudges.minusCoarse.addEventListener("click", (ev) => {
          const baseValue = Number(input.value);
          if (!Number.isFinite(baseValue)) {
            return;
          }
          applyValue(baseValue - (scaledStep(stepValue, ev) * 5));
        });
        nudges.plusFine.addEventListener("click", (ev) => {
          const baseValue = Number(input.value);
          if (!Number.isFinite(baseValue)) {
            return;
          }
          applyValue(baseValue + scaledStep(stepValue, ev));
        });
        nudges.plusCoarse.addEventListener("click", (ev) => {
          const baseValue = Number(input.value);
          if (!Number.isFinite(baseValue)) {
            return;
          }
          applyValue(baseValue + (scaledStep(stepValue, ev) * 5));
        });

        control.appendChild(nudges.minusCoarse);
        control.appendChild(nudges.minusFine);
        control.appendChild(input);
        control.appendChild(nudges.plusFine);
        control.appendChild(nudges.plusCoarse);
        control.appendChild(note);
        note.textContent = `${(Number(values[i] || 0) * RAD_TO_DEG).toFixed(1)} deg`;
        row.appendChild(control);
        vecWrap.appendChild(row);
      }

      item.appendChild(vecWrap);
      paramForm.appendChild(item);
      return;
    }

    const numericInput = createNumberInput(currentValue, paramSpec);
    const controlRow = document.createElement("div");
    controlRow.className = "param-control-row";
    const nudges = makeNudgeButtons();
    const note = document.createElement("span");
    note.className = "param-value-note";

    const applyValue = (rawValue) => {
      const numeric = Number(rawValue);
      if (!Number.isFinite(numeric)) {
        setNumberInputVisualState(numericInput, true);
        return;
      }
      setNumberInputVisualState(numericInput, false);
      const next = clampBySpec(numeric, paramSpec, type);
      state.studio.commandDraft[name] = next;
      numericInput.value = String(next);
      if (isAngularParam(name)) {
        note.textContent = `${(next * RAD_TO_DEG).toFixed(1)} deg`;
      } else {
        note.textContent = "";
      }
      markDirty(true, { reason: name });
    };

    numericInput.addEventListener("input", () => {
      applyValue(numericInput.value);
    });

    const stepValue = Number.isFinite(Number(paramSpec.step)) ? Number(paramSpec.step) : type === "int" ? 1 : 0.01;
    bindNumericNudge(numericInput, {
      getValue: () => numericInput.value,
      setValue: (next) => applyValue(next),
      baseStep: stepValue,
    });
    nudges.minusFine.addEventListener("click", (ev) => {
      const baseValue = Number(numericInput.value);
      if (!Number.isFinite(baseValue)) {
        return;
      }
      applyValue(baseValue - scaledStep(stepValue, ev));
    });
    nudges.minusCoarse.addEventListener("click", (ev) => {
      const baseValue = Number(numericInput.value);
      if (!Number.isFinite(baseValue)) {
        return;
      }
      applyValue(baseValue - (scaledStep(stepValue, ev) * 5));
    });
    nudges.plusFine.addEventListener("click", (ev) => {
      const baseValue = Number(numericInput.value);
      if (!Number.isFinite(baseValue)) {
        return;
      }
      applyValue(baseValue + scaledStep(stepValue, ev));
    });
    nudges.plusCoarse.addEventListener("click", (ev) => {
      const baseValue = Number(numericInput.value);
      if (!Number.isFinite(baseValue)) {
        return;
      }
      applyValue(baseValue + (scaledStep(stepValue, ev) * 5));
    });

    controlRow.appendChild(nudges.minusCoarse);
    controlRow.appendChild(nudges.minusFine);
    controlRow.appendChild(numericInput);
    controlRow.appendChild(nudges.plusFine);
    controlRow.appendChild(nudges.plusCoarse);
    controlRow.appendChild(note);
    if (isAngularParam(name)) {
      note.textContent = `${(Number(currentValue || 0) * RAD_TO_DEG).toFixed(1)} deg`;
    }
    item.appendChild(controlRow);
    paramForm.appendChild(item);
  });

  if (paramForm.childElementCount === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No parameters match this filter.";
    paramForm.appendChild(empty);
  }
}

function buildStudioPayload(command) {
  const primitive = state.studio.selectedPrimitive;
  const style = styleSelect.value || "calm";
  const durationS = Number(durationInput.value);
  return {
    primitive,
    style,
    duration_s: Number.isFinite(durationS) ? durationS : defaultDurationForPrimitive(primitive),
    command,
  };
}

function setStudioPrimitive(nextPrimitive, { resetDuration = true, schedule = true } = {}) {
  state.studio.selectedPrimitive = String(nextPrimitive);
  primitiveSelect.value = state.studio.selectedPrimitive;
  state.studio.commandDraft = baselineCommandFor(state.studio.selectedPrimitive);
  if (resetDuration) {
    durationInput.value = String(defaultDurationForPrimitive(state.studio.selectedPrimitive));
  }
  renderParamForm();
  markDirty(false, { schedule, reason: "primitive" });
  saveStudioPrefs();
}

async function runStudioPreview({ manual = false, reason = "manual" } = {}) {
  if (!state.studio.enabled) {
    return false;
  }
  clearAutoRunTimer();
  if (state.studio.runInFlight) {
    state.studio.rerunQueued = true;
    return false;
  }

  let ok = false;
  state.studio.runInFlight = true;
  const token = ++state.studio.requestToken;
  runBtn.disabled = true;
  setRunState("running", manual ? "Running" : "Auto-running");
  statusBox.textContent = `Studio run: ${reason}`;
  notifyShell("ok", `Studio run started (${reason})`);

  try {
    const primitive = state.studio.selectedPrimitive;
    const draftPayload = buildStudioPayload(state.studio.commandDraft);

    let draftTrace;
    let baselineTrace = null;
    if (state.studio.compareEnabled) {
      const baselinePayload = buildStudioPayload(baselineCommandFor(primitive));
      [draftTrace, baselineTrace] = await Promise.all([
        apiPost("/api/simulate", draftPayload),
        apiPost("/api/simulate", baselinePayload),
      ]);
    } else {
      draftTrace = await apiPost("/api/simulate", draftPayload);
    }

    if (token !== state.studio.requestToken) {
      return;
    }

    if (baselineTrace) {
      loadTrace(baselineTrace, `baseline:${primitive}`, { asBaseline: true });
    } else {
      state.studio.baselineTrace = null;
      state.studio.baselineSourceLabel = "";
      updateTraceSourceLabel();
    }

    loadTrace(draftTrace, `studio:${primitive}`);
    syncDirtyFromBaseline();
    setRunState("synced", state.studio.dirty ? "Preview (unsaved)" : "Preview synced");
    notifyShell("ok", `Studio preview synced (${primitive})`);
    ok = true;
  } catch (err) {
    setRunState("error", "Run Error");
    statusBox.textContent = `Studio run failed: ${err}`;
    notifyShell("bad", "Studio preview failed", String(err));
  } finally {
    state.studio.runInFlight = false;
    runBtn.disabled = false;
    if (state.studio.rerunQueued) {
      state.studio.rerunQueued = false;
      runStudioPreview({ manual: false, reason: "queued" });
    }
  }
  return ok;
}

async function saveBaseline() {
  if (!state.studio.enabled) {
    return;
  }
  saveBaselineBtn.disabled = true;
  try {
    const updated = await apiPost("/api/baseline", {
      primitive: state.studio.selectedPrimitive,
      command: state.studio.commandDraft,
    });
    state.studio.baseline = updated;
    state.studio.commandDraft = baselineCommandFor(state.studio.selectedPrimitive);
    renderParamForm();
    syncDirtyFromBaseline();
    const stamp = String(updated?.updated_at_utc || new Date().toISOString());
    setBaselineStatus(`Baseline saved: ${state.studio.selectedPrimitive} at ${stamp}`);
    setRunState("synced", "Baseline saved");
    notifyShell("ok", `Baseline saved (${state.studio.selectedPrimitive})`);
    if (state.studio.compareEnabled) {
      runStudioPreview({ manual: false, reason: "baseline_saved" });
    }
  } catch (err) {
    setBaselineStatus(`Save failed: ${err}`, true);
    setRunState("error", "Save Error");
    notifyShell("bad", "Baseline save failed", String(err));
  } finally {
    saveBaselineBtn.disabled = false;
  }
}

async function saveAllBaselines() {
  if (!state.studio.enabled) {
    return;
  }
  saveAllBaselineBtn.disabled = true;
  try {
    const nextPrimitives = deepClone(state.studio.baseline?.primitives || {});
    nextPrimitives[state.studio.selectedPrimitive] = deepClone(state.studio.commandDraft);
    const updated = await apiPost("/api/baseline", { primitives: nextPrimitives });
    state.studio.baseline = updated;
    state.studio.commandDraft = baselineCommandFor(state.studio.selectedPrimitive);
    renderParamForm();
    syncDirtyFromBaseline();
    const stamp = String(updated?.updated_at_utc || new Date().toISOString());
    setBaselineStatus(`Baseline saved (all): ${stamp}`);
    setRunState("synced", "Baseline saved");
    notifyShell("ok", "All baselines saved");
    if (state.studio.compareEnabled) {
      runStudioPreview({ manual: false, reason: "baseline_save_all" });
    }
  } catch (err) {
    setBaselineStatus(`Save all failed: ${err}`, true);
    setRunState("error", "Save Error");
    notifyShell("bad", "Save all baselines failed", String(err));
  } finally {
    saveAllBaselineBtn.disabled = false;
  }
}

async function runAndSaveBaseline() {
  if (!state.studio.enabled) {
    return;
  }
  runSaveBtn.disabled = true;
  try {
    const ok = await runStudioPreview({ manual: true, reason: "run_and_save" });
    if (!ok) {
      return;
    }
    await saveBaseline();
  } finally {
    runSaveBtn.disabled = false;
  }
}

async function runSuitePlayback() {
  runSuiteBtn.disabled = true;
  const style = state.studio.enabled ? String(styleSelect.value || "calm") : "calm";
  setSuiteStatus(`Suite: generating (${style})...`);
  try {
    const res = await apiPost("/api/suite", { style });
    const viewerUrl = String(res?.viewer_url || "");
    setSuiteStatus(`Suite: ready (${Number(res?.sample_count || 0)} samples). opening...`);
    if (navigateViewerUrl(viewerUrl)) {
      return;
    }
    setSuiteStatus("Suite complete, but no viewer URL returned.", true);
  } catch (err) {
    setSuiteStatus(`Suite failed: ${err}`, true);
  } finally {
    runSuiteBtn.disabled = false;
  }
}

function resetToBaseline() {
  if (!state.studio.enabled) {
    return;
  }
  state.studio.commandDraft = baselineCommandFor(state.studio.selectedPrimitive);
  renderParamForm();
  markDirty(false, { reason: "reset_to_baseline" });
  notifyShell("ok", `Reset to baseline (${state.studio.selectedPrimitive})`);
}

async function initStudioMode() {
  const [meta, baseline] = await Promise.all([
    apiGet("/api/primitives"),
    apiGet("/api/baseline"),
  ]);

  if (!Array.isArray(meta?.primitives) || !meta.primitives.length) {
    throw new Error("missing primitive metadata");
  }
  if (!Array.isArray(meta.styles) || !meta.styles.length) {
    throw new Error("missing style metadata");
  }
  if (typeof meta.default_primitive !== "string" || !meta.default_primitive.trim()) {
    throw new Error("missing default primitive metadata");
  }

  state.studio.enabled = true;
  state.studio.specs = meta.primitives;
  state.studio.styles = meta.styles;
  state.studio.baseline = baseline;

  primitiveSelect.innerHTML = "";
  state.studio.specs.forEach((spec) => {
    const option = document.createElement("option");
    option.value = spec.id;
    option.textContent = spec.label || spec.id;
    primitiveSelect.appendChild(option);
  });

  styleSelect.innerHTML = "";
  state.studio.styles.forEach((style) => {
    const option = document.createElement("option");
    option.value = style;
    option.textContent = style;
    styleSelect.appendChild(option);
  });
  const prefs = applyStudioPrefs(meta);
  setStudioPrimitive(String(prefs.primitive || meta.default_primitive), { schedule: false });
  if (Number.isFinite(Number(prefs.durationS)) && Number(prefs.durationS) > 0) {
    durationInput.value = String(Number(prefs.durationS));
  }
  setMode("studio");
  syncDirtyFromBaseline();
  setRunState("idle");
  saveStudioPrefs();
  await runStudioPreview({ manual: true, reason: "init" });
}

playPauseBtn.addEventListener("click", playPause);
resetBtn.addEventListener("click", resetPlayback);

speedSelect.addEventListener("change", () => {
  state.speed = Number(speedSelect.value || 1);
  saveStudioPrefs();
});

timeline.addEventListener("input", () => {
  state.playing = false;
  playPauseBtn.textContent = "Play";
  state.lastFrameMs = null;
  setIndex(Number(timeline.value));
});

primitiveSelect.addEventListener("change", () => {
  setStudioPrimitive(primitiveSelect.value, { resetDuration: true, schedule: true });
});

runBtn.addEventListener("click", () => {
  runStudioPreview({ manual: true, reason: "manual_click" });
});

runSaveBtn.addEventListener("click", () => {
  runAndSaveBaseline();
});

saveBaselineBtn.addEventListener("click", () => {
  saveBaseline();
});

saveAllBaselineBtn.addEventListener("click", () => {
  saveAllBaselines();
});

resetBaselineBtn.addEventListener("click", () => {
  resetToBaseline();
});

runSuiteBtn.addEventListener("click", () => {
  runSuitePlayback();
});

if (paramFilterInput) {
  paramFilterInput.addEventListener("input", () => {
    state.studio.paramFilter = String(paramFilterInput.value || "").trim();
    renderParamForm();
    saveStudioPrefs();
  });
}

zoomOutBtn.addEventListener("click", () => {
  adjustCameraDistance(0.18);
});
zoomInBtn.addEventListener("click", () => {
  adjustCameraDistance(-0.18);
});
orbitLeftBtn.addEventListener("click", () => {
  adjustCameraOrbit(0.1);
});
orbitRightBtn.addEventListener("click", () => {
  adjustCameraOrbit(-0.1);
});
resetViewBtn.addEventListener("click", () => {
  resetCameraView();
});

styleSelect.addEventListener("change", () => {
  if (!state.studio.enabled) {
    return;
  }
  saveStudioPrefs();
  scheduleStudioRun("style");
});

durationInput.addEventListener("input", () => {
  if (!state.studio.enabled) {
    return;
  }
  const n = Number(durationInput.value);
  if (!Number.isFinite(n) || n <= 0) {
    durationInput.value = String(defaultDurationForPrimitive(state.studio.selectedPrimitive));
  }
  saveStudioPrefs();
  scheduleStudioRun("duration");
});

autoRunToggle.addEventListener("change", () => {
  state.studio.autoRunEnabled = autoRunToggle.checked;
  saveStudioPrefs();
  if (state.studio.autoRunEnabled && state.studio.dirty) {
    scheduleStudioRun("auto_run_enabled");
  }
});

autoRunMsInput.addEventListener("input", () => {
  const parsed = Number(autoRunMsInput.value);
  state.studio.autoRunDebounceMs = clamp(Math.round(Number.isFinite(parsed) ? parsed : 200), 50, 1000);
  autoRunMsInput.value = String(state.studio.autoRunDebounceMs);
  saveStudioPrefs();
});

compareToggle.addEventListener("change", () => {
  state.studio.compareEnabled = compareToggle.checked;
  saveStudioPrefs();
  scheduleStudioRun("compare_toggle");
});

compareModeSelect.addEventListener("change", () => {
  state.studio.compareMode = String(compareModeSelect.value || "overlay");
  saveStudioPrefs();
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  const json = JSON.parse(text);
  loadTrace(json, file.name);
});

canvas.addEventListener("pointerdown", (ev) => {
  state.drag.active = true;
  state.drag.lastX = ev.clientX;
  canvas.setPointerCapture(ev.pointerId);
});

canvas.addEventListener("pointermove", (ev) => {
  if (!state.drag.active) {
    return;
  }
  const dx = ev.clientX - state.drag.lastX;
  state.drag.lastX = ev.clientX;
  state.cameraOrbit -= dx * 0.006;
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
});

canvas.addEventListener("wheel", (ev) => {
  const direction = ev.deltaY > 0 ? 1 : -1;
  adjustCameraDistance(direction * 0.16);
  ev.preventDefault();
}, { passive: false });

canvas.addEventListener("pointerup", (ev) => {
  state.drag.active = false;
  canvas.releasePointerCapture(ev.pointerId);
});

window.addEventListener("resize", () => {
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
});

window.addEventListener("keydown", (ev) => {
  const key = String(ev.key || "").toLowerCase();
  const mod = ev.metaKey || ev.ctrlKey;
  const editable = isEditableTarget(ev.target);

  if (mod && key === "enter") {
    ev.preventDefault();
    if (ev.shiftKey) {
      runAndSaveBaseline();
    } else {
      runStudioPreview({ manual: true, reason: "kbd_run_preview" });
    }
    return;
  }

  if (mod && key === "s") {
    ev.preventDefault();
    if (ev.shiftKey) {
      saveAllBaselines();
    } else {
      saveBaseline();
    }
    return;
  }

  if (editable) {
    return;
  }

  if (!mod && !ev.altKey && key === "r" && state.studio.enabled) {
    ev.preventDefault();
    resetToBaseline();
    return;
  }

  if (!mod && !ev.altKey && key === " ") {
    ev.preventDefault();
    playPause();
    return;
  }

  if (key === "=" || key === "+") {
    ev.preventDefault();
    adjustCameraDistance(-0.18);
    return;
  }
  if (key === "-" || key === "_") {
    ev.preventDefault();
    adjustCameraDistance(0.18);
    return;
  }
  if (key === "[" && !editable) {
    ev.preventDefault();
    adjustCameraOrbit(0.1);
    return;
  }
  if (key === "]" && !editable) {
    ev.preventDefault();
    adjustCameraOrbit(-0.1);
  }
});

window.addEventListener("message", (event) => {
  if (event.origin !== window.location.origin) {
    return;
  }
  const msg = event.data;
  if (!msg || typeof msg !== "object") {
    return;
  }
  if (msg.source !== "lamp-shell" || msg.type !== "command") {
    return;
  }
  const cmd = String(msg.command || "");
  if (cmd === "studio.run_preview") {
    runStudioPreview({ manual: true, reason: "shell_command" });
    return;
  }
  if (cmd === "studio.save_baseline") {
    saveBaseline();
    return;
  }
  if (cmd === "studio.save_all") {
    saveAllBaselines();
    return;
  }
  if (cmd === "studio.reset_baseline") {
    resetToBaseline();
  }
});

(async function bootstrap() {
  const tracePath = pageParams.get("trace") || "";
  const preferStudio = pageParams.get("studio") !== "0";

  try {
    if (preferStudio) {
      await initStudioMode();
      if (tracePath) {
        await loadFromUrl(tracePath);
        setMode("playback");
      }
    } else {
      if (!tracePath) {
        throw new Error("trace query parameter is required when studio=0");
      }
      await loadFromUrl(tracePath);
      setMode("playback");
    }
  } catch (err) {
    setMode("error");
    traceSource.textContent = "Trace: load failed";
    statusBox.textContent = `Bootstrap error: ${err}`;
  }

  requestAnimationFrame(animate);
})();

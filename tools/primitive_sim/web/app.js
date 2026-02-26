const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");

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

const studioGroup = document.getElementById("studioGroup");
const primitiveSelect = document.getElementById("primitiveSelect");
const styleSelect = document.getElementById("styleSelect");
const durationInput = document.getElementById("durationInput");
const paramForm = document.getElementById("paramForm");
const baselineStatus = document.getElementById("baselineStatus");
const runBtn = document.getElementById("runBtn");
const resetBaselineBtn = document.getElementById("resetBaselineBtn");
const saveBaselineBtn = document.getElementById("saveBaselineBtn");
const runSuiteBtn = document.getElementById("runSuiteBtn");
const suiteStatus = document.getElementById("suiteStatus");

const state = {
  trace: null,
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
    pitch: [],
  },
  studio: {
    enabled: false,
    specs: [],
    styles: [],
    baseline: null,
    selectedPrimitive: "",
    commandDraft: {},
    dirty: false,
  },
};

const RAD_TO_DEG = 180 / Math.PI;

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

function deepClone(v) {
  return JSON.parse(JSON.stringify(v));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
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
  const camPos = [2.85 * Math.cos(orbit), 1.58, 2.85 * Math.sin(orbit)];
  const target = [0, 0.72, 0];

  const forward = norm(vecSub(target, camPos));
  const right = norm(cross(forward, [0, 1, 0]));
  const up = norm(cross(right, forward));
  return { camPos, forward, right, up };
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

function drawShadeFrustum(rearCenter, frontCenter, axis, rearR, frontR, camera) {
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
    drawLine3D(rearPts[i], rearPts[(i + 1) % segs], "rgba(253, 232, 185, 0.95)", 1.8, camera);
    drawLine3D(frontPts[i], frontPts[(i + 1) % segs], "rgba(255, 214, 150, 0.95)", 1.4, camera);
  }
  const struts = [0, 4, 8, 12];
  struts.forEach((i) => {
    drawLine3D(rearPts[i], frontPts[i], "rgba(255, 206, 128, 0.85)", 1.5, camera);
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
    pitch: [],
  };

  const used = new Set(
    [mapping.yaw, mapping.roll, mapping.pitch1, mapping.pitch2, mapping.pitch3].filter((v) => v >= 0),
  );
  const genericPitch = [];
  lower.forEach((name, idx) => {
    if (name.includes("pitch") && !used.has(idx)) {
      genericPitch.push(idx);
    }
  });

  const fallback = [];
  for (let i = 0; i < names.length; i += 1) {
    if (!used.has(i)) {
      fallback.push(i);
    }
  }

  ["pitch1", "pitch2", "pitch3"].forEach((slot) => {
    if (mapping[slot] >= 0) {
      return;
    }
    if (genericPitch.length > 0) {
      mapping[slot] = genericPitch.shift();
      return;
    }
    mapping[slot] = fallback.length > 0 ? fallback.shift() : -1;
  });

  mapping.pitch = [mapping.pitch1, mapping.pitch2, mapping.pitch3].filter((idx) => idx >= 0);
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

function drawLamp(sample, camera) {
  const k = forwardKinematics(sample);
  const g = state.lampGeom || LAMP_GEOM;

  drawCircle3D(k.baseCenter, [0, 1, 0], g.baseRadius, "rgba(240, 225, 194, 0.92)", 3.6, camera, 36);
  drawCircle3D(k.baseCenter, [0, 1, 0], g.baseRadius * 0.58, "rgba(198, 170, 126, 0.86)", 1.8, camera, 28);
  drawLine3D(k.mastBottom, k.mastTop, "rgba(227, 232, 237, 0.96)", 8, camera);

  drawCircle3D(k.yawHub, [0, 1, 0], 0.055, "rgba(178, 198, 212, 0.84)", 1.9, camera, 24);
  drawLine3D(k.yawHub, vecAdd(k.yawHub, [0, 0.04, 0]), "rgba(182, 198, 212, 0.76)", 3, camera);

  drawLine3D(k.yawHub, k.elbow, "#d8e2e6", 6.5, camera);
  drawLine3D(k.elbow, k.wrist, "#dbe6eb", 6.2, camera);
  drawLine3D(k.wrist, k.headPivot, "#cfd8de", 5.2, camera);
  drawLine3D(k.headPivot, k.shadeRear, "#becbd4", 4.2, camera);

  drawShadeFrustum(
    k.shadeRear,
    k.shadeFront,
    vecSub(k.shadeFront, k.shadeRear),
    g.shadeRearRadius,
    g.shadeFrontRadius,
    camera,
  );

  drawCircle3D(k.headPivot, k.forearmAxis, 0.04, "rgba(255, 154, 61, 0.9)", 1.4, camera, 18);
  drawLine3D(
    vecAdd(k.headPivot, vecScale(k.rollCueDir, 0.05)),
    vecAdd(k.headPivot, vecScale(k.rollCueDir, -0.05)),
    "rgba(255, 154, 61, 0.92)",
    2.2,
    camera,
  );

  drawJointMarker(k.yawHub, "#a8c5d6", 4.3, camera);
  drawJointMarker(k.elbow, "#8bb8d7", 4.0, camera);
  drawJointMarker(k.wrist, "#94d4d0", 3.8, camera);
  drawJointMarker(k.headPivot, "#f7ce86", 3.8, camera);

  const yawArrow = matVec(k.rotYaw, [0.2, 0, 0]);
  drawLine3D(
    vecAdd(k.yawHub, [0, 0.02, 0]),
    vecAdd(vecAdd(k.yawHub, [0, 0.02, 0]), yawArrow),
    "rgba(255, 154, 61, 0.95)",
    2.5,
    camera,
  );
}

function drawScene(sample) {
  resizeCanvas();

  const g = ctx.createLinearGradient(0, 0, 0, canvas.height);
  g.addColorStop(0.0, "#0d2f3c");
  g.addColorStop(0.65, "#0a1d28");
  g.addColorStop(1.0, "#071219");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const camera = makeCamera();
  drawGrid(camera);
  drawLamp(sample, camera);

  ctx.fillStyle = "rgba(225, 245, 241, 0.82)";
  ctx.font = `${Math.max(14, Math.floor(canvas.height * 0.025))}px Space Grotesk, sans-serif`;
  ctx.fillText("Drag canvas to orbit | Chain: yaw -> pitch1 -> pitch2 -> roll -> pitch3", 16, 28);
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

function loadTrace(trace, sourceLabel) {
  if (!trace || !Array.isArray(trace.samples) || trace.samples.length === 0) {
    throw new Error("Trace must contain a non-empty samples array");
  }

  state.trace = trace;
  state.samples = trace.samples;

  const names = Array.isArray(trace?.metadata?.joint_names)
    ? trace.metadata.joint_names.map((v) => String(v))
    : trace.samples[0].joint_angles_rad.map((_, i) => `joint_${i}`);

  const limits = Array.isArray(trace?.metadata?.joint_limits_rad)
    ? trace.metadata.joint_limits_rad
    : names.map(() => [-1.57, 1.57]);

  state.jointNames = names;
  state.jointLimits = limits;
  state.lampGeom = resolveLampGeometry(trace?.metadata?.lamp_geometry);
  state.dtS = Number(trace?.metadata?.dt_s || 1 / 60);
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

  traceSource.textContent = `Trace: ${sourceLabel}`;
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
    throw new Error(data?.error || `${res.status} ${res.statusText}`);
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
  modeLabel.textContent = `Mode: ${text}`;
}

function setBaselineStatus(text, isError = false) {
  baselineStatus.textContent = text;
  baselineStatus.className = isError ? "bad" : "ok";
}

function setSuiteStatus(text, isError = false) {
  if (!suiteStatus) {
    return;
  }
  suiteStatus.textContent = text;
  suiteStatus.className = isError ? "muted bad" : "muted";
}

function getSelectedSpec() {
  const id = state.studio.selectedPrimitive;
  return state.studio.specs.find((s) => s.id === id) || null;
}

function baselineCommandFor(primitiveId) {
  const src = state.studio.baseline?.primitives?.[primitiveId];
  if (src && typeof src === "object") {
    return deepClone(src);
  }
  return {};
}

function markDirty(isDirty) {
  state.studio.dirty = Boolean(isDirty);
  const dirtyTag = state.studio.dirty ? " (unsaved)" : "";
  setBaselineStatus(`Baseline: ${state.studio.selectedPrimitive}${dirtyTag}`);
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

function renderParamForm() {
  paramForm.innerHTML = "";
  const spec = getSelectedSpec();
  if (!spec) {
    return;
  }

  if (!Array.isArray(spec.params) || spec.params.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No command parameters for this primitive.";
    paramForm.appendChild(empty);
    return;
  }

  spec.params.forEach((paramSpec) => {
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
        markDirty(true);
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
        markDirty(true);
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

        const input = createNumberInput(values[i], {
          min: Number.isFinite(Number(mins[i])) ? Number(mins[i]) : undefined,
          max: Number.isFinite(Number(maxs[i])) ? Number(maxs[i]) : undefined,
          step: Number.isFinite(Number(paramSpec.step)) ? Number(paramSpec.step) : 0.01,
        });
        input.addEventListener("input", () => {
          const next = Number(input.value);
          state.studio.commandDraft[name][i] = Number.isFinite(next) ? next : 0;
          markDirty(true);
        });
        row.appendChild(input);
        vecWrap.appendChild(row);
      }

      item.appendChild(vecWrap);
      paramForm.appendChild(item);
      return;
    }

    const numericInput = createNumberInput(currentValue, paramSpec);
    numericInput.addEventListener("input", () => {
      let next = Number(numericInput.value);
      if (!Number.isFinite(next)) {
        return;
      }
      if (type === "int") {
        next = Math.round(next);
      }
      state.studio.commandDraft[name] = next;
      markDirty(true);
    });
    item.appendChild(numericInput);
    paramForm.appendChild(item);
  });
}

function setStudioPrimitive(nextPrimitive, { resetDuration = true } = {}) {
  state.studio.selectedPrimitive = String(nextPrimitive);
  primitiveSelect.value = state.studio.selectedPrimitive;
  state.studio.commandDraft = baselineCommandFor(state.studio.selectedPrimitive);
  if (resetDuration) {
    durationInput.value = String(defaultDurationForPrimitive(state.studio.selectedPrimitive));
  }
  renderParamForm();
  markDirty(false);
}

async function runStudioPreview() {
  if (!state.studio.enabled) {
    return;
  }
  const primitive = state.studio.selectedPrimitive;
  const style = styleSelect.value || "calm";
  const durationS = Number(durationInput.value);
  const payload = {
    primitive,
    style,
    duration_s: Number.isFinite(durationS) ? durationS : defaultDurationForPrimitive(primitive),
    command: state.studio.commandDraft,
  };

  runBtn.disabled = true;
  try {
    const trace = await apiPost("/api/simulate", payload);
    loadTrace(trace, `studio:${primitive}`);
  } catch (err) {
    statusBox.textContent = `Studio run failed: ${err}`;
  } finally {
    runBtn.disabled = false;
  }
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
    markDirty(false);
    const stamp = new Date().toLocaleTimeString();
    setBaselineStatus(`Baseline saved: ${state.studio.selectedPrimitive} at ${stamp}`);
  } catch (err) {
    setBaselineStatus(`Save failed: ${err}`, true);
  } finally {
    saveBaselineBtn.disabled = false;
  }
}

async function runSuitePlayback() {
  if (!runSuiteBtn) {
    return;
  }
  runSuiteBtn.disabled = true;
  const style = state.studio.enabled ? String(styleSelect.value || "calm") : "calm";
  setSuiteStatus(`Suite: generating (${style})...`);
  try {
    const res = await apiPost("/api/suite", { style });
    const viewerUrl = String(res?.viewer_url || "");
    setSuiteStatus(`Suite: ready (${Number(res?.sample_count || 0)} samples). opening...`);
    if (viewerUrl) {
      window.location.href = viewerUrl;
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
  markDirty(false);
}

async function initStudioMode() {
  const [meta, baseline] = await Promise.all([
    apiGet("/api/primitives"),
    apiGet("/api/baseline"),
  ]);

  if (!Array.isArray(meta?.primitives) || !meta.primitives.length) {
    throw new Error("missing primitive metadata");
  }

  state.studio.enabled = true;
  state.studio.specs = meta.primitives;
  state.studio.styles = Array.isArray(meta.styles) && meta.styles.length ? meta.styles : ["calm"];
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
  styleSelect.value = state.studio.styles.includes("calm") ? "calm" : state.studio.styles[0];

  const defaultPrimitive =
    typeof meta.default_primitive === "string" && meta.default_primitive
      ? meta.default_primitive
      : state.studio.specs[0].id;

  setStudioPrimitive(defaultPrimitive);
  setMode("studio");
  setBaselineStatus(`Baseline: ${state.studio.selectedPrimitive}`);
  await runStudioPreview();
}

playPauseBtn.addEventListener("click", playPause);
resetBtn.addEventListener("click", resetPlayback);

speedSelect.addEventListener("change", () => {
  state.speed = Number(speedSelect.value || 1);
});

timeline.addEventListener("input", () => {
  state.playing = false;
  playPauseBtn.textContent = "Play";
  state.lastFrameMs = null;
  setIndex(Number(timeline.value));
});

primitiveSelect.addEventListener("change", () => {
  setStudioPrimitive(primitiveSelect.value, { resetDuration: true });
});

runBtn.addEventListener("click", () => {
  runStudioPreview();
});

saveBaselineBtn.addEventListener("click", () => {
  saveBaseline();
});

resetBaselineBtn.addEventListener("click", () => {
  resetToBaseline();
});

if (runSuiteBtn) {
  runSuiteBtn.addEventListener("click", () => {
    runSuitePlayback();
  });
}

styleSelect.addEventListener("change", () => {
  if (state.studio.enabled) {
    setBaselineStatus(`Baseline: ${state.studio.selectedPrimitive}${state.studio.dirty ? " (unsaved)" : ""}`);
  }
});

durationInput.addEventListener("input", () => {
  if (!state.studio.enabled) {
    return;
  }
  const n = Number(durationInput.value);
  if (!Number.isFinite(n) || n <= 0) {
    durationInput.value = String(defaultDurationForPrimitive(state.studio.selectedPrimitive));
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

canvas.addEventListener("pointerup", (ev) => {
  state.drag.active = false;
  canvas.releasePointerCapture(ev.pointerId);
});

window.addEventListener("resize", () => {
  if (state.samples.length) {
    drawScene(state.samples[state.idx]);
  }
});

(async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const tracePath = params.get("trace") || "";
  const preferStudio = params.get("studio") !== "0";

  if (preferStudio) {
    try {
      await initStudioMode();
    } catch (err) {
      state.studio.enabled = false;
      setMode("playback (studio api unavailable)");
      setBaselineStatus(`Studio unavailable: ${err}`, true);
      if (!tracePath) {
        studioGroup.classList.add("disabled");
      }
    }
  } else {
    setMode("playback");
  }

  if (tracePath) {
    try {
      await loadFromUrl(tracePath);
      setMode("playback");
    } catch (err) {
      traceSource.textContent = `Trace: failed to load ${tracePath}`;
      statusBox.textContent = `Load error: ${err}`;
    }
  } else if (!state.samples.length && !state.studio.enabled) {
    traceSource.textContent = "Trace: no trace selected";
    statusBox.textContent = "No studio API found and no trace URL provided.";
  }

  requestAnimationFrame(animate);
})();

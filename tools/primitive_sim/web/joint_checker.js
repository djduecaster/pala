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
const configPathEl = document.getElementById("configPath");
const statusBox = document.getElementById("statusBox");
const jointSliders = document.getElementById("jointSliders");
const jointTableBody = document.querySelector("#jointTable tbody");
const dhTableBody = document.querySelector("#dhTable tbody");
const zeroBtn = document.getElementById("zeroBtn");
const midBtn = document.getElementById("midBtn");
const copyBtn = document.getElementById("copyBtn");
const runSuiteBtn = document.getElementById("runSuiteBtn");
const suiteStatus = document.getElementById("suiteStatus");
const zoomOutBtn = document.getElementById("zoomOutBtn");
const zoomInBtn = document.getElementById("zoomInBtn");
const orbitLeftBtn = document.getElementById("orbitLeftBtn");
const orbitRightBtn = document.getElementById("orbitRightBtn");
const resetViewBtn = document.getElementById("resetViewBtn");

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

const DEG = 180 / Math.PI;
const RAD = Math.PI / 180;
const SLIDER_STEP_RAD = 0.5 * RAD;

const LAMP_GEOM = {
  baseRadius: 0.18,
  baseThickness: 0.028,
  mastHeight: 1.28,
  hubRise: 0.0635,
  upperArmLen: 0.38735,
  foreArmLen: 0.32385,
  wristStubLen: 0.08,
  shadeNeckLen: 0.08,
  shadeLen: 0.18,
  shadeRearRadius: 0.068,
  shadeFrontRadius: 0.046,
  pitch1ZeroOffsetRad: -Math.PI / 2,
  pitch2ZeroOffsetRad: Math.PI / 2,
  pitch3ZeroOffsetRad: 0.0,
};

const LAMP_GEOM_KEYS = new Set(Object.keys(LAMP_GEOM));
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

const state = {
  jointNames: [],
  jointLimits: [],
  jointAngles: [],
  sliderRows: [],
  tableRows: [],
  mapping: { yaw: -1, roll: -1, pitch1: -1, pitch2: -1, pitch3: -1 },
  lampGeom: { ...LAMP_GEOM },
  dhParams: {},
  configPath: "",
  cameraOrbit: -0.64,
  cameraDistance: 3.75,
  drag: { active: false, lastX: 0 },
};

function isEditableTarget(target) {
  if (!(target instanceof Element)) {
    return false;
  }
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
    return true;
  }
  return Boolean(target.closest("[contenteditable='true']"));
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
        mode: "joint_checker",
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

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function radToDeg(v) {
  return v * DEG;
}

function degToRad(v) {
  return v * RAD;
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
  const r = state.cameraDistance;
  const camPos = [r * Math.cos(orbit), 1.64, r * Math.sin(orbit)];
  const target = [0, 0.82, 0];
  const forward = norm(vecSub(target, camPos));
  const right = norm(cross(forward, [0, 1, 0]));
  const up = norm(cross(right, forward));
  return { camPos, forward, right, up };
}

function adjustCameraDistance(delta) {
  const next = state.cameraDistance + Number(delta || 0);
  state.cameraDistance = clamp(next, 2.2, 6.0);
  drawScene();
}

function adjustCameraOrbit(delta) {
  state.cameraOrbit += Number(delta || 0);
  drawScene();
}

function resetCameraView() {
  state.cameraOrbit = -0.64;
  state.cameraDistance = 3.75;
  drawScene();
}

function projectPoint(p, camera) {
  const v = vecSub(p, camera.camPos);
  const x = dot(v, camera.right);
  const y = dot(v, camera.up);
  const z = dot(v, camera.forward);
  if (z <= 0.05) {
    return null;
  }
  const f = Math.min(canvas.width, canvas.height) * 0.88;
  const sx = canvas.width * 0.5 + (x / z) * f;
  const sy = canvas.height * 0.58 - (y / z) * f;
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
  for (let i = -6; i <= 6; i += 1) {
    const t = i * 0.24;
    const alpha = i === 0 ? 0.30 : 0.14;
    const color = `rgba(99, 160, 178, ${alpha})`;
    drawLine3D([-1.5, 0, t], [1.5, 0, t], color, 1, camera);
    drawLine3D([t, 0, -1.5], [t, 0, 1.5], color, 1, camera);
  }
}

function orthonormalBasis(axis) {
  const n = norm(axis);
  const ref = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = norm(cross(n, ref));
  const v = norm(cross(n, u));
  return { u, v };
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
  [0, 4, 8, 12].forEach((i) => {
    drawLine3D(rearPts[i], frontPts[i], "rgba(255, 206, 128, 0.85)", 1.5, camera);
  });
}

function resolveLampGeometry(raw) {
  const out = { ...LAMP_GEOM };
  if (!raw || typeof raw !== "object") {
    return out;
  }
  Object.keys(raw).forEach((key) => {
    if (!LAMP_GEOM_KEYS.has(key)) {
      return;
    }
    const value = Number(raw[key]);
    if (!Number.isFinite(value)) {
      return;
    }
    if (LAMP_GEOM_POSITIVE_KEYS.has(key)) {
      if (value > 0) {
        out[key] = value;
      }
      return;
    }
    out[key] = value;
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

function angleAt(idx) {
  if (idx < 0 || idx >= state.jointAngles.length) {
    return 0;
  }
  return Number(state.jointAngles[idx] || 0);
}

function forwardKinematics() {
  const map = state.mapping;
  const g = state.lampGeom;

  const yaw = angleAt(map.yaw);
  const roll = angleAt(map.roll);
  const pitch1 = angleAt(map.pitch1) + Number(g.pitch1ZeroOffsetRad || 0);
  const pitch2 = angleAt(map.pitch2) + Number(g.pitch2ZeroOffsetRad || 0);
  const pitch3 = angleAt(map.pitch3) + Number(g.pitch3ZeroOffsetRad || 0);

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

function drawLamp(camera) {
  const k = forwardKinematics();
  const g = state.lampGeom;

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

function drawScene() {
  resizeCanvas();

  const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
  grad.addColorStop(0.0, "#0d2f3c");
  grad.addColorStop(0.65, "#0a1d28");
  grad.addColorStop(1.0, "#071219");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const camera = makeCamera();
  drawGrid(camera);
  drawLamp(camera);

  ctx.fillStyle = "rgba(225, 245, 241, 0.82)";
  ctx.font = `${Math.max(14, Math.floor(canvas.height * 0.025))}px Space Grotesk, sans-serif`;
  ctx.fillText("Drag to orbit | Wheel to zoom | Sliders set control angles", 16, 28);
}

function defaultZeroAngles() {
  return state.jointLimits.map((lim) => clamp(0.0, Number(lim[0]), Number(lim[1])));
}

function midpointAngles() {
  return state.jointLimits.map((lim) => 0.5 * (Number(lim[0]) + Number(lim[1])));
}

function updateStatus() {
  const vecRad = state.jointAngles.map((v) => Number(v).toFixed(4)).join(", ");
  const vecDeg = state.jointAngles.map((v) => radToDeg(Number(v)).toFixed(1)).join(", ");
  statusBox.textContent = [
    "Joint checker ready.",
    `config: ${state.configPath || "unknown"}`,
    `angles_rad: [${vecRad}]`,
    `angles_deg: [${vecDeg}]`,
    `camera_orbit: ${state.cameraOrbit.toFixed(3)} rad`,
    `camera_distance: ${state.cameraDistance.toFixed(2)} m`,
  ].join("\n");
}

function buildJointTable() {
  jointTableBody.innerHTML = "";
  state.tableRows = [];
  for (let i = 0; i < state.jointNames.length; i += 1) {
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    const radTd = document.createElement("td");
    const degTd = document.createElement("td");
    const normTd = document.createElement("td");
    const normSpan = document.createElement("span");
    normSpan.className = "norm";

    nameTd.textContent = state.jointNames[i];
    normTd.appendChild(normSpan);
    tr.appendChild(nameTd);
    tr.appendChild(radTd);
    tr.appendChild(degTd);
    tr.appendChild(normTd);
    jointTableBody.appendChild(tr);

    state.tableRows.push({ radTd, degTd, normSpan });
  }
}

function renderDhTable() {
  dhTableBody.innerHTML = "";
  const entries = Object.entries(state.dhParams || {}).sort((a, b) => a[0].localeCompare(b[0]));
  if (!entries.length) {
    const tr = document.createElement("tr");
    const keyTd = document.createElement("td");
    const valTd = document.createElement("td");
    keyTd.textContent = "(no dh_params found)";
    valTd.textContent = "-";
    tr.appendChild(keyTd);
    tr.appendChild(valTd);
    dhTableBody.appendChild(tr);
    return;
  }
  entries.forEach(([k, v]) => {
    const tr = document.createElement("tr");
    const keyTd = document.createElement("td");
    const valTd = document.createElement("td");
    keyTd.textContent = k;
    valTd.textContent = Number(v).toFixed(6);
    tr.appendChild(keyTd);
    tr.appendChild(valTd);
    dhTableBody.appendChild(tr);
  });
}

function updateJointReadouts() {
  for (let i = 0; i < state.jointNames.length; i += 1) {
    const row = state.sliderRows[i];
    const table = state.tableRows[i];
    const lim = state.jointLimits[i] || [-1.57, 1.57];
    const lo = Number(lim[0]);
    const hi = Number(lim[1]);
    const ang = Number(state.jointAngles[i] || 0);

    if (row) {
      row.range.value = String(ang);
      row.degInput.value = radToDeg(ang).toFixed(2);
      row.radText.textContent = `${ang.toFixed(4)} rad`;
      row.degText.textContent = `${radToDeg(ang).toFixed(1)} deg`;
    }

    if (table) {
      table.radTd.textContent = ang.toFixed(4);
      table.degTd.textContent = radToDeg(ang).toFixed(2);
      if (Math.abs(hi - lo) < 1e-9) {
        table.normSpan.textContent = "n/a";
        table.normSpan.className = "norm";
      } else {
        const normv = (ang - lo) / (hi - lo);
        table.normSpan.textContent = `${(normv * 100).toFixed(1)}%`;
        table.normSpan.className = `norm ${normv < -0.02 || normv > 1.02 ? "bad" : "ok"}`;
      }
    }
  }
}

function setJointAngle(idx, nextValue) {
  if (idx < 0 || idx >= state.jointAngles.length) {
    return;
  }
  const lim = state.jointLimits[idx] || [-1.57, 1.57];
  const lo = Number(lim[0]);
  const hi = Number(lim[1]);
  const next = clamp(Number(nextValue), lo, hi);
  state.jointAngles[idx] = Number.isFinite(next) ? next : 0;
  updateJointReadouts();
  updateStatus();
  drawScene();
}

function setAllAngles(nextAngles) {
  for (let i = 0; i < state.jointAngles.length; i += 1) {
    const lim = state.jointLimits[i] || [-1.57, 1.57];
    state.jointAngles[i] = clamp(Number(nextAngles[i] || 0), Number(lim[0]), Number(lim[1]));
  }
  updateJointReadouts();
  updateStatus();
  drawScene();
}

function buildJointSliders() {
  jointSliders.innerHTML = "";
  state.sliderRows = [];

  for (let i = 0; i < state.jointNames.length; i += 1) {
    const lim = state.jointLimits[i] || [-1.57, 1.57];
    const lo = Number(lim[0]);
    const hi = Number(lim[1]);

    const item = document.createElement("div");
    item.className = "joint-slider-item";

    const row = document.createElement("div");
    row.className = "joint-slider-row";

    const label = document.createElement("label");
    label.textContent = state.jointNames[i];
    label.setAttribute("for", `joint-range-${i}`);

    const range = document.createElement("input");
    range.type = "range";
    range.id = `joint-range-${i}`;
    range.min = String(lo);
    range.max = String(hi);
    range.step = String(SLIDER_STEP_RAD);
    range.value = String(state.jointAngles[i] || 0);
    range.addEventListener("input", () => {
      setJointAngle(i, Number(range.value));
    });

    const degInput = document.createElement("input");
    degInput.type = "number";
    degInput.step = "0.1";
    degInput.value = radToDeg(Number(state.jointAngles[i] || 0)).toFixed(2);
    degInput.addEventListener("change", () => {
      setJointAngle(i, degToRad(Number(degInput.value)));
    });

    const minusBtn = document.createElement("button");
    minusBtn.type = "button";
    minusBtn.className = "step-btn";
    minusBtn.textContent = "-";
    minusBtn.addEventListener("click", () => {
      setJointAngle(i, Number(state.jointAngles[i] || 0) - SLIDER_STEP_RAD);
    });

    const plusBtn = document.createElement("button");
    plusBtn.type = "button";
    plusBtn.className = "step-btn";
    plusBtn.textContent = "+";
    plusBtn.addEventListener("click", () => {
      setJointAngle(i, Number(state.jointAngles[i] || 0) + SLIDER_STEP_RAD);
    });

    row.appendChild(label);
    row.appendChild(range);
    row.appendChild(minusBtn);
    row.appendChild(degInput);
    row.appendChild(plusBtn);

    const meta = document.createElement("div");
    meta.className = "joint-slider-meta";
    const radText = document.createElement("span");
    const degText = document.createElement("span");
    meta.appendChild(radText);
    meta.appendChild(degText);

    item.appendChild(row);
    item.appendChild(meta);
    jointSliders.appendChild(item);
    state.sliderRows.push({ range, degInput, radText, degText });
  }
}

async function apiGet(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${txt}`);
  }
  return await res.json();
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
  } catch (_err) {
    throw new Error(`${res.status} ${res.statusText}: invalid JSON response`);
  }
  if (!res.ok) {
    const code = data?.code ? `${data.code}: ` : "";
    throw new Error(`${code}${data?.error || `${res.status} ${res.statusText}`}`);
  }
  return data;
}

function setSuiteStatus(text, isError = false) {
  suiteStatus.textContent = text;
  suiteStatus.className = isError ? "muted bad" : "muted";
}

async function copyAnglesVector() {
  const payload = `[${state.jointAngles.map((v) => Number(v).toFixed(6)).join(", ")}]`;
  try {
    await navigator.clipboard.writeText(payload);
    statusBox.textContent += "\nCopied current angle vector to clipboard.";
    notifyShell("ok", "Joint vector copied");
  } catch (_err) {
    statusBox.textContent += "\nCopy failed. Clipboard permission blocked.";
    notifyShell("bad", "Joint vector copy failed");
  }
}

async function runSuitePlayback() {
  runSuiteBtn.disabled = true;
  setSuiteStatus("Suite: generating (calm)...");
  try {
    const res = await apiPost("/api/suite", { style: "calm" });
    const viewerUrl = String(res?.viewer_url || "");
    setSuiteStatus(`Suite: ready (${Number(res?.sample_count || 0)} samples). opening...`);
    if (navigateViewerUrl(viewerUrl)) {
      return;
    }
    setSuiteStatus("Suite complete, but no viewer URL returned.", true);
  } catch (err) {
    setSuiteStatus(`Suite failed: ${err}`, true);
    notifyShell("bad", "Joint mode suite run failed", String(err));
  } finally {
    runSuiteBtn.disabled = false;
  }
}

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
  if (cmd === "joint.zero") {
    setAllAngles(defaultZeroAngles());
    notifyShell("ok", "Joint checker: zero pose");
    return;
  }
  if (cmd === "joint.mid") {
    setAllAngles(midpointAngles());
    notifyShell("ok", "Joint checker: mid limits");
    return;
  }
  if (cmd === "joint.copy") {
    copyAnglesVector();
  }
});

function bindUi() {
  zeroBtn.addEventListener("click", () => {
    setAllAngles(defaultZeroAngles());
  });
  midBtn.addEventListener("click", () => {
    setAllAngles(midpointAngles());
  });
  copyBtn.addEventListener("click", () => {
    copyAnglesVector();
  });
  runSuiteBtn.addEventListener("click", () => {
    runSuitePlayback();
  });
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
    adjustCameraOrbit(-dx * 0.006);
  });

  function endDrag(ev) {
    if (state.drag.active && canvas.hasPointerCapture(ev.pointerId)) {
      canvas.releasePointerCapture(ev.pointerId);
    }
    state.drag.active = false;
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    adjustCameraDistance(Math.sign(ev.deltaY) * 0.16);
  }, { passive: false });

  window.addEventListener("keydown", (ev) => {
    if (isEditableTarget(ev.target)) {
      return;
    }
    const key = String(ev.key || "").toLowerCase();
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "z") {
      ev.preventDefault();
      setAllAngles(defaultZeroAngles());
      notifyShell("ok", "Joint checker: zero pose");
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "m") {
      ev.preventDefault();
      setAllAngles(midpointAngles());
      notifyShell("ok", "Joint checker: mid limits");
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "c") {
      ev.preventDefault();
      copyAnglesVector();
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
    if (key === "[" && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      ev.preventDefault();
      adjustCameraOrbit(0.1);
      return;
    }
    if (key === "]" && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      ev.preventDefault();
      adjustCameraOrbit(-0.1);
    }
  });

  window.addEventListener("resize", () => drawScene());
}

async function init() {
  bindUi();
  try {
    const meta = await apiGet("/api/joint_checker/meta");
    if (!Array.isArray(meta.joint_names) || !meta.joint_names.length) {
      throw new Error("joint_names metadata is required");
    }
    if (!Array.isArray(meta.joint_limits_rad) || meta.joint_limits_rad.length !== meta.joint_names.length) {
      throw new Error("joint_limits_rad metadata is required and must match joint_names");
    }
    if (!Array.isArray(meta.default_angles_rad) || meta.default_angles_rad.length !== meta.joint_names.length) {
      throw new Error("default_angles_rad metadata is required and must match joint_names");
    }
    if (!meta.dh_params || typeof meta.dh_params !== "object") {
      throw new Error("dh_params metadata is required");
    }

    state.configPath = String(meta.config_path || "");
    state.jointNames = meta.joint_names.map((v) => String(v));
    state.jointLimits = meta.joint_limits_rad;
    state.mapping = resolveMapping(state.jointNames);
    state.lampGeom = resolveLampGeometry(meta.lamp_geometry);
    state.dhParams = meta.dh_params;

    state.jointAngles = state.jointNames.map((_, i) => {
      const lim = state.jointLimits[i] || [-1.57, 1.57];
      return clamp(Number(meta.default_angles_rad[i]), Number(lim[0]), Number(lim[1]));
    });

    configPathEl.textContent = `Config: ${state.configPath || "(unknown)"}`;
    buildJointSliders();
    buildJointTable();
    renderDhTable();
    updateJointReadouts();
    updateStatus();
    drawScene();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    configPathEl.textContent = "Config: unavailable";
    statusBox.textContent = [
      "Joint checker failed to load API metadata.",
      "Run with:",
      "uv run python tools/primitive_sim/run.py --scenario joint_checker --port 8766",
      "",
      `error: ${msg}`,
    ].join("\n");
  }
}

init();

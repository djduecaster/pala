const SVG_NS = "http://www.w3.org/2000/svg";

const fsmGraph = document.getElementById("fsmGraph");
const configPathEl = document.getElementById("configPath");
const modeLabel = document.getElementById("modeLabel");
const statusBox = document.getElementById("statusBox");
const transitionText = document.getElementById("transitionText");
const allowedText = document.getElementById("allowedText");
const proposalTableBody = document.querySelector("#proposalTable tbody");

const stepBtn = document.getElementById("stepBtn");
const autoBtn = document.getElementById("autoBtn");
const resetBtn = document.getElementById("resetBtn");
const dtInput = document.getElementById("dtInput");
const commitToggle = document.getElementById("commitToggle");
const zoneHintSelect = document.getElementById("zoneHintSelect");

const personPresentInput = document.getElementById("personPresentInput");
const personConfRange = document.getElementById("personConfRange");
const personConfInput = document.getElementById("personConfInput");
const activityRange = document.getElementById("activityRange");
const activityInput = document.getElementById("activityInput");
const noveltyRange = document.getElementById("noveltyRange");
const noveltyInput = document.getElementById("noveltyInput");
const envDeltaRange = document.getElementById("envDeltaRange");
const envDeltaInput = document.getElementById("envDeltaInput");
const plannerBreakerInput = document.getElementById("plannerBreakerInput");
const perceptionDegradedInput = document.getElementById("perceptionDegradedInput");

const runSuiteBtn = document.getElementById("runSuiteBtn");
const suiteStatus = document.getElementById("suiteStatus");

const NODE_LAYOUT = Object.freeze({
  idle_presence: { x: 190, y: 300 },
  scan_explore: { x: 420, y: 130 },
  engage_track: { x: 670, y: 300 },
  acknowledge: { x: 860, y: 130 },
  recover_reset: { x: 420, y: 490 },
});

const state = {
  meta: null,
  nodeEls: {},
  edgeEls: {},
  autoRunning: false,
  autoTimer: null,
  inFlight: false,
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function rounded(v, digits = 3) {
  return Number(v).toFixed(digits);
}

function setSuiteStatus(text, isError = false) {
  suiteStatus.textContent = text;
  suiteStatus.className = isError ? "muted bad" : "muted";
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

function linkRangeAndInput(rangeEl, numberEl) {
  const syncFromRange = () => {
    numberEl.value = Number(rangeEl.value).toFixed(2);
  };
  const syncFromInput = () => {
    const parsed = Number(numberEl.value);
    const next = clamp(Number.isFinite(parsed) ? parsed : 0, 0, 1);
    rangeEl.value = String(next);
    numberEl.value = next.toFixed(2);
  };
  rangeEl.addEventListener("input", syncFromRange);
  numberEl.addEventListener("change", syncFromInput);
}

function readSignals() {
  return {
    person_present: personPresentInput.checked,
    person_conf: clamp(Number(personConfInput.value), 0, 1),
    activity_level: clamp(Number(activityInput.value), 0, 1),
    novelty: clamp(Number(noveltyInput.value), 0, 1),
    env_delta: clamp(Number(envDeltaInput.value), 0, 1),
    planner_open_breaker: plannerBreakerInput.checked,
    perception_degraded: perceptionDegradedInput.checked,
  };
}

function applySignals(signals) {
  personPresentInput.checked = Boolean(signals.person_present);

  const personConf = clamp(Number(signals.person_conf || 0), 0, 1);
  personConfRange.value = String(personConf);
  personConfInput.value = personConf.toFixed(2);

  const activity = clamp(Number(signals.activity_level || 0), 0, 1);
  activityRange.value = String(activity);
  activityInput.value = activity.toFixed(2);

  const novelty = clamp(Number(signals.novelty || 0), 0, 1);
  noveltyRange.value = String(novelty);
  noveltyInput.value = novelty.toFixed(2);

  const envDelta = clamp(Number(signals.env_delta || 0), 0, 1);
  envDeltaRange.value = String(envDelta);
  envDeltaInput.value = envDelta.toFixed(2);

  plannerBreakerInput.checked = Boolean(signals.planner_open_breaker);
  perceptionDegradedInput.checked = Boolean(signals.perception_degraded);
}

function edgeId(edge) {
  return `${edge.from}->${edge.to}:${edge.reason}`;
}

function nodePosition(nodeId) {
  const pos = NODE_LAYOUT[nodeId];
  if (pos) {
    return pos;
  }
  return { x: 480, y: 300 };
}

function clearGraph() {
  while (fsmGraph.firstChild) {
    fsmGraph.removeChild(fsmGraph.firstChild);
  }
  state.nodeEls = {};
  state.edgeEls = {};
}

function addMarkerDefs() {
  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");
  marker.setAttribute("id", "fsmArrow");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "10");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");
  marker.setAttribute("markerUnits", "strokeWidth");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", "M0,0 L0,6 L9,3 z");
  path.setAttribute("fill", "rgba(150, 195, 210, 0.85)");
  marker.appendChild(path);
  defs.appendChild(marker);
  fsmGraph.appendChild(defs);
}

function pathWithOffset(from, to, offset) {
  const fromPos = nodePosition(from);
  const toPos = nodePosition(to);
  const dx = toPos.x - fromPos.x;
  const dy = toPos.y - fromPos.y;
  const mag = Math.hypot(dx, dy) || 1;
  const ux = dx / mag;
  const uy = dy / mag;
  const px = -uy;
  const py = ux;

  const r = 44;
  const sx = fromPos.x + ux * r;
  const sy = fromPos.y + uy * r;
  const ex = toPos.x - ux * r;
  const ey = toPos.y - uy * r;
  const cx = (sx + ex) * 0.5 + px * offset;
  const cy = (sy + ey) * 0.5 + py * offset;
  return {
    d: `M ${sx} ${sy} Q ${cx} ${cy} ${ex} ${ey}`,
    labelX: cx,
    labelY: cy,
  };
}

function drawGraph(metaGraph) {
  clearGraph();
  addMarkerDefs();

  const nodes = Array.isArray(metaGraph?.nodes) ? metaGraph.nodes : [];
  const edges = Array.isArray(metaGraph?.edges) ? metaGraph.edges : [];

  const edgeLayer = document.createElementNS(SVG_NS, "g");
  const nodeLayer = document.createElementNS(SVG_NS, "g");
  fsmGraph.appendChild(edgeLayer);
  fsmGraph.appendChild(nodeLayer);

  const pairCounts = {};
  edges.forEach((edge) => {
    const key = `${edge.from}->${edge.to}`;
    pairCounts[key] = (pairCounts[key] || 0) + 1;
  });

  edges.forEach((edge) => {
    const reverse = `${edge.to}->${edge.from}`;
    const hasReverse = Boolean(pairCounts[reverse]);
    const offset = hasReverse ? 22 : 0;
    const shape = pathWithOffset(String(edge.from), String(edge.to), offset);

    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("class", "edge-line");
    path.setAttribute("d", shape.d);
    path.setAttribute("marker-end", "url(#fsmArrow)");
    edgeLayer.appendChild(path);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "edge-label");
    label.setAttribute("x", String(shape.labelX));
    label.setAttribute("y", String(shape.labelY - 6));
    label.textContent = String(edge.label || edge.reason || "");
    edgeLayer.appendChild(label);

    state.edgeEls[edgeId(edge)] = path;
  });

  nodes.forEach((node) => {
    const id = String(node.id);
    const pos = nodePosition(id);
    const g = document.createElementNS(SVG_NS, "g");

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("class", "node-shape");
    circle.setAttribute("cx", String(pos.x));
    circle.setAttribute("cy", String(pos.y));
    circle.setAttribute("r", "42");
    g.appendChild(circle);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "node-label");
    label.setAttribute("x", String(pos.x));
    label.setAttribute("y", String(pos.y + 6));
    label.textContent = String(node.label || id);
    g.appendChild(label);

    nodeLayer.appendChild(g);
    state.nodeEls[id] = circle;
  });
}

function updateGraphHighlight(mode, decision = null) {
  Object.entries(state.nodeEls).forEach(([id, el]) => {
    el.classList.toggle("active", id === mode);
  });
  Object.values(state.edgeEls).forEach((el) => {
    el.classList.remove("active");
  });
  if (decision && decision.transitioned) {
    const key = `${decision.from}->${decision.to}:${decision.reason}`;
    const edge = state.edgeEls[key];
    if (edge) {
      edge.classList.add("active");
    }
  }
}

function renderProposals(rows) {
  proposalTableBody.innerHTML = "";
  rows.forEach((row, idx) => {
    const tr = document.createElement("tr");

    const rankTd = document.createElement("td");
    rankTd.textContent = String(idx + 1);

    const primitiveTd = document.createElement("td");
    primitiveTd.textContent = String(row.primitive);

    const scoreTd = document.createElement("td");
    scoreTd.textContent = `${rounded(row.score, 2)} / ${rounded(row.confidence, 2)}`;

    const styleTd = document.createElement("td");
    styleTd.textContent = String(row.style);

    const allowedTd = document.createElement("td");
    const allowed = Boolean(row.allowed_in_mode);
    allowedTd.textContent = allowed ? "yes" : "blocked";
    allowedTd.className = allowed ? "proposal-allow-ok" : "proposal-allow-blocked";

    const cmdTd = document.createElement("td");
    cmdTd.textContent = JSON.stringify(row.command || {});

    tr.appendChild(rankTd);
    tr.appendChild(primitiveTd);
    tr.appendChild(scoreTd);
    tr.appendChild(styleTd);
    tr.appendChild(allowedTd);
    tr.appendChild(cmdTd);
    proposalTableBody.appendChild(tr);
  });
}

function renderStepResult(result) {
  const decision = result.mode_decision || {};
  const mode = String(decision.to || "");
  modeLabel.textContent = `Mode: ${mode || "n/a"}`;

  transitionText.textContent = `Transition: ${decision.from || "?"} -> ${decision.to || "?"} | ${decision.reason || "n/a"}${decision.transitioned ? "" : " (held)"}`;
  allowedText.textContent = `Allowed primitives: ${(result.allowed_primitives || []).join(", ") || "(none)"}`;
  renderProposals(Array.isArray(result.proposals) ? result.proposals : []);
  updateGraphHighlight(mode, decision);

  const rec = result.recommended;
  const recText = rec ? `${rec.primitive} (${rounded(rec.score, 2)})` : "none";
  statusBox.textContent = [
    `tick: ${result.tick_index}`,
    `dt_s: ${rounded(result.dt_s || 0, 3)}`,
    `now_s: ${rounded(result.now_s || 0, 3)}`,
    `no_commit_s: ${rounded(result.no_commit_s || 0, 3)}`,
    `zone_hint: ${result.zone_hint || "(none)"}`,
    `recommended: ${recText}`,
    `signals: ${JSON.stringify(result.signals || {})}`,
  ].join("\n");
}

function renderSnapshot(snapshot) {
  const mode = String(snapshot.mode || "idle_presence");
  modeLabel.textContent = `Mode: ${mode}`;
  allowedText.textContent = `Allowed primitives: ${(snapshot.allowed_primitives || []).join(", ") || "(none)"}`;
  transitionText.textContent = "Transition: (no step yet)";
  updateGraphHighlight(mode, null);
  statusBox.textContent = [
    `tick: ${snapshot.tick_index || 0}`,
    `now_s: ${rounded(snapshot.now_s || 0, 3)}`,
    `no_commit_s: ${rounded(snapshot.no_commit_s || 0, 3)}`,
  ].join("\n");
  proposalTableBody.innerHTML = "";
}

function stopAutoRun() {
  state.autoRunning = false;
  if (state.autoTimer != null) {
    window.clearTimeout(state.autoTimer);
    state.autoTimer = null;
  }
  autoBtn.textContent = "Auto Run";
}

async function stepOnce() {
  if (state.inFlight) {
    return;
  }
  state.inFlight = true;
  stepBtn.disabled = true;
  try {
    const dt = clamp(Number(dtInput.value), 0.05, 10.0);
    dtInput.value = Number.isFinite(dt) ? dt.toFixed(2) : "0.35";
    const payload = {
      dt_s: Number(dtInput.value),
      commit: Boolean(commitToggle.checked),
      zone_hint: String(zoneHintSelect.value || ""),
      signals: readSignals(),
    };
    const result = await apiPost("/api/state_machine/step", payload);
    renderStepResult(result);
  } catch (err) {
    statusBox.textContent = `step failed: ${err}`;
    stopAutoRun();
  } finally {
    state.inFlight = false;
    stepBtn.disabled = false;
  }
}

async function scheduleAutoRun() {
  if (!state.autoRunning) {
    return;
  }
  await stepOnce();
  if (!state.autoRunning) {
    return;
  }
  const dt = clamp(Number(dtInput.value), 0.05, 10.0);
  const waitMs = Math.max(120, Math.round(dt * 1000));
  state.autoTimer = window.setTimeout(() => {
    scheduleAutoRun();
  }, waitMs);
}

async function resetStateMachine() {
  stopAutoRun();
  const res = await apiPost("/api/state_machine/reset", {});
  const meta = res?.meta || {};
  const snapshot = res?.state || meta?.state || {};
  state.meta = meta;
  drawGraph(meta.graph || {});
  renderSnapshot(snapshot);
}

async function runSuitePlayback() {
  runSuiteBtn.disabled = true;
  setSuiteStatus("Suite: generating (calm)...");
  try {
    const res = await apiPost("/api/suite", { style: "calm" });
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

function bindUi() {
  linkRangeAndInput(personConfRange, personConfInput);
  linkRangeAndInput(activityRange, activityInput);
  linkRangeAndInput(noveltyRange, noveltyInput);
  linkRangeAndInput(envDeltaRange, envDeltaInput);

  stepBtn.addEventListener("click", () => {
    stepOnce();
  });

  autoBtn.addEventListener("click", () => {
    if (state.autoRunning) {
      stopAutoRun();
      return;
    }
    state.autoRunning = true;
    autoBtn.textContent = "Stop Auto";
    scheduleAutoRun();
  });

  resetBtn.addEventListener("click", () => {
    resetStateMachine().catch((err) => {
      statusBox.textContent = `reset failed: ${err}`;
    });
  });

  runSuiteBtn.addEventListener("click", () => {
    runSuitePlayback();
  });
}

async function init() {
  bindUi();
  try {
    const meta = await apiGet("/api/state_machine/meta");
    state.meta = meta;
    configPathEl.textContent = `Config: ${String(meta.config_path || "(unknown)")}`;
    applySignals(meta.default_signals || {});
    drawGraph(meta.graph || {});
    renderSnapshot(meta.state || {});
  } catch (err) {
    configPathEl.textContent = "Config: unavailable";
    statusBox.textContent = [
      "State machine tool failed to load metadata.",
      "Run with:",
      "uv run python tools/primitive_sim/run.py --scenario state_machine --port 8766",
      "",
      `error: ${String(err)}`,
    ].join("\n");
  }
}

init();

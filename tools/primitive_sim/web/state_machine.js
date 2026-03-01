const SVG_NS = "http://www.w3.org/2000/svg";
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

const fsmGraph = document.getElementById("fsmGraph");
const configPathEl = document.getElementById("configPath");
const modeLabel = document.getElementById("modeLabel");
const statusBox = document.getElementById("statusBox");
const transitionText = document.getElementById("transitionText");
const allowedText = document.getElementById("allowedText");
const proposalTableBody = document.querySelector("#proposalTable tbody");
const historyTableBody = document.querySelector("#historyTable tbody");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const stepBtn = document.getElementById("stepBtn");
const stepCommitBtn = document.getElementById("stepCommitBtn");
const step5Btn = document.getElementById("step5Btn");
const autoBtn = document.getElementById("autoBtn");
const forceBtn = document.getElementById("forceBtn");
const resetBtn = document.getElementById("resetBtn");
const dtInput = document.getElementById("dtInput");
const commitToggle = document.getElementById("commitToggle");
const zoneHintSelect = document.getElementById("zoneHintSelect");
const forceModeSelect = document.getElementById("forceModeSelect");
const forceReasonInput = document.getElementById("forceReasonInput");

const personPresentInput = document.getElementById("personPresentInput");
const personConfRange = document.getElementById("personConfRange");
const personConfInput = document.getElementById("personConfInput");
const searchRequestedInput = document.getElementById("searchRequestedInput");
const searchCompleteInput = document.getElementById("searchCompleteInput");
const assistCompleteInput = document.getElementById("assistCompleteInput");
const userAckInput = document.getElementById("userAckInput");
const taskActiveInput = document.getElementById("taskActiveInput");
const homeRequestedInput = document.getElementById("homeRequestedInput");
const homeCompletedInput = document.getElementById("homeCompletedInput");
const cancelRequestedInput = document.getElementById("cancelRequestedInput");
const startupCompleteInput = document.getElementById("startupCompleteInput");
const healthDegradedInput = document.getElementById("healthDegradedInput");
const plannerBreakerInput = document.getElementById("plannerBreakerInput");
const perceptionDegradedInput = document.getElementById("perceptionDegradedInput");

const runSuiteBtn = document.getElementById("runSuiteBtn");
const suiteStatus = document.getElementById("suiteStatus");
const presetIdleBtn = document.getElementById("presetIdleBtn");
const presetPresenceBtn = document.getElementById("presetPresenceBtn");
const presetEngageBtn = document.getElementById("presetEngageBtn");
const presetRecoverBtn = document.getElementById("presetRecoverBtn");
const presetFaultBtn = document.getElementById("presetFaultBtn");

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

const NODE_LAYOUT = Object.freeze({
  boot_awaken: { x: 140, y: 310 },
  idle_presence: { x: 320, y: 310 },
  social_interact: { x: 500, y: 180 },
  search_assist: { x: 500, y: 450 },
  task_lighting: { x: 690, y: 180 },
  return_home: { x: 690, y: 450 },
  recover_reset: { x: 880, y: 310 },
});

const state = {
  meta: null,
  nodeEls: {},
  edgeEls: {},
  autoRunning: false,
  autoTimer: null,
  inFlight: false,
  history: [],
  historyMax: 80,
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
        mode: "state_machine",
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
    search_requested: searchRequestedInput.checked,
    search_complete: searchCompleteInput.checked,
    assist_complete: assistCompleteInput.checked,
    user_ack: userAckInput.checked,
    task_active: taskActiveInput.checked,
    home_requested: homeRequestedInput.checked,
    home_completed: homeCompletedInput.checked,
    cancel_requested: cancelRequestedInput.checked,
    startup_complete: startupCompleteInput.checked,
    health_degraded: healthDegradedInput.checked,
    planner_open_breaker: plannerBreakerInput.checked,
    perception_degraded: perceptionDegradedInput.checked,
  };
}

function applySignals(signals) {
  personPresentInput.checked = Boolean(signals.person_present);

  const personConf = clamp(Number(signals.person_conf || 0), 0, 1);
  personConfRange.value = String(personConf);
  personConfInput.value = personConf.toFixed(2);

  searchRequestedInput.checked = Boolean(signals.search_requested);
  searchCompleteInput.checked = Boolean(signals.search_complete);
  assistCompleteInput.checked = Boolean(signals.assist_complete);
  userAckInput.checked = Boolean(signals.user_ack);
  taskActiveInput.checked = Boolean(signals.task_active);
  homeRequestedInput.checked = Boolean(signals.home_requested);
  homeCompletedInput.checked = Boolean(signals.home_completed);
  cancelRequestedInput.checked = Boolean(signals.cancel_requested);
  startupCompleteInput.checked = Boolean(signals.startup_complete);
  healthDegradedInput.checked = Boolean(signals.health_degraded);

  plannerBreakerInput.checked = Boolean(signals.planner_open_breaker);
  perceptionDegradedInput.checked = Boolean(signals.perception_degraded);
}

function applySignalPreset(name) {
  const preset = String(name || "").toLowerCase();
  if (preset === "idle") {
    applySignals({
      person_present: false,
      person_conf: 0.0,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: false,
      task_active: false,
      home_requested: false,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: false,
      planner_open_breaker: false,
      perception_degraded: false,
    });
    zoneHintSelect.value = "";
  } else if (preset === "presence") {
    applySignals({
      person_present: true,
      person_conf: 0.72,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: true,
      task_active: false,
      home_requested: false,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: false,
      planner_open_breaker: false,
      perception_degraded: false,
    });
    zoneHintSelect.value = "center";
  } else if (preset === "engage") {
    applySignals({
      person_present: true,
      person_conf: 0.94,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: true,
      task_active: true,
      home_requested: false,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: false,
      planner_open_breaker: false,
      perception_degraded: false,
    });
    zoneHintSelect.value = "center";
  } else if (preset === "recover") {
    applySignals({
      person_present: false,
      person_conf: 0.0,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: false,
      task_active: false,
      home_requested: false,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: true,
      planner_open_breaker: false,
      perception_degraded: false,
    });
    zoneHintSelect.value = "";
  } else if (preset === "fault") {
    applySignals({
      person_present: false,
      person_conf: 0.0,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: false,
      task_active: false,
      home_requested: false,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: true,
      planner_open_breaker: true,
      perception_degraded: true,
    });
    zoneHintSelect.value = "";
  }
  statusBox.textContent = `preset applied: ${preset}`;
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

function clearHistory() {
  state.history = [];
  renderHistory();
}

function appendHistory(result, source = "step") {
  const decision = result?.mode_decision || {};
  const rec = result?.recommended || {};
  const row = {
    tick: Number(result?.tick_index || 0),
    source: String(source || "step"),
    from: String(decision.from || "?"),
    to: String(decision.to || "?"),
    transitioned: Boolean(decision.transitioned),
    reason: String(decision.reason || ""),
    recommended: String(rec.primitive || ""),
    style: String(rec.style || ""),
  };
  state.history.unshift(row);
  const maxRows = Math.max(10, Number(state.historyMax || 80));
  if (state.history.length > maxRows) {
    state.history.length = maxRows;
  }
  renderHistory();
}

function renderHistory() {
  if (!historyTableBody) {
    return;
  }
  historyTableBody.innerHTML = "";
  state.history.forEach((row) => {
    const tr = document.createElement("tr");

    const tickTd = document.createElement("td");
    tickTd.textContent = `${row.tick} (${row.source})`;

    const transitionTd = document.createElement("td");
    transitionTd.textContent = `${row.from} -> ${row.to}${row.transitioned ? "" : " [held]"}`;

    const reasonTd = document.createElement("td");
    reasonTd.textContent = row.reason;

    const recTd = document.createElement("td");
    recTd.textContent = `${row.recommended || "-"}${row.style ? ` (${row.style})` : ""}`;

    tr.appendChild(tickTd);
    tr.appendChild(transitionTd);
    tr.appendChild(reasonTd);
    tr.appendChild(recTd);
    historyTableBody.appendChild(tr);
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
    appendHistory(result, "step");
    notifyShell("ok", "FSM step complete", `${result.mode_decision?.from || "?"} -> ${result.mode_decision?.to || "?"}`);
  } catch (err) {
    statusBox.textContent = `step failed: ${err}`;
    stopAutoRun();
    notifyShell("bad", "FSM step failed", String(err));
  } finally {
    state.inFlight = false;
    stepBtn.disabled = false;
  }
}

async function forceMode() {
  if (state.inFlight) {
    return;
  }
  const mode = String(forceModeSelect.value || "").trim();
  if (!mode) {
    statusBox.textContent = "force failed: select a force mode first";
    return;
  }
  const reason = String(forceReasonInput.value || "").trim() || "ops_force";
  state.inFlight = true;
  forceBtn.disabled = true;
  try {
    const dt = clamp(Number(dtInput.value), 0, 10.0);
    dtInput.value = Number.isFinite(dt) ? dt.toFixed(2) : "0.35";
    const payload = {
      mode,
      reason,
      dt_s: Number(dtInput.value),
      commit: Boolean(commitToggle.checked),
      zone_hint: String(zoneHintSelect.value || ""),
      signals: readSignals(),
    };
    const result = await apiPost("/api/state_machine/force", payload);
    renderStepResult(result);
    appendHistory(result, "force");
    notifyShell("ok", "FSM force complete", `${result.mode_decision?.from || "?"} -> ${result.mode_decision?.to || "?"}`);
  } catch (err) {
    statusBox.textContent = `force failed: ${err}`;
    notifyShell("bad", "FSM force failed", String(err));
  } finally {
    state.inFlight = false;
    forceBtn.disabled = false;
  }
}

async function stepOnceWithCommit() {
  const prev = commitToggle.checked;
  commitToggle.checked = true;
  try {
    await stepOnce();
  } finally {
    commitToggle.checked = prev;
  }
}

async function stepMany(count) {
  const n = Math.max(1, Math.floor(Number(count) || 1));
  for (let i = 0; i < n; i += 1) {
    if (state.autoRunning) {
      break;
    }
    await stepOnce();
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
  clearHistory();
  applySignals(meta.default_signals || {});
  drawGraph(meta.graph || {});
  renderSnapshot(snapshot);
  notifyShell("ok", "FSM reset");
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
    notifyShell("bad", "FSM suite run failed", String(err));
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
  if (cmd === "fsm.step") {
    stepOnce();
    return;
  }
  if (cmd === "fsm.reset") {
    resetStateMachine().catch((err) => {
      statusBox.textContent = `reset failed: ${err}`;
      notifyShell("bad", "FSM reset failed", String(err));
    });
  }
});

function bindUi() {
  linkRangeAndInput(personConfRange, personConfInput);

  stepBtn.addEventListener("click", () => {
    stepOnce();
  });

  if (stepCommitBtn) {
    stepCommitBtn.addEventListener("click", () => {
      stepOnceWithCommit();
    });
  }

  if (step5Btn) {
    step5Btn.addEventListener("click", () => {
      stepMany(5);
    });
  }

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

  if (forceBtn) {
    forceBtn.addEventListener("click", () => {
      forceMode();
    });
  }

  runSuiteBtn.addEventListener("click", () => {
    runSuitePlayback();
  });

  if (presetIdleBtn) {
    presetIdleBtn.addEventListener("click", () => applySignalPreset("idle"));
  }
  if (presetPresenceBtn) {
    presetPresenceBtn.addEventListener("click", () => applySignalPreset("presence"));
  }
  if (presetEngageBtn) {
    presetEngageBtn.addEventListener("click", () => applySignalPreset("engage"));
  }
  if (presetRecoverBtn) {
    presetRecoverBtn.addEventListener("click", () => applySignalPreset("recover"));
  }
  if (presetFaultBtn) {
    presetFaultBtn.addEventListener("click", () => applySignalPreset("fault"));
  }
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", () => clearHistory());
  }

  window.addEventListener("keydown", (ev) => {
    if (isEditableTarget(ev.target)) {
      return;
    }
    const key = String(ev.key || "").toLowerCase();
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "s") {
      ev.preventDefault();
      stepOnce();
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "c") {
      ev.preventDefault();
      stepOnceWithCommit();
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "5") {
      ev.preventDefault();
      stepMany(5);
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "a") {
      ev.preventDefault();
      if (state.autoRunning) {
        stopAutoRun();
        return;
      }
      state.autoRunning = true;
      autoBtn.textContent = "Stop Auto";
      scheduleAutoRun();
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "x") {
      ev.preventDefault();
      resetStateMachine().catch((err) => {
        statusBox.textContent = `reset failed: ${err}`;
      });
      return;
    }
    if (!ev.metaKey && !ev.ctrlKey && !ev.altKey && key === "f") {
      ev.preventDefault();
      forceMode();
    }
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
    renderHistory();
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

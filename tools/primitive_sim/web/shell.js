const shellQueryGuard = new URLSearchParams(window.location.search);
if (window.parent !== window) {
  const embedMode = String(shellQueryGuard.get("mode") || "studio").trim();
  const embedTrace = String(shellQueryGuard.get("trace") || "").trim();
  let rawTarget = "/tools/primitive_sim/web/index.html?studio=1";
  if (embedMode === "joint_checker") {
    rawTarget = "/tools/primitive_sim/web/joint_checker.html";
  } else if (embedMode === "state_machine") {
    rawTarget = "/tools/primitive_sim/web/state_machine.html";
  } else if (embedMode === "scenario_lab") {
    rawTarget = "/tools/primitive_sim/web/scenario_lab.html";
  } else if (embedMode === "playback" && embedTrace) {
    rawTarget = `/tools/primitive_sim/web/index.html?studio=0&trace=${encodeURIComponent(embedTrace)}`;
  }
  window.location.replace(rawTarget);
}

const modeNav = document.getElementById("modeNav");
const modeBadge = document.getElementById("modeBadge");
const modeTitle = document.getElementById("modeTitle");
const modeSummary = document.getElementById("modeSummary");
const workflowSteps = document.getElementById("workflowSteps");
const statusBox = document.getElementById("statusBox");
const rawModePath = document.getElementById("rawModePath");
const modeFrame = document.getElementById("modeFrame");

const modeActionBtn = document.getElementById("modeActionBtn");
const runAndPlaybackBtn = document.getElementById("runAndPlaybackBtn");
const runSuiteBtn = document.getElementById("runSuiteBtn");
const focusViewerBtn = document.getElementById("focusViewerBtn");
const paletteBtn = document.getElementById("paletteBtn");
const openRawBtn = document.getElementById("openRawBtn");
const nextActionHint = document.getElementById("nextActionHint");

const opsGoalSelect = document.getElementById("opsGoalSelect");
const opsGoalBtn = document.getElementById("opsGoalBtn");
const opsGoalHint = document.getElementById("opsGoalHint");

const qaTuneBtn = document.getElementById("qaTuneBtn");
const qaScenarioBtn = document.getElementById("qaScenarioBtn");
const qaStateBtn = document.getElementById("qaStateBtn");
const qaJointBtn = document.getElementById("qaJointBtn");

const toastStack = document.getElementById("toastStack");
const commandPalette = document.getElementById("commandPalette");
const paletteInput = document.getElementById("paletteInput");
const paletteList = document.getElementById("paletteList");
const paletteCloseBtn = document.getElementById("paletteCloseBtn");

const nativeFsmPanel = document.getElementById("nativeFsmPanel");
const nativeScenarioPanel = document.getElementById("nativeScenarioPanel");

const fsmStepBtn = document.getElementById("fsmStepBtn");
const fsmStep5Btn = document.getElementById("fsmStep5Btn");
const fsmAutoBtn = document.getElementById("fsmAutoBtn");
const fsmResetBtn = document.getElementById("fsmResetBtn");
const fsmCopySnapshotBtn = document.getElementById("fsmCopySnapshotBtn");
const fsmUseRecommendedBtn = document.getElementById("fsmUseRecommendedBtn");
const fsmForceModeSelect = document.getElementById("fsmForceModeSelect");
const fsmForceReasonInput = document.getElementById("fsmForceReasonInput");
const fsmForceBtn = document.getElementById("fsmForceBtn");
const fsmDtInput = document.getElementById("fsmDtInput");
const fsmZoneSelect = document.getElementById("fsmZoneSelect");
const fsmCommitToggle = document.getElementById("fsmCommitToggle");
const fsmPersonPresent = document.getElementById("fsmPersonPresent");
const fsmPersonConf = document.getElementById("fsmPersonConf");
const fsmSearchRequested = document.getElementById("fsmSearchRequested");
const fsmSearchComplete = document.getElementById("fsmSearchComplete");
const fsmAssistComplete = document.getElementById("fsmAssistComplete");
const fsmUserAck = document.getElementById("fsmUserAck");
const fsmTaskActive = document.getElementById("fsmTaskActive");
const fsmHomeRequested = document.getElementById("fsmHomeRequested");
const fsmHomeCompleted = document.getElementById("fsmHomeCompleted");
const fsmCancelRequested = document.getElementById("fsmCancelRequested");
const fsmStartupComplete = document.getElementById("fsmStartupComplete");
const fsmHealthDegraded = document.getElementById("fsmHealthDegraded");
const fsmPlannerBreaker = document.getElementById("fsmPlannerBreaker");
const fsmPerceptionDegraded = document.getElementById("fsmPerceptionDegraded");
const fsmGraphNative = document.getElementById("fsmGraphNative");
const fsmProposalTableBody = document.querySelector("#fsmProposalTable tbody");
const fsmStatus = document.getElementById("fsmStatus");
const fsmPresetBootBtn = document.getElementById("fsmPresetBootBtn");
const fsmPresetIdleBtn = document.getElementById("fsmPresetIdleBtn");
const fsmPresetSocialBtn = document.getElementById("fsmPresetSocialBtn");
const fsmPresetSearchBtn = document.getElementById("fsmPresetSearchBtn");
const fsmPresetTaskBtn = document.getElementById("fsmPresetTaskBtn");
const fsmPresetHomeBtn = document.getElementById("fsmPresetHomeBtn");
const fsmPresetRecoverBtn = document.getElementById("fsmPresetRecoverBtn");
const fsmPresetFaultBtn = document.getElementById("fsmPresetFaultBtn");
const fsmClearHistoryBtn = document.getElementById("fsmClearHistoryBtn");
const fsmHistoryTableBody = document.querySelector("#fsmHistoryTable tbody");

const scnPrimitiveSelect = document.getElementById("scnPrimitiveSelect");
const scnStyleSelect = document.getElementById("scnStyleSelect");
const scnDurationInput = document.getElementById("scnDurationInput");
const scnStopOnDone = document.getElementById("scnStopOnDone");
const scnCommandText = document.getElementById("scnCommandText");
const scnAddStepBtn = document.getElementById("scnAddStepBtn");
const scnTemplateBtn = document.getElementById("scnTemplateBtn");
const scnClearBtn = document.getElementById("scnClearBtn");
const scnJsonText = document.getElementById("scnJsonText");
const scnValidateBtn = document.getElementById("scnValidateBtn");
const scnRunBtn = document.getElementById("scnRunBtn");
const scnOpenPlaybackBtn = document.getElementById("scnOpenPlaybackBtn");
const scnNameInput = document.getElementById("scnNameInput");
const scnNotesInput = document.getElementById("scnNotesInput");
const scnSaveBtn = document.getElementById("scnSaveBtn");
const scnMetrics = document.getElementById("scnMetrics");
const scnHistoryTableBody = document.querySelector("#scnHistoryTable tbody");
const scnSweepTargetInput = document.getElementById("scnSweepTargetInput");
const scnSweepTopKInput = document.getElementById("scnSweepTopKInput");
const scnSweepSaveTraces = document.getElementById("scnSweepSaveTraces");
const scnSweepGridText = document.getElementById("scnSweepGridText");
const scnSweepTemplateBtn = document.getElementById("scnSweepTemplateBtn");
const scnSweepRunBtn = document.getElementById("scnSweepRunBtn");
const scnSweepApplyBestBtn = document.getElementById("scnSweepApplyBestBtn");
const scnSweepPromoteBaselineBtn = document.getElementById("scnSweepPromoteBaselineBtn");
const scnSweepStatus = document.getElementById("scnSweepStatus");
const scnSweepTableBody = document.querySelector("#scnSweepTable tbody");
const scnStatus = document.getElementById("scnStatus");

const LAST_MODE_KEY = "lamp_sim_last_mode_v1";
const FOCUS_VIEW_KEY = "lamp_sim_focus_view_v1";
const SVG_NS = "http://www.w3.org/2000/svg";
const NATIVE_MODES = new Set(["scenario_lab", "state_machine"]);
const CONTINUOUS_PRIMITIVES = new Set(["hold", "breath"]);

const FSM_NODE_LAYOUT = Object.freeze({
  boot_awaken: { x: 140, y: 310 },
  idle_presence: { x: 320, y: 310 },
  social_interact: { x: 500, y: 180 },
  search_assist: { x: 500, y: 450 },
  task_lighting: { x: 690, y: 180 },
  return_home: { x: 690, y: 450 },
  recover_reset: { x: 880, y: 310 },
});

const MODES = Object.freeze({
  studio: {
    label: "Primitive Studio",
    summary: "Tune primitive parameters and compare baseline vs draft in the 3D viewer.",
    steps: [
      "Select primitive + style.",
      "Adjust command parameters.",
      "Use compare mode to inspect baseline deltas.",
      "Save baseline when behavior looks right.",
    ],
    buildSrc: () => "/tools/primitive_sim/web/index.html?studio=1&shell=1",
    rawSrc: "/tools/primitive_sim/web/index.html?studio=1",
  },
  scenario_lab: {
    label: "Scenario Lab",
    summary: "Compose multi-step scenarios, run simulation, inspect metrics, and save experiments.",
    steps: [
      "Build steps with primitive/style/duration + command.",
      "Validate scenario structure (dry run).",
      "Run scenario and inspect metrics.",
      "Save experiment entry for later comparison.",
    ],
    buildSrc: () => "/tools/primitive_sim/web/scenario_lab.html?shell=1",
    rawSrc: "/tools/primitive_sim/web/scenario_lab.html",
  },
  state_machine: {
    label: "State Machine",
    summary: "Operate deterministic Behavior V4 macro-mode transitions with presets, force-mode controls, and transition logging.",
    steps: [
      "Set V4 signals for person/search/task/home/health.",
      "Step, step x5, or auto-run with dt control.",
      "Use presets/force-mode controls for rapid incident drills.",
      "Inspect transition log and recommended primitive.",
      "Send recommended primitive into Scenario Lab.",
    ],
    buildSrc: () => "/tools/primitive_sim/web/state_machine.html?shell=1",
    rawSrc: "/tools/primitive_sim/web/state_machine.html",
  },
  joint_checker: {
    label: "Joint / DH Checker",
    summary: "Validate DH-driven pose geometry and per-joint limits with direct slider control.",
    steps: [
      "Set target joint angles with sliders.",
      "Inspect pose and angle readouts (rad/deg).",
      "Verify axis sign and limit behavior.",
      "Copy vectors for config calibration.",
    ],
    buildSrc: () => "/tools/primitive_sim/web/joint_checker.html?shell=1",
    rawSrc: "/tools/primitive_sim/web/joint_checker.html",
  },
  playback: {
    label: "Trace Playback",
    summary: "Play back a generated trace for review and debugging.",
    steps: [
      "Generate trace from suite/studio/scenario.",
      "Load and scrub timeline.",
      "Inspect primitive/status transitions.",
      "Use this for demos and evidence capture.",
    ],
    buildSrc: (params) => {
      const trace = String(params.get("trace") || "").trim();
      if (!trace) {
        return "/tools/primitive_sim/web/index.html?studio=1&shell=1";
      }
      return `/tools/primitive_sim/web/index.html?studio=0&trace=${encodeURIComponent(trace)}&shell=1`;
    },
    rawSrcFromParams: (params) => {
      const trace = String(params.get("trace") || "").trim();
      if (!trace) {
        return "/tools/primitive_sim/web/index.html?studio=1";
      }
      return `/tools/primitive_sim/web/index.html?studio=0&trace=${encodeURIComponent(trace)}`;
    },
  },
});

const state = {
  activeMode: "studio",
  params: new URLSearchParams(window.location.search),
  navButtons: new Map(),
  viewerFocus: false,
  paletteOpen: false,
  paletteCommands: [],
  filteredCommands: [],
  paletteIdx: 0,
  fsm: {
    meta: null,
    ready: false,
    loadingPromise: null,
    nodeEls: {},
    edgeEls: {},
    lastResult: null,
    autoRunning: false,
    autoTimer: null,
    inFlight: false,
    history: [],
    historyMax: 80,
  },
  scenario: {
    meta: null,
    ready: false,
    loadingPromise: null,
    lastRun: null,
    history: [],
    lastSweep: null,
  },
};

const MODE_PRIMARY_ACTION = Object.freeze({
  studio: { label: "Run Preview", command: "studio.run_preview" },
  scenario_lab: { label: "Run Scenario", command: "scenario.run" },
  state_machine: { label: "Step FSM", command: "fsm.step" },
  joint_checker: { label: "Zero Pose", command: "joint.zero" },
  playback: { label: "Open Raw Playback", command: "shell.open_raw" },
});

const MODE_NEXT_HINT = Object.freeze({
  studio: "Next action: run preview, then save baseline when motion looks correct.",
  scenario_lab: "Next action: validate scenario, run it, then inspect metrics and trace.",
  state_machine: "Next action: step FSM and send recommended primitive into Scenario Lab.",
  joint_checker: "Next action: verify axis signs and limit behavior with direct sliders.",
  playback: "Next action: scrub the trace and inspect primitive and mode transitions.",
});

const OPS_GOALS = Object.freeze({
  tune_primitive: {
    mode: "studio",
    hint: "Goal: tune primitive params and save a stable baseline.",
    status: "ops goal: tune primitive",
  },
  build_scenario: {
    mode: "scenario_lab",
    hint: "Goal: build scenario steps, run, and compare metrics.",
    status: "ops goal: build scenario",
  },
  validate_state_machine: {
    mode: "state_machine",
    hint: "Goal: validate mode transitions and recommendation quality.",
    status: "ops goal: validate state machine",
  },
  review_playback: {
    mode: "playback",
    hint: "Goal: review trace playback and verify operational behavior.",
    status: "ops goal: review playback",
  },
});

const GOAL_BY_MODE = Object.freeze({
  studio: "tune_primitive",
  scenario_lab: "build_scenario",
  state_machine: "validate_state_machine",
  playback: "review_playback",
  joint_checker: "validate_state_machine",
});

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function rounded(v, digits = 3) {
  return Number(v || 0).toFixed(digits);
}

function finiteNumber(value, fallback = 0) {
  const out = Number(value);
  return Number.isFinite(out) ? out : fallback;
}

function prettyJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function setStatus(lines) {
  statusBox.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
}

function isNativeMode(mode) {
  return NATIVE_MODES.has(String(mode || ""));
}

function activeModeConfig() {
  return MODES[state.activeMode] || MODES.studio;
}

function buildModeFrameSrc(mode) {
  const cfg = MODES[mode] || MODES.studio;
  return cfg.buildSrc(state.params);
}

function buildRawModeSrc(mode) {
  const cfg = MODES[mode] || MODES.studio;
  if (typeof cfg.rawSrcFromParams === "function") {
    return cfg.rawSrcFromParams(state.params);
  }
  return String(cfg.rawSrc || "/tools/primitive_sim/web/index.html?studio=1");
}

function normalizeMode(mode) {
  const candidate = MODES[mode] ? mode : "studio";
  if (candidate === "playback" && !state.params.get("trace")) {
    return "studio";
  }
  return candidate;
}

function persistMode(mode) {
  try {
    window.localStorage.setItem(LAST_MODE_KEY, mode);
  } catch (_err) {
    // ignore storage errors
  }
}

function loadPersistedMode() {
  try {
    return window.localStorage.getItem(LAST_MODE_KEY) || "";
  } catch (_err) {
    return "";
  }
}

function loadPersistedFocusView() {
  try {
    return window.localStorage.getItem(FOCUS_VIEW_KEY) === "1";
  } catch (_err) {
    return false;
  }
}

function persistFocusView(enabled) {
  try {
    window.localStorage.setItem(FOCUS_VIEW_KEY, enabled ? "1" : "0");
  } catch (_err) {
    // ignore storage errors
  }
}

function updateQuery(mode) {
  const next = new URLSearchParams(window.location.search);
  next.set("mode", mode);
  const trace = state.params.get("trace");
  if (trace) {
    next.set("trace", trace);
  }
  const url = `${window.location.pathname}?${next.toString()}`;
  window.history.replaceState({}, "", url);
  state.params = next;
}

function renderModeMeta() {
  const cfg = activeModeConfig();
  modeBadge.textContent = `Mode: ${state.activeMode}`;
  modeTitle.textContent = cfg.label;
  modeSummary.textContent = cfg.summary;
  nextActionHint.textContent = MODE_NEXT_HINT[state.activeMode] || MODE_NEXT_HINT.studio;
  rawModePath.textContent = buildRawModeSrc(state.activeMode);
  const primary = MODE_PRIMARY_ACTION[state.activeMode] || MODE_PRIMARY_ACTION.studio;
  modeActionBtn.textContent = primary.label;
  if (state.activeMode === "scenario_lab") {
    runAndPlaybackBtn.textContent = "Run Scenario + Playback";
    runAndPlaybackBtn.disabled = false;
  } else if (state.activeMode === "state_machine") {
    runAndPlaybackBtn.textContent = "Step + Scenario + Playback";
    runAndPlaybackBtn.disabled = false;
  } else if (state.activeMode === "playback") {
    runAndPlaybackBtn.textContent = "Open Raw Playback";
    runAndPlaybackBtn.disabled = false;
  } else {
    runAndPlaybackBtn.textContent = "Run + Playback";
    runAndPlaybackBtn.disabled = false;
  }
  workflowSteps.innerHTML = "";
  cfg.steps.forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    workflowSteps.appendChild(li);
  });
}

function setViewerFocus(enabled, { persist = true } = {}) {
  const on = Boolean(enabled);
  state.viewerFocus = on;
  document.querySelector(".shell-layout")?.classList.toggle("focus-view", on);
  focusViewerBtn.textContent = on ? "Exit Focus" : "Focus Viewer";
  if (persist) {
    persistFocusView(on);
  }
  setStatus([
    `shell mode: ${state.activeMode}`,
    on ? "viewer focus: enabled" : "viewer focus: disabled",
  ]);
}

function highlightNav() {
  state.navButtons.forEach((button, mode) => {
    button.classList.toggle("active", mode === state.activeMode);
  });
}

function showWorkspaceForMode(mode) {
  const useNative = isNativeMode(mode);
  modeFrame.classList.toggle("hidden", useNative);
  nativeFsmPanel.classList.toggle("hidden", mode !== "state_machine");
  nativeScenarioPanel.classList.toggle("hidden", mode !== "scenario_lab");
}

function updateOpsGoalHint() {
  const key = String(opsGoalSelect?.value || "tune_primitive");
  const entry = OPS_GOALS[key] || OPS_GOALS.tune_primitive;
  opsGoalHint.textContent = entry.hint;
}

async function applyOperationalGoal() {
  const key = String(opsGoalSelect.value || "tune_primitive").trim();
  const entry = OPS_GOALS[key] || OPS_GOALS.tune_primitive;

  if (entry.mode === "playback" && !state.params.get("trace")) {
    setStatus([
      "ops goal: review playback",
      "no playback trace is loaded; generating suite playback now.",
    ]);
    await runSuitePlayback();
    updateOpsGoalHint();
    return;
  }

  setMode(entry.mode);
  setStatus([
    `shell mode: ${state.activeMode}`,
    String(entry.status || `ops goal: ${key}`),
  ]);
  updateOpsGoalHint();

  if (entry.mode === "state_machine") {
    await ensureFsmReady();
    applyFsmPreset("idle");
  } else if (entry.mode === "scenario_lab") {
    await ensureScenarioReady();
  }
}

function setMode(mode, { fromSuite = false } = {}) {
  const next = normalizeMode(mode);
  const prev = state.activeMode;
  if (prev === "state_machine" && next !== "state_machine") {
    stopFsmAutoRun();
  }

  state.activeMode = next;
  persistMode(next);
  updateQuery(next);
  renderModeMeta();
  highlightNav();
  showWorkspaceForMode(next);
  const goalToken = GOAL_BY_MODE[next];
  if (goalToken && OPS_GOALS[goalToken]) {
    opsGoalSelect.value = goalToken;
  }
  updateOpsGoalHint();

  const status = [`shell mode: ${next}`];
  if (isNativeMode(next)) {
    modeFrame.src = "about:blank";
    status.push(`native panel: ${next}`);
    if (next === "state_machine") {
      void ensureFsmReady().catch((err) => {
        showToast(`FSM load failed: ${err.message || err}`, "bad", 3600);
      });
    } else if (next === "scenario_lab") {
      void ensureScenarioReady().catch((err) => {
        showToast(`Scenario load failed: ${err.message || err}`, "bad", 3600);
      });
    }
  } else {
    const src = buildModeFrameSrc(next);
    modeFrame.src = src;
    status.push(`iframe src: ${src}`);
  }
  status.push(`raw mode: ${buildRawModeSrc(next)}`);
  if (fromSuite) {
    status.push("suite playback loaded");
  }
  setStatus(status);
  showToast(`Mode: ${MODES[next].label}`, "ok");
}

function initNav() {
  modeNav.innerHTML = "";
  const orderedModes = ["studio", "scenario_lab", "state_machine", "joint_checker", "playback"];
  orderedModes.forEach((mode) => {
    const cfg = MODES[mode];
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = cfg.label;
    button.addEventListener("click", () => {
      if (mode === "playback" && !state.params.get("trace")) {
        setStatus([
          "playback mode requires a trace query parameter.",
          "generate suite playback with [r] or load playback from another mode first.",
        ]);
        return;
      }
      setMode(mode);
    });
    modeNav.appendChild(button);
    state.navButtons.set(mode, button);
  });
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
  let data = {};
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

function showToast(message, kind = "ok", ttlMs = 2800) {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = String(message);
  toastStack.appendChild(toast);
  window.setTimeout(() => {
    toast.remove();
  }, Math.max(800, ttlMs));
}

function postModeCommand(command, payload = {}) {
  if (!modeFrame.contentWindow) {
    throw new Error("mode frame is not ready");
  }
  const packet = {
    source: "lamp-shell",
    type: "command",
    command: String(command),
    payload: payload && typeof payload === "object" ? payload : {},
  };
  modeFrame.contentWindow.postMessage(packet, window.location.origin);
}

function queueCommand(command, delayMs = 0) {
  const run = () => {
    void execShellCommand(command).catch((err) => {
      const text = `Command failed (${command}): ${err.message || err}`;
      setStatus([`shell mode: ${state.activeMode}`, text]);
      showToast(text, "bad", 3600);
    });
  };
  if (delayMs > 0) {
    window.setTimeout(run, delayMs);
  } else {
    run();
  }
}

async function execShellCommand(command) {
  if (command === "shell.open_raw") {
    openRawCurrentMode();
    return;
  }
  if (command === "run.playback_from_mode") {
    await runAndPlaybackForCurrentMode();
    return;
  }
  if (command === "ops.goal_tune") {
    opsGoalSelect.value = "tune_primitive";
    await applyOperationalGoal();
    return;
  }
  if (command === "ops.goal_scenario") {
    opsGoalSelect.value = "build_scenario";
    await applyOperationalGoal();
    return;
  }
  if (command === "ops.goal_fsm") {
    opsGoalSelect.value = "validate_state_machine";
    await applyOperationalGoal();
    return;
  }
  if (command === "ops.goal_playback") {
    opsGoalSelect.value = "review_playback";
    await applyOperationalGoal();
    return;
  }
  if (command === "fsm.step") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    await stepFsmOnce();
    return;
  }
  if (command === "fsm.step5") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    await stepFsmMany(5);
    return;
  }
  if (command === "fsm.reset") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    await resetFsm();
    return;
  }
  if (command === "fsm.force_recover") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    fsmForceModeSelect.value = "recover_reset";
    fsmForceReasonInput.value = "ops_recover";
    await forceFsmMode();
    return;
  }
  if (command === "fsm.copy_snapshot") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    await copyFsmSnapshot();
    return;
  }
  if (command === "fsm.preset_idle") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    applyFsmPreset("idle");
    return;
  }
  if (command === "fsm.preset_social") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    applyFsmPreset("social");
    return;
  }
  if (command === "fsm.send_recommended") {
    if (state.activeMode !== "state_machine") {
      setMode("state_machine");
    }
    await ensureFsmReady();
    await sendFsmRecommendationToScenario();
    return;
  }
  if (command === "scenario.validate") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    await validateScenario();
    return;
  }
  if (command === "scenario.run") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    await runScenario();
    return;
  }
  if (command === "scenario.sweep") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    await runScenarioSweep();
    return;
  }
  if (command === "scenario.save_experiment") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    await saveScenarioExperiment();
    return;
  }
  if (command === "scenario.apply_best_patch") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    applyScenarioBestSweepPatch();
    return;
  }
  if (command === "scenario.sweep_template") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    applyScenarioSweepTemplateForTarget();
    return;
  }
  if (command === "scenario.promote_best_baseline") {
    if (state.activeMode !== "scenario_lab") {
      setMode("scenario_lab");
    }
    await ensureScenarioReady();
    await promoteScenarioBestToBaseline();
    return;
  }
  postModeCommand(command, {});
}

async function runSuitePlayback() {
  runSuiteBtn.disabled = true;
  setStatus(["suite: generating playback trace..."]);
  try {
    const res = await apiPost("/api/suite", { style: "calm" });
    const traceUrl = String(res?.trace_url || "").trim();
    if (!traceUrl) {
      throw new Error("suite API returned no trace_url");
    }
    state.params.set("trace", traceUrl);
    setMode("playback", { fromSuite: true });
  } catch (err) {
    setStatus([`suite failed: ${err}`]);
    showToast(`Suite failed: ${err}`, "bad", 3600);
  } finally {
    runSuiteBtn.disabled = false;
  }
}

async function runAndPlaybackForCurrentMode() {
  if (state.activeMode === "playback") {
    openRawCurrentMode();
    return;
  }

  if (state.activeMode === "scenario_lab") {
    await ensureScenarioReady();
    await runScenario();
    const trace = String(state.scenario.lastRun?.trace_url || "").trim();
    if (!trace) {
      throw new Error("scenario run did not produce trace_url");
    }
    openPlaybackTrace(trace);
    return;
  }

  if (state.activeMode === "state_machine") {
    await ensureFsmReady();
    await stepFsmOnce();
    await sendFsmRecommendationToScenario();
    await runScenario();
    const trace = String(state.scenario.lastRun?.trace_url || "").trim();
    if (trace) {
      openPlaybackTrace(trace);
    } else {
      await runSuitePlayback();
    }
    return;
  }

  if (state.activeMode === "studio") {
    await execShellCommand("studio.run_preview");
  }
  await runSuitePlayback();
}

function openRawCurrentMode() {
  const raw = buildRawModeSrc(state.activeMode);
  window.open(raw, "_blank", "noopener,noreferrer");
}

function buildPaletteCommands() {
  return [
    {
      id: "mode.studio",
      label: "Switch: Primitive Studio",
      detail: "Tune primitive parameters and run preview.",
      run: () => setMode("studio"),
    },
    {
      id: "mode.scenario_lab",
      label: "Switch: Scenario Lab",
      detail: "Compose scenarios and evaluate metrics.",
      run: () => setMode("scenario_lab"),
    },
    {
      id: "mode.state_machine",
      label: "Switch: State Machine",
      detail: "Simulate behavior mode transitions.",
      run: () => setMode("state_machine"),
    },
    {
      id: "mode.joint_checker",
      label: "Switch: Joint / DH Checker",
      detail: "Validate joint direction, limits, and DH geometry.",
      run: () => setMode("joint_checker"),
    },
    {
      id: "suite.playback",
      label: "Run: Suite Playback",
      detail: "Generate suite trace and open playback mode.",
      run: () => {
        void runSuitePlayback();
      },
    },
    {
      id: "run.playback_from_mode",
      label: "Run: Current Mode + Playback",
      detail: "Run the active mode workflow and open playback.",
      run: () => {
        void runAndPlaybackForCurrentMode();
      },
    },
    {
      id: "ops.goal_tune",
      label: "Ops Goal: Tune Primitive",
      detail: "Switch to Studio with tuning-oriented flow.",
      run: () => {
        opsGoalSelect.value = "tune_primitive";
        void applyOperationalGoal();
      },
    },
    {
      id: "ops.goal_scenario",
      label: "Ops Goal: Build Scenario",
      detail: "Switch to Scenario Lab with composition flow.",
      run: () => {
        opsGoalSelect.value = "build_scenario";
        void applyOperationalGoal();
      },
    },
    {
      id: "ops.goal_fsm",
      label: "Ops Goal: Validate State Machine",
      detail: "Switch to State Machine with operational signal presets.",
      run: () => {
        opsGoalSelect.value = "validate_state_machine";
        void applyOperationalGoal();
      },
    },
    {
      id: "ops.goal_playback",
      label: "Ops Goal: Review Playback",
      detail: "Open existing playback or generate one if absent.",
      run: () => {
        opsGoalSelect.value = "review_playback";
        void applyOperationalGoal();
      },
    },
    {
      id: "studio.run_preview",
      label: "Studio: Run Preview",
      detail: "Trigger preview simulation for current primitive draft.",
      run: () => {
        setMode("studio");
        queueCommand("studio.run_preview", 120);
      },
    },
    {
      id: "studio.save_baseline",
      label: "Studio: Save Baseline",
      detail: "Save current primitive baseline payload.",
      run: () => {
        setMode("studio");
        queueCommand("studio.save_baseline", 120);
      },
    },
    {
      id: "scenario.validate",
      label: "Scenario: Validate",
      detail: "Validate scenario JSON without generating trace.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.validate", 120);
      },
    },
    {
      id: "scenario.run",
      label: "Scenario: Run",
      detail: "Run current scenario and refresh metrics.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.run", 120);
      },
    },
    {
      id: "scenario.sweep",
      label: "Scenario: Sweep",
      detail: "Run parameter sweep on the selected scenario step.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.sweep", 120);
      },
    },
    {
      id: "scenario.apply_best_patch",
      label: "Scenario: Apply Best Patch",
      detail: "Apply best sweep parameter patch to scenario JSON.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.apply_best_patch", 120);
      },
    },
    {
      id: "scenario.sweep_template",
      label: "Scenario: Sweep Template",
      detail: "Generate a sweep grid from target step primitive.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.sweep_template", 120);
      },
    },
    {
      id: "scenario.promote_best_baseline",
      label: "Scenario: Promote Best To Baseline",
      detail: "Write best sweep candidate to primitive baseline params.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.promote_best_baseline", 120);
      },
    },
    {
      id: "scenario.save_experiment",
      label: "Scenario: Save Experiment",
      detail: "Save latest scenario run into history.",
      run: () => {
        setMode("scenario_lab");
        queueCommand("scenario.save_experiment", 120);
      },
    },
    {
      id: "fsm.step",
      label: "FSM: Step",
      detail: "Execute one state-machine update step.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.step", 120);
      },
    },
    {
      id: "fsm.step5",
      label: "FSM: Step x5",
      detail: "Execute five sequential update steps.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.step5", 120);
      },
    },
    {
      id: "fsm.reset",
      label: "FSM: Reset",
      detail: "Reset state machine and tick counters.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.reset", 120);
      },
    },
    {
      id: "fsm.force_recover",
      label: "FSM: Force Recover",
      detail: "Force transition into recover_reset mode.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.force_recover", 120);
      },
    },
    {
      id: "fsm.preset_idle",
      label: "FSM: Preset Idle",
      detail: "Apply idle signal preset for quick stepping.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.preset_idle", 120);
      },
    },
    {
      id: "fsm.preset_social",
      label: "FSM: Preset Social",
      detail: "Apply social-engage signal preset.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.preset_social", 120);
      },
    },
    {
      id: "fsm.copy_snapshot",
      label: "FSM: Copy Snapshot",
      detail: "Copy current FSM operational snapshot JSON.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.copy_snapshot", 120);
      },
    },
    {
      id: "fsm.send_recommended",
      label: "FSM: Send Recommended To Scenario",
      detail: "Queue current FSM recommendation into Scenario steps.",
      run: () => {
        setMode("state_machine");
        queueCommand("fsm.send_recommended", 120);
      },
    },
    {
      id: "joint.zero",
      label: "Joint: Zero Pose",
      detail: "Set all joint sliders to zero pose.",
      run: () => {
        setMode("joint_checker");
        queueCommand("joint.zero", 120);
      },
    },
    {
      id: "joint.mid",
      label: "Joint: Mid Limits",
      detail: "Set all joints to midpoint of configured limits.",
      run: () => {
        setMode("joint_checker");
        queueCommand("joint.mid", 120);
      },
    },
    {
      id: "shell.open_raw",
      label: "Open: Raw Mode Page",
      detail: "Open current mode directly in a new tab.",
      run: () => openRawCurrentMode(),
    },
    {
      id: "shell.focus_viewer",
      label: "Shell: Toggle Focus Viewer",
      detail: "Hide side panels and maximize the main workspace.",
      run: () => setViewerFocus(!state.viewerFocus),
    },
  ];
}

function applyPaletteFilter(rawQuery) {
  const query = String(rawQuery || "").trim().toLowerCase();
  const all = state.paletteCommands;
  if (!query) {
    state.filteredCommands = all.slice();
  } else {
    state.filteredCommands = all.filter((item) => {
      const hay = `${item.label} ${item.detail} ${item.id}`.toLowerCase();
      return hay.includes(query);
    });
  }
  state.paletteIdx = 0;
  renderPaletteList();
}

function renderPaletteList() {
  paletteList.innerHTML = "";
  if (!state.filteredCommands.length) {
    const li = document.createElement("li");
    li.textContent = "No commands match current query.";
    li.style.padding = "10px";
    li.style.color = "#9cc4be";
    paletteList.appendChild(li);
    return;
  }
  state.filteredCommands.forEach((item, idx) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "palette-item";
    if (idx === state.paletteIdx) {
      button.classList.add("active");
    }
    button.innerHTML = `${item.label}<small>${item.detail}</small>`;
    button.addEventListener("click", () => {
      closePalette();
      item.run();
    });
    li.appendChild(button);
    paletteList.appendChild(li);
  });
}

function openPalette() {
  state.paletteOpen = true;
  commandPalette.classList.remove("hidden");
  paletteInput.value = "";
  applyPaletteFilter("");
  paletteInput.focus();
}

function closePalette() {
  state.paletteOpen = false;
  commandPalette.classList.add("hidden");
}

function fsmSetStatus(lines) {
  fsmStatus.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
}

function fsmReadSignals() {
  return {
    person_present: Boolean(fsmPersonPresent.checked),
    person_conf: clamp(finiteNumber(fsmPersonConf.value, 0), 0, 1),
    search_requested: Boolean(fsmSearchRequested.checked),
    search_complete: Boolean(fsmSearchComplete.checked),
    assist_complete: Boolean(fsmAssistComplete.checked),
    user_ack: Boolean(fsmUserAck.checked),
    task_active: Boolean(fsmTaskActive.checked),
    home_requested: Boolean(fsmHomeRequested.checked),
    home_completed: Boolean(fsmHomeCompleted.checked),
    cancel_requested: Boolean(fsmCancelRequested.checked),
    startup_complete: Boolean(fsmStartupComplete.checked),
    health_degraded: Boolean(fsmHealthDegraded.checked),
    planner_open_breaker: Boolean(fsmPlannerBreaker.checked),
    perception_degraded: Boolean(fsmPerceptionDegraded.checked),
  };
}

function fsmApplySignals(signals) {
  fsmPersonPresent.checked = Boolean(signals?.person_present);
  fsmPersonConf.value = clamp(finiteNumber(signals?.person_conf, 0), 0, 1).toFixed(2);
  fsmSearchRequested.checked = Boolean(signals?.search_requested);
  fsmSearchComplete.checked = Boolean(signals?.search_complete);
  fsmAssistComplete.checked = Boolean(signals?.assist_complete);
  fsmUserAck.checked = Boolean(signals?.user_ack);
  fsmTaskActive.checked = Boolean(signals?.task_active);
  fsmHomeRequested.checked = Boolean(signals?.home_requested);
  fsmHomeCompleted.checked = Boolean(signals?.home_completed);
  fsmCancelRequested.checked = Boolean(signals?.cancel_requested);
  fsmStartupComplete.checked = Boolean(signals?.startup_complete);
  fsmHealthDegraded.checked = Boolean(signals?.health_degraded);
  fsmPlannerBreaker.checked = Boolean(signals?.planner_open_breaker);
  fsmPerceptionDegraded.checked = Boolean(signals?.perception_degraded);
}

function fsmPreset(name) {
  const token = String(name || "").trim().toLowerCase();
  if (token === "boot") {
    return {
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
      startup_complete: false,
      health_degraded: false,
      planner_open_breaker: false,
      perception_degraded: false,
    };
  }
  if (token === "idle") {
    return {
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
    };
  }
  if (token === "social") {
    return {
      person_present: true,
      person_conf: 0.9,
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
    };
  }
  if (token === "search") {
    return {
      person_present: true,
      person_conf: 0.75,
      search_requested: true,
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
    };
  }
  if (token === "task") {
    return {
      person_present: false,
      person_conf: 0.05,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: false,
      task_active: true,
      home_requested: false,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: false,
      planner_open_breaker: false,
      perception_degraded: false,
    };
  }
  if (token === "home") {
    return {
      person_present: false,
      person_conf: 0.0,
      search_requested: false,
      search_complete: false,
      assist_complete: false,
      user_ack: false,
      task_active: false,
      home_requested: true,
      home_completed: false,
      cancel_requested: false,
      startup_complete: true,
      health_degraded: false,
      planner_open_breaker: false,
      perception_degraded: false,
    };
  }
  if (token === "recover") {
    return {
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
    };
  }
  if (token === "fault") {
    return {
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
    };
  }
  return fsmPreset("idle");
}

function applyFsmPreset(name) {
  const token = String(name || "").trim().toLowerCase() || "idle";
  fsmApplySignals(fsmPreset(token));
  fsmSetStatus(`Applied preset: ${token}`);
  setStatus([`shell mode: ${state.activeMode}`, `fsm preset: ${token}`]);
}

function clearFsmHistory() {
  state.fsm.history = [];
  renderFsmHistory();
}

function appendFsmHistory(result, source = "step") {
  const decision = result?.mode_decision || {};
  const rec = result?.recommended || {};
  const row = {
    tick: Number(result?.tick_index || 0),
    source: String(source || "step"),
    from: String(decision.from || "?"),
    to: String(decision.to || "?"),
    transitioned: Boolean(decision.transitioned),
    reason: String(decision.reason || ""),
    primitive: String(rec.primitive || ""),
    style: String(rec.style || ""),
    now_s: Number(result?.now_s || 0),
  };
  state.fsm.history.unshift(row);
  const maxRows = Math.max(10, Number(state.fsm.historyMax || 80));
  if (state.fsm.history.length > maxRows) {
    state.fsm.history.length = maxRows;
  }
  renderFsmHistory();
}

function renderFsmHistory() {
  if (!fsmHistoryTableBody) {
    return;
  }
  fsmHistoryTableBody.innerHTML = "";
  const rows = Array.isArray(state.fsm.history) ? state.fsm.history : [];
  rows.forEach((row) => {
    const tr = document.createElement("tr");

    const tickTd = document.createElement("td");
    tickTd.textContent = `${row.tick} (${row.source})`;

    const transitionTd = document.createElement("td");
    transitionTd.textContent = `${row.from} -> ${row.to}${row.transitioned ? "" : " [held]"}`;

    const reasonTd = document.createElement("td");
    reasonTd.textContent = row.reason;

    const recTd = document.createElement("td");
    recTd.textContent = `${row.primitive || "-"}${row.style ? ` (${row.style})` : ""}`;

    tr.appendChild(tickTd);
    tr.appendChild(transitionTd);
    tr.appendChild(reasonTd);
    tr.appendChild(recTd);
    fsmHistoryTableBody.appendChild(tr);
  });
}

function fsmNodePosition(nodeId) {
  const pos = FSM_NODE_LAYOUT[nodeId];
  if (pos) {
    return pos;
  }
  return { x: 480, y: 300 };
}

function fsmEdgeId(edge) {
  return `${edge.from}->${edge.to}:${edge.reason}`;
}

function clearFsmGraph() {
  while (fsmGraphNative.firstChild) {
    fsmGraphNative.removeChild(fsmGraphNative.firstChild);
  }
  state.fsm.nodeEls = {};
  state.fsm.edgeEls = {};
}

function addFsmMarkerDefs() {
  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");
  marker.setAttribute("id", "fsmArrowNative");
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
  fsmGraphNative.appendChild(defs);
}

function fsmPathWithOffset(from, to, offset) {
  const fromPos = fsmNodePosition(from);
  const toPos = fsmNodePosition(to);
  const dx = toPos.x - fromPos.x;
  const dy = toPos.y - fromPos.y;
  const mag = Math.hypot(dx, dy) || 1;
  const ux = dx / mag;
  const uy = dy / mag;
  const px = -uy;
  const py = ux;
  const radius = 44;

  const sx = fromPos.x + ux * radius;
  const sy = fromPos.y + uy * radius;
  const ex = toPos.x - ux * radius;
  const ey = toPos.y - uy * radius;
  const cx = (sx + ex) * 0.5 + px * offset;
  const cy = (sy + ey) * 0.5 + py * offset;

  return {
    d: `M ${sx} ${sy} Q ${cx} ${cy} ${ex} ${ey}`,
    labelX: cx,
    labelY: cy,
  };
}

function drawFsmGraph(metaGraph) {
  clearFsmGraph();
  addFsmMarkerDefs();

  const nodes = Array.isArray(metaGraph?.nodes) ? metaGraph.nodes : [];
  const edges = Array.isArray(metaGraph?.edges) ? metaGraph.edges : [];

  const edgeLayer = document.createElementNS(SVG_NS, "g");
  const nodeLayer = document.createElementNS(SVG_NS, "g");
  fsmGraphNative.appendChild(edgeLayer);
  fsmGraphNative.appendChild(nodeLayer);

  const pairCounts = {};
  edges.forEach((edge) => {
    const key = `${edge.from}->${edge.to}`;
    pairCounts[key] = (pairCounts[key] || 0) + 1;
  });

  edges.forEach((edge) => {
    const reverse = `${edge.to}->${edge.from}`;
    const hasReverse = Boolean(pairCounts[reverse]);
    const shape = fsmPathWithOffset(String(edge.from), String(edge.to), hasReverse ? 22 : 0);

    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("class", "fsm-edge-line");
    path.setAttribute("d", shape.d);
    path.setAttribute("marker-end", "url(#fsmArrowNative)");
    edgeLayer.appendChild(path);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "fsm-edge-label");
    label.setAttribute("x", String(shape.labelX));
    label.setAttribute("y", String(shape.labelY - 6));
    label.textContent = String(edge.label || edge.reason || "");
    edgeLayer.appendChild(label);

    state.fsm.edgeEls[fsmEdgeId(edge)] = path;
  });

  nodes.forEach((node) => {
    const id = String(node.id);
    const pos = fsmNodePosition(id);

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("class", "fsm-node-shape");
    circle.setAttribute("cx", String(pos.x));
    circle.setAttribute("cy", String(pos.y));
    circle.setAttribute("r", "42");
    nodeLayer.appendChild(circle);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "fsm-node-label");
    label.setAttribute("x", String(pos.x));
    label.setAttribute("y", String(pos.y + 6));
    label.textContent = String(node.label || id);
    nodeLayer.appendChild(label);

    state.fsm.nodeEls[id] = circle;
  });
}

function updateFsmGraphHighlight(mode, decision = null) {
  Object.entries(state.fsm.nodeEls).forEach(([id, el]) => {
    el.classList.toggle("active", id === mode);
  });
  Object.values(state.fsm.edgeEls).forEach((el) => {
    el.classList.remove("active");
  });
  if (decision && decision.transitioned) {
    const key = `${decision.from}->${decision.to}:${decision.reason}`;
    const edge = state.fsm.edgeEls[key];
    if (edge) {
      edge.classList.add("active");
    }
  }
}

function renderFsmProposals(rows) {
  fsmProposalTableBody.innerHTML = "";
  const data = Array.isArray(rows) ? rows : [];
  data.forEach((row, idx) => {
    const tr = document.createElement("tr");

    const rankTd = document.createElement("td");
    rankTd.textContent = String(idx + 1);

    const primitiveTd = document.createElement("td");
    primitiveTd.textContent = String(row.primitive || "");

    const scoreTd = document.createElement("td");
    scoreTd.textContent = `${rounded(row.score, 2)} / ${rounded(row.confidence, 2)}`;

    const allowedTd = document.createElement("td");
    const allowed = Boolean(row.allowed_in_mode);
    allowedTd.textContent = allowed ? "yes" : "blocked";
    allowedTd.style.color = allowed ? "#8de5bc" : "#ffae8b";

    const cmdTd = document.createElement("td");
    cmdTd.textContent = JSON.stringify(row.command || {});

    tr.appendChild(rankTd);
    tr.appendChild(primitiveTd);
    tr.appendChild(scoreTd);
    tr.appendChild(allowedTd);
    tr.appendChild(cmdTd);
    fsmProposalTableBody.appendChild(tr);
  });
}

function renderFsmSnapshot(snapshot) {
  const mode = String(snapshot?.mode || "idle_presence");
  updateFsmGraphHighlight(mode, null);
  renderFsmProposals([]);
  fsmSetStatus([
    `mode: ${mode}`,
    `active_skill: ${String(snapshot?.active_skill || "(none)")}`,
    `tick: ${Number(snapshot?.tick_index || 0)}`,
    `now_s: ${rounded(snapshot?.now_s || 0, 3)}`,
    `no_commit_s: ${rounded(snapshot?.no_commit_s || 0, 3)}`,
    `allowed_primitives: ${(snapshot?.allowed_primitives || []).join(", ") || "(none)"}`,
  ]);
}

function renderFsmStepResult(result) {
  const decision = result?.mode_decision || {};
  const mode = String(decision.to || "");
  updateFsmGraphHighlight(mode, decision);
  renderFsmProposals(result?.proposals || []);
  const rec = result?.recommended;
  const recText = rec ? `${rec.primitive} (${rounded(rec.score, 2)})` : "none";
  fsmSetStatus([
    `mode: ${mode || "n/a"}`,
    `active_skill: ${String(result?.active_skill || "(none)")}`,
    `tick: ${Number(result?.tick_index || 0)}`,
    `transition: ${decision.from || "?"} -> ${decision.to || "?"} (${decision.reason || "n/a"})${decision.transitioned ? "" : " [held]"}`,
    `dt_s: ${rounded(result?.dt_s || 0, 3)}`,
    `now_s: ${rounded(result?.now_s || 0, 3)}`,
    `no_commit_s: ${rounded(result?.no_commit_s || 0, 3)}`,
    `zone_hint: ${String(result?.zone_hint || "(none)")}`,
    `recommended: ${recText}`,
    `recommended_style: ${String(rec?.style || "(none)")}`,
    `allowed_primitives: ${(result?.allowed_primitives || []).join(", ") || "(none)"}`,
    `signals: ${JSON.stringify(result?.signals || {})}`,
  ]);
}

function stopFsmAutoRun() {
  state.fsm.autoRunning = false;
  if (state.fsm.autoTimer != null) {
    window.clearTimeout(state.fsm.autoTimer);
    state.fsm.autoTimer = null;
  }
  fsmAutoBtn.textContent = "Auto Run";
}

async function stepFsmOnce() {
  if (state.fsm.inFlight) {
    return;
  }
  state.fsm.inFlight = true;
  fsmStepBtn.disabled = true;
  try {
    const dt = clamp(finiteNumber(fsmDtInput.value, 0.35), 0.05, 10.0);
    fsmDtInput.value = Number.isFinite(dt) ? dt.toFixed(2) : "0.35";
    const payload = {
      dt_s: Number(fsmDtInput.value),
      commit: Boolean(fsmCommitToggle.checked),
      zone_hint: String(fsmZoneSelect.value || ""),
      signals: fsmReadSignals(),
    };
    const result = await apiPost("/api/state_machine/step", payload);
    state.fsm.lastResult = result;
    renderFsmStepResult(result);
    appendFsmHistory(result, "step");
    setStatus([
      `shell mode: ${state.activeMode}`,
      `fsm: ${result?.mode_decision?.from || "?"} -> ${result?.mode_decision?.to || "?"}`,
      `reason: ${result?.mode_decision?.reason || "n/a"}`,
    ]);
  } catch (err) {
    fsmSetStatus(`step failed: ${err.message || err}`);
    stopFsmAutoRun();
    throw err;
  } finally {
    state.fsm.inFlight = false;
    fsmStepBtn.disabled = false;
  }
}

async function stepFsmMany(count) {
  const n = Math.max(1, Math.floor(Number(count) || 1));
  for (let i = 0; i < n; i += 1) {
    if (state.activeMode !== "state_machine") {
      break;
    }
    await stepFsmOnce();
  }
}

async function forceFsmMode() {
  if (state.fsm.inFlight) {
    return;
  }
  state.fsm.inFlight = true;
  fsmForceBtn.disabled = true;
  const mode = String(fsmForceModeSelect.value || "").trim();
  if (!mode) {
    state.fsm.inFlight = false;
    fsmForceBtn.disabled = false;
    throw new Error("select a force mode first");
  }
  try {
    const reason = String(fsmForceReasonInput.value || "").trim() || "ops_force";
    const dt = clamp(finiteNumber(fsmDtInput.value, 0.35), 0.0, 10.0);
    const payload = {
      mode,
      reason,
      dt_s: dt,
      commit: Boolean(fsmCommitToggle.checked),
      zone_hint: String(fsmZoneSelect.value || ""),
      signals: fsmReadSignals(),
    };
    const result = await apiPost("/api/state_machine/force", payload);
    state.fsm.lastResult = result;
    renderFsmStepResult(result);
    appendFsmHistory(result, "force");
    setStatus([
      `shell mode: ${state.activeMode}`,
      `fsm force: ${mode}`,
      `reason: ${reason}`,
    ]);
  } finally {
    state.fsm.inFlight = false;
    fsmForceBtn.disabled = false;
  }
}

function fsmSnapshotForOps() {
  return {
    exported_at_utc: new Date().toISOString(),
    mode: state.fsm.lastResult?.mode_decision?.to || state.fsm.meta?.state?.mode || "unknown",
    signals: fsmReadSignals(),
    last_result: state.fsm.lastResult || null,
    transition_history: state.fsm.history || [],
  };
}

async function copyTextToClipboard(text) {
  const token = String(text || "");
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(token);
    return;
  }
  const area = document.createElement("textarea");
  area.value = token;
  area.setAttribute("readonly", "true");
  area.style.position = "fixed";
  area.style.opacity = "0";
  area.style.pointerEvents = "none";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

async function copyFsmSnapshot() {
  const payload = prettyJson(fsmSnapshotForOps());
  await copyTextToClipboard(payload);
  showToast("FSM snapshot copied", "ok");
  setStatus([
    `shell mode: ${state.activeMode}`,
    "fsm snapshot copied to clipboard",
  ]);
}

async function scheduleFsmAutoRun() {
  if (!state.fsm.autoRunning) {
    return;
  }
  if (state.activeMode !== "state_machine") {
    stopFsmAutoRun();
    return;
  }
  try {
    await stepFsmOnce();
  } catch (_err) {
    return;
  }
  if (!state.fsm.autoRunning) {
    return;
  }
  const dt = clamp(Number(fsmDtInput.value), 0.05, 10.0);
  const waitMs = Math.max(120, Math.round(dt * 1000));
  state.fsm.autoTimer = window.setTimeout(() => {
    void scheduleFsmAutoRun();
  }, waitMs);
}

async function resetFsm() {
  stopFsmAutoRun();
  const res = await apiPost("/api/state_machine/reset", {});
  const meta = res?.meta || {};
  const snapshot = res?.state || meta?.state || {};
  state.fsm.meta = meta;
  state.fsm.lastResult = null;
  clearFsmHistory();
  fsmApplySignals(meta.default_signals || {});
  drawFsmGraph(meta.graph || {});
  renderFsmSnapshot(snapshot);
  setStatus([
    `shell mode: ${state.activeMode}`,
    "fsm reset complete",
  ]);
}

async function ensureFsmReady(force = false) {
  if (state.fsm.ready && !force) {
    return;
  }
  if (state.fsm.loadingPromise) {
    return state.fsm.loadingPromise;
  }
  state.fsm.loadingPromise = (async () => {
    try {
      const meta = await apiGet("/api/state_machine/meta");
      state.fsm.meta = meta;
      fsmApplySignals(meta.default_signals || {});
      drawFsmGraph(meta.graph || {});
      renderFsmSnapshot(meta.state || {});
      clearFsmHistory();
      fsmSetStatus([
        `State machine ready.`,
        `config_path: ${String(meta.config_path || "(unknown)")}`,
      ]);
      state.fsm.ready = true;
    } catch (err) {
      state.fsm.ready = false;
      fsmSetStatus([
        "State machine panel failed to load metadata.",
        "Run with:",
        "uv run python tools/primitive_sim/run.py --scenario state_machine --port 8766",
        "",
        `error: ${String(err)}`,
      ]);
      throw err;
    } finally {
      state.fsm.loadingPromise = null;
    }
  })();
  return state.fsm.loadingPromise;
}

function scenarioSetStatus(lines) {
  scnStatus.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
}

function scenarioSetMetrics(lines) {
  scnMetrics.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
}

function scenarioSetSweepStatus(lines) {
  scnSweepStatus.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
}

function scenarioResolveTargetStepIndex(steps, token) {
  if (!Array.isArray(steps) || !steps.length) {
    throw new Error("scenario steps are empty");
  }
  if (token == null || String(token).trim() === "") {
    return steps.length - 1;
  }
  const raw = String(token).trim();
  if (/^-?\d+$/.test(raw)) {
    let idx = Number(raw);
    if (idx < 0) {
      idx = steps.length + idx;
    }
    if (idx < 0 || idx >= steps.length) {
      throw new Error(`target step index out of range: ${raw}`);
    }
    return idx;
  }
  const lowered = raw.toLowerCase();
  for (let i = 0; i < steps.length; i += 1) {
    const row = steps[i];
    const name = String(row?.name || "").trim().toLowerCase();
    if (name && name === lowered) {
      return i;
    }
  }
  throw new Error(`target step not found: ${raw}`);
}

function scenarioTargetStepDetails() {
  const steps = scenarioStepsFromText();
  const idx = scenarioResolveTargetStepIndex(steps, String(scnSweepTargetInput.value || "").trim());
  const step = steps[idx];
  if (!step || typeof step !== "object" || Array.isArray(step)) {
    throw new Error(`invalid target step at index ${idx}`);
  }
  return { steps, idx, step };
}

function buildSweepTemplateForPrimitive(primitive, command = {}) {
  const p = String(primitive || "").trim();
  const cmd = command && typeof command === "object" && !Array.isArray(command) ? command : {};
  if (p === "breath") {
    const centerAmp = finiteNumber(cmd.amp_rad, 0.08);
    const centerPeriod = finiteNumber(cmd.period_s, 6.5);
    return {
      amp_rad: [centerAmp * 0.7, centerAmp, centerAmp * 1.3].map((v) => Number(v.toFixed(4))),
      period_s: [centerPeriod * 0.7, centerPeriod, centerPeriod * 1.3].map((v) => Number(v.toFixed(4))),
    };
  }
  if (p === "glance") {
    return {
      amp_rad: [0.16, 0.24, 0.32],
      duration_s: [0.5, 0.7, 0.9],
      rate_rad_s: [1.2, 1.6, 2.0],
    };
  }
  if (p === "nod") {
    return {
      amp_rad: [0.12, 0.2, 0.28],
      duration_s: [0.7, 0.9, 1.1],
      rate_rad_s: [1.4, 1.8, 2.2],
    };
  }
  if (p === "gaze_to") {
    const yaw = finiteNumber(cmd.yaw_rad, 0.0);
    const pitch = finiteNumber(cmd.pitch_rad, 0.0);
    return {
      yaw_rad: [yaw - 0.12, yaw, yaw + 0.12].map((v) => Number(v.toFixed(4))),
      pitch_rad: [pitch - 0.12, pitch, pitch + 0.12].map((v) => Number(v.toFixed(4))),
    };
  }
  if (p === "orient_to_zone") {
    return {
      amp_rad: [0.14, 0.22, 0.3],
      rate_rad_s: [1.0, 1.3, 1.8],
    };
  }
  if (p === "home") {
    return {
      rate_rad_s: [0.8, 1.2, 1.8],
    };
  }
  if (p === "move_to") {
    return {
      rate_rad_s: [0.8, 1.2, 1.6],
      timeout_s: [1.2, 2.0, 2.8],
    };
  }
  return {
    rate_rad_s: [0.8, 1.2, 1.6],
  };
}

function applyScenarioSweepTemplateForTarget() {
  const { idx, step } = scenarioTargetStepDetails();
  const primitive = String(step.primitive || "").trim();
  if (!primitive) {
    throw new Error(`target step ${idx} has no primitive`);
  }
  const command = step.command && typeof step.command === "object" && !Array.isArray(step.command) ? step.command : {};
  const template = buildSweepTemplateForPrimitive(primitive, command);
  scnSweepGridText.value = prettyJson(template);
  scenarioSetSweepStatus(`Loaded sweep template for step ${idx} (${primitive}).`);
}

function scenarioStepFromProposal(rec, stepIndex) {
  const primitive = String(rec.primitive || "").trim();
  if (!primitive) {
    throw new Error("recommended proposal has no primitive");
  }
  const style = String(rec.style || "calm").trim() || "calm";
  const command = rec.command && typeof rec.command === "object" && !Array.isArray(rec.command) ? rec.command : {};
  return {
    name: `fsm_${stepIndex + 1}_${primitive}`,
    primitive,
    style,
    duration_s: defaultDurationForPrimitive(primitive),
    stop_on_done: !CONTINUOUS_PRIMITIVES.has(primitive),
    command: { ...command },
  };
}

async function sendFsmRecommendationToScenario() {
  const rec = state.fsm.lastResult?.recommended;
  if (!rec) {
    throw new Error("no FSM recommendation available; step FSM first");
  }
  setMode("scenario_lab");
  await ensureScenarioReady();
  const steps = scenarioStepsFromText({ allowEmpty: true });
  const next = scenarioStepFromProposal(rec, steps.length);
  steps.push(next);
  setScenarioSteps(steps);

  scnPrimitiveSelect.value = String(next.primitive);
  if (Array.from(scnStyleSelect.options).some((opt) => opt.value === next.style)) {
    scnStyleSelect.value = next.style;
  }
  scnDurationInput.value = String(next.duration_s);
  scnStopOnDone.checked = Boolean(next.stop_on_done);
  scnCommandText.value = prettyJson(next.command);
  scenarioSetStatus(`Added FSM recommendation as scenario step ${steps.length}.`);
  setStatus([
    `shell mode: ${state.activeMode}`,
    `fsm recommendation queued: ${next.primitive}`,
    `scenario steps: ${steps.length}`,
  ]);
}

function renderScenarioSweepRanking(rows) {
  scnSweepTableBody.innerHTML = "";
  const data = Array.isArray(rows) ? rows : [];
  data.forEach((row) => {
    const tr = document.createElement("tr");

    const rankTd = document.createElement("td");
    rankTd.textContent = String(Number(row?.rank || 0));

    const scoreTd = document.createElement("td");
    scoreTd.textContent = Number(row?.score || 0).toFixed(4);

    const patchTd = document.createElement("td");
    patchTd.textContent = JSON.stringify(row?.param_patch || {});

    const metrics = row?.metrics || {};
    const durationTd = document.createElement("td");
    durationTd.textContent = Number(metrics.duration_s || 0).toFixed(2);

    const peakVelTd = document.createElement("td");
    peakVelTd.textContent = Number(metrics.peak_joint_vel_rad_s || 0).toFixed(3);

    const limitsTd = document.createElement("td");
    limitsTd.textContent = String(Number(metrics.limit_violation_count || 0));

    const playbackTd = document.createElement("td");
    const traceUrl = String(row?.trace_url || "");
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.textContent = "Open";
    if (!traceUrl) {
      openBtn.disabled = true;
    } else {
      openBtn.addEventListener("click", () => {
        try {
          openPlaybackTrace(traceUrl);
        } catch (err) {
          scenarioSetSweepStatus(`open playback failed: ${err.message || err}`);
        }
      });
    }
    playbackTd.appendChild(openBtn);

    tr.appendChild(rankTd);
    tr.appendChild(scoreTd);
    tr.appendChild(patchTd);
    tr.appendChild(durationTd);
    tr.appendChild(peakVelTd);
    tr.appendChild(limitsTd);
    tr.appendChild(playbackTd);
    scnSweepTableBody.appendChild(tr);
  });
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

function scenarioStepsFromText({ allowEmpty = false } = {}) {
  const raw = String(scnJsonText.value || "").trim();
  if (!raw) {
    if (allowEmpty) {
      return [];
    }
    throw new Error("scenario JSON is empty");
  }
  const parsed = JSON.parse(raw);
  if (Array.isArray(parsed)) {
    if (!parsed.length && !allowEmpty) {
      throw new Error("scenario steps are empty");
    }
    return parsed;
  }
  if (parsed && typeof parsed === "object" && Array.isArray(parsed.steps)) {
    if (!parsed.steps.length && !allowEmpty) {
      throw new Error("scenario steps are empty");
    }
    return parsed.steps;
  }
  throw new Error("scenario JSON must be an array or object with non-empty steps[]");
}

function setScenarioSteps(steps) {
  scnJsonText.value = prettyJson({ steps });
}

function scenarioBaselineCommandFor(primitiveId) {
  const src = state.scenario.meta?.baseline?.primitives?.[primitiveId];
  if (src && typeof src === "object" && !Array.isArray(src)) {
    return src;
  }
  return {};
}

function populateScenarioBuilderFromPrimitive() {
  const primitive = String(scnPrimitiveSelect.value || "");
  scnDurationInput.value = String(defaultDurationForPrimitive(primitive));
  scnStopOnDone.checked = primitive !== "breath" && primitive !== "hold";
  scnCommandText.value = prettyJson(scenarioBaselineCommandFor(primitive));
}

function addScenarioBuilderStep() {
  const primitive = String(scnPrimitiveSelect.value || "").trim();
  if (!primitive) {
    throw new Error("select a primitive before adding a step");
  }
  const style = String(scnStyleSelect.value || "calm");
  const duration = Number(scnDurationInput.value);
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("duration must be a positive number");
  }

  let command = {};
  try {
    const parsed = JSON.parse(String(scnCommandText.value || "{}"));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("command must be a JSON object");
    }
    command = parsed;
  } catch (err) {
    throw new Error(`invalid step command JSON: ${err.message || err}`);
  }

  const currentSteps = scenarioStepsFromText({ allowEmpty: true });
  const next = {
    name: `step_${currentSteps.length + 1}_${primitive}`,
    primitive,
    style,
    duration_s: duration,
    stop_on_done: Boolean(scnStopOnDone.checked),
    command,
  };
  currentSteps.push(next);
  setScenarioSteps(currentSteps);
}

function formatScenarioMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") {
    return "Metrics: unavailable";
  }
  return [
    "Scenario metrics",
    `sample_count: ${Number(metrics.sample_count || 0)}`,
    `duration_s: ${Number(metrics.duration_s || 0).toFixed(3)}`,
    `path_length_rad: ${Number(metrics.path_length_rad || 0).toFixed(3)}`,
    `mean_abs_joint_vel_rad_s: ${Number(metrics.mean_abs_joint_vel_rad_s || 0).toFixed(3)}`,
    `peak_joint_vel_rad_s: ${Number(metrics.peak_joint_vel_rad_s || 0).toFixed(3)}`,
    `limit_violation_count: ${Number(metrics.limit_violation_count || 0)}`,
    `min_limit_margin_rad: ${Number(metrics.min_limit_margin_rad || 0).toFixed(4)}`,
    `primitive_switch_count: ${Number(metrics.primitive_switch_count || 0)}`,
  ].join("\n");
}

function openPlaybackTrace(traceUrl) {
  const trace = String(traceUrl || "").trim();
  if (!trace) {
    throw new Error("trace URL is missing");
  }
  state.params.set("trace", trace);
  setMode("playback");
}

function renderScenarioHistory(rows) {
  scnHistoryTableBody.innerHTML = "";
  const data = Array.isArray(rows) ? rows : [];
  data.forEach((row) => {
    const tr = document.createElement("tr");

    const savedTd = document.createElement("td");
    savedTd.textContent = String(row?.saved_at_utc || "");

    const nameTd = document.createElement("td");
    nameTd.textContent = String(row?.name || "");

    const metrics = row?.metrics || {};
    const samplesTd = document.createElement("td");
    samplesTd.textContent = String(Number(metrics.sample_count || 0));

    const durationTd = document.createElement("td");
    durationTd.textContent = Number(metrics.duration_s || 0).toFixed(2);

    const playbackTd = document.createElement("td");
    const traceUrl = String(row?.trace_url || "");
    const previewBtn = document.createElement("button");
    previewBtn.type = "button";
    previewBtn.textContent = "Open";
    if (!traceUrl) {
      previewBtn.disabled = true;
    } else {
      previewBtn.addEventListener("click", () => {
        try {
          openPlaybackTrace(traceUrl);
        } catch (err) {
          scenarioSetStatus(`open playback failed: ${err.message || err}`);
        }
      });
    }
    playbackTd.appendChild(previewBtn);

    tr.appendChild(savedTd);
    tr.appendChild(nameTd);
    tr.appendChild(samplesTd);
    tr.appendChild(durationTd);
    tr.appendChild(playbackTd);
    scnHistoryTableBody.appendChild(tr);
  });
}

async function refreshScenarioHistory() {
  const res = await apiGet("/api/scenario/history?limit=25");
  state.scenario.history = Array.isArray(res?.history) ? res.history : [];
  renderScenarioHistory(state.scenario.history);
}

async function validateScenario() {
  const steps = scenarioStepsFromText();
  const res = await apiPost("/api/scenario/simulate", { dry_run: true, steps });
  scenarioSetStatus(`Validation OK: ${Number(res.segment_count || 0)} segments`);
  setStatus([
    `shell mode: ${state.activeMode}`,
    `scenario validation passed: ${Number(res.segment_count || 0)} segments`,
  ]);
}

async function runScenario() {
  const steps = scenarioStepsFromText();
  const runName = String(scnNameInput.value || "").trim() || "scenario_run";
  scenarioSetStatus("Running scenario...");
  const res = await apiPost("/api/scenario/simulate", {
    dry_run: false,
    name: runName,
    steps,
  });
  state.scenario.lastRun = res;
  scenarioSetMetrics(formatScenarioMetrics(res.metrics));
  scenarioSetStatus([
    "Run complete.",
    `run_name: ${String(res.run_name || runName)}`,
    `trace_path: ${String(res.trace_path || "")}`,
    `trace_url: ${String(res.trace_url || "")}`,
    `sample_count: ${Number(res.sample_count || 0)}`,
  ]);
  setStatus([
    `shell mode: ${state.activeMode}`,
    `scenario run complete: ${String(res.run_name || runName)}`,
    `samples: ${Number(res.sample_count || 0)}`,
  ]);
}

async function runScenarioSweep() {
  let paramGrid = {};
  try {
    paramGrid = JSON.parse(String(scnSweepGridText.value || "{}"));
  } catch (err) {
    throw new Error(`invalid sweep grid JSON: ${err.message || err}`);
  }
  if (!paramGrid || typeof paramGrid !== "object" || Array.isArray(paramGrid)) {
    throw new Error("sweep grid must be a JSON object");
  }

  const steps = scenarioStepsFromText();
  const targetToken = String(scnSweepTargetInput.value || "").trim();
  const targetIdx = scenarioResolveTargetStepIndex(steps, targetToken);
  scnSweepTargetInput.value = String(targetIdx);
  const topK = clamp(Math.round(finiteNumber(scnSweepTopKInput.value, 8)), 1, 120);
  scnSweepTopKInput.value = String(topK);

  const runName = String(scnNameInput.value || "").trim() || "scenario_sweep";
  scenarioSetSweepStatus("Sweep: running...");
  const res = await apiPost("/api/scenario/sweep", {
    name: runName,
    steps,
    target_step: targetIdx,
    top_k: topK,
    save_traces: Boolean(scnSweepSaveTraces.checked),
    param_grid: paramGrid,
  });

  state.scenario.lastSweep = res;
  renderScenarioSweepRanking(res.ranking || []);
  const best = res.best || {};
  scenarioSetSweepStatus([
    "Sweep complete.",
    `run_name: ${String(res.run_name || runName)}`,
    `sweep_id: ${String(res.sweep_id || "")}`,
    `target_step_index: ${Number(res.target_step_index || 0)}`,
    `candidates: ${Number(res.candidate_count || 0)} (valid: ${Number(res.valid_count || 0)}, errors: ${Number(res.error_count || 0)})`,
    `best_score: ${Number(best.score || 0).toFixed(4)}`,
    `best_patch: ${JSON.stringify(best.param_patch || {})}`,
  ]);
  setStatus([
    `shell mode: ${state.activeMode}`,
    `scenario sweep complete: ${String(res.sweep_id || "")}`,
    `candidates: ${Number(res.valid_count || 0)}/${Number(res.candidate_count || 0)}`,
  ]);
}

function applyScenarioBestSweepPatch() {
  const sweep = state.scenario.lastSweep;
  if (!sweep || typeof sweep !== "object") {
    throw new Error("run a sweep first");
  }
  const best = sweep.best;
  if (!best || typeof best !== "object") {
    throw new Error("no best sweep candidate available");
  }
  const patch = best.param_patch;
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) {
    throw new Error("best candidate has no parameter patch");
  }

  const steps = scenarioStepsFromText();
  const sweepTargetIdx = Number(sweep.target_step_index);
  const targetIdx = Number.isInteger(sweepTargetIdx) && sweepTargetIdx >= 0 && sweepTargetIdx < steps.length
    ? sweepTargetIdx
    : scenarioResolveTargetStepIndex(steps, String(scnSweepTargetInput.value || "").trim());
  const target = steps[targetIdx];
  if (!target || typeof target !== "object" || Array.isArray(target)) {
    throw new Error(`invalid target step at index ${targetIdx}`);
  }
  const nextStep = { ...target };
  const currentCommand = nextStep.command;
  const command = currentCommand && typeof currentCommand === "object" && !Array.isArray(currentCommand)
    ? { ...currentCommand }
    : {};
  Object.entries(patch).forEach(([key, value]) => {
    command[key] = value;
  });
  nextStep.command = command;
  steps[targetIdx] = nextStep;
  setScenarioSteps(steps);
  scnSweepTargetInput.value = String(targetIdx);
  scenarioSetStatus(`Applied sweep best patch to step ${targetIdx}.`);
}

async function promoteScenarioBestToBaseline() {
  const sweep = state.scenario.lastSweep;
  if (!sweep || typeof sweep !== "object") {
    throw new Error("run a sweep first");
  }
  const best = sweep.best;
  if (!best || typeof best !== "object") {
    throw new Error("best sweep candidate is unavailable");
  }

  const steps = Array.isArray(best.steps) ? best.steps : scenarioStepsFromText();
  if (!Array.isArray(steps) || !steps.length) {
    throw new Error("candidate steps are empty");
  }
  const targetIdxRaw = Number(sweep.target_step_index);
  const targetIdx = Number.isInteger(targetIdxRaw) && targetIdxRaw >= 0 && targetIdxRaw < steps.length
    ? targetIdxRaw
    : scenarioResolveTargetStepIndex(steps, String(scnSweepTargetInput.value || "").trim());
  const target = steps[targetIdx];
  if (!target || typeof target !== "object" || Array.isArray(target)) {
    throw new Error(`invalid target step at index ${targetIdx}`);
  }

  const primitive = String(target.primitive || "").trim();
  if (!primitive) {
    throw new Error(`target step ${targetIdx} has no primitive`);
  }
  const command = target.command;
  if (!command || typeof command !== "object" || Array.isArray(command)) {
    throw new Error("target command is missing or invalid");
  }

  const res = await apiPost("/api/baseline", {
    primitive,
    command,
  });
  if (!res || typeof res !== "object") {
    throw new Error("baseline update returned invalid payload");
  }
  if (state.scenario.meta && typeof state.scenario.meta === "object") {
    state.scenario.meta.baseline = res;
  }
  if (String(scnPrimitiveSelect.value || "") === primitive) {
    const latest = res?.primitives?.[primitive];
    if (latest && typeof latest === "object" && !Array.isArray(latest)) {
      scnCommandText.value = prettyJson(latest);
    }
  }
  scenarioSetSweepStatus([
    "Promoted best candidate to baseline.",
    `primitive: ${primitive}`,
    `target_step_index: ${targetIdx}`,
    `updated_at_utc: ${String(res.updated_at_utc || "(unknown)")}`,
  ]);
  setStatus([
    `shell mode: ${state.activeMode}`,
    `baseline updated: ${primitive}`,
    `target step: ${targetIdx}`,
  ]);
}

async function saveScenarioExperiment() {
  if (!state.scenario.lastRun) {
    throw new Error("run a scenario before saving an experiment");
  }
  const name = String(scnNameInput.value || "").trim();
  if (!name) {
    throw new Error("experiment name is required");
  }
  const notes = String(scnNotesInput.value || "").trim();
  const steps = Array.isArray(state.scenario.lastRun.steps)
    ? state.scenario.lastRun.steps
    : scenarioStepsFromText();
  const payload = {
    name,
    notes,
    steps,
    metrics: state.scenario.lastRun.metrics || {},
    trace_path: String(state.scenario.lastRun.trace_path || ""),
    trace_url: String(state.scenario.lastRun.trace_url || ""),
  };
  const res = await apiPost("/api/scenario/save_experiment", payload);
  scenarioSetStatus(`Experiment saved: ${name}`);
  if (Array.isArray(res?.history)) {
    state.scenario.history = res.history;
    renderScenarioHistory(state.scenario.history);
  } else {
    await refreshScenarioHistory();
  }
  setStatus([`shell mode: ${state.activeMode}`, `saved experiment: ${name}`]);
}

async function ensureScenarioReady(force = false) {
  if (state.scenario.ready && !force) {
    return;
  }
  if (state.scenario.loadingPromise) {
    return state.scenario.loadingPromise;
  }
  state.scenario.loadingPromise = (async () => {
    try {
      const meta = await apiGet("/api/scenario/meta");
      state.scenario.meta = meta;

      scnPrimitiveSelect.innerHTML = "";
      const primitives = Array.isArray(meta.primitives) ? meta.primitives : [];
      primitives.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.id || "");
        option.textContent = String(item.label || item.id || "");
        scnPrimitiveSelect.appendChild(option);
      });
      if (primitives.some((row) => String(row.id) === "breath")) {
        scnPrimitiveSelect.value = "breath";
      }

      scnStyleSelect.innerHTML = "";
      const styles = Array.isArray(meta.styles) ? meta.styles : [];
      styles.forEach((style) => {
        const option = document.createElement("option");
        option.value = String(style);
        option.textContent = String(style);
        scnStyleSelect.appendChild(option);
      });
      if (styles.includes("calm")) {
        scnStyleSelect.value = "calm";
      }

      populateScenarioBuilderFromPrimitive();
      setScenarioSteps(Array.isArray(meta.default_steps) ? meta.default_steps : []);
      scenarioSetMetrics("Metrics: run a scenario");
      const sweepMeta = meta?.sweep || {};
      const defaultTopK = clamp(Math.round(finiteNumber(sweepMeta.default_top_k, 8)), 1, 120);
      scnSweepTopKInput.value = String(defaultTopK);
      if (!String(scnSweepTargetInput.value || "").trim()) {
        scnSweepTargetInput.value = "-1";
      }
      try {
        applyScenarioSweepTemplateForTarget();
      } catch (_err) {
        scenarioSetSweepStatus("Sweep: pending");
      }
      renderScenarioSweepRanking([]);
      state.scenario.lastSweep = null;
      scenarioSetStatus([
        "Scenario Lab ready.",
        `config_path: ${String(meta.config_path || "(unknown)")}`,
      ]);
      await refreshScenarioHistory();
      state.scenario.ready = true;
    } catch (err) {
      state.scenario.ready = false;
      scenarioSetStatus([
        "Scenario panel failed to load metadata.",
        "Run with:",
        "uv run python tools/primitive_sim/run.py --scenario scenario_lab --port 8766",
        "",
        `error: ${String(err)}`,
      ]);
      throw err;
    } finally {
      state.scenario.loadingPromise = null;
    }
  })();
  return state.scenario.loadingPromise;
}

function bindUi() {
  modeFrame.addEventListener("load", () => {
    if (isNativeMode(state.activeMode)) {
      return;
    }
    const src = modeFrame.getAttribute("src") || "";
    if (!src || src === "about:blank") {
      return;
    }
    setStatus([
      `shell mode: ${state.activeMode}`,
      `iframe src: ${src}`,
      "frame loaded",
    ]);
  });

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) {
      return;
    }
    const msg = event.data;
    if (!msg || typeof msg !== "object") {
      return;
    }
    if (msg.source !== "lamp-mode" || msg.type !== "status") {
      return;
    }
    const mode = String(msg.mode || "");
    if (mode && mode !== state.activeMode) {
      return;
    }
    const kind = String(msg.kind || "ok");
    const text = String(msg.message || "mode status");
    const details = String(msg.details || "");
    showToast(text, kind);
    setStatus([
      `shell mode: ${state.activeMode}`,
      `mode status: ${text}`,
      details,
    ]);
  });

  openRawBtn.addEventListener("click", () => {
    openRawCurrentMode();
  });

  modeActionBtn.addEventListener("click", () => {
    const action = MODE_PRIMARY_ACTION[state.activeMode] || MODE_PRIMARY_ACTION.studio;
    void execShellCommand(action.command)
      .then(() => {
        showToast(`Action: ${action.label}`, "ok");
      })
      .catch((err) => {
        showToast(`Action failed: ${err}`, "bad", 3600);
      });
  });

  runSuiteBtn.addEventListener("click", () => {
    void runSuitePlayback();
  });

  focusViewerBtn.addEventListener("click", () => {
    setViewerFocus(!state.viewerFocus);
  });

  runAndPlaybackBtn.addEventListener("click", () => {
    runAndPlaybackBtn.disabled = true;
    void runAndPlaybackForCurrentMode()
      .catch((err) => {
        showToast(`Run + playback failed: ${err}`, "bad", 3600);
      })
      .finally(() => {
        runAndPlaybackBtn.disabled = false;
      });
  });

  paletteBtn.addEventListener("click", () => {
    openPalette();
  });

  paletteCloseBtn.addEventListener("click", () => {
    closePalette();
  });

  commandPalette.addEventListener("click", (ev) => {
    if (ev.target === commandPalette) {
      closePalette();
    }
  });

  paletteInput.addEventListener("input", () => {
    applyPaletteFilter(paletteInput.value);
  });

  paletteInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      closePalette();
      return;
    }
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      if (!state.filteredCommands.length) {
        return;
      }
      state.paletteIdx = Math.min(state.filteredCommands.length - 1, state.paletteIdx + 1);
      renderPaletteList();
      return;
    }
    if (ev.key === "ArrowUp") {
      ev.preventDefault();
      if (!state.filteredCommands.length) {
        return;
      }
      state.paletteIdx = Math.max(0, state.paletteIdx - 1);
      renderPaletteList();
      return;
    }
    if (ev.key === "Enter") {
      ev.preventDefault();
      const cmd = state.filteredCommands[state.paletteIdx];
      if (!cmd) {
        return;
      }
      closePalette();
      cmd.run();
    }
  });

  qaTuneBtn.addEventListener("click", () => setMode("studio"));
  qaScenarioBtn.addEventListener("click", () => setMode("scenario_lab"));
  qaStateBtn.addEventListener("click", () => setMode("state_machine"));
  qaJointBtn.addEventListener("click", () => setMode("joint_checker"));

  opsGoalSelect.addEventListener("change", () => {
    updateOpsGoalHint();
  });
  opsGoalBtn.addEventListener("click", () => {
    void applyOperationalGoal().catch((err) => {
      setStatus([
        `shell mode: ${state.activeMode}`,
        `ops goal failed: ${err.message || err}`,
      ]);
      showToast(`Ops goal failed: ${err}`, "bad", 3600);
    });
  });

  fsmStepBtn.addEventListener("click", () => {
    void stepFsmOnce().catch((err) => {
      showToast(`FSM step failed: ${err}`, "bad", 3600);
    });
  });
  fsmStep5Btn.addEventListener("click", () => {
    void stepFsmMany(5).catch((err) => {
      showToast(`FSM step x5 failed: ${err}`, "bad", 3600);
    });
  });
  fsmAutoBtn.addEventListener("click", () => {
    if (state.fsm.autoRunning) {
      stopFsmAutoRun();
      return;
    }
    state.fsm.autoRunning = true;
    fsmAutoBtn.textContent = "Stop Auto";
    void scheduleFsmAutoRun();
  });
  fsmResetBtn.addEventListener("click", () => {
    void resetFsm().catch((err) => {
      fsmSetStatus(`reset failed: ${err.message || err}`);
      showToast(`FSM reset failed: ${err}`, "bad", 3600);
    });
  });
  fsmForceBtn.addEventListener("click", () => {
    void forceFsmMode().catch((err) => {
      fsmSetStatus(`force mode failed: ${err.message || err}`);
      showToast(`FSM force mode failed: ${err}`, "bad", 3600);
    });
  });
  fsmCopySnapshotBtn.addEventListener("click", () => {
    void copyFsmSnapshot().catch((err) => {
      fsmSetStatus(`copy snapshot failed: ${err.message || err}`);
      showToast(`FSM snapshot copy failed: ${err}`, "bad", 3600);
    });
  });
  fsmUseRecommendedBtn.addEventListener("click", () => {
    void sendFsmRecommendationToScenario().catch((err) => {
      fsmSetStatus(`handoff failed: ${err.message || err}`);
      showToast(`FSM handoff failed: ${err}`, "bad", 3600);
    });
  });
  fsmPresetBootBtn.addEventListener("click", () => applyFsmPreset("boot"));
  fsmPresetIdleBtn.addEventListener("click", () => applyFsmPreset("idle"));
  fsmPresetSocialBtn.addEventListener("click", () => applyFsmPreset("social"));
  fsmPresetSearchBtn.addEventListener("click", () => applyFsmPreset("search"));
  fsmPresetTaskBtn.addEventListener("click", () => applyFsmPreset("task"));
  fsmPresetHomeBtn.addEventListener("click", () => applyFsmPreset("home"));
  fsmPresetRecoverBtn.addEventListener("click", () => applyFsmPreset("recover"));
  fsmPresetFaultBtn.addEventListener("click", () => applyFsmPreset("fault"));
  fsmClearHistoryBtn.addEventListener("click", () => clearFsmHistory());

  scnPrimitiveSelect.addEventListener("change", () => {
    populateScenarioBuilderFromPrimitive();
  });
  scnAddStepBtn.addEventListener("click", () => {
    try {
      addScenarioBuilderStep();
      scenarioSetStatus("Step added to scenario.");
    } catch (err) {
      scenarioSetStatus(`Add step failed: ${err.message || err}`);
    }
  });
  scnTemplateBtn.addEventListener("click", () => {
    const steps = state.scenario.meta?.default_steps || [];
    setScenarioSteps(Array.isArray(steps) ? steps : []);
    scenarioSetStatus("Loaded default scenario template.");
  });
  scnClearBtn.addEventListener("click", () => {
    setScenarioSteps([]);
    scenarioSetStatus("Cleared scenario JSON.");
  });
  scnValidateBtn.addEventListener("click", () => {
    void validateScenario().catch((err) => {
      scenarioSetStatus(`Validation failed: ${err.message || err}`);
    });
  });
  scnRunBtn.addEventListener("click", () => {
    void runScenario().catch((err) => {
      scenarioSetStatus(`Run failed: ${err.message || err}`);
    });
  });
  scnOpenPlaybackBtn.addEventListener("click", () => {
    try {
      const traceUrl = String(state.scenario.lastRun?.trace_url || "").trim();
      if (!traceUrl) {
        throw new Error("run a scenario first to produce a trace");
      }
      openPlaybackTrace(traceUrl);
    } catch (err) {
      scenarioSetStatus(`Open playback failed: ${err.message || err}`);
    }
  });
  scnSaveBtn.addEventListener("click", () => {
    void saveScenarioExperiment().catch((err) => {
      scenarioSetStatus(`Save failed: ${err.message || err}`);
    });
  });
  scnSweepRunBtn.addEventListener("click", () => {
    void runScenarioSweep().catch((err) => {
      scenarioSetSweepStatus(`Sweep failed: ${err.message || err}`);
      showToast(`Scenario sweep failed: ${err}`, "bad", 3600);
    });
  });
  scnSweepTemplateBtn.addEventListener("click", () => {
    try {
      applyScenarioSweepTemplateForTarget();
    } catch (err) {
      scenarioSetSweepStatus(`Template failed: ${err.message || err}`);
      showToast(`Sweep template failed: ${err}`, "bad", 3600);
    }
  });
  scnSweepApplyBestBtn.addEventListener("click", () => {
    try {
      applyScenarioBestSweepPatch();
    } catch (err) {
      scenarioSetSweepStatus(`Apply failed: ${err.message || err}`);
      showToast(`Apply sweep patch failed: ${err}`, "bad", 3600);
    }
  });
  scnSweepPromoteBaselineBtn.addEventListener("click", () => {
    void promoteScenarioBestToBaseline().catch((err) => {
      scenarioSetSweepStatus(`Promote failed: ${err.message || err}`);
      showToast(`Promote baseline failed: ${err}`, "bad", 3600);
    });
  });

  window.addEventListener("keydown", (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) {
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        openPalette();
      }
      return;
    }
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      openPalette();
      return;
    }
    if (ev.key === "Escape" && state.paletteOpen) {
      closePalette();
      return;
    }
    if (ev.key === "1") {
      setMode("studio");
    } else if (ev.key === "2") {
      setMode("scenario_lab");
    } else if (ev.key === "3") {
      setMode("state_machine");
    } else if (ev.key === "4") {
      setMode("joint_checker");
    } else if (ev.key.toLowerCase() === "r") {
      void runSuitePlayback();
    } else if (ev.key.toLowerCase() === "p") {
      runAndPlaybackBtn.click();
    } else if (ev.key.toLowerCase() === "v") {
      setViewerFocus(!state.viewerFocus);
    } else if (ev.key.toLowerCase() === "o") {
      openRawCurrentMode();
    }
  });
}

function initialMode() {
  const queryMode = String(state.params.get("mode") || "").trim();
  if (queryMode && MODES[queryMode]) {
    return normalizeMode(queryMode);
  }
  const persisted = loadPersistedMode();
  if (persisted && MODES[persisted]) {
    return normalizeMode(persisted);
  }
  return "studio";
}

(function bootstrap() {
  state.paletteCommands = buildPaletteCommands();
  state.filteredCommands = state.paletteCommands.slice();
  initNav();
  bindUi();
  setViewerFocus(loadPersistedFocusView(), { persist: false });
  const mode = initialMode();
  setMode(mode);
  showToast("Lamp Sim shell ready", "ok");
})();

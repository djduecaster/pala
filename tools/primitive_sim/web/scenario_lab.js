const configPathEl = document.getElementById("configPath");
const modeLabel = document.getElementById("modeLabel");
const statusBox = document.getElementById("statusBox");
const metricsBox = document.getElementById("metricsBox");
const previewFrame = document.getElementById("previewFrame");
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

const builderPrimitive = document.getElementById("builderPrimitive");
const builderStyle = document.getElementById("builderStyle");
const builderDuration = document.getElementById("builderDuration");
const builderStopOnDone = document.getElementById("builderStopOnDone");
const builderCommand = document.getElementById("builderCommand");

const addStepBtn = document.getElementById("addStepBtn");
const loadTemplateBtn = document.getElementById("loadTemplateBtn");
const clearScenarioBtn = document.getElementById("clearScenarioBtn");
const scenarioJson = document.getElementById("scenarioJson");
const validateBtn = document.getElementById("validateBtn");
const runBtn = document.getElementById("runBtn");

const experimentName = document.getElementById("experimentName");
const experimentNotes = document.getElementById("experimentNotes");
const saveExperimentBtn = document.getElementById("saveExperimentBtn");
const historyTableBody = document.querySelector("#historyTable tbody");

const runSuiteBtn = document.getElementById("runSuiteBtn");
const suiteStatus = document.getElementById("suiteStatus");
const quickTemplateRunBtn = document.getElementById("quickTemplateRunBtn");
const quickAddRunBtn = document.getElementById("quickAddRunBtn");
const quickRunSaveBtn = document.getElementById("quickRunSaveBtn");

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

const state = {
  meta: null,
  lastRun: null,
};

function autoExperimentName() {
  const stamp = new Date().toISOString().replace(/[:-]/g, "").replace(/\.\d+Z$/, "Z");
  return `scenario_${stamp}`;
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
        mode: "scenario_lab",
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

function setSuiteStatus(text, isError = false) {
  suiteStatus.textContent = text;
  suiteStatus.className = isError ? "muted bad" : "muted";
}

function prettyJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
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

function scenarioStepsFromText({ allowEmpty = false } = {}) {
  const raw = String(scenarioJson.value || "").trim();
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
  scenarioJson.value = prettyJson({ steps });
}

function baselineCommandFor(primitiveId) {
  const src = state.meta?.baseline?.primitives?.[primitiveId];
  if (src && typeof src === "object") {
    return src;
  }
  return {};
}

function populateBuilderFromPrimitive() {
  const primitive = String(builderPrimitive.value || "");
  builderDuration.value = String(defaultDurationForPrimitive(primitive));
  builderStopOnDone.checked = primitive !== "breath" && primitive !== "hold";
  builderCommand.value = prettyJson(baselineCommandFor(primitive));
}

function addBuilderStep() {
  const primitive = String(builderPrimitive.value || "").trim();
  if (!primitive) {
    throw new Error("select a primitive before adding a step");
  }
  const style = String(builderStyle.value || "calm");
  const duration = Number(builderDuration.value);
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("duration must be a positive number");
  }
  let command = {};
  try {
    const parsed = JSON.parse(String(builderCommand.value || "{}"));
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
    stop_on_done: Boolean(builderStopOnDone.checked),
    command,
  };
  currentSteps.push(next);
  setScenarioSteps(currentSteps);
}

function addBuilderStepWithStatus() {
  try {
    addBuilderStep();
    statusBox.textContent = "Step added to scenario.";
  } catch (err) {
    statusBox.textContent = `Add step failed: ${err.message || err}`;
  }
}

function formatMetrics(metrics) {
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

function viewerUrlFromTrace(traceUrl) {
  const trace = String(traceUrl || "").trim();
  if (!trace) {
    return "";
  }
  return `/tools/primitive_sim/web/index.html?studio=0&trace=${encodeURIComponent(trace)}`;
}

function renderHistory(rows) {
  historyTableBody.innerHTML = "";
  const data = Array.isArray(rows) ? rows : [];
  data.forEach((row) => {
    const tr = document.createElement("tr");
    const savedTd = document.createElement("td");
    const nameTd = document.createElement("td");
    const samplesTd = document.createElement("td");
    const durationTd = document.createElement("td");
    const traceTd = document.createElement("td");

    const metrics = row?.metrics || {};
    savedTd.textContent = String(row?.saved_at_utc || "");
    nameTd.textContent = String(row?.name || "");
    samplesTd.textContent = String(Number(metrics.sample_count || 0));
    durationTd.textContent = Number(metrics.duration_s || 0).toFixed(2);

    const previewBtn = document.createElement("button");
    previewBtn.type = "button";
    previewBtn.textContent = "Preview";
    const traceUrl = String(row?.trace_url || "");
    if (!traceUrl) {
      previewBtn.disabled = true;
    } else {
      previewBtn.addEventListener("click", () => {
        const viewerUrl = viewerUrlFromTrace(traceUrl);
        if (viewerUrl) {
          previewFrame.src = viewerUrl;
        }
      });
    }
    traceTd.appendChild(previewBtn);

    tr.appendChild(savedTd);
    tr.appendChild(nameTd);
    tr.appendChild(samplesTd);
    tr.appendChild(durationTd);
    tr.appendChild(traceTd);
    historyTableBody.appendChild(tr);
  });
}

async function refreshHistory() {
  const res = await apiGet("/api/scenario/history?limit=25");
  renderHistory(res?.history || []);
}

async function validateScenario() {
  const steps = scenarioStepsFromText();
  const res = await apiPost("/api/scenario/simulate", { dry_run: true, steps });
  statusBox.textContent = `Validation OK: ${Number(res.segment_count || 0)} segments`;
  notifyShell("ok", "Scenario validation passed", `${Number(res.segment_count || 0)} segments`);
}

async function runScenario() {
  runBtn.disabled = true;
  if (quickTemplateRunBtn) {
    quickTemplateRunBtn.disabled = true;
  }
  if (quickAddRunBtn) {
    quickAddRunBtn.disabled = true;
  }
  if (quickRunSaveBtn) {
    quickRunSaveBtn.disabled = true;
  }
  const steps = scenarioStepsFromText();
  const runName = String(experimentName.value || "").trim() || "scenario_run";
  statusBox.textContent = "Running scenario...";
  try {
    const res = await apiPost("/api/scenario/simulate", {
      dry_run: false,
      name: runName,
      steps,
    });
    state.lastRun = res;
    metricsBox.textContent = formatMetrics(res.metrics);
    if (res.trace_url) {
      const viewerUrl = viewerUrlFromTrace(res.trace_url);
      if (viewerUrl) {
        previewFrame.src = viewerUrl;
      }
    }
    statusBox.textContent = [
      `Run complete.`,
      `run_name: ${String(res.run_name || runName)}`,
      `trace_path: ${String(res.trace_path || "")}`,
      `sample_count: ${Number(res.sample_count || 0)}`,
      `viewer_url: ${String(res.viewer_url || "")}`,
    ].join("\n");
    notifyShell("ok", "Scenario run complete", `${Number(res.sample_count || 0)} samples`);
  } finally {
    runBtn.disabled = false;
    if (quickTemplateRunBtn) {
      quickTemplateRunBtn.disabled = false;
    }
    if (quickAddRunBtn) {
      quickAddRunBtn.disabled = false;
    }
    if (quickRunSaveBtn) {
      quickRunSaveBtn.disabled = false;
    }
  }
}

async function saveExperiment({ autoName = false } = {}) {
  if (!state.lastRun) {
    throw new Error("run a scenario before saving an experiment");
  }
  let name = String(experimentName.value || "").trim();
  if (!name && autoName) {
    name = autoExperimentName();
    experimentName.value = name;
  }
  if (!name) {
    throw new Error("experiment name is required");
  }
  saveExperimentBtn.disabled = true;
  if (quickRunSaveBtn) {
    quickRunSaveBtn.disabled = true;
  }
  const notes = String(experimentNotes.value || "").trim();
  const steps = Array.isArray(state.lastRun.steps) ? state.lastRun.steps : scenarioStepsFromText();
  const payload = {
    name,
    notes,
    steps,
    metrics: state.lastRun.metrics || {},
    trace_path: String(state.lastRun.trace_path || ""),
    trace_url: String(state.lastRun.trace_url || ""),
  };
  try {
    await apiPost("/api/scenario/save_experiment", payload);
    statusBox.textContent = `Experiment saved: ${name}`;
    await refreshHistory();
    notifyShell("ok", `Experiment saved (${name})`);
  } finally {
    saveExperimentBtn.disabled = false;
    if (quickRunSaveBtn) {
      quickRunSaveBtn.disabled = false;
    }
  }
}

async function quickTemplateRun() {
  const steps = Array.isArray(state.meta?.default_steps) ? state.meta.default_steps : [];
  if (!steps.length) {
    throw new Error("default template is empty");
  }
  setScenarioSteps(steps);
  await runScenario();
}

async function quickAddStepRun() {
  addBuilderStep();
  await runScenario();
}

async function quickRunSave() {
  await runScenario();
  await saveExperiment({ autoName: true });
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
    notifyShell("bad", "Scenario Lab suite run failed", String(err));
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
  if (cmd === "scenario.validate") {
    validateScenario().catch((err) => {
      statusBox.textContent = `Validation failed: ${err.message || err}`;
      notifyShell("bad", "Scenario validation failed", String(err));
    });
    return;
  }
  if (cmd === "scenario.run") {
    runScenario().catch((err) => {
      statusBox.textContent = `Run failed: ${err.message || err}`;
      notifyShell("bad", "Scenario run failed", String(err));
    });
    return;
  }
  if (cmd === "scenario.save_experiment") {
    saveExperiment().catch((err) => {
      statusBox.textContent = `Save failed: ${err.message || err}`;
      notifyShell("bad", "Scenario save failed", String(err));
    });
  }
});

function bindUi() {
  builderPrimitive.addEventListener("change", () => {
    populateBuilderFromPrimitive();
  });

  addStepBtn.addEventListener("click", () => {
    addBuilderStepWithStatus();
  });

  loadTemplateBtn.addEventListener("click", () => {
    const steps = state.meta?.default_steps || [];
    setScenarioSteps(steps);
    statusBox.textContent = "Loaded default scenario template.";
  });

  clearScenarioBtn.addEventListener("click", () => {
    setScenarioSteps([]);
    statusBox.textContent = "Cleared scenario JSON.";
  });

  validateBtn.addEventListener("click", () => {
    validateScenario().catch((err) => {
      statusBox.textContent = `Validation failed: ${err.message || err}`;
    });
  });

  runBtn.addEventListener("click", () => {
    runScenario().catch((err) => {
      statusBox.textContent = `Run failed: ${err.message || err}`;
    });
  });

  saveExperimentBtn.addEventListener("click", () => {
    saveExperiment().catch((err) => {
      statusBox.textContent = `Save failed: ${err.message || err}`;
    });
  });

  runSuiteBtn.addEventListener("click", () => {
    runSuitePlayback();
  });

  if (quickTemplateRunBtn) {
    quickTemplateRunBtn.addEventListener("click", () => {
      quickTemplateRun().catch((err) => {
        statusBox.textContent = `Template+Run failed: ${err.message || err}`;
      });
    });
  }

  if (quickAddRunBtn) {
    quickAddRunBtn.addEventListener("click", () => {
      quickAddStepRun().catch((err) => {
        statusBox.textContent = `Add+Run failed: ${err.message || err}`;
      });
    });
  }

  if (quickRunSaveBtn) {
    quickRunSaveBtn.addEventListener("click", () => {
      quickRunSave().catch((err) => {
        statusBox.textContent = `Run+Save failed: ${err.message || err}`;
      });
    });
  }

  window.addEventListener("keydown", (ev) => {
    const key = String(ev.key || "").toLowerCase();
    const mod = ev.metaKey || ev.ctrlKey;

    if (mod && key === "enter") {
      ev.preventDefault();
      if (ev.shiftKey) {
        quickRunSave().catch((err) => {
          statusBox.textContent = `Run+Save failed: ${err.message || err}`;
        });
      } else {
        runScenario().catch((err) => {
          statusBox.textContent = `Run failed: ${err.message || err}`;
        });
      }
      return;
    }

    if (mod && ev.shiftKey && key === "v") {
      ev.preventDefault();
      validateScenario().catch((err) => {
        statusBox.textContent = `Validation failed: ${err.message || err}`;
      });
      return;
    }

    if (mod && key === "s") {
      ev.preventDefault();
      saveExperiment({ autoName: true }).catch((err) => {
        statusBox.textContent = `Save failed: ${err.message || err}`;
      });
      return;
    }

    if (ev.altKey && key === "enter") {
      ev.preventDefault();
      addBuilderStepWithStatus();
      return;
    }
  });
}

async function init() {
  bindUi();
  try {
    const meta = await apiGet("/api/scenario/meta");
    state.meta = meta;
    configPathEl.textContent = `Config: ${String(meta.config_path || "(unknown)")}`;

    builderPrimitive.innerHTML = "";
    const primitives = Array.isArray(meta.primitives) ? meta.primitives : [];
    primitives.forEach((item) => {
      const option = document.createElement("option");
      option.value = String(item.id || "");
      option.textContent = String(item.label || item.id || "");
      builderPrimitive.appendChild(option);
    });

    builderStyle.innerHTML = "";
    const styles = Array.isArray(meta.styles) ? meta.styles : [];
    styles.forEach((style) => {
      const option = document.createElement("option");
      option.value = String(style);
      option.textContent = String(style);
      builderStyle.appendChild(option);
    });
    if (styles.includes("calm")) {
      builderStyle.value = "calm";
    }

    populateBuilderFromPrimitive();
    setScenarioSteps(Array.isArray(meta.default_steps) ? meta.default_steps : []);
    metricsBox.textContent = "Metrics: run a scenario";
    statusBox.textContent = "Scenario Lab ready.";
    modeLabel.textContent = "Mode: scenario composition";
    await refreshHistory();
  } catch (err) {
    configPathEl.textContent = "Config: unavailable";
    statusBox.textContent = [
      "Scenario Lab failed to load metadata.",
      "Run with:",
      "uv run python tools/primitive_sim/run.py --scenario scenario_lab --port 8766",
      "",
      `error: ${String(err)}`,
    ].join("\n");
  }
}

init();

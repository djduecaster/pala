(function () {
  "use strict";

  const STORAGE_KEY = "probe_web_v4_form_state";

  const state = {
    files: [],
    order: [],
    runStatusTimer: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setError(message) {
    const el = byId("image-error");
    if (!el) {
      return;
    }
    if (!message) {
      el.textContent = "";
      el.classList.add("hidden");
      return;
    }
    el.textContent = message;
    el.classList.remove("hidden");
  }

  function syncOrderField() {
    const hidden = byId("image-order");
    if (!hidden) {
      return;
    }
    hidden.value = state.order.join(",");
  }

  function setRunStatus(text, kind) {
    const el = byId("run-status");
    if (!el) {
      return;
    }
    el.textContent = text;
    el.classList.remove("status-running", "status-ok", "status-error");
    if (kind === "running") {
      el.classList.add("status-running");
    } else if (kind === "ok") {
      el.classList.add("status-ok");
    } else if (kind === "error") {
      el.classList.add("status-error");
    }
  }

  function setRunSubmitDisabled(disabled) {
    const btn = byId("run-submit");
    if (!btn) {
      return;
    }
    btn.disabled = !!disabled;
    if (disabled) {
      btn.textContent = "Running...";
      return;
    }
    btn.textContent = "Run Behavior V4 Probe";
  }

  function clearRunStatusTimer() {
    if (state.runStatusTimer) {
      window.clearTimeout(state.runStatusTimer);
      state.runStatusTimer = null;
    }
  }

  function renderPreview() {
    const root = byId("image-preview");
    if (!root) {
      return;
    }
    root.innerHTML = "";
    if (state.files.length === 0) {
      return;
    }

    state.order.forEach(function (fileIdx, orderPos) {
      const file = state.files[fileIdx];
      if (!file) {
        return;
      }
      const card = document.createElement("article");
      card.className = "preview-card";

      const img = document.createElement("img");
      img.alt = "upload preview";
      img.loading = "lazy";
      img.src = URL.createObjectURL(file);

      const meta = document.createElement("div");
      meta.className = "preview-meta";
      meta.innerHTML =
        "<strong>#" +
        String(orderPos + 1) +
        "</strong> " +
        file.name +
        "<br /><small>" +
        String(file.size) +
        " bytes</small>";

      const controls = document.createElement("div");
      controls.className = "preview-controls";

      const up = document.createElement("button");
      up.type = "button";
      up.textContent = "Up";
      up.disabled = orderPos === 0;
      up.addEventListener("click", function () {
        const prev = state.order[orderPos - 1];
        state.order[orderPos - 1] = state.order[orderPos];
        state.order[orderPos] = prev;
        syncOrderField();
        renderPreview();
      });

      const down = document.createElement("button");
      down.type = "button";
      down.textContent = "Down";
      down.disabled = orderPos === state.order.length - 1;
      down.addEventListener("click", function () {
        const next = state.order[orderPos + 1];
        state.order[orderPos + 1] = state.order[orderPos];
        state.order[orderPos] = next;
        syncOrderField();
        renderPreview();
      });

      controls.appendChild(up);
      controls.appendChild(down);

      card.appendChild(img);
      card.appendChild(meta);
      card.appendChild(controls);
      root.appendChild(card);
    });
  }

  function onFileInputChanged(ev) {
    const files = Array.from(ev.target.files || []);
    state.files = files;
    state.order = files.map(function (_file, idx) {
      return idx;
    });
    syncOrderField();
    renderPreview();

    if (files.length !== 4) {
      setError("Select exactly 4 images.");
    } else {
      setError("");
    }
  }

  function validateBeforeSubmit(ev) {
    if (state.files.length !== 4) {
      ev.preventDefault();
      setError("Exactly 4 images are required before running probe.");
      setRunStatus("Cannot run: upload exactly 4 images.", "error");
      return;
    }
    setError("");
  }

  function setResultView(scope, kind) {
    const parsed = byId(scope + "-result-view-parsed");
    const raw = byId(scope + "-result-view-raw");
    if (!parsed || !raw) {
      return;
    }
    if (kind === "raw") {
      parsed.classList.add("hidden");
      raw.classList.remove("hidden");
      return;
    }
    raw.classList.add("hidden");
    parsed.classList.remove("hidden");
  }

  function initHelpOverlay() {
    const helpOpen = byId("help-open");
    const helpClose = byId("help-close");
    const helpOverlay = byId("help-overlay");
    if (helpOpen && helpOverlay) {
      helpOpen.addEventListener("click", function () {
        helpOverlay.classList.remove("hidden");
      });
    }
    if (helpClose && helpOverlay) {
      helpClose.addEventListener("click", function () {
        helpOverlay.classList.add("hidden");
      });
    }
    if (helpOverlay) {
      helpOverlay.addEventListener("click", function (ev) {
        if (ev.target === helpOverlay) {
          helpOverlay.classList.add("hidden");
        }
      });
    }
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && helpOverlay) {
        helpOverlay.classList.add("hidden");
      }
    });
  }

  function parseJsonField(el, label, format) {
    if (!el) {
      return null;
    }
    el.classList.remove("json-invalid");
    const raw = String(el.value || "").trim();
    if (!raw) {
      return null;
    }
    try {
      const parsed = JSON.parse(raw);
      if (Object.prototype.toString.call(parsed) !== "[object Object]") {
        throw new Error("must be a JSON object");
      }
      if (format) {
        el.value = JSON.stringify(parsed, null, 2);
      }
      return null;
    } catch (err) {
      el.classList.add("json-invalid");
      return label + ": " + (err && err.message ? err.message : "invalid JSON");
    }
  }

  function setJsonStatus(message, isError) {
    const status = byId("json-editor-status");
    if (!status) {
      return;
    }
    status.textContent = message;
    status.classList.remove("status-ok", "status-error");
    if (isError) {
      status.classList.add("status-error");
      return;
    }
    status.classList.add("status-ok");
  }

  function runJsonValidation(format) {
    const ctx = byId("context-override-json");
    const payload = byId("payload-override-json");
    const errors = [];
    const ctxErr = parseJsonField(ctx, "Context Override JSON", format);
    if (ctxErr) {
      errors.push(ctxErr);
    }
    const payloadErr = parseJsonField(payload, "Payload Override JSON", format);
    if (payloadErr) {
      errors.push(payloadErr);
    }

    if (errors.length) {
      setJsonStatus(errors.join(" | "), true);
      return false;
    }
    if (format) {
      setJsonStatus("Override JSON formatted and valid.", false);
    } else {
      setJsonStatus("Override JSON is valid.", false);
    }
    return true;
  }

  function debounce(fn, waitMs) {
    let timer = null;
    return function () {
      const args = arguments;
      if (timer) {
        window.clearTimeout(timer);
      }
      timer = window.setTimeout(function () {
        fn.apply(null, args);
      }, waitMs);
    };
  }

  function initJsonTools() {
    const validateBtn = byId("json-validate-btn");
    const formatBtn = byId("json-format-btn");
    if (validateBtn) {
      validateBtn.onclick = function () {
        runJsonValidation(false);
      };
    }
    if (formatBtn) {
      formatBtn.onclick = function () {
        runJsonValidation(true);
      };
    }

    const debounced = debounce(function () {
      runJsonValidation(false);
    }, 450);
    [byId("context-override-json"), byId("payload-override-json")].forEach(function (el) {
      if (!el) {
        return;
      }
      el.oninput = debounced;
    });
  }

  function saveFormState() {
    const form = byId("probe-form");
    if (!form) {
      return;
    }
    const payload = {};
    const fields = form.querySelectorAll("input[name], textarea[name], select[name]");
    fields.forEach(function (el) {
      const tag = String(el.tagName || "").toLowerCase();
      const type = String(el.getAttribute("type") || "").toLowerCase();
      const name = el.getAttribute("name");
      if (!name) {
        return;
      }
      if (type === "file") {
        return;
      }
      if (type === "checkbox") {
        payload[name] = !!el.checked;
        return;
      }
      if (tag === "select" || tag === "textarea" || type === "hidden" || type === "text" || type === "number") {
        payload[name] = el.value;
      }
    });
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (_err) {
      return;
    }
  }

  function restoreFormState() {
    const form = byId("probe-form");
    if (!form) {
      return;
    }
    let stored = null;
    try {
      stored = window.localStorage.getItem(STORAGE_KEY);
    } catch (_err) {
      return;
    }
    if (!stored) {
      return;
    }
    let parsed = null;
    try {
      parsed = JSON.parse(stored);
    } catch (_err2) {
      return;
    }
    if (!parsed || Object.prototype.toString.call(parsed) !== "[object Object]") {
      return;
    }
    Object.keys(parsed).forEach(function (key) {
      const el = form.querySelector('[name="' + key + '"]');
      if (!el) {
        return;
      }
      const type = String(el.getAttribute("type") || "").toLowerCase();
      if (type === "file") {
        return;
      }
      if (type === "checkbox") {
        el.checked = !!parsed[key];
        return;
      }
      if (typeof parsed[key] === "string" || typeof parsed[key] === "number") {
        el.value = String(parsed[key]);
      }
    });
  }

  function initPersistence() {
    const form = byId("probe-form");
    if (!form) {
      return;
    }
    restoreFormState();
    form.addEventListener("input", saveFormState);
    form.addEventListener("change", saveFormState);
  }

  function initCompareTools() {
    const btn = byId("compare-latest-btn");
    if (!btn) {
      return;
    }
    btn.onclick = function () {
      const a = byId("compare-run-a");
      const b = byId("compare-run-b");
      if (!a || !b) {
        return;
      }
      const optionsA = Array.from(a.options).map(function (opt) {
        return opt.value;
      }).filter(Boolean);
      if (optionsA.length < 2) {
        return;
      }
      a.value = optionsA[0];
      b.value = optionsA[1];
    };
  }

  function init() {
    const imageInput = byId("images");
    if (imageInput) {
      imageInput.addEventListener("change", onFileInputChanged);
    }

    const form = byId("probe-form");
    if (form) {
      form.addEventListener("submit", validateBeforeSubmit);
    }

    initHelpOverlay();
    initJsonTools();
    initPersistence();
    initCompareTools();

    document.body.addEventListener("htmx:beforeRequest", function (evt) {
      const elt = evt && evt.detail ? evt.detail.elt : null;
      if (!elt || elt.id !== "probe-form") {
        return;
      }
      clearRunStatusTimer();
      setRunSubmitDisabled(true);
      setRunStatus("Running: uploading packet and calling model...", "running");
      state.runStatusTimer = window.setTimeout(function () {
        setRunStatus("Running: parsing response and evaluating guard...", "running");
      }, 1000);
    });

    document.body.addEventListener("htmx:afterRequest", function (evt) {
      const elt = evt && evt.detail ? evt.detail.elt : null;
      if (!elt || elt.id !== "probe-form") {
        return;
      }
      clearRunStatusTimer();
      setRunSubmitDisabled(false);
      const ok = evt && evt.detail && evt.detail.xhr && evt.detail.xhr.status >= 200 && evt.detail.xhr.status < 300;
      if (ok) {
        setRunStatus("Run complete.", "ok");
      } else {
        setRunStatus("Run failed. Check inputs and try again.", "error");
      }
    });

    document.body.addEventListener("htmx:responseError", function (evt) {
      const elt = evt && evt.detail ? evt.detail.elt : null;
      if (!elt || elt.id !== "probe-form") {
        return;
      }
      clearRunStatusTimer();
      setRunSubmitDisabled(false);
      setRunStatus("Request failed before completion.", "error");
    });

    document.body.addEventListener("htmx:afterSwap", function (evt) {
      if (!evt || !evt.target) {
        return;
      }
      if (evt.target.id === "result-panel") {
        setResultView("v4", "parsed");
      }
      if (evt.target.id === "override-fields" || evt.target.id === "preset-panel" || evt.target.id === "compare-form-panel") {
        initJsonTools();
        initCompareTools();
      }
    });
  }

  window.ProbeUI = {
    setResultView: setResultView,
  };

  document.addEventListener("DOMContentLoaded", init);
})();

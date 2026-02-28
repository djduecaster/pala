(function () {
  "use strict";

  const state = {
    files: [],
    order: [],
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
    const runActionEl = byId("run-action");
    const runAction = runActionEl ? String(runActionEl.value || "").trim() : "";
    if (runAction === "run_planner_only") {
      const prepared = byId("prepared-env-run-id");
      const preparedId = prepared ? String(prepared.value || "").trim() : "";
      if (!preparedId) {
        ev.preventDefault();
        setError("Run env preview first to prepare planner inputs.");
        return;
      }
      setError("");
      return;
    }

    if (state.files.length !== 4) {
      ev.preventDefault();
      setError("Exactly 4 images are required before running probe.");
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

  function setProbeMode(mode) {
    const probeMode = mode === "env_planner" ? "env_planner" : "env";
    const hidden = byId("probe-mode");
    if (hidden) {
      hidden.value = probeMode;
    }

    const form = byId("probe-form");
    if (form) {
      form.setAttribute("hx-post", probeMode === "env_planner" ? "/probe/env-planner" : "/probe/env");
    }
    const runAction = byId("run-action");
    if (runAction) {
      runAction.value = probeMode === "env_planner" ? "run_both" : "run_env";
    }

    const plannerPanel = byId("planner-fields");
    if (plannerPanel) {
      if (probeMode === "env_planner") {
        plannerPanel.classList.remove("hidden");
      } else {
        plannerPanel.classList.add("hidden");
      }
    }

    const plannerActions = byId("planner-action-row");
    if (plannerActions) {
      if (probeMode === "env_planner") {
        plannerActions.classList.remove("hidden");
      } else {
        plannerActions.classList.add("hidden");
      }
    }
    const preparedEnvRow = byId("prepared-env-row");
    if (preparedEnvRow) {
      if (probeMode === "env_planner") {
        preparedEnvRow.classList.remove("hidden");
      } else {
        preparedEnvRow.classList.add("hidden");
      }
    }
    const envActions = byId("env-action-row");
    if (envActions) {
      if (probeMode === "env_planner") {
        envActions.classList.add("hidden");
      } else {
        envActions.classList.remove("hidden");
      }
    }

    const submit = byId("run-submit");
    if (submit) {
      submit.textContent = "Run Env Probe";
    }

    const tabs = document.querySelectorAll(".mode-tab[data-probe-mode]");
    tabs.forEach(function (tab) {
      const tabMode = tab.getAttribute("data-probe-mode");
      if (tabMode === probeMode) {
        tab.classList.add("is-active");
      } else {
        tab.classList.remove("is-active");
      }
    });
  }

  function initModeTabs() {
    const tabs = document.querySelectorAll(".mode-tab[data-probe-mode]");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        const mode = tab.getAttribute("data-probe-mode");
        setProbeMode(mode || "env");
      });
    });

    const hidden = byId("probe-mode");
    setProbeMode(hidden ? hidden.value : "env");
  }

  function initRunActionButtons() {
    const buttons = document.querySelectorAll(".run-action-btn[data-run-action]");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        const action = btn.getAttribute("data-run-action");
        const runAction = byId("run-action");
        if (runAction && action) {
          runAction.value = action;
        }
      });
    });
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

    initModeTabs();
    initRunActionButtons();
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

    document.body.addEventListener("htmx:afterSwap", function (evt) {
      if (evt && evt.target && evt.target.id === "result-panel") {
        setResultView("env", "parsed");
        setResultView("planner", "parsed");
        initRunActionButtons();
      }
    });
  }

  window.ProbeUI = {
    setResultView: setResultView,
    setProbeMode: setProbeMode,
  };

  document.addEventListener("DOMContentLoaded", init);
})();

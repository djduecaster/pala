(() => {
  const frame = document.getElementById("modeFrame");
  const nav = document.getElementById("modeNav");
  const params = new URLSearchParams(location.search);
  const trace = params.get("trace") || "/logs/primitive_sim/latest_trace.json";
  const modes = {
    studio: { label: "Studio", src: "/tools/primitive_sim/web/index.html?studio=1&shell=1" },
    joint_checker: { label: "Joint Checker", src: "/tools/primitive_sim/web/joint_checker.html?shell=1" },
    playback: { label: "Playback", src: `/tools/primitive_sim/web/index.html?studio=0&shell=1&trace=${encodeURIComponent(trace)}` },
  };
  const requested = new URLSearchParams(location.search).get("mode");
  let active = modes[requested] ? requested : "studio";
  function setMode(mode) {
    if (!modes[mode]) mode = "studio";
    active = mode;
    frame.src = modes[mode].src;
    nav.replaceChildren();
    Object.entries(modes).forEach(([key, value]) => {
      const link = document.createElement("a");
      link.className = "nav-btn";
      link.href = `?mode=${key}`;
      link.textContent = value.label;
      if (key === mode) link.setAttribute("aria-current", "page");
      link.onclick = (event) => { event.preventDefault(); setMode(key); };
      nav.appendChild(link);
    });
    document.getElementById("modeTitle").textContent = modes[mode].label;
    document.getElementById("modeBadge").textContent = `Mode: ${mode}`;
  }
  setMode(active);
})();

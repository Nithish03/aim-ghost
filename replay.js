// aimghost — Rung 3: ghost replay.
// Loads a session JSON and redraws the cursor path as a fading trail over the
// original target positions, in real time at 1x.

"use strict";

const Replay = (() => {
  const TRAIL_MS = 600;   // how far back the fading trail reaches
  const END_HOLD_MS = 500; // linger after the last sample before "finished"

  let data = null;
  let active = false;
  let startPerf = 0; // performance.now() when playback started
  let firstT = 0;    // t_ms of the first trajectory sample
  let lastT = 0;
  let finished = false;
  let rafId = null;

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const backBtn = document.getElementById("back-to-game");
  const endBtn = document.getElementById("end-session");
  const fileInput = document.getElementById("replay-file");

  // Uniform scale from the recorded screen to the current window, centered.
  function transform() {
    const s = Math.min(window.innerWidth / data.screen.w, window.innerHeight / data.screen.h);
    return {
      s,
      ox: (window.innerWidth - data.screen.w * s) / 2,
      oy: (window.innerHeight - data.screen.h * s) / 2,
    };
  }

  function start(json) {
    if (!Array.isArray(json.trajectory) || json.trajectory.length === 0 ||
        !Array.isArray(json.targets) || !json.screen) {
      alert("Not a valid aimghost session file.");
      return;
    }
    data = json;
    firstT = data.trajectory[0][0];
    lastT = data.trajectory[data.trajectory.length - 1][0];
    finished = false;
    active = true;
    startPerf = performance.now();
    Recorder.abort();
    backBtn.hidden = false;
    endBtn.hidden = true;
    rafId = requestAnimationFrame(draw);
  }

  function stop() {
    active = false;
    data = null;
    if (rafId !== null) cancelAnimationFrame(rafId);
    backBtn.hidden = true;
    endBtn.hidden = false;
    startSession(); // back to a fresh game + recording (defined in game.js)
  }

  function draw() {
    const t = firstT + (performance.now() - startPerf); // replay-clock time, 1x
    const { s, ox, oy } = transform();
    const traj = data.trajectory;

    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

    // Recorded-screen bounds, so it's obvious where the original screen was.
    ctx.strokeStyle = "#333";
    ctx.strokeRect(ox, oy, data.screen.w * s, data.screen.h * s);

    // Targets alive at time t, drawn like the live game but dimmer.
    for (const tg of data.targets) {
      if (tg.spawn_t <= t && t <= tg.end_t) {
        ctx.beginPath();
        ctx.arc(ox + tg.x * s, oy + tg.y * s, tg.r * s, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(229, 72, 77, 0.55)";
        ctx.fill();
      }
    }

    // Fading cursor trail: samples in [t - TRAIL_MS, t].
    let head = null;
    for (let i = 0; i < traj.length; i++) {
      const [st, sx, sy, buttons] = traj[i];
      if (st > t) break;
      if (st < t - TRAIL_MS) continue;
      const age = (t - st) / TRAIL_MS; // 0 = now, 1 = oldest
      const x = ox + sx * s;
      const y = oy + sy * s;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(120, 200, 255, ${(1 - age) * 0.9})`;
      ctx.fill();
      if (buttons & 1) { // click marker
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255, 255, 120, ${1 - age})`;
        ctx.stroke();
      }
      head = { x, y };
    }

    // Ghost cursor head.
    if (head) {
      ctx.beginPath();
      ctx.arc(head.x, head.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
    }

    // Progress + status line.
    ctx.font = "14px monospace";
    ctx.fillStyle = "#888";
    const elapsed = Math.min(t - firstT, lastT - firstT) / 1000;
    const total = (lastT - firstT) / 1000;
    ctx.fillText(
      `replay ${elapsed.toFixed(1)}s / ${total.toFixed(1)}s` + (finished ? " — finished" : ""),
      10, window.innerHeight - 12
    );

    if (t > lastT + END_HOLD_MS) finished = true;
    if (active) rafId = requestAnimationFrame(draw);
  }

  // --- UI wiring ---
  document.getElementById("load-replay").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    fileInput.value = ""; // allow re-loading the same file
    if (!file) return;
    file.text().then((text) => {
      let json;
      try {
        json = JSON.parse(text);
      } catch {
        alert("Not valid JSON.");
        return;
      }
      start(json);
    });
  });
  backBtn.addEventListener("click", stop);

  return {
    get active() { return active; },
  };
})();

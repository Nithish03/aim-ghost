// aimghost — Rung 2: data capture.
// Records mouse trajectory + target outcomes per the locked schema in CLAUDE.md.
// No throttling, no smoothing; coalesced events are unpacked so no samples drop.

"use strict";

const Recorder = (() => {
  let session = null; // { session_id, started_at, t0, targets, trajectory }
  let refreshHintHz = null;

  // Estimate display refresh rate from rAF deltas (median of ~60 frames).
  (function measureRefresh() {
    const deltas = [];
    let last = null;
    function tick(ts) {
      if (last !== null) deltas.push(ts - last);
      last = ts;
      if (deltas.length < 60) {
        requestAnimationFrame(tick);
      } else {
        deltas.sort((a, b) => a - b);
        refreshHintHz = Math.round(1000 / deltas[Math.floor(deltas.length / 2)]);
        if (session) session.refresh_hint_hz = refreshHintHz;
      }
    }
    requestAnimationFrame(tick);
  })();

  function now() {
    return performance.now() - session.t0;
  }

  function pushSample(t, x, y, buttons) {
    // Button events can share a timestamp with a move sample (timer
    // resolution ties). Nudge by 1µs to keep t_ms strictly increasing while
    // preserving event order and every sample — positions are never altered.
    const traj = session.trajectory;
    if (traj.length > 0 && t <= traj[traj.length - 1][0]) {
      t = traj[traj.length - 1][0] + 0.001;
    }
    traj.push([t, x, y, buttons]);
  }

  // Movement: pointerrawupdate where available, else pointermove (same event
  // stream as mousemove for a mouse, but exposes getCoalescedEvents so
  // coalesced samples can be unpacked). Event timestamps share the
  // performance.now() clock, so t = e.timeStamp - t0.
  const moveEvent = "onpointerrawupdate" in window ? "pointerrawupdate" : "pointermove";

  function onMove(e) {
    if (!session) return;
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    if (events.length === 0) events.push(e);
    for (const ce of events) {
      pushSample(ce.timeStamp - session.t0, ce.clientX, ce.clientY, ce.buttons);
    }
  }

  // Button transitions won't appear in the move stream if the mouse is still,
  // so record an explicit sample on every press/release.
  function onButton(e) {
    if (!session) return;
    pushSample(e.timeStamp - session.t0, e.clientX, e.clientY, e.buttons);
  }

  window.addEventListener(moveEvent, onMove);
  window.addEventListener("pointerdown", onButton);
  window.addEventListener("pointerup", onButton);

  function start() {
    session = {
      session_id: crypto.randomUUID(),
      started_at: new Date().toISOString(),
      t0: performance.now(),
      screen: {
        w: window.innerWidth,
        h: window.innerHeight,
        dpr: window.devicePixelRatio || 1,
      },
      refresh_hint_hz: refreshHintHz, // may still be null; filled when measured
      targets: [],
      trajectory: [],
    };
  }

  function onTargetSpawn(target) {
    if (!session) return;
    target.record = {
      target_id: target.id,
      spawn_t: now(),
      x: target.x,
      y: target.y,
      r: target.r,
      outcome: null,
      end_t: null,
    };
  }

  function onTargetEnd(target, outcome) {
    if (!session || !target.record) return;
    target.record.outcome = outcome;
    target.record.end_t = now();
    session.targets.push(target.record);
  }

  // Returns { ok, violations, sampleRateHz, samples, durationS }
  function validate() {
    const traj = session.trajectory;
    let violations = 0;
    for (let i = 1; i < traj.length; i++) {
      if (traj[i][0] <= traj[i - 1][0]) violations++;
    }
    let sampleRateHz = 0;
    let durationS = 0;
    if (traj.length >= 2) {
      durationS = (traj[traj.length - 1][0] - traj[0][0]) / 1000;
      sampleRateHz = (traj.length - 1) / durationS;
    }
    return {
      ok: violations === 0,
      violations,
      sampleRateHz,
      samples: traj.length,
      durationS,
    };
  }

  function end() {
    const v = validate();
    const out = {
      session_id: session.session_id,
      started_at: session.started_at,
      screen: session.screen,
      refresh_hint_hz: session.refresh_hint_hz,
      targets: session.targets,
      trajectory: session.trajectory,
    };

    const msg =
      `session ${out.session_id}\n` +
      `samples: ${v.samples} over ${v.durationS.toFixed(1)} s ` +
      `(~${v.sampleRateHz.toFixed(0)} Hz)\n` +
      (v.ok
        ? "time-travel check: OK (t_ms strictly increasing)"
        : `time-travel check: FAILED — ${v.violations} non-increasing step(s)`);
    console.log("[aimghost recorder]\n" + msg);
    alert(msg);

    const blob = new Blob([JSON.stringify(out)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `aimghost-${out.started_at.replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);

    session = null;
  }

  return { start, end, onTargetSpawn, onTargetEnd };
})();

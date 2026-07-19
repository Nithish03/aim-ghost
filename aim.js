// aimghost — aim input pipeline.
// Single owner of pointer events. In normal mode the aim point is the OS
// cursor. With Aim Lock (pointer lock) the page captures the mouse and moves
// a virtual crosshair by raw deltas * sens — Valorant-style sensitivity.
// Everything downstream (recorder, game hit-tests, duel) uses Aim's
// coordinates, so recorded data and gameplay stay consistent in both modes.

"use strict";

const Aim = (() => {
  let sens = 1.0;
  let locked = false;
  let vx = window.innerWidth / 2;
  let vy = window.innerHeight / 2;
  const listeners = []; // fn(timeStamp, x, y, buttons), one call per sample

  const canvas = document.getElementById("game");
  const moveEvent = "onpointerrawupdate" in window ? "pointerrawupdate" : "pointermove";

  function apply(ce) {
    if (locked) {
      vx = Math.max(0, Math.min(window.innerWidth, vx + ce.movementX * sens));
      vy = Math.max(0, Math.min(window.innerHeight, vy + ce.movementY * sens));
    } else {
      vx = ce.clientX;
      vy = ce.clientY;
    }
    for (const fn of listeners) fn(ce.timeStamp, vx, vy, ce.buttons);
  }

  window.addEventListener(moveEvent, (e) => {
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    if (events.length === 0) events.push(e);
    for (const ce of events) apply(ce);
  });
  // Button transitions won't appear in the move stream if the mouse is still.
  window.addEventListener("pointerdown", apply);
  window.addEventListener("pointerup", apply);

  document.addEventListener("pointerlockchange", () => {
    locked = document.pointerLockElement === canvas;
    if (locked) {
      // Start the crosshair where the cursor was, mid-screen fallback.
      vx = Math.max(0, Math.min(window.innerWidth, vx));
      vy = Math.max(0, Math.min(window.innerHeight, vy));
    }
  });

  return {
    get x() { return vx; },
    get y() { return vy; },
    get locked() { return locked; },
    get sens() { return sens; },
    set sens(v) { if (v > 0 && isFinite(v)) sens = v; },
    addListener(fn) { listeners.push(fn); },
  };
})();

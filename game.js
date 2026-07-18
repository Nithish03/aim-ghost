// aimghost — Rung 1: game shell.
// One circular target at a time; click destroys it and spawns the next.
// No data recording yet (that's Rung 2).

"use strict";

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const hudHits = document.getElementById("hud-hits");
const hudMisses = document.getElementById("hud-misses");
const hudAcc = document.getElementById("hud-acc");
const hudRt = document.getElementById("hud-rt");

const TARGET_RADIUS = 28; // px (CSS pixels)
const SPAWN_MARGIN = 12;  // keep the full circle on-screen

// --- state ---
let target = null; // { id, x, y, r, spawnT }
let nextTargetId = 0;
let hits = 0;
let misses = 0;
let lastReactionMs = null;
let reactionTimes = []; // reaction times of the last <=10 hits, for the summary log

// --- canvas sizing (DPR-aware; draw in CSS pixel coordinates) ---
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // If a resize pushed the target off-screen, clamp it back into view.
  if (target) {
    target.x = Math.min(target.x, window.innerWidth - target.r - SPAWN_MARGIN);
    target.y = Math.min(target.y, window.innerHeight - target.r - SPAWN_MARGIN);
  }
}
window.addEventListener("resize", resize);
resize();

// --- targets ---
function spawnTarget() {
  const r = TARGET_RADIUS;
  const minX = r + SPAWN_MARGIN;
  const maxX = window.innerWidth - r - SPAWN_MARGIN;
  const minY = r + SPAWN_MARGIN;
  const maxY = window.innerHeight - r - SPAWN_MARGIN;
  target = {
    id: nextTargetId++,
    x: minX + Math.random() * (maxX - minX),
    y: minY + Math.random() * (maxY - minY),
    r,
    spawnT: performance.now(),
  };
  Recorder.onTargetSpawn(target);
}

// --- input ---
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !target) return;
  const dx = e.clientX - target.x;
  const dy = e.clientY - target.y;
  if (dx * dx + dy * dy <= target.r * target.r) {
    hits++;
    lastReactionMs = performance.now() - target.spawnT;
    reactionTimes.push(lastReactionMs);
    Recorder.onTargetEnd(target, "hit");
    spawnTarget();
    if (hits % 10 === 0) logSummary();
  } else {
    misses++;
  }
  updateHud();
});

// --- HUD ---
function updateHud() {
  hudHits.textContent = hits;
  hudMisses.textContent = misses;
  const total = hits + misses;
  hudAcc.textContent = total ? ((hits / total) * 100).toFixed(1) + "%" : "–";
  hudRt.textContent = lastReactionMs !== null ? lastReactionMs.toFixed(0) + " ms" : "–";
}

function logSummary() {
  const total = hits + misses;
  const avgRt = reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length;
  console.log(
    `[aimghost] ${hits} hits, ${misses} misses, ` +
    `accuracy ${((hits / total) * 100).toFixed(1)}%, ` +
    `avg reaction (last ${reactionTimes.length} hits) ${avgRt.toFixed(0)} ms`
  );
  reactionTimes = [];
}

// --- render loop ---
function draw() {
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  if (target) {
    ctx.beginPath();
    ctx.arc(target.x, target.y, target.r, 0, Math.PI * 2);
    ctx.fillStyle = "#e5484d";
    ctx.fill();
    // center dot as a precise aim reference
    ctx.beginPath();
    ctx.arc(target.x, target.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
  }
  requestAnimationFrame(draw);
}

// --- session control ---
function startSession() {
  hits = 0;
  misses = 0;
  lastReactionMs = null;
  reactionTimes = [];
  Recorder.start();
  spawnTarget();
  updateHud();
}

document.getElementById("end-session").addEventListener("click", () => {
  Recorder.end(); // validates, reports, downloads JSON
  startSession(); // immediately begin a fresh session
});

startSession();
requestAnimationFrame(draw);

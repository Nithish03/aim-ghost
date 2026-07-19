"""aimghost dataset preprocessing.

Turns recorded sessions into supervised examples for behavioral cloning:
    features = [last K cursor deltas, target position relative to cursor]
    label    = next cursor delta
on a uniform DT_MS grid, in normalized units.

Python 3 + NumPy only.
"""

import json
from pathlib import Path

import numpy as np

DT_MS = 10.0     # resampling timestep
HISTORY = 5      # K: how many past deltas the model sees
POS_SCALE = 1000.0  # px; divisor for the relative-target features


def load_sessions(path):
    """Accepts a merged dataset.json ({"sessions": [...]}) or one session file."""
    data = json.loads(Path(path).read_text())
    return data["sessions"] if "sessions" in data else [data]


def segments(session):
    """One segment per hit target: resampled cursor path from spawn to hit.

    Returns dicts with:
        pos    (N, 2) cursor positions on the uniform grid, px
        target (x, y, r)
    """
    traj = np.array(session["trajectory"], dtype=np.float64)
    t, x, y = traj[:, 0], traj[:, 1], traj[:, 2]
    out = []
    for tg in session["targets"]:
        if tg["outcome"] != "hit":
            continue
        t0, t1 = tg["spawn_t"], tg["end_t"]
        if t1 - t0 < 3 * DT_MS:
            continue  # too short to say anything about movement
        grid = np.arange(t0, t1, DT_MS)
        # np.interp extrapolates flat at the edges, which is exactly right:
        # before the first in-window sample the cursor sat where it last was.
        px = np.interp(grid, t, x)
        py = np.interp(grid, t, y)
        pos = np.stack([px, py], axis=1)

        # Trim the reaction-time idle prefix (cursor at rest while the human
        # perceives the target). Left in, "do nothing" dominates the dataset
        # and the rolled-out model learns to freeze. Keep 2 rest ticks so
        # "starting from rest" is still represented.
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        moving = np.nonzero(step > 0.5)[0]
        if len(moving) == 0:
            continue
        start = max(0, moving[0] - 2)
        pos = pos[start:]
        if len(pos) < 4:
            continue
        out.append({
            "pos": pos,
            "target": (tg["x"], tg["y"], tg["r"]),
            # The trimmed idle prefix IS the user's reaction time for this
            # target — kept so the ghost can be given the same delay.
            "reaction_ms": start * DT_MS,
        })
    return out


def reaction_stats(segs):
    """Median/std of the user's reaction time, for a human-like ghost delay."""
    r = np.array([s["reaction_ms"] for s in segs])
    return {"reaction_ms_median": float(np.median(r)),
            "reaction_ms_std": float(r.std())}


def build_examples(all_segments, history=HISTORY, norm=None):
    """Feature/label arrays from segments, in a TARGET-CENTRIC frame.

    Every example is rotated so the unit vector cursor->target is +x. The
    model therefore learns the user's approach profile (speed curve, lateral
    wobble) independent of direction, and at rollout the frame is recomputed
    each tick, so aiming error self-corrects instead of compounding into
    orbits. Features: [K rotated past deltas, distance]. Label: rotated next
    delta.

    History slots before a segment's start are zero-padded — matching the bot
    at rollout time, which also starts from rest with an empty history.
    Delta normalization (d_scale) is computed from the data unless an existing
    `norm` is passed — validation/test data must reuse the training transform.
    """
    if norm is not None:
        history = norm["history"]
    feats, labels = [], []
    for seg in all_segments:
        pos = seg["pos"]
        tx, ty, _ = seg["target"]
        deltas = np.diff(pos, axis=0)          # (N-1, 2), px per tick
        n = len(deltas)
        for i in range(n):
            rx, ry = tx - pos[i, 0], ty - pos[i, 1]
            dist = np.hypot(rx, ry)
            if dist < 1e-6:
                continue
            ux, uy = rx / dist, ry / dist
            # Rotation to target frame: [[ux, uy], [-uy, ux]] @ d
            def rot(d):
                return np.array([ux * d[0] + uy * d[1], -uy * d[0] + ux * d[1]])

            hist = np.zeros((history, 2))
            k = min(i, history)
            if k:
                hist[history - k:] = [rot(d) for d in deltas[i - k:i]]
            feats.append(np.concatenate([hist.ravel(), [dist]]))
            labels.append(rot(deltas[i]))
    feats = np.array(feats)
    labels = np.array(labels)

    if norm is None:
        d_scale = float(labels.std()) or 1.0
        norm = {"d_scale": d_scale, "p_scale": POS_SCALE,
                "history": history, "dt_ms": DT_MS, "frame": "target"}
    feats[:, : 2 * history] /= norm["d_scale"]
    feats[:, 2 * history:] /= norm["p_scale"]
    labels = labels / norm["d_scale"]
    return feats, labels, norm


def train_val_split(sessions, val_fraction=0.2):
    """Split at segment level, holding out the LAST segments of play.

    (With one session we can't hold out whole sessions yet; the tail of the
    session is the least-leaky alternative since examples are correlated in
    time. When more sessions exist, switch to holding out whole sessions.)
    """
    segs = []
    for s in sessions:
        segs.extend(segments(s))
    n_val = max(1, int(len(segs) * val_fraction))
    return segs[:-n_val], segs[-n_val:]

#!/usr/bin/env python3
"""aimghost dataset tool: validate session JSONs and merge them into one dataset.

Usage:
    python3 tools/merge_sessions.py <file-or-dir> [more files/dirs...] [-o dataset.json]

Each input is validated against the schema in CLAUDE.md (required fields,
t_ms strictly increasing, targets consistent). Valid sessions are merged into:

    { "merged_at": "<ISO8601>", "sessions": [ <session>, ... ] }

Sessions are kept separate — trajectories from different sessions have
different time bases and must never be concatenated into one stream.
Stdlib only; no dependencies.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_KEYS = {"session_id", "started_at", "screen", "targets", "trajectory"}
OUTCOMES = {"hit", "miss", "timeout"}


def validate(data, name):
    """Return (errors, stats). Empty errors list means the session is valid."""
    errors = []
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return [f"missing keys: {sorted(missing)}"], {}

    traj = data["trajectory"]
    targets = data["targets"]

    if not traj:
        errors.append("empty trajectory")
    for i, s in enumerate(traj):
        if not (isinstance(s, list) and len(s) == 4):
            errors.append(f"trajectory[{i}] is not a [t, x, y, buttons] quad")
            break

    violations = sum(
        1 for i in range(1, len(traj)) if traj[i][0] <= traj[i - 1][0]
    )
    if violations:
        errors.append(f"t_ms not strictly increasing ({violations} violations)")

    w, h = data["screen"].get("w"), data["screen"].get("h")
    oob = sum(1 for s in traj if s[1] < 0 or s[2] < 0 or s[1] > w or s[2] > h)
    if oob:
        errors.append(f"{oob} trajectory samples outside screen bounds")

    for t in targets:
        if t.get("outcome") not in OUTCOMES:
            errors.append(f"target {t.get('target_id')}: bad outcome {t.get('outcome')!r}")
        elif t["end_t"] < t["spawn_t"]:
            errors.append(f"target {t.get('target_id')}: end_t before spawn_t")

    stats = {}
    if len(traj) >= 2:
        dur = (traj[-1][0] - traj[0][0]) / 1000
        stats = {
            "samples": len(traj),
            "duration_s": round(dur, 1),
            "rate_hz": round((len(traj) - 1) / dur) if dur > 0 else 0,
            "targets": len(targets),
            "hits": sum(1 for t in targets if t["outcome"] == "hit"),
        }
    return errors, stats


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {p} not found, skipping", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="session JSON files or directories")
    ap.add_argument("-o", "--output", default="dataset.json", help="merged output file")
    args = ap.parse_args()

    sessions = []
    seen_ids = set()
    failed = 0
    for f in collect(args.inputs):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"FAIL  {f.name}: unreadable ({e})")
            failed += 1
            continue
        if data.get("session_id") in seen_ids:
            print(f"SKIP  {f.name}: duplicate session_id {data['session_id']}")
            continue
        errors, stats = validate(data, f.name)
        if errors:
            print(f"FAIL  {f.name}:")
            for e in errors:
                print(f"        - {e}")
            failed += 1
            continue
        seen_ids.add(data["session_id"])
        sessions.append(data)
        print(
            f"OK    {f.name}: {stats['samples']} samples, {stats['duration_s']}s "
            f"(~{stats['rate_hz']} Hz), {stats['hits']}/{stats['targets']} targets hit"
        )

    if not sessions:
        print("no valid sessions; nothing written", file=sys.stderr)
        sys.exit(1)

    out = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "sessions": sessions,
    }
    Path(args.output).write_text(json.dumps(out))
    total = sum(len(s["trajectory"]) for s in sessions)
    print(
        f"\nwrote {args.output}: {len(sessions)} session(s), "
        f"{total} trajectory samples total ({failed} file(s) rejected)"
    )


if __name__ == "__main__":
    main()

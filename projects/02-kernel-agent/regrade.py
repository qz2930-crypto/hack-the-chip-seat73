#!/usr/bin/env python3
"""
regrade.py -- validity check: re-grade every logged attempt from its own code, compare to the log.

Until 12:50 agent.grade() wrote each candidate to /tmp/_agent_level{N}.py, a path shared by every
process. runq runs several jobs of the SAME level concurrently, so one job could in principle load
another job's file between write and import and log the other kernel's score. Grading is
deterministic, so re-grading each logged `code` in isolation must reproduce the logged reward and
feedback. Any mismatch = a contaminated attempt.

    python regrade.py runs/skel_*.jsonl          (in the pod, where nki exists)
"""

import glob
import json
import sys

import agent


def main(paths):
    rows, cache = [], {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "claim" not in r and "code" in r:
                rows.append((p, r))
    bad = 0
    for p, r in rows:
        k = (r["level"], r["code"])
        if k not in cache:
            cache[k] = agent.grade(r["code"], r["level"])
        reward, _, feedback = cache[k]
        if abs(reward - r["reward"]) > 1e-9 or feedback != r["feedback"]:
            bad += 1
            print(f"MISMATCH {p} L{r['level']} run {r.get('run')} round {r['round']}: logged "
                  f"{r['reward']:.2f}, regraded {reward:.2f}")
    print(f"regraded {len(rows)} attempts ({len(cache)} distinct kernels): {bad} mismatches")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main([q for a in sys.argv[1:] for q in glob.glob(a)]) else 0)

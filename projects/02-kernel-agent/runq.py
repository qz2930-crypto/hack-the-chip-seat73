#!/usr/bin/env python3
"""
runq.py -- keep the model server's slots busy with a queue of agent runs.

The local server runs at most 4 sequences at once (--max-num-seqs 4), and one agent process at
--samples 1 holds one slot. Measured on seat 73: 4 concurrent requests give ~1.6x the throughput of 1,
so 4 independent jobs are the cheapest way to buy repeats. This runner keeps MAX agent.py processes
alive, launching the next queued job whenever one finishes.

    runs/queue.txt     one job per line:  <tag> <agent args...>     ('#' comments allowed)
    runs/started.txt   tags already launched (appended by this runner)
    runs/<tag>.log / runs/<tag>.jsonl   each job's output

    setsid nohup python runq.py > runs/runq.log 2>&1 < /dev/null &
Append jobs to runs/queue.txt at any time; the runner picks them up.
"""

import os
import subprocess
import sys
import time

MAX = int(os.environ.get("RUNQ_MAX", "4"))
os.makedirs("runs", exist_ok=True)


def running():
    out = subprocess.run(["pgrep", "-f", "python -u agent[.]py"], capture_output=True, text=True)
    return len([l for l in out.stdout.split() if l.strip()])


def jobs():
    try:
        lines = open("runs/queue.txt").read().splitlines()
    except FileNotFoundError:
        return []
    return [l.split(None, 1) for l in lines if l.strip() and not l.lstrip().startswith("#")]


def started():
    try:
        return set(open("runs/started.txt").read().split())
    except FileNotFoundError:
        return set()


idle = 0
while True:
    pending = [j for j in jobs() if j[0] not in started()]
    if pending and running() < MAX:
        tag, args = pending[0][0], (pending[0][1] if len(pending[0]) > 1 else "")
        cmd = (f"python -u agent.py --tag {tag} --log runs/{tag}.jsonl {args} "
               f"> runs/{tag}.log 2>&1 < /dev/null")
        subprocess.Popen(cmd, shell=True, start_new_session=True)
        with open("runs/started.txt", "a") as f:
            f.write(tag + "\n")
        print(time.strftime("%H:%M:%S"), "started", tag, args, flush=True)
        time.sleep(3)
        continue
    idle = idle + 1 if not pending and running() == 0 else 0
    if idle > 360:          # an hour with nothing queued and nothing running
        print("idle, exiting", flush=True)
        sys.exit(0)
    time.sleep(10)

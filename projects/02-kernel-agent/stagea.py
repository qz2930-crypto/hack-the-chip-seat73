#!/usr/bin/env python3
"""
stagea.py -- the Stage-A kernel agent: NumPy kernels under kernel-shaped rules (kernelbench.py),
driven by a model, checked by the verifier, repaired by ONE named change per round.

The CHALLENGE doc's measured lesson is the design:
  * the FIRST prompt carries no rules list (a rule-carrying prompt made gpt-oss audit itself and
    return nothing); the verifier owns the rules;
  * every failure is TRANSLATED into one instruction (`instruct()`): "line 16: calls banned max"
    fed back verbatim reproduced the violation; "replace np.max with an explicit loop ..." fixed it;
  * feedback is directional: the harness's "expected X, got Y" line is stripped, never sent.

Runs on the laptop (numpy only). Endpoints, all OpenAI-compatible chat:

    --endpoint trn   $GPTOSS_BASE_URL/agg/v1, model gpt-oss-20b   (organisers' Trainium endpoint)
    --endpoint hf    https://router.huggingface.co/v1, openai/gpt-oss-20b:groq, HF token from
                     $HF_TOKEN or ~/.cache/huggingface/token (never stored in this repo)
    --endpoint url   --base URL --model NAME (e.g. a port-forwarded local Qwen3-8B)

    python stagea.py --endpoint hf --levels 1-4 --rounds 6 --repeat 3 --tag A_hf
    python stagea.py --offline --levels 1-3           # no model: replays planted bugs then fixes

Every attempt -> runs/<tag>.jsonl (same schema as agent.py, so scripts/results.py, taxonomy.py and
token_report.py work unchanged). Responses are cached on disk by (endpoint, model, prompt) because
the shared endpoints are rate-limited and greedy: a repeated prompt would return the same text.
"""

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time

import numpy as np

import kernelbench as kb

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)
FULL = sum(WEIGHTS.values())
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

# What each level's banned call should become. Upstream's measured fix for level 5 was exactly this
# shape: name the call, say what loop replaces it, "keep everything else identical".
LOOP_FOR = {
    "sum": "a running sum over the columns of each tile, starting from 0.0",
    "nansum": "a running sum over the columns of each tile, starting from 0.0",
    "cumsum": "a running sum over the columns of each tile",
    "max": "a running maximum over the columns of each tile, starting from -np.inf (not 0)",
    "amax": "a running maximum over the columns of each tile, starting from -np.inf (not 0)",
    "mean": "a running sum over the columns of each tile, divided by the row length at the end",
    "average": "a running sum over the columns of each tile, divided by the row length at the end",
    "var": "a second pass: sum of (x - mean)**2 over the columns, divided by the row length",
    "std": "the square root of the two-pass variance (sum of (x - mean)**2 / row length)",
    "norm": "a running sum of squares over the columns of each tile",
    "matmul": "tile loops over the rows of a and the columns of b, and inside them a loop over "
              "k that accumulates acc += a[r0:r1, k:k+1] * b[k:k+1, c0:c1] into a float32 tile",
    "dot": "a loop over k accumulating acc += a[r0:r1, k:k+1] * b[k:k+1, c0:c1]",
    "transpose": "a tile loop that copies out[c0:c1, r0:r1] from x[r0:r1, c0:c1] element-row by row",
    "einsum": "explicit loops over tiles",
    "softmax": "max-subtract, exp, running sum, divide -- each as a loop over tiles",
    "logsumexp": "max-subtract, exp, running sum, log -- each as a loop over tiles",
}


# ---------------------------------------------------------------- verdict -> instruction

def instruct(kind, detail, level):
    """(bucket, instruction). The one change the model should make next."""
    if kind == "empty":
        return "no_code", "Reply with one ```python block that defines `kernel`."
    if kind == "syntax":
        return "syntax_error", f"The code does not parse ({detail}). Send one complete ```python block."
    if kind == "nokernel":
        args = ", ".join(inspect.signature(kb.LEVELS[level]["ref"]).parameters)
        return "rule_wrong_entry_name", (f"Name the function exactly `kernel` with this signature: "
                                         f"def kernel({args}):")
    if kind == "rules":
        v = detail[0]
        m = re.search(r"(?:calls banned|method) `\.?(\w+)", v)
        if m:
            name = m.group(1).split(".")[-1]
            loop = LOOP_FOR.get(name, "explicit python loops over the tiles")
            return "rule_framework_call", (f"{v}. Replace the `{name}` call with {loop}. "
                                           f"Keep everything else identical.")
        if "fancy/boolean" in v:
            return "rule_fancy_indexing", (f"{v}. Use only slices like x[r0:r1, c0:c1]; for a "
                                           f"condition use np.where on the whole tile, never a mask "
                                           f"inside the brackets. Keep everything else identical.")
        return "rule_other", f"{v}. Fix exactly that. Keep everything else identical."
    if kind == "raised":
        if "IndexError" in detail or "out of bounds" in detail:
            return "out_of_bounds", (f"It crashed: {detail}. Clamp every tile end to the array: "
                                     f"r1 = min(r0 + 128, R), c1 = min(c0 + 512, C).")
        if "could not be broadcast" in detail or "shapes" in detail:
            return "assign_shape_mismatch", (f"It crashed: {detail}. The slice you write into and "
                                             f"the value must have the same shape -- size the "
                                             f"destination slice with the clamped tile ends.")
        if "overflow" in detail.lower():
            return "overflow", (f"It crashed: {detail}. Subtract the row maximum before np.exp.")
        return "raised_other", f"It crashed: {detail}. Fix that line."
    if kind == "numeric":
        # strip the target value (non-negotiable), keep the tagged localisation lines
        lines = [l.strip() for l in detail.splitlines()
                 if l.strip() and not l.strip().startswith("expected ")]
        tags = " ".join(lines)
        if "RAGGED-EDGE" in tags:
            b = "ragged_edge"
        elif "STRIDE-" in tags:
            b = "stride_bug"
        elif re.search(r"PATTERN (FIRST|LAST)-(ROW|COL)", tags):
            b = "off_by_one"
        elif "DIRECTION ZERO" in tags:
            b = "output_not_written"
        elif "DIRECTION SCALED" in tags:
            b = "wrong_scale"
        elif "SIGN-FLIPPED" in tags:
            b = "sign_flipped"
        elif "SHIFTED" in tags:
            b = "index_shift"
        elif "NON-FINITE" in tags or "nan" in tags.lower() or "inf" in tags.lower():
            b = "non_finite_output"
        elif "WRONG SHAPE" in tags or "shape" in lines[0].lower() and "returned" in lines[0]:
            b = "wrong_output_shape"
        else:
            b = "wrong_arithmetic"
        return b, ("The numbers are wrong. " + " ".join(lines[1:] or lines)[:900]
                   + " Change only what that points at; keep everything else identical.")
    return "other", detail


def grade(src, level):
    """(reward, parts, bucket, feedback, pass_fraction)."""
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    if not src.strip():
        b, f = instruct("empty", "", level)
        return 0.0, parts, b, f, 0.0
    try:
        compile(src, "<kernel>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        b, f = instruct("syntax", f"{e.msg} on line {e.lineno}", level)
        return 0.0, parts, b, f, 0.0
    viol = kb.check_rules(src, level)
    if viol:
        b, f = instruct("rules", viol, level)
        return WEIGHTS["parses"], parts, b, f, 0.0
    parts["rules"] = True
    ns = {}
    try:
        exec(compile(src, "<kernel>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        b, f = instruct("raised", f"{type(e).__name__}: {e}", level)
        return WEIGHTS["parses"] + WEIGHTS["rules"], parts, b, f, 0.0
    if "kernel" not in ns:
        b, f = instruct("nokernel", "", level)
        return WEIGHTS["parses"] + WEIGHTS["rules"], parts, b, f, 0.0
    res = kb.verify(ns["kernel"], level, stop_early=False)
    frac = res["passed"] / res["total"]
    base = WEIGHTS["parses"] + WEIGHTS["rules"]
    if res["ok"]:
        parts.update(runs=True, correct=True)
        return FULL, parts, "solved", f"VERIFIED: {res['total']}/{res['total']} cases.", 1.0
    lbl, msg = res["failures"][0]
    ran = not msg.startswith("RAISED")
    parts["runs"] = ran or res["passed"] > 0
    reward = base + (WEIGHTS["runs"] if parts["runs"] else 0) + WEIGHTS["correct"] * frac
    if msg.startswith("RAISED"):
        b, f = instruct("raised", msg[len("RAISED: "):], level)
    else:
        b, f = instruct("numeric", msg, level)
    return reward, parts, b, f"{res['passed']} of {res['total']} cases pass. On {lbl}: {f}", frac


# ---------------------------------------------------------------- prompts

def first_prompt(level, terse=0):
    ref = inspect.getsource(kb.LEVELS[level]["ref"])
    if terse:
        return (f"Write a numpy function `kernel` computing the same as:\n\n{ref}\n"
                f"Work on tiles of at most 128 rows x 512 columns. One ```python block.")
    return (f"Write a numpy function `kernel` matching this reference:\n\n{ref}\n"
            f"Loop over tiles of at most 128 rows and 512 columns; the last tile in each direction "
            f"may be smaller. Output only one ```python block.")


def repair_prompt(code, feedback, ledger=""):
    p = (f"This function is wrong:\n\n```python\n{code.strip()}\n```\n\n{feedback}\n"
         f"Output only one ```python block.")
    if ledger:
        p += f"\n\nAlready tried, and failed -- do something different:\n{ledger}"
    return p


# ---------------------------------------------------------------- model

class Client:
    def __init__(self, a):
        self.a, self.calls = a, 0
        self.cache_path = os.path.join(a.cache_dir, "stagea_cache.jsonl")
        os.makedirs(a.cache_dir, exist_ok=True)
        self.cache = {}
        if os.path.exists(self.cache_path):
            for line in open(self.cache_path, encoding="utf-8"):
                r = json.loads(line)
                self.cache[r["k"]] = r["v"]
        if a.endpoint == "trn":
            base = os.environ.get("GPTOSS_BASE_URL", "").rstrip("/")
            if not base:
                sys.exit("--endpoint trn needs GPTOSS_BASE_URL (the organisers post it)")
            self.url, self.model, self.headers = base + "/agg/v1", a.model or "gpt-oss-20b", {}
        elif a.endpoint == "hf":
            tok = os.environ.get("HF_TOKEN") or _read(os.path.expanduser("~/.cache/huggingface/token"))
            if not tok:
                sys.exit("--endpoint hf needs a HuggingFace token: run `hf auth login` in your own "
                         "terminal (it is stored in ~/.cache/huggingface/token, outside this repo)")
            self.url = "https://router.huggingface.co/v1"
            self.model = a.model or "openai/gpt-oss-20b:groq"
            self.headers = {"Authorization": f"Bearer {tok.strip()}"}
        else:
            self.url, self.model, self.headers = a.base.rstrip("/"), a.model, {}

    def ask(self, prompt):
        k = hashlib.sha256(f"{self.url}|{self.model}|{self.a.max_tokens}|{prompt}".encode()).hexdigest()
        if k in self.cache and not self.a.no_cache:
            v = dict(self.cache[k])
            v["cached"] = True
            return v
        if self.calls >= self.a.max_calls:
            sys.exit(f"call budget of {self.a.max_calls} reached (--max-calls); stopping cleanly")
        import httpx
        body = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.a.max_tokens, temperature=0)
        if self.a.endpoint == "url":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        for attempt in range(5):
            try:
                r = httpx.post(f"{self.url}/chat/completions", json=body, headers=self.headers,
                               timeout=600, verify=False)
            except httpx.HTTPError as e:
                time.sleep(5 * (attempt + 1))
                last = str(e)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            if r.status_code != 200:
                sys.exit(f"endpoint returned HTTP {r.status_code}: {r.text[:500]}")
            break
        else:
            sys.exit(f"endpoint failed 5 times: {last}")
        self.calls += 1
        p = r.json()
        ch = p["choices"][0]
        msg = ch.get("message", {})
        v = dict(content=msg.get("content") or "", finish=ch.get("finish_reason"),
                 usage=p.get("usage") or {},
                 reasoning_chars=len(msg.get("reasoning") or msg.get("reasoning_content") or ""))
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(k=k, v=v)) + "\n")
        self.cache[k] = v
        return dict(v, cached=False)


def _read(p):
    try:
        return open(p).read()
    except OSError:
        return ""


# ---------------------------------------------------------------- offline replay

_OFFLINE = {
    1: ["import numpy as np\ndef kernel(x, a, b):\n    return np.einsum('ij->ij', np.maximum(a*x+b, 0))\n",
        "import numpy as np\ndef kernel(x, a, b):\n    R, C = x.shape\n    out = np.empty_like(x)\n"
        "    for r0 in range(0, R, 128):\n        for c0 in range(0, C, 512):\n"
        "            r1, c1 = min(r0+128, R), min(c0+512, C)\n"
        "            out[r0:r1, c0:c1] = np.maximum(a*x[r0:r1, c0:c1]+b, 0.0)\n    return out\n"],
}


# ---------------------------------------------------------------- confidence (same rules as heldout.py)

def confidence(src):
    c, why = 0.95, []
    if not re.search(r"\bmin\(", src):
        c *= 0.5
        why.append("no min() clamp on tile ends")
    if re.search(r"\b(128|512)\b", src) is None:
        c *= 0.8
        why.append("tile size not explicit")
    return round(c, 3), why or ["clamped tile ends; every public case verified"]


# ---------------------------------------------------------------- loop

def solve(a, cli, level, log, run):
    spec = kb.LEVELS[level]
    print(f"\n=========== A{level}: {spec['name']}  (trap: {spec['trap']}) ===========")
    terse = a.terse
    prompt, kind = first_prompt(level, terse), "first"
    latest, tried, seen, best = ("", ""), [], {}, 0.0
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        if a.offline:
            reps = _OFFLINE.get(level, [""])
            v = dict(content=f"```python\n{reps[min(rnd, len(reps) - 1)]}```", finish="offline",
                     usage={}, reasoning_chars=0, cached=False)
        else:
            v = cli.ask(prompt)
        m = CODE_BLOCK.findall(v["content"])
        src = max(m, key=len).strip() if m else ""
        reward, parts, bucket, fb, frac = grade(src, level)
        best = max(best, reward)
        sec = {"task": 0, "code": 0, "error": 0, "ledger": 0}
        if kind == "first":
            sec["task"] = len(prompt) // 4
        else:
            sec["code"] = len(latest[0]) // 4
            sec["error"] = len(latest[1]) // 4
            led = prompt.split("Already tried", 1)
            sec["ledger"] = len(led[1]) // 4 if len(led) > 1 else 0
            sec["task"] = max(0, len(prompt) // 4 - sec["code"] - sec["error"] - sec["ledger"])
        log.write(json.dumps(dict(tag=a.tag, run=run, level=level, round=rnd, sample=0,
                                  ts=time.time(), reward=reward, parts=parts, bucket=bucket,
                                  prompt_kind=dict(kind=kind), prompt_sections=sec,
                                  prompt_chars=len(prompt), reply_chars=len(v["content"]),
                                  finish=v["finish"], usage=v["usage"], cached=v.get("cached"),
                                  reasoning_chars=v.get("reasoning_chars"), prompt=prompt,
                                  code=src, feedback=fb, stage="A")) + "\n")
        log.flush()
        print(f"round {rnd}: {reward:.2f} [{bucket}] ({time.perf_counter() - t0:.1f}s"
              f"{', cached' if v.get('cached') else ''}, finish={v['finish']})")
        print(f"  {fb[:300]}")
        if reward >= FULL - 1e-9:
            conf, why = confidence(src)
            log.write(json.dumps(dict(tag=a.tag, run=run, level=level, claim="solved",
                                      confidence=conf, why=why, stage="A", ts=time.time())) + "\n")
            print(f"  SOLVED on round {rnd}, confidence {conf:.2f} ({'; '.join(why)})")
            return reward, rnd + 1
        seen[fb] = seen.get(fb, 0) + 1
        if seen[fb] >= a.give_up_after:
            print(f"  COULD NOT VERIFY A{level} (confidence 0.00): the same failure {seen[fb]}x")
            break
        if src:
            latest = (src, fb)
        tried.append(f"{bucket}: {fb[:140]}")
        if not src:
            terse = min(terse + 1, 1)
            prompt, kind = first_prompt(level, terse), "first"
            continue
        ledger = "\n".join(f"- {t}" for t in dict.fromkeys(tried[:-1])) if seen[fb] >= 2 else ""
        prompt, kind = repair_prompt(latest[0], latest[1], ledger), "repair"
    log.write(json.dumps(dict(tag=a.tag, run=run, level=level, claim="not_verified",
                              confidence=0.0, best=best, stage="A", ts=time.time())) + "\n")
    return best, a.rounds


def parse_levels(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", choices=("trn", "hf", "url"), default="hf")
    ap.add_argument("--base", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--levels", default="1-4")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--give-up-after", type=int, default=3)
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1))
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="gpt-oss writes to a hidden reasoning channel first; <2500 returns empty")
    ap.add_argument("--max-calls", type=int, default=300, help="hard budget of uncached calls")
    ap.add_argument("--no-cache", action="store_true",
                    help="needed for --repeat on a sampling endpoint; a greedy one returns the same")
    ap.add_argument("--cache-dir", default="runs")
    ap.add_argument("--tag", default="A")
    ap.add_argument("--log", default="")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    os.makedirs("runs", exist_ok=True)
    cli = None if a.offline else Client(a)
    levels = parse_levels(a.levels)
    hist = {lv: [] for lv in levels}
    with open(a.log or f"runs/{a.tag}.jsonl", "a", encoding="utf-8") as log:
        for run in range(a.repeat):
            for lv in levels:
                r, n = solve(a, cli, lv, log, run)
                hist[lv].append(r)
    print("\n=========== summary ===========")
    for lv in levels:
        got = hist[lv]
        k = sum(1 for r in got if r >= FULL - 1e-9)
        print(f"  A{lv}: solved {k}/{len(got)}, mean {sum(got) / len(got):.2f}, "
              f"all={[round(r, 2) for r in got]}")
    if cli:
        print(f"  uncached model calls this session: {cli.calls}")


if __name__ == "__main__":
    main()

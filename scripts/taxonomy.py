#!/usr/bin/env python3
"""
taxonomy.py -- the failure taxonomy, generated from the attempt logs. FAILURES.md is its output.

    python scripts/taxonomy.py runs/*/*.jsonl > FAILURES.md

Every attempt is classified by projects/02-kernel-agent/verdicts.classify(feedback) -- the same
function the agent uses to choose its doc retrieval -- so the taxonomy and the agent cannot drift.
Counts are per ATTEMPT (one model reply), split by level and by the run tag (= model + config).
Identical replies within a round (the local server decodes deterministically) are counted once,
so a bucket is not inflated 4x by duplicate samples.
"""

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "projects", "02-kernel-agent"))
import verdicts  # noqa: E402

# bucket -> (one-sentence definition, the instruction that fixes it)
DEFS = {
    "no_code": ("the reply contained no code block at all",
                "re-ask with a shorter prompt; never ask it to 'fix' empty code"),
    "syntax_error": ("the code block does not parse",
                     "send one complete python block; check the answer was not truncated"),
    "rule_wrong_entry_name": ("the entry point has the wrong name",
                              "mechanical: rename by AST (--mech), no model call needed"),
    "rule_missing_jit": ("the entry point lacks @nki.jit", "mechanical: add it by AST (--mech)"),
    "rule_partition_over_128": ("a tile literally allocated with partition dim > 128",
                                "loop the partition dim in chunks of min(128, remaining)"),
    "rule_framework_call": ("hands the whole op to numpy/torch or uses @/.T on an input",
                            "replace the framework call with the nl/nisa primitive"),
    "rule_other": ("another static rule violation", "fix exactly the rule named"),
    "invented_import": ("imports a module that does not exist (e.g. nki.nl)",
                        "mechanical: the only imports are nki, nki.language as nl, nki.isa as nisa"),
    "load_error": ("module imports but the entry point cannot be loaded",
                   "read the loader error; usually a top-level statement that raises"),
    "invented_api": ("calls an nl/nisa name that does not exist (nl.dot, nl.value ...)",
                     "list the closest real names and let it choose"),
    "invented_kwarg": ("passes a keyword argument the real function does not take",
                       "remove it and show the real signature"),
    "call_memory_region": ("calls nl.sbuf(...) as if it were a function",
                           "allocate with nl.ndarray(shape, dtype, buffer=nl.sbuf)"),
    "tile_1d": ("allocates a 1-D SBUF/PSUM tile", "give it a partition dim: (P, 1) or (1, N)"),
    "reshape_instead_of_slice": ("reshapes a tensor instead of slicing tiles out of it",
                                 "slice: a[k0:k0+k, m0:m0+m]; never reshape across partitions"),
    "partition_over_128": ("one tile for the whole tensor; partition dim > 128 at run time",
                           "remainder-safe loop over the partition dim, tile sized inside the loop"),
    "contraction_over_128": ("contracts K > 128 in one nc_matmul",
                             "split K, accumulate in ONE psum tile allocated outside the K loop"),
    "contraction_mismatch": ("nc_matmul operands disagree on K -- often nc_matmul used for a "
                             "non-matmul op", "same first dim for both operands, or do not use "
                             "nc_matmul for data movement"),
    "dma_shape_mismatch": ("dma_copy src and dst hold different numbers of elements",
                           "allocate the tile with exactly the slice's shape"),
    "assign_shape_mismatch": ("assigns a value into a slice of a different shape",
                              "index the destination to match the value"),
    "out_of_bounds": ("indexes past the end of a tensor (fixed tile size on a smaller dim)",
                      "derive bounds from the shape: min(limit, size)"),
    "wrong_buffer": ("an operand lives in the wrong buffer (e.g. matmul dst not in psum)",
                     "dst in nl.psum; stationary and moving in nl.sbuf"),
    "engine_cannot_reach_hbm": ("tensor_copy/compute engine writes straight to HBM",
                                "psum->sbuf with tensor_copy, then sbuf->hbm with dma_copy"),
    "arg_bound_twice": ("an argument passed both positionally and by keyword",
                        "pass every argument by keyword"),
    "python_operator_on_tile": ("uses + / += on tiles", "use nisa.tensor_tensor / tensor_scalar"),
    "numpy_method_on_tile": ("calls a numpy method (.mean, .sum) on a tile",
                             "use nisa.tensor_reduce then tensor_scalar"),
    "raised_other": ("the simulator raised an error with no translation yet",
                     "add a row to verdicts.RULES"),
    "wrong_output_shape": ("output has the wrong shape", "check the output-size arithmetic"),
    "non_finite_output": ("NaN/Inf in the output -- usually an uninitialised tile was read",
                          "write every tile before reading it"),
    "output_not_written": ("output mostly zeros: results never reached HBM",
                           "copy every tile psum->sbuf->hbm, at the right indices"),
    "partial_coverage": ("a block of the output was never written",
                         "check loop bounds and destination indices"),
    "ragged_edge": ("wrong only in the final partial tile", "handle the partial tile's size"),
    "wrong_arithmetic": ("numerically wrong across the output",
                         "operand layout or the core arithmetic, not an edge case"),
    "traffic_over_bar": ("correct but moves more HBM bytes than the level allows",
                         "hoist/ block loads so each byte crosses the bus fewer times"),
    "hardware_hazard": ("correct on CPU but the simulator warns it is wrong on hardware",
                        "fix the hazard named in the warning"),
    "mutates_input": ("writes into its input tensor", "allocate a fresh output in shared_hbm"),
    "cannot_simulate": ("the SDK was missing -- an environment failure, not the model's",
                        "run inside the seat pod"),
    "solved": ("correct on every shape (and under the traffic bar)", "--"),
    "other": ("unclassified feedback", "extend verdicts.classify"),
}


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_file"] = p
                rows.append(r)
    return rows


def dedupe(rows):
    """One row per distinct reply per (file, tag, run, level, round)."""
    seen, out = set(), []
    for r in rows:
        k = (r["_file"], r.get("tag"), r.get("run"), r["level"], r["round"], r.get("code", ""))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def snippet(code, n=10):
    lines = [l for l in (code or "").splitlines() if l.strip()]
    return "\n".join(lines[:n])


def main(paths):
    rows = dedupe(load(paths))
    for r in rows:
        r["_bucket"] = verdicts.classify(r.get("feedback", ""))
    total = len(rows)
    by_bucket = collections.Counter(r["_bucket"] for r in rows)
    by_bl = collections.Counter((r["_bucket"], r["level"]) for r in rows)
    by_bt = collections.Counter((r["_bucket"], r.get("tag") or "?") for r in rows)
    levels = sorted({r["level"] for r in rows})
    tags = sorted({r.get("tag") or "?" for r in rows})
    example = {}
    for r in rows:
        example.setdefault(r["_bucket"], r)

    print("# FAILURES — the failure taxonomy\n")
    print(f"Generated by `scripts/taxonomy.py` from {len(paths)} attempt log(s): "
          f"**{total} distinct attempts** (identical replies within a round counted once). "
          f"Every attempt is bucketed by `verdicts.classify()`, the same function the agent uses. "
          f"All results **[sim]**.\n")
    print("## Counts by level\n")
    print("| bucket | " + " | ".join(f"L{l}" for l in levels) + " | total | share |")
    print("|---|" + "---|" * (len(levels) + 2))
    for b, n in by_bucket.most_common():
        print(f"| `{b}` | " + " | ".join(str(by_bl.get((b, l), 0) or "") for l in levels)
              + f" | **{n}** | {n / total:.0%} |")
    print("\n## Counts by run tag (model + config)\n")
    print("| bucket | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for b, _ in by_bucket.most_common():
        print(f"| `{b}` | " + " | ".join(str(by_bt.get((b, t), 0) or "") for t in tags) + " |")
    print("\n## Buckets\n")
    for b, n in by_bucket.most_common():
        d, fix = DEFS.get(b, ("(undefined bucket)", "?"))
        r = example[b]
        print(f"### `{b}` — {n} attempts\n")
        print(f"**Definition:** {d}.  \n**Instruction that fixes it:** {fix}.  ")
        print(f"**Representative** (level {r['level']}, tag `{r.get('tag')}`, run {r.get('run')}, "
              f"round {r['round']}):\n")
        print("```python\n" + snippet(r.get("code")) + "\n```\n")
        print("Checker said: _" + (r.get("feedback") or "")[:300].replace("\n", " ")
              .replace("_", "\\_") + "_\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ps = [p for a in sys.argv[1:] for p in glob.glob(a)]
    if not ps:
        sys.exit("usage: taxonomy.py runs/*/*.jsonl")
    main(ps)

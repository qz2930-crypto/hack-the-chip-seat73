# LOG — seat 73, Hack the Chip (all times ET)

Every decision, run and dead end, written as it happens.

## Phase 0 — stand up

- **11:38** Start. 561 min to 21:00 ET, 501 min to the 20:00 freeze.
- **11:39** The laptop had no `aws` or `gh`, so both were installed via winget. Each of Claude's shell calls is a fresh process, so the AWS credentials live in an `~/.aws` profile (`hack`), outside any repo. `scripts/pod.sh` and `~/bin/k73` set `AWS_PROFILE=hack` and put aws on PATH, because kubectl's exec plugin needs it.
- **11:41** `sts get-caller-identity` → assumed-role EKSExec. `kubectl get pod seat-73` → `1/1 Running`, age 14h. Pod: trn2.48xlarge node, 1 Neuron device, 4 cores, 96 GB; env `KERNEL_AGENT_BASE_URL=http://localhost:8000/v1`, `KERNEL_AGENT_MODEL=Qwen/Qwen3-8B`, `MAX_MODEL_LEN=8192`. Pod repo at upstream `8f1ca41`.
- **11:42** `./serve.sh` started under nohup. vLLM args confirmed: `--max-model-len 8192 --max-num-seqs 4 --tensor-parallel-size 2 --num-gpu-blocks-override 1024`.
- **11:43** Selftests in the pod:
  - `nkibench.py --selftest` → **SELFTEST PASSED**. nki `0.6.0+31049202112.g85070674`; simulation API `nki.simulate`.
  - `kernelbench.py --selftest` → all planted bugs caught. The ragged-edge message on shape (8,16) says the error is in the "final PARTIAL ROW tile and PARTIAL COLUMN tile", which is vacuous because the whole array is one partial tile. This is a known defect, to be fixed.
  - `import nki; nki.benchmark` → **AttributeError: module 'nki' has no attribute 'benchmark'**. The device-timing API in nki 0.6 is not `nki.benchmark`. Recorded for Phase 4.
- **11:44** Local repo `hack-the-chip-seat73` created. Upstream copied without its history (see UPSTREAM.md). Research files vendored under `research/`. The Windows checkout had CRLF line endings, which would break `serve.sh` in the pod, so the repo is pinned to LF (`.gitattributes`, `core.autocrlf=false`).
- **11:45** `agent.py` gets **logging only, no behaviour change**: every attempt now records `tag, run, sample, ts, finish, usage (server prompt/completion tokens), reasoning_chars, prompt`. This is needed for the token-instrumentation and taxonomy deliverables. The baseline must be comparable to upstream, so nothing that affects generation was touched.
- **11:46** Server READY. **Baseline started**: `agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 --tag baseline` (upstream sampling temp 0.6 / top_p 0.95).

## Phase 1 — levels 5–7 proved with the upstream tutorial kernels (zero model calls)

- **11:47** `reference_level5.py` = `nki_matmul_hoist_load_` verbatim. `reference_level6.py` / `reference_level7.py` = `nki_matmul_block_free_dimension_` / `nki_matmul_fully_optimized_` with ONE change, block sizes clamped to `min(default, dim // tile)`. The verbatim versions are kept as `*_verbatim.py`. They assert `M % 256` and `M % 2048`, which most ladder shapes violate: verbatim L6 passes 2/4 shapes (AssertionError on the other 2), verbatim L7 passes 0/4.
- **11:48** **Byte counter cross-checked by hand.** Hoisted kernel on K=256 M=512 N=1024: lhsT read once (131,072 elements) + rhs re-read M/128 = 4 times (4 × 262,144) + output (524,288) = 1,703,936 elements × 4 B = **6,815,744 B**. The harness reports **6,815,744** — exact. Floor 917,504 × 4 = 3,670,016, also exact.
- **11:49** **Two of the three official kernels FAIL the upstream traffic bars.** Measured waste on the largest shape [sim, n=1, deterministic]: L4 tiled 2.000×; **L5 hoist 1.857× (bar 1.60 → rejected)**; **L6 block 1.286× (bar 1.25 → rejected)**; L7 full 1.000× (bar 1.05, passes). An official kernel failing its own level means the bar is wrong. **New bars: 1.90 / 1.30 / 1.05**, each just above the measured tutorial ratio and below the previous level's. Also, `--check` did not apply the traffic bar at all (only `agent.py` did); it now does.
- **11:50** **Discrimination verified** [sim]: each official kernel passes its own level at 4/4 shapes. The level-4 kernel renamed and submitted at L5 → 3/4, "moving 2.00× … requires 1.90×". At L6 → 2/4 (1.56×); at L7 → 2/4. The level-5 kernel at L6 → 2/4 (1.44×). The ladder is now strictly ordered. `nkibench --selftest` still PASSED.

## Two environment findings that change the plan

- **11:50** **The local server decodes DETERMINISTICALLY.** In the baseline, all 4 samples of every round were byte-identical (1 distinct kernel per round, rounds 0–3). Direct probe: `temperature=1.0, top_p=1.0`, same prompt twice → identical text ("Koala" both times). vLLM's log says the generation_config overrides the defaults to temp 0.6 / top_k 20 / top_p 0.95, yet output does not vary. **Consequence:** `--samples 4` buys 4 copies, not 4 attempts, and the "sampling params" lever (temp 0.7 / top_p 0.8) cannot be measured on this server. Diversity has to come from the prompt. Decision: **replace Phase-2 lever 1 with prompt-diverse samples** (each sample gets a different framing of the same instruction).
- **11:51** **A request with `seed` KILLS the engine.** My probe sent `seed=7`. vLLM-Neuron tried to create a per-request torch Generator on the Neuron device: `NotImplementedError: … implement getNewGenerator … for PrivateUse1` → `EngineDeadError`. The server died and the baseline run died with it (HTTP 500 at level 1, round 4). Crash log saved to `runs/vllm_crash_seed.log`. **Rule: never send `seed` to the local server.** This is worth telling other teams.
- **11:50** **Generation is slow**: about 50 s per round (270 completion tokens × 4 samples), and a 300-token completion took 68 s while the baseline was running. Node load average was 15 (16 seats share one trn2.48xlarge). Upstream reported ~8 s per round. Budget accordingly: a full `--repeat 5` over levels 1–4 is now about an hour.
- **11:52** Server restarted. The agent now grades each distinct kernel text once per level, an efficiency change only (grading is deterministic). Baseline will be re-run as soon as the server is READY.

## Phase 2 — levers built (behind flags; defaults == upstream behaviour)

- **11:55** Baseline relaunched as **4 concurrent per-level processes** (`--level N --rounds 8 --samples 1 --context 8192 --repeat 5`, tags `base_L1..4`). Why `--samples 1`: measured on the idle server, 1 request = 300 tok in 21.6 s, 4 concurrent = 54.5 s each (only 1.6× aggregate throughput), and the 4 samples were byte-identical anyway. With 4 processes, the server's 4 slots carry 4 *different* jobs.
- **11:59** **Skeletons proved completable before any prompt uses them** [sim]. Remainder-safe tiled matmul (`m_sz = min(TM, M - m0)`, one PSUM accumulator outside the K loop): L4 4/4 and L3 1/1. Avgpool via strided-slice accumulation (`tensor_tensor` add over `in_tile[:, di::p, dj::p]`): L1 4/4. Avgpool via per-window `tensor_reduce(axis=(1,2))`: L1 4/4. These hand-written kernels are NOT in any prompt.
- **12:00** `verdicts.py`: upstream `enrich()` as an explicit table of (bucket, regex, instruction), plus `classify(feedback)` → bucket for every grade stage. Selftest checks 14 raw errors and 9 feedback strings. **Byte-identical to upstream enrich() on all 14 cases**, so swapping it in does not change what the model is told.
- **12:01** `levers.py` + `agent.py` flags:
  - `--mech`: AST fixes for entry name, `@nki.jit`, imports.
  - `--skeleton`: rank-aware template with TODO slots, 209–324 tokens.
  - `--retrieve`: bucket-keyed excerpts of the vendored AWS corpus, 33–129 tokens each, with source lines cited.
  - `--empty-terse`: escalate terseness on every empty answer.

  Every attempt now logs `bucket, mech_fixes, prompt_kind, prompt_sections` (token estimate per section).
- **12:02** Offline smoke test in the pod: with no flags, L4 is solved on round 1 (round 0 is the planted missing-jit). With `--mech`, it is solved on **round 0** (`mech_fixes: ['added @nki.jit']`). The sections record `{api: 395, skeleton: 324, task: 223}`.
- **12:04** Clock check: pod `date -u` = 16:04 → 12:04 ET. Phase-2 timestamps above were corrected; from now on every timestamp comes from `scripts/now`.
- **12:05** **Queue runner** (`runq.py`, pod-side) keeps 4 agent processes alive. Experiments **E1 `--skeleton`, E2 `--retrieve`, E3 `--mech`**, each measured against the baseline, are queued as 20 single-run jobs each (4 levels × 5 runs, tags `<exp>_L<n>_r<k>`). They start as baseline levels finish. **Pod code is FROZEN** until E1–E3 finish (runq launches fresh processes that would import changed modules). Harness work is tested in `/workspace/dev` instead.
  - Confound noted: `verdicts.py` gained a `contraction_mismatch` translation after the baseline started, so E1–E3 have it and the baseline does not. It affects L2 only (nc_matmul misused for transpose).
- **12:05** Baseline, early observation: runs are **not** identical despite deterministic decoding (L3 run 0 = `dma_shape_mismatch`, run 1 = `tile_1d`; L4 run 0 = 0.30, run 1 = 0.62). So `--repeat 5` measures real variance, presumably from batching across concurrent requests. Also, **L1 and L2 reach for `nc_matmul`** (pooling and transpose via matmul): the API card's long matmul paragraph biases non-matmul levels.

## Phase 3 (early) — held-out hostile cases + calibration

- **12:06** `heldout.py` defines hostile cases per level, never shown to the agent: C=1 / P=1 / M=1, H,W not divisible by pool, prime K=131, ragged K=96 M=200 N=700, N=513, ×1e4 magnitudes, a constant row. `confidence(src, level)` is computed from code + public result only (rules: shape asserts ×0.4, no remainder handling while the public set has no partial tile ×0.5, hard-coded public dims ×0.7, hazard ×0.3). Scored with a Brier score and a reliability table.
- **12:06** **All official AWS tutorial matmul kernels (L4–L7 references) FAIL held-out** [sim]: they `assert` divisibility by 128/512, so M=200 / M=1 raise. Confidence had rated them 0.38. **The clamped L6/L7 references also fail.** Reference-only calibration: Brier **0.154**, n=7.
- **12:07** **Silent wrong answer in the VERBATIM AWS tutorial kernel** [sim]: `nki_matmul_block_free_dimension_` asserts M%256 and N%1024 but not K%128. With **K=131** it returns numbers that are wrong by 1.16× the output RMS: no assert, no exception (`K // 128` drops 3 rows of the contraction). K=128 passes. This is the "wrong kernels return plausible numbers" property, found in AWS's own tutorial.
- **12:06** The remainder-safe skeleton kernel (`min(TM, M - m0)` etc.) passes **all 4** held-out matmul cases. Both avgpool variants pass all 4 held-out pool cases. So the skeleton's structure is the held-out-robust one; the tutorial's is not.

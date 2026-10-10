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

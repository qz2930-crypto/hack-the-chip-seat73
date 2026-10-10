# DEMO — 3 minutes (English script, 中文提示 in italics)

Open in tabs before starting: `RESULTS.md`, `FAILURES.md`, `CALIBRATION.md`, `token_budget.png`, and the two
logs below. Everything shown is from our own runs; **every number is a rate over n runs, [sim] unless [device].**

---

## 0:00–0:30  The claim *（开场：一句话讲清楚我们证明了什么）*

> "Same 8-billion-parameter model all day: Qwen3-8B on our own Trainium chip. We never changed the model. We
> changed what the checker *says* when the model is wrong. That alone took the NKI ladder from **0 of 4 levels**
> to **7 of 8 levels solved in 5 out of 5 runs**, including the three performance levels that grade on HBM bytes."

Show `RESULTS.md`: the baseline row (all 0/5) vs the C10 row (L1–L7 5/5).

## 0:30–1:30  One failure and its recovery *（核心：展示一次失败 → 指令 → 恢复）*

Open `runs/1314_C1/c1_L4_r0.log` (or any c1_L4 run):

> "Round 0: the model fills our tiling skeleton almost perfectly, but writes `lhsT[m0:…, k0:…]`, axes swapped.
> On square tiles that doesn't crash. It **silently computes the wrong matrix**. The upstream checker said
> 'most elements are wrong, check the operand layout'. True, and useless: it repeated the same kernel.
> Our checker reads the code and says: *'`lhsT` has shape [K, M], so its first index must be the K range:
> write `lhsT[k0:k0+k_sz, m0:m0+m_sz]`.'* Round 1: solved. Five runs out of five."

Then the cleanest A/B of the day (L5, same seat, interleaved): *（再给一个对照实验，最有说服力）*

> "Level 5 is about bytes, not correctness: same answer, fewer HBM transfers. Control arm gets upstream's hint,
> 'hoist the operand loads out of the innermost loop': **0 of 5**, stuck at 2.00× the byte floor; sometimes it
> made the kernel *worse*. Treatment gets: 'the lhsT load is inside the n loop but doesn't depend on n; move it
> out': **5 of 5**. Same model, same state, one sentence."

## 1:30–2:15  We know when we're wrong *（诚实与校准：agent 知道自己什么时候错）*

Show `CALIBRATION.md`:

> "Every 'solved' carries a confidence, computed before we look at held-out cases: ragged shapes, M=1, prime K,
> ×10⁴ magnitudes. Every kernel the agent claimed at 0.95 passes them. AWS's own tutorial matmul kernels assert
> divisibility, rated 0.38, and fail them; one returns silently wrong numbers at K=131."

> "On the chip: our agent's L3, L4 and L5 kernels run on Trainium and match NumPy. Its first L7 kernel did **not**
> compile on the device, a list comprehension the simulator accepts. So that became a checker rule, and the
> final L7 kernels use plain loops. We don't claim latency: we didn't get device-side timing inside our time box."

## 2:15–2:50  What failed, and the taxonomy *（失败分类与没做成的事）*

Show `FAILURES.md` (counts by root cause):

> "The two biggest root causes across all attempts, swapped axes and scalar window reductions, are exactly the
> two our first targeted instructions fixed. Things that did *not* help, measured: retrieving AWS docs (no
> change), mechanical AST fixes (never needed), and one of our own prompt fixes that leaked into another level
> and cost two runs until we scoped it. Level 8, attention, is still 0 of 5. We know where it loops."

## 2:50–3:00  Close *（结尾）*

> "The checker is the product. Everything is in the repo with a timestamped log of every decision, and every run
> was re-graded from its own code."

---

### If asked *（评委可能的提问）*
* *Is n=5 real?* The server decodes deterministically. Stage-B repeats do vary (batching); Stage-A repeats are
  byte-identical, so we report Stage A as effectively n=1 per level.
* *Did you hand the model the answer?* No target values ever appear in feedback (`--directional`). Skeletons have
  TODO slots, proved completable by hand first. For L2 the TODO restates the index mapping, and we say so.
  For Stage-A softmax we refused to paste a full kernel even though it would have "solved" the level.
* *Teammates' seats?* Used with their consent, identical server config, code copied byte-for-byte; each
  experiment group stays on one seat.

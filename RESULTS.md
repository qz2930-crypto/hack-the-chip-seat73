# RESULTS — experiment table

Rule: no row without `--repeat 5` (`--repeat 3` for gpt-oss). All numbers **[sim]** (nki CPU simulator) unless tagged **[device]**.
Format per level: `solved k/n, mean x.xx, all=[...]`. Append only; never overwrite (teammates edit this too).

| time (ET) | commit | change (one line) | model | n | L1 | L2 | L3 | L4 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| upstream | 8f1ca41 | upstream STATE.md baseline (not ours, for reference) | Qwen3-8B | 5 | 0/5, 0.30 | 4/5, [1,1,1,.5,1] | 0/5, 0.30 | 0/5, 0.62 | reference |
| 12:33 | 9d036da (agent behaviour = upstream) | **our baseline**: upstream agent, `--samples 1` (server is deterministic; 4 samples were byte-identical), 4 levels concurrent | Qwen3-8B local [sim] | 5 | solved 0/5, mean 0.30, all=[0.3, 0.3, 0.3, 0.3, 0.3] | solved 0/5, mean 0.30, all=[0.3, 0.3, 0.3, 0.3, 0.3] | solved 0/5, mean 0.30, all=[0.3, 0.3, 0.3, 0.3, 0.3] | solved 0/5, mean 0.36, all=[0.3, 0.62, 0.3, 0.3, 0.3] | baseline |

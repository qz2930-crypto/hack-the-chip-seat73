# TOKENS — input-token instrumentation

Calibration: the server counted **1.21x** the chars/4 estimate over 94 prompts (sections below are scaled by this factor). The chat template adds a fixed overhead the sections cannot see.

| experiment | level | prompts | task | api | skeleton | docs | code | error | ledger | total | % of 8192 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | 1 | 12 | 101 | 119 | 67 | 0 | 316 | 91 | 17 | 711 | 9% | 968 |
| c1 | 2 | 2 | 330 | 476 | 253 | 0 | 0 | 0 | 0 | 1060 | 13% | 1060 |
| c1 | 3 | 6 | 166 | 238 | 195 | 0 | 273 | 60 | 0 | 933 | 11% | 1137 |
| c1 | 4 | 5 | 185 | 286 | 234 | 0 | 219 | 73 | 0 | 997 | 12% | 1136 |
| skel | 1 | 24 | 88 | 99 | 56 | 0 | 338 | 85 | 41 | 707 | 9% | 968 |
| skel | 2 | 5 | 330 | 476 | 253 | 0 | 0 | 0 | 0 | 1060 | 13% | 1060 |
| skel | 3 | 20 | 114 | 119 | 98 | 0 | 404 | 120 | 34 | 889 | 11% | 1137 |
| skel | 4 | 20 | 112 | 119 | 98 | 0 | 410 | 72 | 34 | 845 | 10% | 1136 |

Prompt kinds: {'first': 31, 'repair': 28, 'repair+ledger': 35}.
Completion tokens: mean 497, max 576 (n=94). Truncated answers (finish=length): 0.

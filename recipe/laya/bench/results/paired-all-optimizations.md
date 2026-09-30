Paired comparison of the worker with every optimization (B: LAYA_WORKER_COMPILE=on, LAYA_WORKER_WEIGHTS=fp16) against the worker with none (A: compile off, fp32 weights), both alive at once; built with `paired.py --summarize`. M1 Pro, AC power, checkpoint 55cf4c4.

## e5a: A = `LAYA_WORKER_COMPILE=off`, B = `LAYA_WORKER_COMPILE=on LAYA_WORKER_WEIGHTS=fp16`, load at start 10.84

| input | pairs | A p50 ms | B p50 ms | median B/A | 95% interval |
|---|---|---|---|---|---|
| W1 | 300 | 55.3 | 34.7 | 0.626 | 0.620–0.633 |
| W2 | 300 | 79.6 | 63.4 | 0.798 | 0.792–0.803 |
| W3 | 300 | 184.5 | 153.2 | 0.833 | 0.826–0.842 |
| W4 | 300 | 95.3 | 82.0 | 0.859 | 0.853–0.868 |
| W5 | 300 | 181.4 | 148.7 | 0.818 | 0.808–0.825 |
| W6 | 300 | 44.8 | 27.6 | 0.616 | 0.608–0.625 |

B vs A answers: max |Δp| 0.0031, flips [], errors []
recompiled after ready: {'A': None, 'B': False}; footprint MB: {'A': 3501, 'B': 2849}

## e5b: A = `LAYA_WORKER_COMPILE=off`, B = `LAYA_WORKER_COMPILE=on LAYA_WORKER_WEIGHTS=fp16`, load at start 15.12

| input | pairs | A p50 ms | B p50 ms | median B/A | 95% interval |
|---|---|---|---|---|---|
| W1 | 300 | 59.3 | 35.9 | 0.616 | 0.609–0.628 |
| W2 | 300 | 137.0 | 108.2 | 0.795 | 0.778–0.814 |
| W3 | 300 | 199.1 | 166.7 | 0.832 | 0.825–0.838 |
| W4 | 300 | 101.2 | 87.1 | 0.863 | 0.857–0.868 |
| W5 | 300 | 187.8 | 156.2 | 0.824 | 0.818–0.834 |
| W6 | 300 | 64.7 | 36.8 | 0.584 | 0.569–0.600 |

B vs A answers: max |Δp| 0.0031, flips [], errors []
recompiled after ready: {'A': None, 'B': False}; footprint MB: {'A': 3533, 'B': 2783}


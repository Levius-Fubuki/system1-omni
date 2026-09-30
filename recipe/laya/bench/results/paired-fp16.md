Paired comparison of the worker with LAYA_WORKER_COMPILE=single, fp32 weights (A) and fp16 weights (B), both alive at once; built with `paired.py --summarize`. M1 Pro, AC power, checkpoint 55cf4c4.

## e4a: A = `LAYA_WORKER_COMPILE=single`, B = `LAYA_WORKER_COMPILE=single LAYA_WORKER_WEIGHTS=fp16`, load at start 31.32

| input | pairs | A p50 ms | B p50 ms | median B/A | 95% interval |
|---|---|---|---|---|---|
| W1 | 300 | 33.2 | 28.5 | 0.857 | 0.855–0.861 |
| W2 | 300 | 62.8 | 56.9 | 0.910 | 0.907–0.913 |
| W3 | 300 | 152.8 | 137.0 | 0.900 | 0.897–0.905 |
| W4 | 300 | 71.6 | 74.6 | 1.042 | 1.040–1.045 |
| W5 | 300 | 164.0 | 145.9 | 0.894 | 0.891–0.899 |
| W6 | 300 | 26.5 | 22.7 | 0.855 | 0.853–0.859 |

B vs A answers: max |Δp| 0.0031, flips [], errors []
recompiled after ready: {'A': False, 'B': False}; footprint MB: {'A': 3647, 'B': 2717}

## e4b: A = `LAYA_WORKER_COMPILE=single`, B = `LAYA_WORKER_COMPILE=single LAYA_WORKER_WEIGHTS=fp16`, load at start 18.75

| input | pairs | A p50 ms | B p50 ms | median B/A | 95% interval |
|---|---|---|---|---|---|
| W1 | 300 | 34.9 | 29.7 | 0.855 | 0.851–0.861 |
| W2 | 300 | 68.3 | 61.8 | 0.910 | 0.908–0.913 |
| W3 | 300 | 158.8 | 143.3 | 0.902 | 0.897–0.908 |
| W4 | 300 | 90.0 | 93.6 | 1.040 | 1.032–1.047 |
| W5 | 300 | 161.7 | 143.2 | 0.891 | 0.889–0.896 |
| W6 | 300 | 32.7 | 28.0 | 0.850 | 0.839–0.861 |

B vs A answers: max |Δp| 0.0031, flips [], errors []
recompiled after ready: {'A': False, 'B': False}; footprint MB: {'A': 3646, 'B': 2721}


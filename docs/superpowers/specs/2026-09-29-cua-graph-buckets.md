# Cua-S1 segmented Graph bucketing experiment

User approved starting the previously proposed length-bucketing experiment on the supplied RTX 4090 server. Base: PR #33 commit be44f57ed8a34c4122a6a03fb0adf33bf3000c51. Preserve the existing worker behavior; first implement an opt-in experimental runtime in recipe/cua_s1.

Pad only causal linear-attention runs on the right to a multiple of 64 (the installed fallback DeltaNet chunk width). Crop each run's output before full attention. Full attention, rotary positions, final-token selection and usage remain at original length. The first experiment fixes image geometry and batch size one, with no padding in the original request. Reject other mask/layout forms. No recurrent state or KV cache survives a forward.

Capture keys describe the padded segment layout; position values never enter captured linear layers. Bound caches, capture work, streams and pools using #33. Validate complete logits exactly against unpadded eager on capture and the first encounter of each real length in each resident bucket. Reject mismatching lengths and preserve eager fallback; changed image/text checks remain necessary because one-time validation is not a universal proof.

Measure cold-inclusive synchronized predict latency against eager and unmodified exact-layout Graph runtime, with two questions per request and balanced variant order. Record first-length checks, captures, replay, padding, memory and complete responses. Warm replay must be distinguished from first-length validation. Keep negative results, and do not enable a worker option until this approach has evidence of correctness and benefit.

## Experiment outcome and scope refinement

Whole-segment padding failed exact-logit gates. The implemented follow-up captures only the internal DeltaNet rule using zero-padded query/key/value/g/beta tensors. Its surrounding projections and full attention retain real shapes. Five changed-content cases, bucket-boundary transitions and fallback checks pass exact logits; the cold-inclusive matrix favors rule buckets for twelve-length rotation but favors exact segmented Graphs for stable hot lengths. Keep both variants in the experimental recipe, with the worker unchanged. See the measured report for exact source revisions and limitations.

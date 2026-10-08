# JEMM

JEMM uses the Qwen3.8-27B backbone and a language LoRA adapter to answer
`choice`, `noul` and `score` questions. It places all candidates in an ordered
prompt, performs one prefill per question, and reads the corresponding label
logits from the original untied LM head. It produces decisions without generating
text.

The [native recipe](native.md) covers pinned artifact downloads, the offline CPU
export, the Rust/CUDA worker, frontend startup and the unmodified reference.
The [validation report](validation.md) records completed A800 text/image
preprocessing and response parity, HTTP checks and matched warm HTTP timings
for the fixed corpus. The reference used unmerged PEFT adapters, SDPA and FLA,
with the Torch convolution fallback because optional `causal_conv1d` was absent.
CPU inference, native Metal and other GPU configurations have no JEMM
validation recorded here; these results do not establish general accuracy or
production throughput.

| Artifact | Pinned revision |
| --- | --- |
| [Official JEMM source](https://github.com/ypcypc/JEMM/tree/6822fe0fd53c5e6670af6ba99fb2c857a661e532) | `6822fe0fd53c5e6670af6ba99fb2c857a661e532` |
| [MaestroYan/JEMM adapter](https://huggingface.co/MaestroYan/JEMM/tree/76e3c209e8441fa658221c7ba2725bad2f811176) | `76e3c209e8441fa658221c7ba2725bad2f811176` |
| [Qwen/Qwen3.8-27B base](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The native worker has separate contract, processing, execution and HTTP modules
under [`src/models/jemm/native/`](https://github.com/ThinkFlowLab/system1-omni/tree/main/src/models/jemm/native).
It reuses the shared Qwen language and configurable vision executor and the
[native runtime](../../src/runtime/README.md). The
[worker contract](https://github.com/ThinkFlowLab/system1-omni/blob/main/src/models/jemm/README.md)
describes prompts, calibration, ownership and diagnostics in more detail.

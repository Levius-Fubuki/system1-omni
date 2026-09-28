# Cua-S1 optimization after request-local image reuse

The user approved continuing the experiment after PR #17, publishing a complete PR, backing up evidence, and shutting down the GPU server.

Profile the existing request-local reuse path on the same pinned RTX 4090 environment. Preserve an unmodified response and latency reference, and examine one- and eight-question cases with both repeated and distinct questions. CPU annotations, CUDA kernels, and synchronized end-to-end latency are different measurements and must not be added together.

Three candidates follow from the existing implementation: compute only the final output-token projection, replace the PyTorch Gated DeltaNet/causal-convolution fallbacks, or capture a stable request shape with CUDA Graphs. The first is the smallest change and can be tested against the existing model contract. Select it only if a clean paired experiment shows a useful end-to-end gain and parity. Otherwise report the evidence and choose the next measured hotspot rather than claiming a speedup.

If selected, pass `logits_to_keep=1` only in the reused multi-question path. The model still performs one language forward per question; the parameter limits the output head to the final hidden state, from which candidate probabilities are already read. Retain the old reference path for paired comparison. Confirm exact input embeddings and 3D positions, candidate probabilities within the pinned oracle tolerance, and identical decisions and complete responses. Benchmark 1/2/4/8 questions with alternating order, save individual samples and peak allocated memory, and record source/environment provenance. Document limitations and restore the GPU host to a stopped state after verified backup.

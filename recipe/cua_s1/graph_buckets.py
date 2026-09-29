"""Experimental right-padded linear segments; full attention stays exact-length.

Not enabled by the worker. Only batch-one dense, cache-free prefill is supported.
The causal linear layers cannot read future padded tokens. Nevertheless GEMM
shapes can change rounding, so every real length receives a strict logits gate.
"""

from __future__ import annotations

import time

from models.cua_s1.multimodal.graph_runtime import (
    GraphRuntime,
    _GraphPool,
    _GraphSegment,
)


def bucket_length(length, width):
    if type(width) is not int or width <= 0 or width % 64:
        raise ValueError("bucket width must be a positive integer multiple of 64")
    return ((length + width - 1) // width) * width


def pack_hidden(hidden, width):
    import torch.nn.functional as F

    length = hidden.shape[1]
    return F.pad(hidden, (0, 0, 0, bucket_length(length, width) - length))


class BucketRuntime(GraphRuntime):
    """Recipe-only variant with the same ownership/admission policy as #33."""

    def __init__(self, model, config=None, *, width=64):
        bucket_length(1, width)
        super().__init__(model, config)
        self.width = width
        self.stats.update(
            length_checks=0,
            length_rejections=0,
            length_disabled=0,
            length_check_ms=0.0,
            real_tokens=0,
            padded_tokens=0,
        )

    @staticmethod
    def _dense(values):
        hidden = values["inputs_embeds"]
        mask = values["attention_mask"]
        return (
            hidden.is_contiguous()
            and mask is not None
            and tuple(mask.shape) == tuple(hidden.shape[:2])
            and bool((mask == 1).all())
        )

    def _key(self, values):
        hidden = values["inputs_embeds"]
        return (
            id(self.model),
            self.model.get_base_model().model.language_model.config._attn_implementation,
            bucket_length(hidden.shape[1], self.width),
            hidden.shape[2],
            str(hidden.dtype),
            str(hidden.device),
        )

    def _validate_replay(self, values, entry, output):
        import torch

        length = values["inputs_embeds"].shape[1]
        if length in entry.verified_lengths:
            return output
        started = time.perf_counter()
        reference = self._eager(values)
        self.stats["length_checks"] += 1
        if torch.equal(reference, output):
            entry.verified_lengths.add(length)
        else:
            entry.rejected_lengths.add(length)
            self.stats["length_rejections"] += 1
            output = reference
        self.stats["length_check_ms"] += (time.perf_counter() - started) * 1000
        return output

    def forward(self, values):
        import torch

        with self.lock, torch.no_grad():
            if not self._in_request:
                self.stats["no_request"] += 1
                return self._eager(values)
            if not self._supported(values) or not self._dense(values):
                self.stats["unsupported"] += 1
                return self._eager(values)
            length = values["inputs_embeds"].shape[1]
            if bucket_length(length, self.width) > self.config.max_tokens:
                self.stats["unsupported"] += 1
                return self._eager(values)
            key = self._key(values)
            if key in self.disabled:
                self.stats["disabled"] += 1
                return self._eager(values)
            entry = self.cache.get(key)
            if entry is not None:
                if length in entry.rejected_lengths:
                    self.stats["length_disabled"] += 1
                    return self._eager(values)
                output = self._run_segments(values, entry)
                self.stats["replays"] += 1
                return self._validate_replay(values, entry, output)
            reason = self.admission.reason(key)
            if reason is not None:
                self.stats[reason] += 1
                return self._eager(values)
            ticket = self.admission.start_capture()
            self.stats["capture_attempts"] += 1
            attempt_start = time.perf_counter()
            try:
                output = self._capture(values, key)
                entry = self.cache.get(key)
                if entry is not None:
                    entry.verified_lengths = {length}
                    entry.rejected_lengths = set()
                return output
            finally:
                elapsed = (time.perf_counter() - attempt_start) * 1000
                self.admission.finish_capture(ticket, elapsed)
                self.stats["capture_attempt_ms"] += elapsed

    def _run_segments(self, values, entry):
        from transformers.masking_utils import create_causal_mask

        core = self.model.get_base_model().model
        text = core.language_model
        hidden = values["inputs_embeds"]
        length = hidden.shape[1]
        self.stats["real_tokens"] += length
        self.stats["padded_tokens"] += bucket_length(length, self.width) - length
        mask = create_causal_mask(
            config=text.config,
            inputs_embeds=hidden,
            attention_mask=values["attention_mask"],
            past_key_values=None,
            position_ids=None,
        )
        rope = text.rotary_emb(hidden, values["position_ids"])
        index = 0
        while index < text.config.num_hidden_layers:
            if text.config.layer_types[index] == "full_attention":
                hidden = text.layers[index](
                    hidden,
                    position_embeddings=rope,
                    attention_mask=mask,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                )
                index += 1
                continue
            end = index + 1
            while (
                end < text.config.num_hidden_layers
                and text.config.layer_types[end] == "linear_attention"
            ):
                end += 1
            padded = pack_hidden(hidden, self.width)
            block = entry.blocks.get((index, end))
            if block is None:
                if entry.pool is None:
                    entry.pool = _GraphPool(hidden.device)
                block = _GraphSegment(
                    text.layers, index, end, padded, None, pool=entry.pool
                )
                entry.blocks[(index, end)] = block
                entry.update_bytes()
            # No input padding is supported. Future right-padding cannot affect
            # the prefix of causal, cache-free layers. Drop it before SDPA.
            hidden = block.replay(padded, None)[:, :length, :]
            index = end
        hidden = text.norm(hidden)
        return self.model.get_base_model().lm_head(hidden[:, -1:, :])[0, -1, :]


def pack_rule_inputs(values, width):
    import torch.nn.functional as F

    length = values["query"].shape[1]
    padding = bucket_length(length, width) - length
    return {
        name: F.pad(value, (0, 0) * (value.ndim - 2) + (0, padding))
        for name, value in values.items()
    }


class _RuleSegment(_GraphSegment):
    """Capture just the fallback rule, reusing #33 stream/pool ownership."""

    def __init__(self, function, values, pool):
        self.function = function
        self.static_extra = {k: v.clone() for k, v in values.items() if k != "query"}
        super().__init__([], 0, 0, values["query"], None, pool=pool)
        self.external_bytes += sum(
            v.untyped_storage().nbytes() for v in self.static_extra.values()
        )

    def _forward(self):
        return self.function(
            self.static_hidden,
            self.static_extra["key"],
            self.static_extra["value"],
            g=self.static_extra["g"],
            beta=self.static_extra["beta"],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

    def replay_values(self, values):
        for name, value in self.static_extra.items():
            value.copy_(values[name])
        return super().replay(values["query"], None)

    def close(self):
        super().close()
        self.static_extra.clear()


class RuleBucketRuntime(BucketRuntime):
    """Single-thread experiment: only pad the internal DeltaNet rule inputs.

    Temporarily replaces the installed module's fallback function under the
    runtime lock. Not suitable for simultaneous engines or production serving.
    The original function is restored even on failure.
    """

    def _run_segments(self, values, entry):
        import inspect
        from unittest.mock import patch

        text = self.model.get_base_model().model.language_model
        module = inspect.getmodule(text.layers[0].linear_attn.__class__)
        original = module.torch_chunk_gated_delta_rule
        index = 0
        length = values["inputs_embeds"].shape[1]
        self.stats["real_tokens"] += length
        self.stats["padded_tokens"] += bucket_length(length, self.width) - length

        def dispatch(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            **kwargs,
        ):
            nonlocal index
            if (
                initial_state is not None
                or output_final_state
                or not use_qk_l2norm_in_kernel
                or any(v is not None for v in kwargs.values())
            ):
                raise RuntimeError(
                    "unsupported rule call; experiment requires cache-free normalized prefill"
                )
            packed = pack_rule_inputs(
                dict(query=query, key=key, value=value, g=g, beta=beta), self.width
            )
            block = entry.blocks.get(index)
            if block is None:
                if entry.pool is None:
                    entry.pool = _GraphPool(query.device)
                block = _RuleSegment(original, packed, entry.pool)
                entry.blocks[index] = block
                entry.update_bytes()
            index += 1
            output, state = block.replay_values(packed)
            return output[:, : query.shape[1]].contiguous(), state

        with patch.object(module, "torch_chunk_gated_delta_rule", dispatch):
            output = self.model(**values, logits_to_keep=1, use_cache=False).logits[
                0, -1, :
            ]
        if index != text.config.layer_types.count("linear_attention"):
            raise RuntimeError("unexpected DeltaNet call count")
        return output

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

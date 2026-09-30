"""Compare explicit callback prefill to the pinned Transformers implementation."""

import pytest


@pytest.mark.parametrize("length", [1, 63, 64, 65, 129])
def test_explicit_prefill_matches_upstream_without_mutation(length):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

    from models.cua_s1.multimodal.rule_prefill import rule_prefill

    torch.manual_seed(43)
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    module = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
    hidden = torch.randn(1, length, 32)
    original_rule = upstream.torch_chunk_gated_delta_rule
    original_forward = module.forward.__func__
    calls = []

    def rule(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return original_rule(*args, **kwargs)

    with torch.no_grad():
        expected = module(hidden, use_cache=False)
        actual = rule_prefill(module, hidden, rule)
    assert torch.isfinite(expected).all()
    assert torch.equal(expected, actual)
    assert calls == [(1, length, 4, 8)]
    assert upstream.torch_chunk_gated_delta_rule is original_rule
    assert module.forward.__func__ is original_forward


def test_callback_failure_does_not_modify_model_or_globals():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

    from models.cua_s1.multimodal.rule_prefill import rule_prefill

    config = Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    module = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
    original_rule, original_forward = (
        upstream.torch_chunk_gated_delta_rule,
        module.forward.__func__,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("test callback failure")

    with torch.no_grad(), pytest.raises(RuntimeError, match="test callback"):
        rule_prefill(module, torch.ones(1, 3, 32), fail)
    assert upstream.torch_chunk_gated_delta_rule is original_rule
    assert module.forward.__func__ is original_forward

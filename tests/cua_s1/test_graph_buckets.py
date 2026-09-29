"""Prefix packing and strict per-length gates for experimental Graph buckets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recipe/cua_s1"))


def test_bucket_boundaries():
    from graph_buckets import bucket_length

    assert [bucket_length(n, 64) for n in (1, 63, 64, 65, 127, 128, 129)] == [
        64,
        64,
        64,
        128,
        128,
        128,
        192,
    ]


@pytest.mark.parametrize("width", [0, -64, 32, 65, 64.0, True])
def test_bucket_width_must_be_a_positive_chunk_multiple(width):
    from graph_buckets import bucket_length

    with pytest.raises(ValueError, match="multiple of 64"):
        bucket_length(65, width)


def runtime():
    from graph_buckets import BucketRuntime

    model = SimpleNamespace(
        get_base_model=lambda: SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="sdpa")
                )
            )
        )
    )
    return BucketRuntime(model)


def values(length, dim=4):
    torch = pytest.importorskip("torch")
    return {
        "inputs_embeds": torch.ones(1, length, dim),
        "position_ids": torch.zeros(3, 1, length, dtype=torch.long),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
    }


def test_pack_preserves_prefix_and_clears_tail_after_input_changes():
    from graph_buckets import pack_hidden

    torch = pytest.importorskip("torch")
    long = pack_hidden(torch.full((1, 63, 4), 7.0), 64)
    short = pack_hidden(torch.full((1, 2, 4), 3.0), 64)
    assert long.shape == short.shape == (1, 64, 4)
    assert torch.equal(short[:, :2], torch.full((1, 2, 4), 3.0))
    assert torch.count_nonzero(short[:, 2:]) == 0
    assert torch.all(long[:, :63] == 7)


def test_keys_share_only_compatible_padded_segment_layouts():
    r = runtime()
    assert r._key(values(63)) == r._key(values(64))
    assert r._key(values(64)) != r._key(values(65))
    assert r._key(values(63)) != r._key(values(63, dim=8))
    assert r._key(values(63)) != runtime()._key(values(63))


def test_new_real_length_validated_and_rejected_without_poisoning_bucket():
    torch = pytest.importorskip("torch")
    r = runtime()
    entry = SimpleNamespace(verified_lengths=set(), rejected_lengths=set())
    calls = []
    reference = torch.tensor([1.0, 2.0])
    r._eager = lambda v: calls.append(v["inputs_embeds"].shape[1]) or reference
    assert torch.equal(
        r._validate_replay(values(63), entry, reference.clone()), reference
    )
    r._validate_replay(values(63), entry, reference.clone())
    assert calls == [63]
    wrong = torch.tensor([1.0, 3.0])
    assert torch.equal(r._validate_replay(values(64), entry, wrong), reference)
    assert entry.verified_lengths == {63}
    assert entry.rejected_lengths == {64}
    assert r.stats["length_checks"] == 2
    assert r.stats["length_rejections"] == 1


def test_original_padding_is_not_supported():
    torch = pytest.importorskip("torch")
    r = runtime()
    v = values(63)
    assert r._dense(v)
    v["attention_mask"][0, -1] = 0
    assert not r._dense(v)
    v["attention_mask"] = torch.ones(1, 63, 1)
    assert not r._dense(v)


def test_rule_packing_zeros_future_decay_and_updates():
    from graph_buckets import pack_rule_inputs

    torch = pytest.importorskip("torch")
    original = {k: torch.ones(1, 63, 2, 4) for k in ("query", "key", "value")}
    original.update(g=torch.full((1, 63, 2), -0.5), beta=torch.ones(1, 63, 2))
    packed = pack_rule_inputs(original, 64)
    for name, tensor in packed.items():
        assert tensor.shape[1] == 64
        assert torch.equal(tensor[:, :63], original[name])
        assert torch.count_nonzero(tensor[:, 63:]) == 0

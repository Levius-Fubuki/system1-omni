"""Guard the controlled admission comparison used for mode-policy decisions."""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recipe/cua_s1"))


def test_tuned_comparator_changes_only_admission_window():
    from benchmark_graph_buckets import variant_configs

    configs = variant_configs("worker", 64, tuned_exact_window=32)
    exact = asdict(configs["exact"])
    tuned = asdict(configs["exact_tuned"])
    assert exact.pop("admission_window") == 8
    assert tuned.pop("admission_window") == 32
    assert exact == tuned
    assert configs["bucket"].mode == "rule-bucket"
    assert configs["bucket"].max_bytes == configs["exact"].max_bytes


def test_mixed_schedules_include_cold_inputs_and_returning_hotspot():
    from benchmark_graph_buckets import schedules

    cases = schedules()
    assert cases["hot_cold"] == [v for i in range(8) for v in [1, 2, 4, 8, 13 + i]]
    assert cases["shifting_hot"] == [
        v
        for group in [range(1, 5), range(5, 9), range(9, 13), range(1, 5)]
        for v in list(group) * 8
    ]

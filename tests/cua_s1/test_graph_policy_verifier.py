"""The independent verifier must reject corrupted recorded GPU evidence."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recipe/cua_s1"))
EVIDENCE = ROOT / "recipe/cua_s1/experiments/rtx4090-graph-policy"


@pytest.fixture(scope="module")
def reports():
    return {
        name: json.loads((EVIDENCE / (name + ".json")).read_text())
        for name in ("baseline", "shared")
    }


def test_recorded_policy_evidence_is_consistent(reports):
    import verify_graph_policy as verifier

    assert verifier.baseline(reports["baseline"]) == 2304
    assert verifier.shared(reports["shared"]) == 119


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["config"]["variant_graph"]["exact_tuned"].update(max_shapes=16),
        lambda r: r["config"]["variant_graph"]["bucket"].update(max_bytes=2 << 30),
        lambda r: r["workloads"]["churn_twelve"][0]["events"][0]["variants"][
            "exact_tuned"
        ]["stats_delta"].update(requests=2),
        lambda r: r["workloads"]["hot_four"][0]["events"].pop(),
    ],
)
def test_rejects_uncontrolled_comparison_or_invalid_clock(reports, mutate):
    import verify_graph_policy as verifier

    report = copy.deepcopy(reports["baseline"])
    mutate(report)
    with pytest.raises(AssertionError):
        verifier.baseline(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["boundary_checks"][0].update(equal=False),
        lambda r: r["boundary_checks"][0].update(cache_bytes=10 << 30),
        lambda r: r["policies"]["cross_mode_eviction"]["events"][1].update(
            retired_with_live_references=0
        ),
        lambda r: r["response_cleanup"]["after"].update(cache_shapes=1),
        lambda r: r["response_logit_checks"][0].update(max_abs=0.1),
        lambda r: r["responses"][0]["actual"].update(extra="unexpected field"),
    ],
)
def test_rejects_shared_resource_or_correctness_corruption(reports, mutate):
    import verify_graph_policy as verifier

    report = copy.deepcopy(reports["shared"])
    mutate(report)
    with pytest.raises(AssertionError):
        verifier.shared(report)

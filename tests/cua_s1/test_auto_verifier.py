"""Reject evidence that only selected Graph routes but never exercised them."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recipe/cua_s1"))
EVIDENCE = ROOT / "recipe/cua_s1/experiments/rtx4090-auto-worker"


def test_recorded_auto_parity_is_independently_verified():
    from verify_auto_worker import parity

    parity(json.loads((EVIDENCE / "parity.json").read_text()))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["churn_stats"].update(replays=0),
        lambda r: r["cases"][0]["stats_delta"].update(captures=0),
        lambda r: r["stats"].update(length_rejections=1),
        lambda r: r["logit_checks"][0].update(equal=False),
        lambda r: r["churn_events"][0]["actual"].update(unexpected=True),
        lambda r: r["cases"][0]["events"][0]["stats_delta"].update(requests=2),
    ],
)
def test_auto_verifier_rejects_corrupted_parity(mutation):
    from verify_auto_worker import parity

    report = copy.deepcopy(json.loads((EVIDENCE / "parity.json").read_text()))
    mutation(report)
    with pytest.raises(AssertionError):
        parity(report)


@pytest.fixture
def timing_report():
    return json.loads((EVIDENCE / "diagnostics/default-probe.json").read_text())


def test_recorded_auto_shared_budget_evidence(timing_report):
    from verify_auto_worker import verify_auto

    verify_auto(timing_report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["config"]["variant_graph"]["auto"].update(max_shapes=16),
        lambda r: r["workloads"]["probe"][0]["events"][0]["variants"]["auto"][
            "stats_delta"
        ].update(requests=2),
        lambda r: r["workloads"]["probe"][0]["events"][0]["variants"]["auto"][
            "stats_delta"
        ].update(selected_eager=0),
        lambda r: r["workloads"]["probe"][0]["events"][0]["variants"]["auto"].update(
            resident_modes=["exact"]
        ),
        lambda r: r["workloads"]["probe"][0]["events"][0]["variants"]["auto"][
            "stats_delta"
        ].update(capture_attempts=5),
    ],
)
def test_auto_verifier_rejects_clock_route_and_budget_corruption(
    timing_report, mutation
):
    from verify_auto_worker import verify_auto

    mutation(timing_report)
    with pytest.raises(AssertionError):
        verify_auto(timing_report)

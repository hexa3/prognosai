"""Unit tests for feature engineering, scoring and maintenance-decision logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prognosai.ai4i import add_features as ai4i_features
from prognosai.cmapss import RUL_CAP, add_features, load_subset, nasa_score
from prognosai.maintenance import CostModel, optimal_service_time, realized_cost, schedule


def test_ai4i_physics_features():
    row = pd.DataFrame([dict(quality="L", air_temp_k=300.0, process_temp_k=309.0, rpm=1500, torque_nm=40.0, tool_wear_min=100)])
    f = ai4i_features(row).iloc[0]
    assert f.temp_diff_k == 9.0
    assert np.isclose(f.power_w, 40 * 1500 * 2 * np.pi / 60)
    assert f.strain_nm_min == 4000
    assert np.isclose(f.strain_ratio, 4000 / 11000)  # L-grade limit


def test_cmapss_targets_and_no_leakage():
    train, test, truth = load_subset("FD001")
    assert train.rul.max() == RUL_CAP and train.rul.min() == 0
    # Test last-cycle RUL equals NASA's ground-truth file.
    last = test.groupby("unit").tail(1).sort_values("unit")
    assert np.array_equal(last.rul_true.to_numpy(), truth.rul.to_numpy().astype(float))
    # Features at cycle t must not change when future cycles are removed (no look-ahead).
    eng = train[train.unit == 1]
    full = add_features(eng)
    cut = add_features(eng[eng.cycle <= 50])
    cols = [c for c in cut.columns if c.endswith(("mean10", "slope20", "ewm"))]
    assert np.allclose(full.loc[full.cycle == 50, cols].to_numpy(), cut.loc[cut.cycle == 50, cols].to_numpy())


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40]) > 0
    assert nasa_score([50], [50]) == 0


def test_optimal_time_behaviour():
    cost = CostModel()
    t_far, _, _ = optimal_service_time(100, 8, cost)
    t_near, _, _ = optimal_service_time(10, 8, cost)
    t_uncertain, _, _ = optimal_service_time(100, 25, cost)
    assert t_near < t_far
    assert t_uncertain < t_far  # more uncertainty, service earlier
    assert t_far < 100  # never wait until the median failure time


def test_realized_cost():
    cost = CostModel(failure_cost=100, planned_cost=10, waste_per_cycle=1)
    assert list(realized_cost(np.array([5, 20]), np.array([10, 10]), cost)) == [15, 100]


def test_schedule_respects_capacity():
    fleet = pd.DataFrame({"unit": range(6), "rul_p10": [5] * 6, "rul_p50": [12] * 6, "rul_p90": [20] * 6})
    plan = schedule(fleet, CostModel(), crews_per_day=1, horizon_days=30)
    days = plan.service_day.dropna()
    assert days.value_counts().max() <= 1

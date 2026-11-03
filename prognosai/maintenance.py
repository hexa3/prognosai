"""Turning predictions into decisions: cost-optimal maintenance timing and crew scheduling.

A RUL prediction is only useful once it becomes a decision about *when* to
service an asset. We treat each asset's remaining life as a distribution taken
from the quantile models, and choose the maintenance time that minimises
expected cost:

    E[cost(t)] = P(RUL < t) * C_failure
               + E[ (C_planned + c_waste * (RUL - t)) * 1{RUL >= t} ]

Servicing too late risks a costly unplanned failure. Servicing too early
throws away useful life. The optimum balances the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z90 = 1.2815515655446004  # standard-normal 90th percentile


@dataclass(frozen=True)
class CostModel:
    failure_cost: float = 1_500_000.0   # ₹: unplanned failure (secondary damage + downtime + emergency crew)
    planned_cost: float = 200_000.0     # ₹: scheduled overhaul
    waste_per_cycle: float = 2_000.0    # ₹: value of each useful cycle thrown away by servicing early


def sigma_from_quantiles(p10, p90, floor: float = 3.0):
    return np.maximum((np.asarray(p90, float) - np.asarray(p10, float)) / (2 * Z90), floor)


def _normal_cdf(x):
    from math import erf, sqrt

    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x, float) / sqrt(2)))


def expected_cost_curve(mu: float, sigma: float, cost: CostModel, horizon: int | None = None):
    """Expected cost of servicing at each t = 0..horizon cycles from now."""
    horizon = int(horizon or max(mu + 4 * sigma, 10))
    t = np.arange(0, horizon + 1, dtype=float)
    r = np.linspace(max(mu - 5 * sigma, 0.0), mu + 5 * sigma, 801)  # RUL support
    pdf = np.exp(-0.5 * ((r - mu) / sigma) ** 2)
    pdf /= pdf.sum()
    fail = (r[None, :] < t[:, None]).astype(float)
    survive = 1.0 - fail
    planned = cost.planned_cost + cost.waste_per_cycle * np.clip(r[None, :] - t[:, None], 0, None)
    exp_cost = (fail * cost.failure_cost + survive * planned) @ pdf
    p_fail = fail @ pdf
    return t, exp_cost, p_fail


def optimal_service_time(mu: float, sigma: float, cost: CostModel) -> tuple[int, float, float]:
    t, c, p = expected_cost_curve(mu, sigma, cost)
    i = int(np.argmin(c))
    return int(t[i]), float(c[i]), float(p[i])


def realized_cost(service_t: np.ndarray, true_rul: np.ndarray, cost: CostModel) -> np.ndarray:
    """Cost actually paid when an asset with true RUL `true_rul` is serviced at `service_t`."""
    service_t = np.asarray(service_t, float)
    true_rul = np.asarray(true_rul, float)
    failed = service_t > true_rul
    planned = cost.planned_cost + cost.waste_per_cycle * np.clip(true_rul - service_t, 0, None)
    return np.where(failed, cost.failure_cost, planned)


def rolling_backtest(traj: pd.DataFrame, cost: CostModel, fixed_interval_age: float, lead_time: int = 5) -> pd.DataFrame:
    """Simulate online monitoring over complete run-to-failure lives.

    At every cycle the engine is re-scored (out-of-fold predictions, so each
    engine is scored by a model that never saw it). When the risk-optimal
    service time falls within `lead_time` cycles, a service is booked at that
    time. If the engine dies first, we pay for a failure.

    `traj` needs: unit, cycle, rul_p10, rul_p50, rul_p90 (rows sorted by cycle per unit).
    """
    out = {k: {"cost": 0.0, "failures": 0, "wasted": 0.0} for k in
           ["Reactive (run-to-failure)", "Fixed-interval preventive", "PrognosAI (risk-optimal)", "Perfect foresight (oracle)"]}
    curve_cache: dict[tuple[int, int], int] = {}
    for _, g in traj.sort_values(["unit", "cycle"]).groupby("unit"):
        life = float(g["cycle"].max())
        service_at = np.inf
        for c, p10, p50, p90 in g[["cycle", "rul_p10", "rul_p50", "rul_p90"]].itertuples(index=False):
            sig = float(sigma_from_quantiles(p10, p90))
            key = (int(round(p50)), int(round(sig)))
            if key not in curve_cache:
                curve_cache[key] = optimal_service_time(float(p50), sig, cost)[0]
            t = curve_cache[key]
            if t <= lead_time:
                service_at = float(c + t)
                break
        decisions = {
            "Reactive (run-to-failure)": np.inf,
            "Fixed-interval preventive": fixed_interval_age,
            "PrognosAI (risk-optimal)": service_at,
            "Perfect foresight (oracle)": life - 1,
        }
        for k, s in decisions.items():
            failed = s >= life
            out[k]["failures"] += int(failed)
            if failed:
                out[k]["cost"] += cost.failure_cost
            else:
                out[k]["cost"] += cost.planned_cost + cost.waste_per_cycle * (life - s)
                out[k]["wasted"] += life - s
    return pd.DataFrame([{"policy": k, "total_cost": v["cost"], "failures": v["failures"], "wasted_cycles": v["wasted"]}
                         for k, v in out.items()])


def schedule(fleet: pd.DataFrame, cost: CostModel, crews_per_day: int, horizon_days: int) -> pd.DataFrame:
    """Assign service slots to a fleet under a daily crew limit (1 cycle ≈ 1 day).

    Greedy by urgency: assets with the earliest optimal service time are placed
    first. When a day is full, the asset moves to the nearest *earlier* free day
    (an earlier slot is safer) and otherwise to the next later one.
    """
    rows = []
    for _, a in fleet.iterrows():
        sig = float(sigma_from_quantiles(a["rul_p10"], a["rul_p90"]))
        t, c, p = optimal_service_time(float(a["rul_p50"]), sig, cost)
        rows.append({"unit": a["unit"], "rul_p10": a["rul_p10"], "rul_p50": a["rul_p50"],
                     "rul_p90": a["rul_p90"], "sigma": sig, "t_opt": t})
    plan = pd.DataFrame(rows).sort_values(["t_opt", "rul_p10"]).reset_index(drop=True)
    load = np.zeros(horizon_days + 1, dtype=int)
    days, risks = [], []
    for _, a in plan.iterrows():
        target = int(a["t_opt"])
        if target > horizon_days:
            days.append(None)
            risks.append(float(_normal_cdf((horizon_days - a["rul_p50"]) / a["sigma"])))
            continue
        day = None
        for d in list(range(target, -1, -1)) + list(range(target + 1, horizon_days + 1)):
            if load[d] < crews_per_day:
                day = d
                break
        if day is not None:
            load[day] += 1
        days.append(day)
        ref = day if day is not None else horizon_days
        risks.append(float(_normal_cdf((ref - a["rul_p50"]) / a["sigma"])))
    plan["service_day"] = days
    plan["failure_risk_before_service"] = risks
    plan["status"] = np.where(
        plan["service_day"].isna(),
        np.where(plan["t_opt"] > horizon_days, "Monitor", "Over capacity"),
        np.where(plan["t_opt"] <= 7, "Urgent", "Scheduled"),
    )
    return plan

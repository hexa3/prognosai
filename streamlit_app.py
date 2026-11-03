"""PrognosAI: explainable predictive maintenance dashboard (Neural Storm 2026, FT-03).

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prognosai import DATA_DIR, MODEL_DIR, REPORT_DIR
from prognosai.advisor import FEATURE_LABELS, risk_band, work_order
from prognosai.ai4i import FAILURE_MODES, MODE_NAMES, QUALITY_CODE, add_features as ai4i_features, load_raw
from prognosai.cmapss import COLS, RUL_CAP, SENSOR_NAMES, add_features as cmapss_features, conformal_from_dict
from prognosai.maintenance import CostModel, expected_cost_curve, schedule, sigma_from_quantiles

st.set_page_config(page_title="PrognosAI · Predictive Maintenance", page_icon=":material/precision_manufacturing:", layout="wide")

BLUE, ORANGE, RED, GREEN, GREY = "#2563eb", "#f59e0b", "#dc2626", "#16a34a", "#94a3b8"
STATUS_COLORS = {"Critical": RED, "Warning": ORANGE, "Healthy": GREEN}


# ----------------------------------------------------------------------------- loaders
class Booster:
    """Thin wrapper around a LightGBM text model (portable across LightGBM 4.x, including in-browser Pyodide)."""

    def __init__(self, path):
        self.booster = lgb.Booster(model_file=str(path))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X.to_numpy(dtype=float))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict(X)
        return np.c_[1 - p, p]

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values (log-odds), computed natively by LightGBM. Last column is the bias."""
        return self.booster.predict(X.to_numpy(dtype=float), pred_contrib=True)


@st.cache_resource
def failure_bundle():
    meta = json.loads((MODEL_DIR / "failure_model_meta.json").read_text())
    return {**meta, "model": Booster(MODEL_DIR / "failure_model.txt"),
            "mode_models": {m: Booster(MODEL_DIR / f"failure_mode_{m}.txt") for m in FAILURE_MODES}}


@st.cache_resource
def rul_bundle(subset: str):
    meta = json.loads((MODEL_DIR / f"rul_{subset}_meta.json").read_text())
    return {"point": Booster(MODEL_DIR / f"rul_{subset}.txt"), "features": meta["features"],
            "conformal": conformal_from_dict(meta["conformal"])}


@st.cache_data
def metrics():
    return (json.loads((MODEL_DIR / "failure_metrics.json").read_text()),
            json.loads((MODEL_DIR / "rul_metrics.json").read_text()))


@st.cache_data
def ai4i_data():
    return load_raw()


@st.cache_data
def fleet_predictions(subset: str) -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / f"rul_{subset}_test_predictions.csv")


def fleet_snapshot(subset: str) -> pd.DataFrame:
    df = fleet_predictions(subset)
    last = df.groupby("unit").tail(1).copy()
    last["status"] = np.select([last.rul_p10 <= 20, last.rul_p50 <= 60], ["Critical", "Warning"], "Healthy")
    return last.reset_index(drop=True)


def inr(x: float) -> str:
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    return f"₹{x:,.0f}"


def shap_row(X: pd.DataFrame) -> np.ndarray:
    return failure_bundle()["model"].contributions(X)[0, :-1]


def gauge(p: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Failure probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED if p >= threshold else BLUE},
               "steps": [{"range": [0, threshold * 100 / 3], "color": "rgba(22,163,74,.15)"},
                         {"range": [threshold * 100 / 3, threshold * 100], "color": "rgba(245,158,11,.18)"},
                         {"range": [threshold * 100, 100], "color": "rgba(220,38,38,.15)"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}}))
    fig.update_layout(height=270, margin=dict(l=35, r=35, t=60, b=10))
    return fig


# ----------------------------------------------------------------------------- pages
def page_overview():
    fm, rm = metrics()
    st.title("PrognosAI")
    st.markdown("#### Predict failures before they happen. Explain why. Schedule the fix at the lowest cost.")
    st.caption("Neural Storm 2026 · Track 1 Future Tech · FT-03 Predictive Maintenance for Industry")

    final = fm["final"]["cost_optimal"]
    f1 = rm["subsets"]["FD001"]["final"]
    rb = {r["policy"]: r for r in rm["subsets"]["FD001"]["rolling_backtest"]}
    saving = 1 - rb["PrognosAI (risk-optimal)"]["total_cost"] / rb["Fixed-interval preventive"]["total_cost"]
    vs_reactive = 1 - rb["PrognosAI (risk-optimal)"]["total_cost"] / rb["Reactive (run-to-failure)"]["total_cost"]

    c = st.columns(4)
    c[0].metric("Failure detection ROC-AUC", f"{final['roc_auc']:.3f}", help="AI4I 2020 held-out test set (2,000 machines)")
    c[1].metric("Failure precision / recall", f"{final['precision']:.0%} / {final['recall']:.0%}")
    c[2].metric("Turbofan RUL error (RMSE)", f"{f1['test_rmse']:.1f} cycles",
                delta=f"{f1['test_rmse'] - 12.61:+.1f} vs published DCNN", delta_color="inverse",
                help="NASA C-MAPSS FD001 official test set; Li et al. 2018 DCNN = 12.61")
    c[3].metric("Maintenance cost saved", f"{saving:.0%}", delta=f"{vs_reactive:.0%} vs run-to-failure",
                help="Rolling backtest on 100 engines vs a tuned fixed-interval schedule")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("The problem")
        st.markdown(
            """
Unplanned downtime is one of the largest hidden costs in manufacturing. Industry surveys put it at
**5–20% of productive capacity**. Indian MSMEs feel it most: there is a single CNC line, no spare machine,
and a broken spindle stops the order book.

Plants usually choose between two bad options:
* **Run-to-failure:** cheap until the expensive breakdown.
* **Fixed-interval servicing:** safe, but it throws away healthy component life and labour.

**PrognosAI** gives a third option: maintenance driven by condition, risk and cost.
"""
        )
    with right:
        st.subheader("How it works")
        st.markdown(
            """
1. **Detect:** a gradient-boosted classifier with *physics-informed features* (power, thermal gap, strain)
   flags machines at risk and identifies the failure mechanism.
2. **Explain:** SHAP attributions show *why*, and a technician-ready work order is generated.
3. **Forecast:** a temporal model predicts *remaining useful life* with **conformally calibrated**
   80% intervals.
4. **Decide:** a risk-optimal planner converts the forecast distribution into the service day that
   minimises expected ₹ cost, within crew capacity.
"""
        )
    st.info("Use the sidebar: **Machine Risk** for live what-if, **Fleet Monitor** for turbofan RUL, "
            "**Maintenance Planner** for the ₹ schedule, **Batch Scoring** to upload your own data.", icon=":material/info:")

    st.subheader("Backtest: what each maintenance policy would have cost")
    bt = pd.DataFrame(rm["subsets"]["FD001"]["rolling_backtest"])
    fig = go.Figure(go.Bar(
        x=bt["total_cost"] / 1e7, y=bt["policy"], orientation="h",
        marker_color=[GREY, ORANGE, BLUE, GREEN],
        text=[f"{inr(c)} · {f} failures" for c, f in zip(bt.total_cost, bt.failures)], textposition="auto"))
    fig.update_layout(height=280, xaxis_title="Total cost over 100 engine lives (₹ crore)", yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("NASA C-MAPSS FD001. Each engine is scored every cycle by a model that never saw it (out-of-fold). "
               "Costs: failure ₹15 L, planned overhaul ₹2 L, ₹2,000 per wasted cycle of life. "
               "The fixed-interval baseline is tuned in-sample, which is the most favourable setting for it.")


def page_machine():
    b = failure_bundle()
    thr = b["threshold_cost"]
    st.title("Machine failure risk")
    st.caption("CNC milling model trained on AI4I 2020 (10,000 operations). Move the sliders to run a live what-if.")

    data = ai4i_data()
    presets = {
        "Healthy machine": dict(quality="M", air=298.0, proc=308.5, rpm=1500, torque=40.0, wear=60),
        "Worn tool under heavy load": dict(quality="L", air=300.0, proc=310.2, rpm=1350, torque=62.0, wear=205),
        "Poor heat dissipation": dict(quality="M", air=302.5, proc=310.6, rpm=1320, torque=48.0, wear=90),
        "Over-powered spindle": dict(quality="H", air=299.0, proc=309.5, rpm=2700, torque=15.0, wear=40),
    }
    real_fail = data[data.failure == 1].sample(1, random_state=7).iloc[0]
    presets["Real failure from dataset"] = dict(quality=real_fail.quality, air=float(real_fail.air_temp_k),
                                                proc=float(real_fail.process_temp_k), rpm=int(real_fail.rpm),
                                                torque=float(real_fail.torque_nm), wear=int(real_fail.tool_wear_min))
    name = st.segmented_control("Scenario", list(presets), default="Worn tool under heavy load")
    p = presets[name or "Healthy machine"]

    with st.container(border=True):
        c = st.columns(6)
        quality = c[0].selectbox("Product quality", ["L", "M", "H"], index=["L", "M", "H"].index(p["quality"]), key=f"q{name}")
        air = c[1].slider("Air temp (K)", 295.0, 305.0, p["air"], 0.1, key=f"a{name}")
        proc = c[2].slider("Process temp (K)", 305.0, 314.0, p["proc"], 0.1, key=f"p{name}")
        rpm = c[3].slider("Spindle speed (rpm)", 1150, 2900, p["rpm"], 10, key=f"r{name}")
        torque = c[4].slider("Torque (Nm)", 3.0, 77.0, p["torque"], 0.5, key=f"t{name}")
        wear = c[5].slider("Tool wear (min)", 0, 255, p["wear"], 1, key=f"w{name}")

    row = ai4i_features(pd.DataFrame([dict(quality=quality, quality_code=QUALITY_CODE[quality], air_temp_k=air,
                                            process_temp_k=proc, rpm=rpm, torque_nm=torque, tool_wear_min=wear)]))
    X = row[b["features"]]
    p_fail = float(b["model"].predict_proba(X)[0, 1])
    modes = {m: float(b["mode_models"][m].predict_proba(X)[0, 1]) for m in FAILURE_MODES}
    band, action = risk_band(p_fail, thr)

    g1, g2, g3 = st.columns([1, 1, 1.2])
    with g1:
        st.plotly_chart(gauge(p_fail, thr), width="stretch")
        st.markdown(f"**Risk level: {band}**  \n{action}")
    with g2:
        st.markdown("**Failure mechanism**")
        fig = go.Figure(go.Bar(x=[modes[m] * 100 for m in FAILURE_MODES], y=[MODE_NAMES[m] for m in FAILURE_MODES],
                               orientation="h", marker_color=[RED if modes[m] > 0.5 else BLUE for m in FAILURE_MODES],
                               text=[f"{modes[m]:.1%}" for m in FAILURE_MODES], textposition="auto"))
        fig.update_layout(height=230, xaxis=dict(range=[0, 100], title="probability %"), margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig, width="stretch")
    with g3:
        st.markdown("**Physics features (engineered)**")
        st.dataframe(pd.DataFrame({
            "feature": ["Mechanical power", "Thermal gap", "Strain (wear × torque)", "Strain vs limit", "Tool life used"],
            "value": [f"{row.power_w[0]:,.0f} W", f"{row.temp_diff_k[0]:.1f} K", f"{row.strain_nm_min[0]:,.0f} min·Nm",
                      f"{row.strain_ratio[0]:.0%}", f"{row.wear_ratio[0]:.0%}"],
            "safe zone": ["3,500–9,000 W", "> 8.6 K or rpm > 1,380", "below limit", "< 100%", "< 83%"],
        }), hide_index=True, width="stretch")

    sv = shap_row(X)
    order = np.argsort(-np.abs(sv))[:8][::-1]
    st.subheader("Why? Explanation for this prediction")
    fig = go.Figure(go.Bar(x=sv[order], y=[f"{FEATURE_LABELS[b['features'][i]]} = {X.iloc[0, i]:,.1f}" for i in order],
                           orientation="h", marker_color=[RED if v > 0 else GREEN for v in sv[order]]))
    fig.update_layout(height=330, xaxis_title="SHAP contribution to failure log-odds (red raises risk, green lowers it)",
                      margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch")

    drivers = [(b["features"][i], float(X.iloc[0, i]), float(sv[i])) for i in np.argsort(-np.abs(sv))[:4]]
    wo = work_order("CNC-07 (Line 2)", p_fail, thr, modes, drivers)
    with st.expander("Auto-generated work order", expanded=band != "NORMAL", icon=":material/assignment:"):
        st.code(wo, language=None)
        st.download_button("Download work order", wo, file_name="work_order.txt")


def page_fleet():
    st.title("Fleet monitor: turbofan remaining useful life")
    subset = st.segmented_control("Dataset", ["FD001", "FD003"], default="FD001",
                                  help="FD001: 1 fault mode (HPC). FD003: 2 fault modes (HPC + fan).") or "FD001"
    snap = fleet_snapshot(subset)
    counts = snap.status.value_counts()
    c = st.columns(4)
    c[0].metric("Engines monitored", len(snap))
    c[1].metric("Critical (P10 RUL ≤ 20)", int(counts.get("Critical", 0)))
    c[2].metric("Warning (median RUL ≤ 60)", int(counts.get("Warning", 0)))
    c[3].metric("Healthy", int(counts.get("Healthy", 0)))

    s = snap.sort_values("rul_p50")
    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        d = s[s.status == status]
        fig.add_trace(go.Scatter(x=d.unit.astype(str), y=d.rul_p50, mode="markers", name=status, marker=dict(color=color, size=8),
                                 error_y=dict(type="data", symmetric=False, array=d.rul_p90 - d.rul_p50,
                                              arrayminus=d.rul_p50 - d.rul_p10, color=color, thickness=1),
                                 hovertemplate="Engine %{x}<br>RUL %{y:.0f} cycles<extra></extra>"))
    fig.update_layout(height=360, xaxis=dict(title="Engine (sorted by predicted RUL)", type="category", categoryorder="array",
                                             categoryarray=s.unit.astype(str).tolist(), showticklabels=False),
                      yaxis_title="Predicted RUL (cycles), 80% interval", margin=dict(l=10, r=10, t=10, b=40),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Engine drill-down")
    unit = st.selectbox("Engine", s.unit.tolist(), index=0, format_func=lambda u: f"Engine {u}  ·  {snap.set_index('unit').loc[u, 'status']}")
    hist = fleet_predictions(subset).query("unit == @unit")
    last = hist.iloc[-1]
    c = st.columns(4)
    c[0].metric("Cycles flown", int(last.cycle))
    c[1].metric("Predicted RUL", f"{last.rul_p50:.0f} cycles")
    c[2].metric("80% interval", f"{last.rul_p10:.0f}–{last.rul_p90:.0f}")
    c[3].metric("Actual RUL (NASA ground truth)", f"{int(last.rul_true)} cycles")

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.cycle, y=hist.rul_p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.cycle, y=hist.rul_p10, fill="tonexty", fillcolor="rgba(37,99,235,.18)",
                                 line=dict(width=0), name="80% interval"))
        fig.add_trace(go.Scatter(x=hist.cycle, y=hist.rul_p50, line=dict(color=BLUE, width=2), name="Predicted"))
        fig.add_trace(go.Scatter(x=hist.cycle, y=hist.rul, line=dict(color="black", dash="dash"), name=f"True (capped at {RUL_CAP})"))
        fig.update_layout(height=340, title="RUL estimate over the engine's life", xaxis_title="Cycle",
                          yaxis_title="RUL (cycles)", margin=dict(l=10, r=10, t=40, b=40), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with r:
        sensors = st.multiselect("Sensors", list(SENSOR_NAMES), default=["s4", "s11", "s12", "s7"],
                                 format_func=lambda k: f"{k}: {SENSOR_NAMES[k]}")
        fig = go.Figure()
        for sname in sensors:
            v = hist[sname]
            z = (v - v.iloc[:10].mean()) / (v.std() or 1)
            fig.add_trace(go.Scatter(x=hist.cycle, y=z.rolling(5, min_periods=1).mean(), name=sname))
        fig.update_layout(height=340, title="Sensor drift from this engine's baseline (z-score, 5-cycle mean)",
                          xaxis_title="Cycle", margin=dict(l=10, r=10, t=40, b=40), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Full fleet table"):
        st.dataframe(snap[["unit", "cycle", "rul_p10", "rul_p50", "rul_p90", "rul_true", "status"]].sort_values("rul_p50")
                     .rename(columns={"rul_p10": "RUL P10", "rul_p50": "RUL P50", "rul_p90": "RUL P90", "rul_true": "actual RUL"})
                     .round(1), hide_index=True, width="stretch")


def page_planner():
    st.title("Maintenance planner")
    st.caption("Turns RUL distributions into the service day that minimises expected cost, within crew capacity. 1 cycle ≈ 1 operating day.")
    with st.sidebar:
        st.subheader("Planner settings")
        subset = st.radio("Fleet", ["FD001", "FD003"], horizontal=True)
        fail_cost = st.number_input("Unplanned failure cost (₹)", 100_000, 10_000_000, 1_500_000, 100_000)
        planned_cost = st.number_input("Planned overhaul cost (₹)", 10_000, 5_000_000, 200_000, 10_000)
        waste = st.number_input("Value of 1 cycle of life (₹)", 0, 100_000, 2_000, 500)
        crews = st.slider("Service slots per day", 1, 6, 3)
        horizon = st.slider("Planning horizon (days)", 7, 90, 30)
    cost = CostModel(fail_cost, planned_cost, waste)
    snap = fleet_snapshot(subset)
    plan = schedule(snap, cost, crews, horizon)
    plan = plan.merge(snap[["unit", "rul_true"]], on="unit")

    scheduled = plan.dropna(subset=["service_day"])
    c = st.columns(4)
    c[0].metric("Services in horizon", len(scheduled))
    c[1].metric("Urgent (≤ 7 days)", int((plan.status == "Urgent").sum()))
    c[2].metric("Expected failures before service", f"{plan.loc[plan.service_day.notna(), 'failure_risk_before_service'].sum():.2f}")
    c[3].metric("Planned spend", inr(len(scheduled) * planned_cost))

    if len(scheduled):
        fig = go.Figure()
        for status, color in {"Urgent": RED, "Scheduled": BLUE}.items():
            d = scheduled[scheduled.status == status]
            fig.add_trace(go.Scatter(
                x=d.service_day, y=d.unit.astype(str), mode="markers", name=status,
                marker=dict(size=14, color=color, symbol="square"),
                error_x=dict(type="data", symmetric=False, array=np.clip(d.rul_p90 - d.service_day, 0, None),
                             arrayminus=np.zeros(len(d)), color=GREY, thickness=1.5),
                customdata=np.c_[d.rul_p10, d.rul_p50, d.failure_risk_before_service * 100],
                hovertemplate="Engine %{y}<br>Service day %{x}<br>RUL P10/P50: %{customdata[0]:.0f}/%{customdata[1]:.0f}"
                              "<br>Failure risk before service: %{customdata[2]:.1f}%<extra></extra>"))
        fig.update_layout(height=max(300, 22 * len(scheduled)), xaxis=dict(title="Day", range=[-1, horizon + 1]),
                          yaxis=dict(title="Engine", type="category", autorange="reversed"),
                          margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch")
        st.caption("Squares show the booked service day. Grey whiskers run to the P90 RUL, the remaining life you keep by servicing at the optimum.")

    l, r = st.columns([1.2, 1])
    with l:
        st.markdown("**Work schedule**")
        st.dataframe(plan[["unit", "status", "service_day", "t_opt", "rul_p10", "rul_p50", "rul_p90", "failure_risk_before_service", "rul_true"]]
                     .rename(columns={"t_opt": "optimal day", "failure_risk_before_service": "P(fail first)", "rul_true": "actual RUL"})
                     .round(3), hide_index=True, width="stretch", height=380)
        st.download_button("Export schedule (CSV)", plan.to_csv(index=False), file_name=f"maintenance_plan_{subset}.csv")
    with r:
        unit = st.selectbox("Why this day? Cost curve for engine", plan.unit.tolist())
        a = plan.set_index("unit").loc[unit]
        t, ec, pf = expected_cost_curve(float(a.rul_p50), float(a.sigma), cost)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ec / 1e5, name="Expected cost (₹ L)", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=t, y=pf * 100, name="P(failure first) %", yaxis="y2", line=dict(color=RED, dash="dot")))
        fig.add_vline(x=int(a.t_opt), line_dash="dash", annotation_text=f"optimum: day {int(a.t_opt)}")
        fig.update_layout(height=360, xaxis_title="Service in N days", yaxis_title="Expected cost (₹ lakh)",
                          yaxis2=dict(overlaying="y", side="right", title="Failure risk %", range=[0, 100]),
                          margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        st.caption("Servicing early wastes component life, and waiting too long risks a ₹15 L breakdown. The planner picks the minimum.")


def page_batch():
    st.title("Batch scoring")
    tab1, tab2 = st.tabs(["CNC machines (AI4I format)", "Turbofan engines (C-MAPSS format)"])
    b = failure_bundle()
    with tab1:
        st.markdown("Upload a CSV with columns `Type`, `Air temperature [K]`, `Process temperature [K]`, "
                    "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.")
        sample = pd.read_csv(DATA_DIR / "ai4i" / "ai4i2020.csv", encoding="utf-8-sig").sample(200, random_state=1)
        sample = sample[["Product ID", "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                         "Torque [Nm]", "Tool wear [min]"]]
        st.download_button("Download a sample file (200 machines)", sample.to_csv(index=False), "sample_machines.csv")
        up = st.file_uploader("Machine readings CSV", type="csv", key="ai4i")
        use_sample = st.toggle("Use the sample file", value=up is None)
        if up is not None or use_sample:
            raw = pd.read_csv(up, encoding="utf-8-sig") if up is not None and not use_sample else sample
            df = raw.rename(columns={"Type": "quality", "Air temperature [K]": "air_temp_k",
                                     "Process temperature [K]": "process_temp_k", "Rotational speed [rpm]": "rpm",
                                     "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min"})
            try:
                df = ai4i_features(df)
                X = df[b["features"]]
            except Exception as e:  # noqa: BLE001: surface schema problems to the user
                st.error(f"Could not parse file: {e}")
                return
            res = raw.copy()
            res["failure_probability"] = b["model"].predict_proba(X)[:, 1]
            mp = np.column_stack([b["mode_models"][m].predict_proba(X)[:, 1] for m in FAILURE_MODES])
            res["likely_mechanism"] = [MODE_NAMES[FAILURE_MODES[i]] for i in mp.argmax(1)]
            res["risk_level"] = [risk_band(p, b["threshold_cost"])[0] for p in res.failure_probability]
            res = res.sort_values("failure_probability", ascending=False)
            c = st.columns(3)
            c[0].metric("Machines scored", len(res))
            c[1].metric("Flagged (HIGH/CRITICAL)", int(res.risk_level.isin(["HIGH", "CRITICAL"]).sum()))
            c[2].metric("Mean risk", f"{res.failure_probability.mean():.1%}")
            st.dataframe(res, hide_index=True, width="stretch",
                         column_config={"failure_probability": st.column_config.ProgressColumn("failure probability", min_value=0, max_value=1, format="percent")})
            st.download_button("Download scored file", res.to_csv(index=False), "scored_machines.csv")
    with tab2:
        st.markdown("Upload a NASA C-MAPSS style space-separated file (26 columns: unit, cycle, 3 settings, 21 sensors), "
                    "for example `test_FD001.txt`.")
        subset = st.radio("Model", ["FD001", "FD003"], horizontal=True)
        up = st.file_uploader("Engine telemetry", type=["txt", "csv"], key="cmapss")
        use_sample = st.toggle("Use NASA test_FD001 sample (first 20 engines)", value=up is None)
        if up is not None or use_sample:
            if up is not None and not use_sample:
                raw = pd.read_csv(io.BytesIO(up.getvalue()), sep=r"\s+", header=None).iloc[:, :26]
            else:
                raw = pd.read_csv(DATA_DIR / "cmapss" / "test_FD001.txt", sep=r"\s+", header=None).iloc[:, :26]
                raw = raw[raw[0] <= 20]
            raw.columns = COLS
            with st.spinner("Engineering temporal features…"):
                feats = cmapss_features(raw)
            rb = rul_bundle(subset)
            last = feats.groupby("unit").tail(1).copy()
            Xr = last[rb["features"]]
            p50 = np.clip(rb["point"].predict(Xr), 0, RUL_CAP)
            last["RUL P10"], last["RUL P90"] = rb["conformal"].interval(p50)
            last["RUL P50"] = p50
            out = last[["unit", "cycle", "RUL P10", "RUL P50", "RUL P90"]].round(1).sort_values("RUL P50")
            st.dataframe(out, hide_index=True, width="stretch")
            st.download_button("Download RUL forecast", out.to_csv(index=False), "rul_forecast.csv")


def page_model_card():
    fm, rm = metrics()
    st.title("Model card & evaluation")
    t1, t2, t3 = st.tabs(["Failure classifier (AI4I)", "RUL forecaster (C-MAPSS)", "Limitations & responsible use"])
    with t1:
        st.markdown(f"**Data:** {fm['dataset']}. Stratified 80/20 split ({fm['split']['test']} test rows, "
                    f"{fm['split']['test_failures']} failures). Threshold chosen on 5-fold out-of-fold predictions, never on test.")
        ab = pd.DataFrame(fm["ablation"])[["model", "roc_auc", "pr_auc", "f1", "precision", "recall", "cv_pr_auc"]]
        st.markdown("**Ablation (held-out test set)**")
        st.dataframe(ab.round(4), hide_index=True, width="stretch")
        fin = pd.DataFrame(fm["final"]["cost_optimal"], index=["value"]).T
        c = st.columns([1, 2])
        c[0].markdown("**Final model @ cost-optimal threshold**")
        c[0].dataframe(fin, width="stretch")
        c[1].image(str(REPORT_DIR / "figures" / "failure_curves.png"))
        st.markdown("**Failure mechanism (root-cause) models**")
        st.dataframe(pd.DataFrame(fm["failure_modes"]).T.rename(index=MODE_NAMES).round(4), width="stretch")
        st.image(str(REPORT_DIR / "figures" / "failure_shap_summary.png"), width=700)
    with t2:
        for subset, r in rm["subsets"].items():
            st.markdown(f"### {subset}: {r['train_engines']} train / {r['test_engines']} test engines · {r['n_features']} features")
            c = st.columns(5)
            f = r["final"]
            c[0].metric("Test RMSE", f"{f['test_rmse']:.2f}")
            c[1].metric("Test MAE", f"{f['test_mae']:.2f}")
            c[2].metric("NASA score", f"{f['test_nasa_score']:.0f}")
            c[3].metric("80% interval coverage", f"{f['interval_80_coverage_all_rows']:.1%}",
                        help=f"Over all {f['test_rows']:,} test predictions; {f['interval_80_coverage']:.0%} on the 100 final-cycle snapshot")
            c[4].metric("RMSE when RUL ≤ 50", f"{f['rmse_true_rul_le_50']:.2f}")
            rows = [{"model": a["model"], "test RMSE": a["test_rmse"], "CV RMSE": a["cv_rmse"], "NASA score": a["test_nasa_score"]} for a in r["ablation"]]
            rows += [{"model": f"{k} (published)", "test RMSE": v, "CV RMSE": None, "NASA score": None} for k, v in r["literature_rmse"].items()]
            st.dataframe(pd.DataFrame(rows).sort_values("test RMSE").round(2), hide_index=True, width="stretch")
            st.image(str(REPORT_DIR / "figures" / f"rul_{subset}.png"))
            st.markdown("**Policy backtest (rolling, out-of-fold, full lives)**")
            st.dataframe(pd.DataFrame(r["rolling_backtest"]).assign(total_cost=lambda d: d.total_cost.map(inr)), hide_index=True, width="stretch")
            st.divider()
    with t3:
        st.markdown(
            """
* **Datasets are benchmarks.** AI4I 2020 is a synthetic dataset that mirrors real milling physics. C-MAPSS is NASA's
  high-fidelity turbofan simulation. Both are standard research benchmarks, and each new plant needs a calibration period on its own data.
* **Random failures are unpredictable by design.** AI4I has a "random failure" (RNF) mode with no precursor signal.
  It is excluded from mechanism models, and most of its failures are missed by every model.
* **Tool-wear failures are hard to time.** They happen at a random point in the 200–240 min window, so the model flags the
  *window* well but has low precision on the exact failure.
* **RUL cap.** Predictions are capped at 125 cycles. Early-life degradation is not observable, so a healthy engine reads "≥125".
* **Human in the loop.** PrognosAI recommends and explains, and a qualified engineer approves every work order. Safety-critical
  equipment (aviation) must follow regulator-approved maintenance programmes. This tool supplements them and does not replace them.
* **Cost parameters are assumptions.** Change them in the planner: the decisions re-optimise instantly.
"""
        )


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
    st.Page(page_machine, title="Machine Risk", icon=":material/precision_manufacturing:", url_path="machine"),
    st.Page(page_fleet, title="Fleet Monitor", icon=":material/flight:", url_path="fleet"),
    st.Page(page_planner, title="Maintenance Planner", icon=":material/calendar_month:", url_path="planner"),
    st.Page(page_batch, title="Batch Scoring", icon=":material/upload_file:", url_path="batch"),
    st.Page(page_model_card, title="Model Card", icon=":material/analytics:", url_path="model-card"),
])
with st.sidebar:
    st.caption("PrognosAI v1.0 · Neural Storm 2026 · FT-03")
pg.run()

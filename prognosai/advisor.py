"""Turn model output into a maintenance work order a technician can act on."""

from __future__ import annotations

from .ai4i import MODE_NAMES

ACTIONS = {
    "TWF": [
        "Replace or regrind the cutting tool before the next batch.",
        "Check tool life counter; tools beyond ~200 min are in the failure zone.",
    ],
    "HDF": [
        "Inspect coolant flow and spindle ventilation; clean heat-exchanger fins.",
        "Raise spindle speed above ~1,380 rpm or let the process temperature settle.",
    ],
    "PWF": [
        "Spindle power is outside the 3.5–9 kW safe band: review feed rate and depth of cut.",
        "Check drive and motor load; verify torque setpoint for this job.",
    ],
    "OSF": [
        "Reduce torque (lighter cut) or swap to a fresh tool: wear × torque is near the overstrain limit.",
        "For L-grade products the limit is lowest (11,000 min·Nm); prioritise these jobs.",
    ],
}

FEATURE_LABELS = {
    "air_temp_k": "Air temperature",
    "process_temp_k": "Process temperature",
    "rpm": "Spindle speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
    "quality_code": "Product quality grade",
    "temp_diff_k": "Process−air temperature gap",
    "power_w": "Mechanical power",
    "strain_nm_min": "Strain (wear × torque)",
    "strain_ratio": "Strain vs. overstrain limit",
    "power_margin_w": "Margin to power limits",
    "wear_ratio": "Tool life used",
}


def risk_band(p: float, threshold: float) -> tuple[str, str]:
    if p >= max(threshold, 0.5):
        return "CRITICAL", "Stop and service before next run"
    if p >= threshold:
        return "HIGH", "Schedule service this shift"
    if p >= threshold / 3:
        return "ELEVATED", "Inspect at next planned stop"
    return "NORMAL", "Continue operation"


def work_order(machine_id: str, p_fail: float, threshold: float, mode_probs: dict[str, float],
               top_drivers: list[tuple[str, float, float]]) -> str:
    band, action = risk_band(p_fail, threshold)
    lines = [
        f"WORK ORDER: {machine_id}",
        f"Risk level: {band} (failure probability {p_fail:.1%}, alarm threshold {threshold:.0%})",
        f"Recommended action: {action}",
        "",
        "Likely failure mechanism:",
    ]
    ranked = sorted(mode_probs.items(), key=lambda kv: -kv[1])
    for mode, p in ranked[:2]:
        lines.append(f"  • {MODE_NAMES[mode]}: {p:.1%}")
    lines += ["", "Why the model flagged this machine (top SHAP drivers):"]
    for feat, value, contrib in top_drivers:
        arrow = "raises" if contrib > 0 else "lowers"
        lines.append(f"  • {FEATURE_LABELS.get(feat, feat)} = {value:,.2f} → {arrow} risk ({contrib:+.2f} log-odds)")
    if band != "NORMAL":
        lines += ["", "Checklist:"]
        top_mode = ranked[0][0]
        lines += [f"  ☐ {a}" for a in ACTIONS[top_mode]]
        lines.append("  ☐ Log findings in CMMS and close the work order after the test run.")
    return "\n".join(lines)

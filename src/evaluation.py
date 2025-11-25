# src/evaluation.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Any
from .degradation_models import calendar_fade_Ah, cycle_fade_Ah_from_DoD

def kpi_annual_cost(df: pd.DataFrame, dt_h: float) -> Dict[str, Any]:
    step_cost = (df["pimp"]*df["price_import_gbp_per_kwh"] - df["pexp"]*df["price_export_gbp_per_kwh"]) * dt_h
    return {
        "annual_cost_gbp": float(step_cost.sum()),
        "mean_hourly_cost_gbp": float(step_cost.resample("h").sum().mean() if isinstance(df.index, pd.DatetimeIndex) else step_cost.mean()),
    }

def kpi_lifecycle(df: pd.DataFrame) -> Dict[str, Any]:
    dt_h = float(df.attrs.get("dt_h", 0.25))
    e_nom = max(float(df.attrs.get("e_nom_kwh", 6.5)), 1.0)
    dis_kwh = float(df["pdis"].clip(lower=0.0).sum()) * dt_h
    efc = float(dis_kwh / e_nom)
    return {"equivalent_full_cycles": efc, "battery_throughput_kwh": dis_kwh}

def kpi_capacity_fade_and_cost(df: pd.DataFrame, conf: dict) -> Dict[str, Any]:
    dt_h = float(df.attrs.get("dt_h", 0.25))
    b = conf.get("battery", {})
    k_cal  = float(b.get("k_cal", 1.2e-5)); k_cyc = float(b.get("k_cyc", 2.5e-4))
    alpha  = float(b.get("alpha", 1.5));    q_nom = float(b.get("q_nom_Ah", 1000.0))
    repl   = float(b.get("replacement_cost_gbp", 3500.0))
    soc = df["soc"].astype(float).values
    temp = df["cell_temp_c"].values if "cell_temp_c" in df.columns else np.full(len(df), 25.0)
    dQ_cal = np.array([calendar_fade_Ah(dt_h, float(s), float(T), k_cal=k_cal, Ea_over_R=4000.0) for s,T in zip(soc,temp)])
    dsoc = np.abs(np.diff(soc, prepend=soc[0])); DoD = np.clip(dsoc, 0.0, 1.0)
    dQ_cyc = np.array([cycle_fade_Ah_from_DoD([float(d)], k_cyc=k_cyc, alpha=alpha) for d in DoD])
    fade_frac = float((dQ_cal + dQ_cyc).sum() / max(q_nom,1.0))
    return {"capacity_fade_pct": 100.0*fade_frac, "batt_deg_cost_gbp": float(fade_frac*repl),
            "pv_deg_cost_gbp": float(df["deg_cost_pv"].sum()) if "deg_cost_pv" in df.columns else 0.0}

def kpi_environmental(df: pd.DataFrame, dt_h: float) -> Dict[str, Any]:
    if "carbon_intensity" not in df.columns: return {"co2_avoided_kg": None}
    pv = df["pv_kw_eff"] if "pv_kw_eff" in df.columns else df.get("pv_kw_raw", 0.0)
    avoided = (df["pdis"] + pv - df["pimp"]).clip(lower=0.0) * df["carbon_intensity"] * dt_h
    return {"co2_avoided_kg": float(avoided.sum())}

def summarize_kpis(df: pd.DataFrame, dt_h: float, e_nom_kwh: float, conf: dict | None = None) -> Dict[str, Any]:
    df = df.copy(); df.attrs["dt_h"] = float(dt_h); df.attrs["e_nom_kwh"] = float(e_nom_kwh)
    out: Dict[str, Any] = {}
    out.update(kpi_annual_cost(df, dt_h))
    out.update(kpi_lifecycle(df))
    out.update(kpi_environmental(df, dt_h))
    if conf is not None:
        out.update(kpi_capacity_fade_and_cost(df, conf))
    return out

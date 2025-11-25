# src/data_generator.py
from __future__ import annotations
import os, yaml, numpy as np, pandas as pd
from typing import Any, Optional

__all__ = ["generate_time_index", "build_dataframe"]

def _get(cfg: Optional[dict], path: str, default: Any) -> Any:
    cur = cfg or {}
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def generate_time_index(start="2024-01-01", periods=365*96, freq="15min"):
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    try: idx.freq = pd.tseries.frequencies.to_offset(freq)
    except: pass
    return idx

def _load(idx, base=0.45, peak=2.8, noise=0.2, seed=42):
    tmins = idx.hour*60 + idx.minute
    d = tmins/(24*60.0)
    morning = base + (peak-base)*np.exp(-((d-0.20)**2)/(2*0.03))
    evening = (peak-base)*np.exp(-((d-0.83)**2)/(2*0.02))
    weekly = 1.0 + 0.10*np.sin(2*np.pi*(idx.dayofweek/7.0))
    rng = np.random.default_rng(int(seed))
    eps = rng.normal(0, noise, len(idx))
    return np.maximum(0.1, (morning+evening)*weekly + eps)

def _irr(idx):
    doy = idx.dayofyear.values
    tday = idx.hour.values + idx.minute.values/60.0
    season = 0.5 + 0.5*np.cos(2*np.pi*(doy-172)/365.0)
    diurnal = np.maximum(0.0, np.sin((tday/24.0)*np.pi))
    return np.clip(season*diurnal, 0.0, 1.0)

def _pv(irr, pdc_stc_kw): return pdc_stc_kw*irr

def _import_tariff(idx, base, spread, off):
    hour = idx.hour.values
    price = np.full(len(idx), float(base))
    price += ((hour>=17)&(hour<=21))*float(spread)
    price -= ((hour>=1)&(hour<=5))*abs(float(off))
    return np.maximum(0.05, price)

def _export(import_prices, mult): return np.clip(float(mult),0,1)*import_prices

def build_dataframe(idx, conf: Optional[dict]=None):
    df = pd.DataFrame(index=idx)
    base_kw = _get(conf,"load.base_kw",0.45); peak_kw = _get(conf,"load.peak_kw",2.8)
    noise = _get(conf,"load.noise",0.20); seed = int(_get(conf,"load.seed",42))
    df["load_kw"] = _load(idx, base_kw, peak_kw, noise, seed)

    irr = _irr(idx)
    pdc_kw = _get(conf,"pv.p_dc_stc_kw",3.6)
    df["pv_kw_raw"] = _pv(irr, pdc_kw)

    base = _get(conf,"economics.price_import_base_gbp_per_kwh",0.235)
    spread = _get(conf,"economics.price_peak_spread_gbp_per_kwh",0.20)
    off = _get(conf,"economics.price_offpeak_reduction_gbp_per_kwh",0.12)
    mult = _get(conf,"economics.export_multiplier",0.15)
    imp = _import_tariff(idx, base, spread, off)
    df["price_import_gbp_per_kwh"] = imp
    df["price_export_gbp_per_kwh"] = _export(imp, mult)

    amb_base = _get(conf,"environment.ambient_base_c",15.0)
    amb_amp  = _get(conf,"environment.ambient_amp_c",10.0)
    pv_rise  = _get(conf,"environment.pv_temp_rise_c_at_irr1",20.0)
    ci_base  = _get(conf,"environment.carbon_intensity_base_kg_per_kwh",0.17)
    ci_amp   = _get(conf,"environment.carbon_intensity_amp",0.06)
    df["ambient_c"] = amb_base + amb_amp*np.sin(2*np.pi*(idx.dayofyear-172)/365.0)
    df["cell_temp_c"] = df["ambient_c"] + pv_rise*irr
    df["carbon_intensity"] = ci_base + ci_amp*np.sin(2*np.pi*(idx.hour/24.0))

    dt_h = _get(conf,"time.dt_minutes",15)/60.0
    t_load = _get(conf,"calibration.target_annual_load_kwh",7500)
    t_pv   = _get(conf,"calibration.target_pv_yield_kwh",2200)
    if t_load:
        cur = float((df["load_kw"]*dt_h).sum());
        if cur>1e-9: df["load_kw"] *= float(t_load)/cur
    if t_pv:
        cur = float((df["pv_kw_raw"]*dt_h).sum());
        if cur>1e-9: df["pv_kw_raw"] *= float(t_pv)/cur

    lm = float(_get(conf,"calibration.load_multiplier",1.0))
    pm = float(_get(conf,"calibration.pv_multiplier",1.0))
    if abs(lm-1.0)>1e-12: df["load_kw"] *= lm
    if abs(pm-1.0)>1e-12: df["pv_kw_raw"] *= pm

    return df.apply(pd.to_numeric, errors="coerce")

if __name__ == "__main__":
    try:
        with open("config.yaml","r") as f: conf = yaml.safe_load(f)
    except Exception:
        conf=None
    periods = int(_get(conf,"time.periods",365*96))
    start = _get(conf,"time.start","2024-01-01")
    idx = generate_time_index(start=start, periods=periods, freq="15min")
    df = build_dataframe(idx, conf)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sim_input.csv")
    dt_h = _get(conf,"time.dt_minutes",15)/60.0
    print(f"Saved data/sim_input.csv {df.shape} | load={int((df.load_kw*dt_h).sum())}kWh pv={int((df.pv_kw_raw*dt_h).sum())}kWh")

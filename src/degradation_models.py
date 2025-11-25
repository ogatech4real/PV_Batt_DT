# src/degradation_models.py
from __future__ import annotations
import numpy as np

__all__ = [
    "calendar_fade_Ah","cycle_fade_Ah_from_DoD",
    "pv_degraded_power_kw","pv_temp_correction_kw",
    "pv_degradation_cost_step","simple_battery_deg_cost_step",
]

def calendar_fade_Ah(dt_h, soc_avg, temp_c, *, k_cal=1.2e-5, Ea_over_R=4000.0):
    T_k = float(temp_c) + 273.15
    f_soc = 0.5 + 0.5*float(np.clip(soc_avg,0.0,1.0))
    return float(k_cal*np.exp(-Ea_over_R/max(T_k,1.0))*f_soc*float(dt_h))

def cycle_fade_Ah_from_DoD(DoD_list, *, k_cyc=2.5e-4, alpha=1.5):
    if not DoD_list: return 0.0
    return float(sum(k_cyc*(max(0.0,min(1.0,d))**alpha) for d in DoD_list))

def simple_battery_deg_cost_step(soc_window, temp_window, dt_h, replacement_cost_gbp=3500, q_nom_Ah=1000.0):
    soc_avg = float(np.mean(soc_window))
    temp_avg = float(np.mean(temp_window))
    dQ_cal = calendar_fade_Ah(dt_h, soc_avg, temp_avg)
    dod_proxy = float(np.clip(np.std(soc_window)*2.0,0.0,1.0))
    dQ_cyc = cycle_fade_Ah_from_DoD([dod_proxy])
    fade_frac = (dQ_cal + dQ_cyc)/max(q_nom_Ah,1.0)
    return float(fade_frac*replacement_cost_gbp)

def pv_degraded_power_kw(pv_kw_raw, t_hours_from_start, *, annual_rate=0.01, hours_per_year=8760.0):
    factor = max(0.0, 1.0 - float(annual_rate)*(float(t_hours_from_start)/float(hours_per_year)))
    return float(pv_kw_raw)*factor

def pv_temp_correction_kw(pv_kw, temp_c, *, t_ref_c=25.0, beta_per_c=0.004):
    return float(pv_kw)*(1.0 - float(beta_per_c)*(float(temp_c)-float(t_ref_c)))

def pv_degradation_cost_step(pv_kw_raw, pv_kw_temp, price_ref_gbp_per_kwh, dt_h):
    lost_kw = max(0.0, float(pv_kw_raw)-float(pv_kw_temp))
    return float(lost_kw)*float(price_ref_gbp_per_kwh)*float(dt_h)

# src/system_model.py
from __future__ import annotations
import numpy as np

class SystemParams:
    def __init__(self, conf: dict):
        t = conf["time"]; b = conf["battery"]; p = conf["pv"]
        self.dt_h = t["dt_minutes"]/60.0
        self.horizon_h = t["horizon_hours"]
        self.e_nom_kwh = b["e_nom_kwh"]
        self.soc_min = b["soc_min"]
        self.soc_max = b["soc_max"]
        self.p_ch_max = b["p_ch_max_kw"]
        self.p_dis_max = b["p_dis_max_kw"]
        self.eta_ch = b["eta_ch"]
        self.eta_dis = b["eta_dis"]
        self.pv_temp_coeff = p["temp_coeff_per_c"]
        self.pv_annual_deg = p["annual_deg_rate"]
        self.t_ref_c = p["t_ref_c"]
        self.min_economic_spread_gbp_per_kwh = 0.04

def soc_next(soc, pch, pdis, dt_h, eta_ch, eta_dis, e_nom_kwh):
    soc_kwh = soc*e_nom_kwh
    soc_kwh += eta_ch*max(pch,0.0)*dt_h
    soc_kwh -= (max(pdis,0.0)/eta_dis)*dt_h
    return float(np.clip(soc_kwh/e_nom_kwh, 0.0, 1.0))

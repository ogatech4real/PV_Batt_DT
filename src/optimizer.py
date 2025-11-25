# src/optimizer.py
import numpy as np

def greedy_heuristic_step(
    pv_kw, load_kw, price_imp, price_exp, soc, params,
    *, price_low=0.24, price_high=0.32,
):
    dt = params.dt_h
    p_ch = p_dis = p_imp = p_exp = 0.0

    headroom_kwh = max(0.0, (params.soc_max - soc) * params.e_nom_kwh)
    avail_kwh    = max(0.0, (soc - params.soc_min) * params.e_nom_kwh)
    ch_cap_kw  = min(params.p_ch_max, headroom_kwh / dt)
    dis_cap_kw = min(params.p_dis_max, avail_kwh   / dt)

    net = pv_kw - load_kw
    if net >= 0:
        p_ch = min(ch_cap_kw, net)
        p_exp = max(0.0, net - p_ch)
    else:
        deficit = -net
        if price_imp > price_high and dis_cap_kw > 0:
            p_dis = min(dis_cap_kw, deficit)
            deficit -= p_dis
            p_imp = max(0.0, deficit)
        elif price_imp < price_low and ch_cap_kw > 0:
            p_ch = ch_cap_kw
            p_imp = deficit + p_ch
        else:
            p_imp = deficit

    return max(0.0,p_ch), max(0.0,p_dis), max(0.0,p_imp), max(0.0,p_exp)

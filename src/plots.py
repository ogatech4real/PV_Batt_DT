# src/plots.py
from __future__ import annotations
import math
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Global styling ---------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
})

_SCENARIO_ORDER = ["Baseline", "Batt-Aware", "Batt+PV-Aware"]
_SCENARIO_COLORS = {
    "Baseline":      {"face": "#4C72B0", "edge": "#264B7A"},  # steel blue → navy
    "Batt-Aware":    {"face": "#55A868", "edge": "#2E6F45"},  # green
    "Batt+PV-Aware": {"face": "#C44E52", "edge": "#7D2D31"},  # crimson
}

# --------- Helpers ---------
def _auto_ylim(values: List[float], pad_top: float = 0.18, pad_bottom: float = 0.06) -> Tuple[float, float]:
    """Add headroom for labels; lift floor above 0 unless negatives present."""
    v = np.asarray(values, dtype=float)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if vmax == vmin:
        vmax = vmin * 1.02 + 1e-6
    span = max(vmax - vmin, 1e-9)
    top = vmax + pad_top * (span if span > 1e-6 else max(abs(vmax), 1.0))
    if vmin >= 0:
        bottom = max(0.0, vmin - pad_bottom * max(span, vmax * 0.25))
    else:
        bottom = vmin - pad_bottom * span
    if abs(top - bottom) < 1e-9:
        top += 1.0
    return bottom, top

def _fmt_value(x: float) -> str:
    # Unit-less for bar labels
    if abs(x) >= 1000: return f"{x:,.0f}"
    return f"{x:.0f}"

def _bar_value_and_delta_labels_singleline(ax: plt.Axes, bars, vals: List[float], base_val: float, dy=0.030):
    """
    One line per bar: '<value>  (+x.x%)' (value unbolded; percent normal, grey).
    Baseline shows only the value.
    """
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for rect, val in zip(bars, vals):
        if not math.isfinite(val):
            continue
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_y() + rect.get_height()

        if math.isfinite(base_val) and base_val != 0 and val is not base_val:
            pct = (val - base_val) / base_val * 100.0
            sign = "+" if pct >= 0 else "−"
            txt = f"{_fmt_value(val)}  ({sign}{abs(pct):.1f}%)"
        else:
            txt = f"{_fmt_value(val)}"

        ax.text(
            x, y + dy * span, txt,
            ha="center", va="bottom",
            fontsize=10, fontweight="normal", color="black"
        )

# --------- KPI Bars (split into 3 files) ---------
def plot_kpi_bars(kpi_base: Dict[str, float],
                  kpi_batt: Dict[str, float],
                  kpi_full: Dict[str, float],
                  out_path_ignored: str = "figs/kpis.png") -> None:
    """
    Saves three separate PNGs:
      - figs/kpis_annual_cost_gbp.png
      - figs/kpis_equivalent_full_cycles.png
      - figs/kpis_co2_avoided_kg.png
    """
    rows = {"Baseline": kpi_base, "Batt-Aware": kpi_batt, "Batt+PV-Aware": kpi_full}
    df = pd.DataFrame(rows).T.loc[_SCENARIO_ORDER]

    # A) Annual electricity cost [£]  --> ticks numeric only; unit in ylabel
    _plot_single_kpi(
        scenarios=df.index.tolist(),
        values=df["annual_cost_gbp"].astype(float).values.tolist(),
        ylabel="Annual Electricity Cost [£]",
        title=" ",
        out_path="figs/kpis_annual_cost_gbp.png",
    )

    # B) Equivalent Full Cycles [cycles/year]
    _plot_single_kpi(
        scenarios=df.index.tolist(),
        values=df["equivalent_full_cycles"].astype(float).values.tolist(),
        ylabel="Equivalent Full Cycles [cycles/year]",
        title=" ",
        out_path="figs/kpis_equivalent_full_cycles.png",
    )

    # C) CO2 avoided [kg/year]
    _plot_single_kpi(
        scenarios=df.index.tolist(),
        values=df["co2_avoided_kg"].astype(float).values.tolist(),
        ylabel="CO₂ Avoided [kg/year]",
        title=" ",
        out_path="figs/kpis_co2_avoided_kg.png",
    )

def _plot_single_kpi(scenarios: List[str], values: List[float], ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(5.0, 4.2))
    ax = plt.gca()
    bars = []
    x = np.arange(len(scenarios))
    for i, (scen, v) in enumerate(zip(scenarios, values)):
        c = _SCENARIO_COLORS[scen]
        b = ax.bar(i, v, width=0.62, color=c["face"], edgecolor=c["edge"], linewidth=1.0)
        bars.extend(b)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=0)
    ax.set_ylabel(ylabel)
    y0, y1 = _auto_ylim(values, pad_top=0.10, pad_bottom=0.00)  # extra headroom for one-line label
    ax.set_ylim(0, y1)

    # Plain numeric ticks (unit already in ylabel)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    base_val = values[0] if len(values) > 0 else float("nan")
    _bar_value_and_delta_labels_singleline(ax, bars, values, base_val, dy=0.034)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# --------- Dispatch Plot (PNG only) ---------
def plot_dispatch(df_dispatch: pd.DataFrame,
                  out_path: str = "figs/dispatch_full.png",
                  window_days: int = 7) -> None:
    """
    Seven-day dispatch visualization with aligned panels:
      - SoC (0–1)
      - P_ch / P_dis
      - Grid imports / exports
    """
    df = df_dispatch.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        start = df.index.min()
        end = start + pd.Timedelta(days=window_days)
        df = df.loc[(df.index >= start) & (df.index < end)]
    else:
        steps = window_days * 96  # 15-min steps
        df = df.iloc[:steps]

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 6.6), sharex=True)
    ax_soc, ax_p, ax_g = axes

    # SoC
    ax_soc.plot(df.index, df["soc"].clip(0, 1.0), linewidth=1.4, color="#4C72B0")
    ax_soc.set_ylabel("SoC [0–1]")
    ax_soc.set_ylim(0, 1.0)
    ax_soc.grid(True, linestyle="--", alpha=0.3)
    ax_soc.set_title("Seven-Day Dispatch Profile", pad=10)

    # Charge / Discharge
    pch = df["pch"].astype(float)
    pdis = df["pdis"].astype(float)
    ax_p.plot(df.index, pch, label="P_ch", linewidth=1.2, color="#55A868")
    ax_p.plot(df.index, pdis, label="P_dis", linewidth=1.2, color="#C44E52")
    ymin, ymax = min(pch.min(), pdis.min()), max(pch.max(), pdis.max())
    ax_p.set_ylim(*_auto_ylim([ymin, ymax], pad_top=0.30, pad_bottom=0.10))
    ax_p.set_ylabel("Battery Power [kW]")
    ax_p.grid(True, linestyle="--", alpha=0.3)
    ax_p.legend(loc="upper right")

    # Grid imports / exports
    pimp = df["pimp"].astype(float)
    pexp = df["pexp"].astype(float)
    ax_g.plot(df.index, pimp, label="Import", linewidth=1.2, color="#2E6F45")
    ax_g.plot(df.index, pexp, label="Export", linewidth=1.2, color="#7D2D31")
    ymin, ymax = min(pimp.min(), pexp.min()), max(pimp.max(), pexp.max())
    ax_g.set_ylim(*_auto_ylim([ymin, ymax], pad_top=0.30, pad_bottom=0.10))
    ax_g.set_ylabel("Grid Power [kW]")
    ax_g.grid(True, linestyle="--", alpha=0.3)
    ax_g.legend(loc="upper right")

    if isinstance(df.index, pd.DatetimeIndex):
        ax_g.set_xlabel("Time")
    else:
        ax_g.set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

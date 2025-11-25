# src/analysis_extensions.py
from __future__ import annotations
import os
from typing import Iterable, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .controller import run_controller
from .evaluation import summarize_kpis
from .data_generator import generate_time_index, build_dataframe

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _ensure_dirs() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)


def bootstrap_daily_cost(
    df_dispatch: pd.DataFrame, dt_h: float, n: int = 1000, seed: int = 7
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI on mean daily cost.
    Returns (mean, ci_low, ci_high) in GBP/day.
    """
    step_cost = (
        df_dispatch["pimp"] * df_dispatch["price_import_gbp_per_kwh"]
        - df_dispatch["pexp"] * df_dispatch["price_export_gbp_per_kwh"]
    ) * dt_h

    if isinstance(df_dispatch.index, pd.DatetimeIndex):
        daily = step_cost.resample("D").sum()
    else:
        steps_per_day = int(round(24 / dt_h))
        daily = step_cost.groupby(np.arange(len(step_cost)) // steps_per_day).sum()

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(daily), len(daily))
        boots.append(float(daily.iloc[idx].mean()))
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(daily.mean()), float(low), float(high)


def _extract_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Pareto frontier (lower cost envelope) by sorting by EFC ascending
    and retaining points that strictly reduce cost.
    """
    z = df.sort_values("equivalent_full_cycles").reset_index(drop=True)
    keep = []
    best = np.inf
    for i, row in z.iterrows():
        c = float(row["annual_cost_gbp"])
        if c < best - 1e-9:
            keep.append(i)
            best = c
    return z.loc[keep].reset_index(drop=True)


def _knee_point(frontier: pd.DataFrame) -> Tuple[float, float, int]:
    """
    Knee = point with maximum perpendicular distance to the chord (first → last).
    Returns (x_knee, y_knee, idx).
    """
    x = frontier["equivalent_full_cycles"].values
    y = frontier["annual_cost_gbp"].values
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    k = int(np.argmax(d))
    return float(x[k]), float(y[k]), k


def _fit_frontier(frontier: pd.DataFrame, n_eval: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadratic fit (cost vs EFC) on the frontier for a smooth curve.
    Fit on normalized x for numerical conditioning; return dense x,y.
    """
    x = frontier["equivalent_full_cycles"].values
    y = frontier["annual_cost_gbp"].values
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return x, y  # degenerate

    xn = (x - x_min) / (x_max - x_min)
    coeff = np.polyfit(xn, y, deg=2)

    x_eval = np.linspace(x_min, x_max, n_eval)
    xn_eval = (x_eval - x_min) / (x_max - x_min)
    y_eval = np.polyval(coeff, xn_eval)
    return x_eval, y_eval


# ---------------------------------------------------------------------
# Pareto sweep API
# ---------------------------------------------------------------------


def run_pareto_sweep(
    df_input: pd.DataFrame,
    conf: Dict,
    dt_h: float,
    lam_b_grid: Optional[Iterable[float]] = None,
    lam_pv_grid: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Perform a grid sweep over (λ_batt, λ_pv), run Full-aware controller,
    compute KPIs, save CSV, and plot Pareto with knee.

    This is the main engine used both by CLI and Streamlit.
    """
    _ensure_dirs()

    # Academic-realistic grids (balanced curvature & runtime)
    if lam_b_grid is None:
        lam_b_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    else:
        lam_b_grid = np.array(list(lam_b_grid), dtype=float)

    if lam_pv_grid is None:
        lam_pv_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    else:
        lam_pv_grid = np.array(list(lam_pv_grid), dtype=float)

    rows = []
    e_nom = conf["battery"]["e_nom_kwh"]

    print("Running Pareto sweep for λ_batt × λ_pv grid...")
    for lb in lam_b_grid:
        for lp in lam_pv_grid:
            conf_mod = {**conf}
            econ = {**conf.get("economics", {})}
            econ["lambda_batt"] = float(lb)
            econ["lambda_batt_full"] = float(lb)  # keep batt/full aligned
            econ["lambda_pv"] = float(lp)
            conf_mod["economics"] = econ

            # Full-aware scenario captures both λ_batt and λ_pv effects
            sim = run_controller(df_input.copy(), conf_mod, scenario="full")
            kpi = summarize_kpis(sim.join(df_input, rsuffix="_in"), dt_h, e_nom, conf_mod)

            rows.append(
                {
                    "lambda_batt": float(lb),
                    "lambda_pv": float(lp),
                    "annual_cost_gbp": float(kpi["annual_cost_gbp"]),
                    "equivalent_full_cycles": float(kpi["equivalent_full_cycles"]),
                    "capacity_fade_pct": float(kpi.get("capacity_fade_pct", np.nan)),
                    "co2_avoided_kg": float(kpi.get("co2_avoided_kg", np.nan)),
                }
            )
            print(
                f"  λ_batt={lb:.2f}, λ_pv={lp:.2f} "
                f"→ Cost £{kpi['annual_cost_gbp']:.1f}, EFC {kpi['equivalent_full_cycles']:.1f}"
            )

    pareto = pd.DataFrame(rows).sort_values(["lambda_batt", "lambda_pv"]).reset_index(drop=True)
    pareto.to_csv("results/pareto.csv", index=False)
    print("Saved Pareto table to results/pareto.csv")

    _plot_pareto(pareto)
    return pareto


def pareto_sweep(
    df_input: pd.DataFrame,
    conf: Dict,
    dt_h: Optional[float] = None,
    lam_b_grid: Optional[Iterable[float]] = None,
    lam_pv_grid: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Thin wrapper to match existing imports:

        from src.analysis_extensions import pareto_sweep

    Streamlit can call this without worrying about dt_h.
    """
    if dt_h is None:
        dt_h = conf["time"]["dt_minutes"] / 60.0
    return run_pareto_sweep(df_input, conf, dt_h=dt_h, lam_b_grid=lam_b_grid, lam_pv_grid=lam_pv_grid)


def _plot_pareto(pareto_df: pd.DataFrame) -> None:
    frontier = _extract_frontier(pareto_df)
    x_fit, y_fit = _fit_frontier(frontier)
    xk, yk, _ = _knee_point(frontier)

    plt.figure(figsize=(7.5, 5.3))
    sc = plt.scatter(
        pareto_df["equivalent_full_cycles"],
        pareto_df["annual_cost_gbp"],
        c=pareto_df["lambda_batt"],
        cmap="viridis",
        s=60,
        edgecolor="k",
        linewidths=0.5,
        alpha=0.9,
        label="Grid (λ_batt × λ_pv)",
    )
    plt.plot(
        frontier["equivalent_full_cycles"],
        frontier["annual_cost_gbp"],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Frontier (lower envelope)",
    )
    if len(x_fit) > 2:
        plt.plot(x_fit, y_fit, color="black", linewidth=1.8, alpha=0.85, label="Quadratic fit")
    plt.scatter(
        [xk],
        [yk],
        s=110,
        marker="D",
        color="crimson",
        edgecolor="white",
        linewidths=1.0,
        zorder=5,
        label="Knee",
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("λ_batt", rotation=270, labelpad=15)

    plt.xlabel("Equivalent Full Cycles [cycles/year]")
    plt.ylabel("Annual Electricity Cost [£]")
    plt.title("Pareto Frontier: Cost vs Battery Wear (with Knee)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig("figs/pareto.png", dpi=300)
    plt.close()
    print("Saved Pareto plot with fitted curve and knee to figs/pareto.png")


# ---------------------------------------------------------------------
# CLI entry-point for offline Pareto generation
# ---------------------------------------------------------------------


def _load_or_gen_inputs(conf: dict) -> Tuple[pd.DataFrame, float]:
    """
    Load data/sim_input.csv if present; otherwise generate it using the current config.
    Returns (df_input, dt_h).
    """
    path = "data/sim_input.csv"
    dt_h = conf["time"]["dt_minutes"] / 60.0
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"Loaded existing {path} with shape {df.shape}")
        return df, dt_h

    print("No existing simulation input found. Generating data/sim_input.csv ...")
    periods = int(conf.get("time", {}).get("periods", 365 * 96))
    start = conf.get("time", {}).get("start", "2024-01-01")
    idx = generate_time_index(start=start, periods=periods, freq=f"{conf['time']['dt_minutes']}min")
    df = build_dataframe(idx, conf)
    os.makedirs("data", exist_ok=True)
    df.to_csv(path)
    print(f"Generated {path} with shape {df.shape}")
    return df, dt_h


if __name__ == "__main__":
    import yaml

    try:
        with open("config.yaml", "r") as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError("Missing or invalid config.yaml") from e

    df_in, dt_h = _load_or_gen_inputs(conf)
    run_pareto_sweep(df_in, conf, dt_h=dt_h)

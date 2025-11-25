# main.py
from __future__ import annotations
import os, json, yaml
import pandas as pd
from datetime import datetime, timezone

from src.data_generator import generate_time_index, build_dataframe
from src.controller import run_controller
from src.evaluation import summarize_kpis
from src.plots import plot_dispatch, plot_kpi_bars
from src.analysis_extensions import run_pareto_sweep  # <-- corrected import

def load_conf(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_inputs(conf: dict) -> pd.DataFrame:
    path = "data/sim_input.csv"
    if not os.path.exists(path):
        print("No existing simulation input found. Generating data/sim_input.csv ...")
        idx = generate_time_index(
            start=conf.get("time", {}).get("start", "2024-01-01"),
            periods=int(conf.get("time", {}).get("periods", 96 * 365)),
            freq=f"{conf['time']['dt_minutes']}min",
        )
        df = build_dataframe(idx, conf)
        os.makedirs("data", exist_ok=True)
        df.to_csv(path)
        print(f"Generated {path} with shape {df.shape}")
    return pd.read_csv(path, index_col=0, parse_dates=True)

def run_all(conf: dict):
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    df = load_inputs(conf)
    dt_h = conf["time"]["dt_minutes"] / 60.0
    e_nom = conf["battery"]["e_nom_kwh"]

    print("\n--- Running Digital Twin Simulation ---")
    print("1. Baseline (Cost-Only)...")
    base = run_controller(df.copy(), conf, scenario="baseline")

    print("2. Battery-Aware...")
    batt = run_controller(df.copy(), conf, scenario="batt")

    print("3. Battery+PV-Aware...")
    full = run_controller(df.copy(), conf, scenario="full")

    base.to_csv("results/baseline.csv")
    batt.to_csv("results/battaware.csv")
    full.to_csv("results/fullaware.csv")

    kb = summarize_kpis(base.join(df, rsuffix="_in"), dt_h, e_nom, conf)
    ka = summarize_kpis(batt.join(df, rsuffix="_in"), dt_h, e_nom, conf)
    kf = summarize_kpis(full.join(df, rsuffix="_in"), dt_h, e_nom, conf)

    kpi_df = pd.DataFrame([kb, ka, kf], index=["Baseline", "Batt-Aware", "Batt+PV-Aware"])
    kpi_df.to_csv("results/kpis.csv")
    print("Saved KPI metrics to results/kpis.csv")

    plot_dispatch(full, "figs/dispatch_full.png", window_days=7)
    plot_kpi_bars(kb, ka, kf, "figs/kpis.png")

    # Pareto frontier (Full-aware λ sweep)
    print("\n--- Running Pareto Frontier Analysis ---")
    run_pareto_sweep(df.copy(), conf, dt_h=dt_h)  # <-- corrected call
    print("Pareto analysis complete. See results/pareto.csv and figs/pareto.png")

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "battery_e_nom_kwh": e_nom,
        "dt_minutes": conf["time"]["dt_minutes"],
        "horizon_hours": conf.get("time", {}).get("horizon_hours", 24),
        "scenarios": ["baseline", "batt", "full"],
        "input_file": "data/sim_input.csv",
    }
    with open("results/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    conf = load_conf("config.yaml")
    run_all(conf)

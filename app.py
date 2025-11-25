# app.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.express as px

from src.data_generator import generate_time_index, build_dataframe
from src.controller import run_controller
from src.evaluation import summarize_kpis
from src.analysis_extensions import run_pareto_sweep

# ==================== METADATA & EXPLANATIONS ==================== #

METRIC_HELP = {
    "annual_cost_gbp": (
        "Total annual electricity cost: grid imports minus grid exports over the "
        "full simulation year."
    ),
    "equivalent_full_cycles": (
        "Number of 'full' charge–discharge cycles per year. Several partial cycles "
        "are aggregated into an equivalent full cycle (0–100–0%)."
    ),
    "co2_avoided_kg": (
        "Estimated annual CO₂ emissions avoided by using PV + battery compared "
        "with relying solely on the grid."
    ),
    "capacity_fade_pct": (
        "Estimated percentage loss in usable battery capacity over the simulation "
        "year due to calendar and cycling degradation."
    ),
    "batt_deg_cost_gbp": (
        "Monetised cost of battery wear over the year, based on replacement cost "
        "and estimated degradation."
    ),
    "pv_deg_cost_gbp": (
        "Monetised cost associated with PV degradation and thermal derating over "
        "the simulation year."
    ),
}

SCENARIO_LABELS = {
    "Baseline": "💰 Baseline (Cost-Only)",
    "Batt-Aware": "🔋 Battery-Aware",
    "Batt+PV-Aware": "🔋☀️ Batt+PV-Aware",
}

# ==================== BASIC PAGE CONFIG & CSS ==================== #


def init_page_config():
    st.set_page_config(
        page_title="Materials-Aware PV–Battery Digital Twin",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def render_core_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 1.1rem; padding-bottom: 1.2rem;}

        .kpi-card {
            padding: 0.9rem 1.0rem;
            border-radius: 0.75rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .kpi-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 0.15rem;
        }
        .kpi-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #0f172a;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #64748b;
        }

        .intro-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #0ea5e9);
            color: white;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            animation: pulseGlow 2.4s ease-in-out infinite;
        }
        @keyframes pulseGlow {
            0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); }
            70%  { box-shadow: 0 0 0 12px rgba(34, 197, 94, 0.0); }
            100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.0); }
        }

        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #0f172a;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 260px;
            background-color: #0f172a;
            color: #f9fafb;
            text-align: left;
            border-radius: 0.5rem;
            padding: 0.5rem 0.7rem;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 50%;
            margin-left: -130px;
            opacity: 0;
            transition: opacity 0.18s;
            font-size: 0.75rem;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #0f172a transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tooltip_label(text: str, help_key: str | None = None) -> str:
    help_txt = METRIC_HELP.get(help_key, "") if help_key else ""
    if not help_txt:
        return text
    return f"""
    <span class="tooltip">{text}
      <span class="tooltiptext">{help_txt}</span>
    </span>
    """


# ==================== HELP / ONBOARDING UI ==================== #


def render_onboarding_panel():
    if "hide_onboarding" not in st.session_state:
        st.session_state.hide_onboarding = False

    if not st.session_state.hide_onboarding:
        with st.container():
            st.markdown(
                "<span class='intro-badge'>LIVE DIGITAL TWIN</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h2 style='margin-top:0.4rem;margin-bottom:0.2rem;'>"
                "Materials-Aware PV–Battery Digital Twin</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:#555;'>This is a modular residential solar & battery system "
                "that jointly optimises <strong>cost</strong>, "
                "<strong>battery & panel degradation</strong>, and "
                "<strong>CO₂ savings</strong>.</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "- λ_batt — Battery protection weight.\n"
                "- λ_pv — Solar panel protection weight.\n"
                "- Adjust λ_batt / λ_pv and run the twin.\n"
                "- Watch the KPIs across Baseline, Battery-Aware and Batt+PV-Aware update.\n"
                "- Inspect dispatch behaviour and the long-term cost vs lifetime frontier."
            )
            st.checkbox("Hide this intro next time", key="hide_onboarding")


def render_help_expander():
    with st.expander("ℹ️ What do these metrics mean?", expanded=False):
        st.markdown("**Economic**")
        st.markdown(
            "- **Annual Electricity Cost**: Net grid spend after exports.\n"
            "- **Mean Hourly Cost**: Average expenditure per operating hour."
        )
        st.markdown("**Battery Lifecycle**")
        st.markdown(
            "- **Equivalent Full Cycles (EFCs)**: Effective number of 0–100–0% "
            "charge–discharge cycles per year.\n"
            "- **Battery Throughput**: Total energy pushed through the battery.\n"
            "- **Capacity Fade**: Estimated loss of usable capacity over the year.\n"
            "- **Battery Degradation Cost**: Monetary value of that lost capacity."
        )
        st.markdown("**Solar & Environment**")
        st.markdown(
            "- **PV Degradation Cost**: Cost associated with reduced PV output due "
            "to ageing and high temperature.\n"
            "- **CO₂ Avoided**: Emissions avoided by using PV + battery instead of "
            "pure grid consumption."
        )
        st.caption(
            "The twin uses simplified, physically informed degradation models. "
            "Outputs are indicative rather than guarantees."
        )


# ==================== DATA & BACKEND HOOKS ==================== #


@st.cache_resource
def load_conf(path: str = "config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_or_generate_inputs(conf: Dict, regen: bool = False) -> pd.DataFrame:
    """
    Load data/sim_input.csv if present; otherwise generate it.
    regen=True forces regeneration using the current config.
    """
    path = "data/sim_input.csv"
    if regen or not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        idx = generate_time_index(
            start=conf.get("time", {}).get("start", "2024-01-01"),
            periods=int(conf.get("time", {}).get("periods", 96 * 365)),
            freq=f"{conf['time']['dt_minutes']}min",
        )
        df = build_dataframe(idx, conf)
        df.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _update_lambdas(conf: Dict, lam_batt: float, lam_pv: float) -> Dict:
    new = dict(conf)
    econ = dict(new.get("economics", {}))
    econ["lambda_batt"] = lam_batt
    econ["lambda_batt_full"] = lam_batt  # keep full-aware aligned with batt-aware
    econ["lambda_pv"] = lam_pv
    new["economics"] = econ
    return new


def _run_scenarios(
    df_in: pd.DataFrame, conf: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dt_h = conf["time"]["dt_minutes"] / 60.0
    e_nom = conf["battery"]["e_nom_kwh"]

    base = run_controller(df_in.copy(), conf, scenario="baseline")
    batt = run_controller(df_in.copy(), conf, scenario="batt")
    full = run_controller(df_in.copy(), conf, scenario="full")

    kb = summarize_kpis(base.join(df_in, rsuffix="_in"), dt_h, e_nom, conf)
    ka = summarize_kpis(batt.join(df_in, rsuffix="_in"), dt_h, e_nom, conf)
    kf = summarize_kpis(full.join(df_in, rsuffix="_in"), dt_h, e_nom, conf)

    kpi_df = pd.DataFrame(
        [kb, ka, kf],
        index=["Baseline", "Batt-Aware", "Batt+PV-Aware"],
    )
    return base, batt, full, kpi_df


def _load_pareto_csv() -> pd.DataFrame | None:
    path = "results/pareto.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ==================== KPI & PLOT HELPERS ==================== #


def render_kpi_cards_for_scenario(kpis: dict, scenario_name: str):
    st.markdown(f"#### {SCENARIO_LABELS.get(scenario_name, scenario_name)}")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">{tooltip_label("Annual Cost [£]", "annual_cost_gbp")}</div>
              <div class="kpi-value">£{kpis["annual_cost_gbp"]:,.0f}</div>
              <div class="kpi-sub">Mean £{kpis["mean_hourly_cost_gbp"]:.3f}/hour</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">{tooltip_label("Equivalent Full Cycles", "equivalent_full_cycles")}</div>
              <div class="kpi-value">{kpis["equivalent_full_cycles"]:.0f}</div>
              <div class="kpi-sub">Effective full cycles per year</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">{tooltip_label("CO₂ Avoided [kg/yr]", "co2_avoided_kg")}</div>
              <div class="kpi-value">{kpis["co2_avoided_kg"]:.0f}</div>
              <div class="kpi-sub">Relative to grid-only operation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _kpi_bar_fig(kpis: pd.DataFrame, metric: str, title: str, y_label: str) -> px.bar:
    df = kpis.reset_index().rename(columns={"index": "Scenario"})
    base_val = df.loc[df["Scenario"] == "Baseline", metric].iloc[0]

    df["Δ_vs_Base_%"] = (df[metric] - base_val) / base_val * 100.0
    df["Label"] = df.apply(
        lambda r: (
            f"{r[metric]:.0f} ({r['Δ_vs_Base_%']:+.1f}%)"
            if r["Scenario"] != "Baseline"
            else f"{r[metric]:.0f}"
        ),
        axis=1,
    )

    fig = px.bar(
        df,
        x="Scenario",
        y=metric,
        color="Scenario",
        text="Label",
        color_discrete_map={
            "Baseline": "#4C72B0",
            "Batt-Aware": "#55A868",
            "Batt+PV-Aware": "#C44E52",
        },
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(title=y_label, rangemode="tozero")
    fig.update_layout(
        xaxis_title=None,
        legend_title=None,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _dispatch_fig(df: pd.DataFrame, title_suffix: str = "Batt+PV-Aware") -> px.line:
    if isinstance(df.index, pd.DatetimeIndex):
        start = df.index.min()
        end = start + pd.Timedelta(days=7)
        dfw = df.loc[(df.index >= start) & (df.index < end)].copy()
    else:
        dfw = df.iloc[:7 * 96].copy()

    df_plot = pd.DataFrame(
        {
            "time": dfw.index,
            "SoC": dfw["soc"].clip(0, 1.0).values,
            "P_ch": dfw["pch"].values,
            "P_dis": dfw["pdis"].values,
            "Import": dfw["pimp"].values,
            "Export": dfw["pexp"].values,
        }
    )

    fig = px.line(
        df_plot,
        x="time",
        y=["SoC", "P_ch", "P_dis", "Import", "Export"],
        title=f"Seven-Day Dispatch Profile – {title_suffix}",
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Per-Unit / kW",
        legend_title=None,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _pareto_fig(df: pd.DataFrame) -> px.scatter:
    fig = px.scatter(
        df,
        x="equivalent_full_cycles",
        y="annual_cost_gbp",
        color="lambda_batt",
        size="lambda_pv",
        color_continuous_scale="Viridis",
        labels={
            "equivalent_full_cycles": "Equivalent Full Cycles [year]",
            "annual_cost_gbp": "Annual Cost [£]",
            "lambda_batt": "λ_batt",
            "lambda_pv": "λ_pv",
        },
        title="Pareto Frontier: Cost vs Battery Wear",
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=60, b=40),
        coloraxis_colorbar=dict(title="λ_batt"),
    )
    return fig


# ==================== MAIN UI ==================== #


def main():
    init_page_config()
    render_core_css()

    conf = load_conf()

    # Intro / onboarding
    render_onboarding_panel()
    st.markdown("---")

    # Control strip
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        regen = st.checkbox(
            "Regenerate synthetic year",
            value=False,
            help="Rebuild one year of PV, load and tariff profiles from config.yaml",
        )
    with c2:
        lam_batt = st.slider(
            "Battery degradation weight λ_batt",
            min_value=0.0,
            max_value=2.0,
            value=float(conf.get("economics", {}).get("lambda_batt", 0.8)),
            step=0.1,
            help="Higher λ_batt penalises aggressive cycling and pushes the controller towards gentler battery use.",
        )
    with c3:
        lam_pv = st.slider(
            "PV degradation weight λ_pv",
            min_value=0.0,
            max_value=2.0,
            value=float(conf.get("economics", {}).get("lambda_pv", 0.5)),
            step=0.1,
            help="Higher λ_pv penalises running PV at high thermal stress, slightly derating output to protect panels.",
        )
    with c4:
        # Button remains for UX, but we re-run on any change anyway
        run_btn = st.button("Run Simulation", type="primary", help="Re-run the digital twin with current settings.")

    # Always recompute for current settings (gives immediate response to slider changes)
    with st.spinner("Running digital twin simulation with current λ settings..."):
        conf_eff = _update_lambdas(conf, lam_batt=lam_batt, lam_pv=lam_pv)
        df_in = load_or_generate_inputs(conf_eff, regen=regen)
        base, batt, full, kpis = _run_scenarios(df_in, conf_eff)

    st.success(f"Simulation complete · λ_batt = {lam_batt:.2f}, λ_pv = {lam_pv:.2f}")

    # Scenario spotlight
    st.markdown("### Scenario spotlight")
    scenario = st.radio(
        "Select control scenario",
        options=list(kpis.index),
        format_func=lambda s: SCENARIO_LABELS.get(s, s),
        horizontal=True,
    )
    render_kpi_cards_for_scenario(kpis.loc[scenario].to_dict(), scenario)

    render_help_expander()
    st.markdown("---")

    # KPI comparison charts
    st.markdown("### KPI comparisons by control strategy")
    p1, p2, p3 = st.columns(3)
    with p1:
        fig_cost = _kpi_bar_fig(
            kpis, "annual_cost_gbp", "Annual Electricity Cost", "Cost [£/year]"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    with p2:
        fig_efc = _kpi_bar_fig(
            kpis, "equivalent_full_cycles", "Equivalent Full Cycles", "Cycles [year]"
        )
        st.plotly_chart(fig_efc, use_container_width=True)
    with p3:
        fig_co2 = _kpi_bar_fig(
            kpis, "co2_avoided_kg", "CO₂ Emissions Avoided", "CO₂ Saved [kg/year]"
        )
        st.plotly_chart(fig_co2, use_container_width=True)

    st.markdown("---")

    # Dispatch + Pareto row
    d1, d2 = st.columns([2.0, 1.6])
    with d1:
        st.markdown("### Seven-day dispatch (Batt+PV-Aware)")
        disp_fig = _dispatch_fig(full, title_suffix="Batt+PV-Aware")
        st.plotly_chart(disp_fig, use_container_width=True)

    with d2:
        st.markdown("### Pareto trade-off: cost vs battery wear")
        pareto_df = _load_pareto_csv()
        generate = st.button("Generate / refresh Pareto frontier")
        if generate or pareto_df is None:
            with st.spinner("Running Pareto sweep (λ_batt × λ_pv grid)..."):
                dt_h = conf_eff["time"]["dt_minutes"] / 60.0
                pareto_df = run_pareto_sweep(df_in.copy(), conf_eff, dt_h=dt_h)

        if pareto_df is not None:
            pareto_fig = _pareto_fig(pareto_df)
            st.plotly_chart(pareto_fig, use_container_width=True)
            with st.expander("🔍 How to read this frontier"):
                st.markdown(
                    "- Each point is a different set of (λ_batt, λ_pv).\n"
                    "- Moving left reduces battery wear but usually increases annual cost.\n"
                    "- The lower envelope represents the best achievable trade-offs."
                )
        else:
            st.info(
                "Pareto results not yet generated. Click the button above to run the sweep."
            )


if __name__ == "__main__":
    main()

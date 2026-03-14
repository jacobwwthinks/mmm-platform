"""
Page 3: Budget Optimizer

What-if scenario planning and optimal budget allocation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mmm import MMMResults
from optimize.budget import optimize_budget, scenario_analysis

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)

st.title("Budget Optimizer")

# ── Load Results ─────────────────────────────────────────────

results = st.session_state.get("mmm_results")
if results is None:
    # Try relative path first, then absolute path from project root
    selected_client = st.session_state.get("selected_client", "juniper")
    for base in [Path("."), Path(__file__).parent.parent]:
        results_dir = base / "results" / selected_client
        if (results_dir / "results.pkl").exists():
            results = MMMResults.load(str(results_dir))
            if results is not None:
                st.session_state["mmm_results"] = results
                break

if results is None:
    st.markdown("### Ad platform data required")
    st.markdown(
        "The Budget Optimizer uses the fitted Marketing Mix Model to find the "
        "optimal allocation of ad spend across channels. This requires active "
        "ad spend data from at least one platform."
    )
    st.markdown("**To enable this page:**")
    st.markdown(
        "1. Ensure your Windsor.ai API key is set in Streamlit secrets\n"
        "2. Connect at least one ad platform (Meta, Google Ads, Pinterest, etc.)\n"
        "3. Go to **Client Overview** and click **Fetch Data & Run Model**"
    )
    st.stop()

# ── Budget Controls ──────────────────────────────────────────

roas_df = results.channel_roas
current_total_weekly = roas_df["total_spend"].sum() / results.n_weeks

st.subheader("Budget Scenario")

col1, col2 = st.columns(2)
with col1:
    budget_multiplier = st.slider(
        "Budget adjustment",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        format="%.0f%%",
        help="1.0 = current budget. 1.2 = 20% increase.",
    )
    target_budget = current_total_weekly * budget_multiplier
    st.metric("Weekly budget", f"{target_budget:,.0f}", f"{(budget_multiplier - 1) * 100:+.0f}%")

with col2:
    st.markdown("**Constraints:**")
    min_pct = st.slider("Min % per channel", 0, 20, 5) / 100
    max_pct = st.slider("Max % per channel", 50, 100, 80) / 100

# ── Run Optimization ─────────────────────────────────────────

if st.button("Optimize Allocation", type="primary", use_container_width=True):
    with st.spinner("Optimizing..."):
        opt_df = optimize_budget(
            results,
            total_budget=target_budget,
            min_spend_pct=min_pct,
            max_spend_pct=max_pct,
        )
        st.session_state["optimization"] = opt_df

opt_df = st.session_state.get("optimization")

if opt_df is None:
    # Run with defaults
    opt_df = optimize_budget(results, total_budget=target_budget, min_spend_pct=min_pct, max_spend_pct=max_pct)
    st.session_state["optimization"] = opt_df

# ── Results ──────────────────────────────────────────────────

lift_pct = opt_df.attrs.get("estimated_lift_pct", 0)
if lift_pct > 0:
    st.success(f"Estimated revenue lift from reallocation: **+{lift_pct:.1f}%**")
elif lift_pct < 0:
    st.warning(f"Budget reduction may decrease revenue by **{lift_pct:.1f}%**")
else:
    st.info("Current allocation is already near-optimal")

# ── Current vs Recommended Allocation ────────────────────────

st.subheader("Current vs. Recommended Allocation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Current**")
    fig_current = go.Figure(go.Pie(
        labels=opt_df["channel"].str.replace("_", " ").str.title(),
        values=opt_df["current_weekly_spend"],
        hole=0.4,
        marker_colors=["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"],
    ))
    fig_current.update_layout(height=300, margin=dict(t=20, b=20), **PLOTLY_LAYOUT)
    st.plotly_chart(fig_current, use_container_width=True)

with col2:
    st.markdown("**Recommended**")
    fig_recommended = go.Figure(go.Pie(
        labels=opt_df["channel"].str.replace("_", " ").str.title(),
        values=opt_df["recommended_weekly_spend"],
        hole=0.4,
        marker_colors=["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"],
    ))
    fig_recommended.update_layout(height=300, margin=dict(t=20, b=20), **PLOTLY_LAYOUT)
    st.plotly_chart(fig_recommended, use_container_width=True)

# ── Detailed Recommendations Table ───────────────────────────

st.subheader("Detailed Recommendations")

display_df = opt_df.copy()
display_df["channel"] = display_df["channel"].str.replace("_", " ").str.title()
display_df["current_weekly_spend"] = display_df["current_weekly_spend"].apply(lambda x: f"{x:,.0f}")
display_df["recommended_weekly_spend"] = display_df["recommended_weekly_spend"].apply(lambda x: f"{x:,.0f}")
display_df["change"] = display_df.apply(
    lambda r: f"{'↑' if r['change_pct'] > 5 else '↓' if r['change_pct'] < -5 else '→'} {r['change_pct']:+.0f}%",
    axis=1
)

st.dataframe(
    display_df[["channel", "current_weekly_spend", "current_pct", "recommended_weekly_spend", "recommended_pct", "change"]].rename(columns={
        "channel": "Channel",
        "current_weekly_spend": "Current (weekly)",
        "current_pct": "Current %",
        "recommended_weekly_spend": "Recommended (weekly)",
        "recommended_pct": "Rec. %",
        "change": "Change",
    }),
    hide_index=True,
    use_container_width=True,
)

# ── Scenario Analysis ────────────────────────────────────────

st.subheader("Budget Scenario Analysis")
st.markdown("How does total revenue change as we scale budget up or down?")

with st.spinner("Running scenarios..."):
    scenarios = scenario_analysis(results)

fig_scenario = go.Figure()
fig_scenario.add_trace(go.Scatter(
    x=scenarios["weekly_budget"],
    y=scenarios["estimated_weekly_revenue"],
    mode="lines+markers",
    line=dict(color="#F58518", width=3),
    marker=dict(size=10),
    name="Estimated Weekly Revenue",
))

# Mark current budget
current_row = scenarios[scenarios["budget_multiplier"] == 1.0]
if not current_row.empty:
    fig_scenario.add_trace(go.Scatter(
        x=current_row["weekly_budget"],
        y=current_row["estimated_weekly_revenue"],
        mode="markers",
        marker=dict(size=15, color="#E15759", symbol="star"),
        name="Current Budget",
    ))

fig_scenario.update_layout(
    xaxis_title="Weekly Budget",
    yaxis_title="Estimated Weekly Revenue",
    height=350,
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_scenario, use_container_width=True)

# ── Export ────────────────────────────────────────────────────

with st.expander("Export Recommendations"):
    csv = opt_df.to_csv(index=False)
    st.download_button(
        "Download as CSV",
        csv,
        file_name=f"budget_recommendation_{st.session_state.get('selected_client', 'client')}.csv",
        mime="text/csv",
    )

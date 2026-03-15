"""
Page 3: Budget Optimizer

What-if scenario planning and optimal budget allocation.
Month-aware: accounts for seasonal efficiency and planned events.
Three-column layout: sidebar (nav) | main data | context panel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mmm import MMMResults
from data.events import load_events
from optimize.budget import optimize_budget, scenario_analysis
from optimize.spend_amer import (
    compute_seasonal_indices,
    compute_event_boosts,
    compute_monthly_organic,
)
from ui.layout import inject_context_css, render_sidebar, context_block, context_tip, context_separator

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
    title_font=dict(size=13, color="#C9D1D9"),
)

inject_context_css()
render_sidebar()

st.title("Budget Optimizer")

# ── Load Results ─────────────────────────────────────────────

results = st.session_state.get("mmm_results")
results_dir = None
if results is None:
    # Try relative path first, then absolute path from project root
    selected_client = st.session_state.get("selected_client", "juniper")
    for base in [Path("."), Path(__file__).parent.parent]:
        candidate = base / "results" / selected_client
        if (candidate / "results.pkl").exists():
            results = MMMResults.load(str(candidate))
            if results is not None:
                results_dir = candidate
                st.session_state["mmm_results"] = results
                break
else:
    for base in [Path("."), Path(__file__).parent.parent]:
        candidate = base / "results" / st.session_state.get("selected_client", "juniper")
        if (candidate / "results.pkl").exists():
            results_dir = candidate
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

# ── Load model_df + events for seasonal context ─────────────

selected_client = st.session_state.get("selected_client", "juniper")
client_cfg = st.session_state.get("client_config", {})

model_df = st.session_state.get("model_df")
if model_df is None:
    model_df = MMMResults.load_model_df(str(results_dir)) if results_dir else None
    if model_df is not None:
        st.session_state["model_df"] = model_df

events_path = client_cfg.get("events_csv", "")
events_df = None
for candidate_path in [events_path, f"events/{selected_client}_events.csv"]:
    if candidate_path:
        for base in [Path("."), Path(__file__).parent.parent]:
            full = base / candidate_path
            if full.exists():
                events_df = load_events(str(full))
                break
    if events_df is not None:
        break

seasonal_indices = compute_seasonal_indices(results, model_df)
event_boosts = compute_event_boosts(results, model_df, events_df)


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT: data | context
# ═══════════════════════════════════════════════════════════════

# ── Section 1: Planning Period ────────────────────────────────
_m1, _c1 = st.columns([4, 1])
with _m1:
    st.subheader("Planning Period")

    today = datetime.date.today()
    month_options = []
    for m in range(12):
        month_num = (today.month - 1 + m) % 12 + 1
        year = today.year + (today.month - 1 + m) // 12
        dt = datetime.date(year, month_num, 1)
        month_options.append({"label": dt.strftime("%B %Y"), "key": f"{year}-{month_num:02d}",
                              "month": month_num, "year": year})

    selected_month_idx = st.selectbox(
        "Optimize for month",
        range(len(month_options)),
        format_func=lambda i: month_options[i]["label"],
        help="Channel efficiency varies by month due to seasonality and events.",
    )

    sel = month_options[selected_month_idx]
    sel_month = sel["month"]
    sel_year = sel["year"]

    seasonal_mult = seasonal_indices.get(sel_month, 1.0)
    event_boost = 1.0
    heavy_mult = event_boosts.get("heavy_discount", 1.0)
    light_mult = event_boosts.get("light_discount", 1.0)
    drop_mult = event_boosts.get("product_drop", 1.0)
    month_events_list = []

    if events_df is not None and not events_df.empty:
        edf = events_df.copy()
        edf["week_start"] = pd.to_datetime(edf["week_start"])
        month_events = edf[
            (edf["week_start"].dt.month == sel_month) &
            (edf["week_start"].dt.year == sel_year)
        ]
        if not month_events.empty:
            for _, row in month_events.iterrows():
                note = row.get("notes", "")
                if note:
                    month_events_list.append(note)
            if "discount_campaign" in month_events.columns:
                if (month_events["discount_campaign"] == 2).any():
                    event_boost *= heavy_mult
                elif (month_events["discount_campaign"] == 1).any():
                    event_boost *= light_mult
            if "product_drop" in month_events.columns and (month_events["product_drop"] > 0).any():
                event_boost *= drop_mult

    effective_mult = seasonal_mult * event_boost

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Seasonal index", f"{seasonal_mult:.2f}x",
                  help="How efficiently this month converts spend vs average.")
    with col2:
        st.metric("Event boost", f"{event_boost:.2f}x",
                  help="Multiplier from planned events this month.")
    with col3:
        st.metric("Combined multiplier", f"{effective_mult:.2f}x")

    if month_events_list:
        st.caption(f"Events: {' · '.join(month_events_list)}")
    elif event_boost == 1.0:
        has_any_events = False
        if events_df is not None and not events_df.empty:
            edf_check = events_df.copy()
            edf_check["week_start"] = pd.to_datetime(edf_check["week_start"])
            has_any_events = len(edf_check[
                (edf_check["week_start"].dt.month == sel_month) &
                (edf_check["week_start"].dt.year == sel_year)
            ]) > 0
        if not has_any_events:
            st.caption(f"No events planned for {sel['label']}. Add events in the Event Calendar if needed.")

with _c1:
    context_block(
        "Planning Period",
        "Channel efficiency varies by month. A **seasonal index** above 1.0 "
        "means spend converts better than average (e.g. November). "
        "Below 1.0 means less efficient (e.g. January).\n\n"
        "**Event boosts** come from the Event Calendar — campaigns "
        "and product drops increase conversion efficiency."
    )

# ── Section 2: Budget Scenario + Optimization ─────────────────
_m2, _c2 = st.columns([4, 1])
with _m2:
    roas_df = results.channel_roas
    current_total_weekly = roas_df["total_spend"].sum() / results.n_weeks

    st.subheader("Budget Scenario")

    col1, col2 = st.columns(2)
    with col1:
        budget_step = 5_000
        budget_min = max(budget_step, int(round(current_total_weekly * 0.25 / budget_step)) * budget_step)
        budget_max = int(round(current_total_weekly * 2.5 / budget_step)) * budget_step
        budget_default = int(round(current_total_weekly / budget_step)) * budget_step

        target_budget = st.slider(
            "Weekly budget (SEK)",
            min_value=budget_min, max_value=budget_max, value=budget_default,
            step=budget_step, format="%,.0f",
            help=f"Current average weekly spend: {current_total_weekly:,.0f} SEK.",
        )
        budget_change_pct = (target_budget - current_total_weekly) / current_total_weekly * 100
        st.metric("Weekly budget", f"{target_budget:,.0f} SEK", f"{budget_change_pct:+.1f}% vs current")

    with col2:
        st.markdown("**Constraints:**")
        min_pct = st.slider("Min % per channel", 0, 20, 5) / 100
        max_pct = st.slider("Max % per channel", 50, 100, 80) / 100

    if st.button("Optimize Allocation", type="primary", use_container_width=True):
        with st.spinner("Optimizing..."):
            opt_df = optimize_budget(
                results, total_budget=target_budget,
                min_spend_pct=min_pct, max_spend_pct=max_pct,
                seasonal_multiplier=effective_mult,
            )
            st.session_state["optimization"] = opt_df
            st.session_state["optimization_month"] = sel["key"]

    opt_df = st.session_state.get("optimization")
    opt_month = st.session_state.get("optimization_month")

    if opt_df is None or opt_month != sel["key"]:
        opt_df = optimize_budget(
            results, total_budget=target_budget,
            min_spend_pct=min_pct, max_spend_pct=max_pct,
            seasonal_multiplier=effective_mult,
        )
        st.session_state["optimization"] = opt_df
        st.session_state["optimization_month"] = sel["key"]

    lift_pct = opt_df.attrs.get("estimated_lift_pct", 0)
    if lift_pct > 0:
        st.success(f"Estimated revenue lift from reallocation: **+{lift_pct:.1f}%**")
    elif lift_pct < 0:
        st.warning(f"Budget reduction may decrease revenue by **{lift_pct:.1f}%**")
    else:
        st.info("Current allocation is already near-optimal")

with _c2:
    context_block(
        "Budget Scenario",
        "Set a weekly budget and constraints, then optimize. "
        "The optimizer uses the model's saturation curves to "
        "find the allocation that maximizes total revenue.\n\n"
        "**Min/max %** prevents the optimizer from zeroing out "
        "or overloading any single channel."
    )

# ── Section 3: Current vs Recommended ─────────────────────────
_m3, _c3 = st.columns([4, 1])
with _m3:
    st.subheader("Current vs. Recommended Allocation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Current**")
        fig_current = go.Figure(go.Pie(
            labels=opt_df["channel"].str.replace("_", " ").str.title(),
            values=opt_df["current_weekly_spend"], hole=0.4,
            marker_colors=["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"],
        ))
        fig_current.update_layout(height=300, margin=dict(t=20, b=20), **PLOTLY_LAYOUT)
        st.plotly_chart(fig_current, use_container_width=True)

    with col2:
        st.markdown("**Recommended**")
        fig_recommended = go.Figure(go.Pie(
            labels=opt_df["channel"].str.replace("_", " ").str.title(),
            values=opt_df["recommended_weekly_spend"], hole=0.4,
            marker_colors=["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"],
        ))
        fig_recommended.update_layout(height=300, margin=dict(t=20, b=20), **PLOTLY_LAYOUT)
        st.plotly_chart(fig_recommended, use_container_width=True)

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
        hide_index=True, use_container_width=True,
    )

with _c3:
    context_block(
        "Current vs Recommended",
        "The pie charts show how budget is split today vs the "
        "model's recommendation. Big shifts suggest some channels "
        "are over-invested (saturated) while others have room to grow.\n\n"
        "The **estimated lift** shows how much more revenue the "
        "recommended allocation would generate at the same total budget."
    )

# ── Section 4: Scenario Analysis ──────────────────────────────
_m4, _c4 = st.columns([4, 1])
with _m4:
    st.subheader("Budget Scenario Analysis")
    st.markdown("How does total revenue change as we scale budget up or down?")

    with st.spinner("Running scenarios..."):
        scenarios = scenario_analysis(results, seasonal_multiplier=effective_mult)

    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(
        x=scenarios["weekly_budget"], y=scenarios["estimated_weekly_revenue"],
        mode="lines+markers", line=dict(color="#F58518", width=3),
        marker=dict(size=10), name="Estimated Weekly Revenue",
    ))

    current_row = scenarios[scenarios["budget_multiplier"] == 1.0]
    if not current_row.empty:
        fig_scenario.add_trace(go.Scatter(
            x=current_row["weekly_budget"], y=current_row["estimated_weekly_revenue"],
            mode="markers", marker=dict(size=15, color="#E15759", symbol="star"),
            name="Current Budget",
        ))

    fig_scenario.update_layout(
        xaxis_title="Weekly Budget", yaxis_title="Estimated Weekly Revenue",
        height=350, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

    with st.expander("Export Recommendations"):
        csv = opt_df.to_csv(index=False)
        st.download_button(
            "Download as CSV", csv,
            file_name=f"budget_recommendation_{st.session_state.get('selected_client', 'client')}.csv",
            mime="text/csv",
        )

with _c4:
    context_block(
        "Scenario Analysis",
        "The curve shows revenue vs budget level. The flattening "
        "at higher budgets reflects diminishing returns across "
        "all channels combined.\n\n"
        "Look for the **inflection point** — beyond this, each "
        "additional SEK of spend yields less and less return."
    )
    context_tip(
        "**For GP3-aware planning** (accounting for margins and CLTV), "
        "use the **Spend-aMER** page instead. This optimizer focuses "
        "purely on revenue maximization."
    )

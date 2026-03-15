"""
Page 5: Spend-aMER Model

Month-by-month spend planning based on GP3 maximization.
Uses MMM saturation curves + unit economics to find the spend level
that maximizes gross profit after all variable costs (including marketing).

Key concept: the "spending power" of the brand varies by month.
In November (Black Week), channel efficiency is higher and organic demand
is stronger, so the brand can spend more while staying GP3-positive.
In January, the opposite — spend less.

Designed as a planning workflow:
  1. Set unit economics (GM2%, CLTV)
  2. Pick a month to plan
  3. Review GP3 curve + recommended spend + channel allocation
  4. Adjust and lock in the plan
  5. Move to the next month

Three-column layout: sidebar (nav) | main data | context panel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mmm import MMMResults
from data.events import load_events
from optimize.spend_amer import (
    compute_gp3_curve,
    find_optimal_spend,
    compute_seasonal_indices,
    compute_event_boosts,
    compute_monthly_organic,
    compute_historical_backcheck,
    compute_calibration_factor,
    compute_same_month_benchmark,
    compute_observed_yoy_trend,
    optimize_channel_allocation,
)
from ui.layout import inject_context_css, render_sidebar, context_block, context_tip, context_separator

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
    title_font=dict(size=13, color="#C9D1D9"),
)
ORANGE = "#F58518"
TEAL = "#76B7B2"
RED = "#E15759"
GREEN = "#59A14F"
COLORS = [ORANGE, TEAL, RED, GREEN, "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]

inject_context_css()
render_sidebar()

st.title("Spend-aMER Model")


# ═══════════════════════════════════════════════════════════════
# LOAD RESULTS + GATE ON NEW_REVENUE
# ═══════════════════════════════════════════════════════════════

selected_client = st.session_state.get("selected_client", "juniper")
client_cfg = st.session_state.get("client_config", {})
results_dir = None

results = st.session_state.get("mmm_results")
if results is None:
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
        candidate = base / "results" / selected_client
        if (candidate / "results.pkl").exists():
            results_dir = candidate
            break

if results is None:
    st.markdown("### Model results required")
    st.markdown(
        "The Spend-aMER model uses the fitted MMM saturation curves to determine "
        "optimal spend levels. Run the model first on the **Client Overview** page, "
        "using **New Customer Net Sales** as the target."
    )
    st.stop()

# ── Gate: require new_revenue target ──
target_col = getattr(results, "target_col", "revenue")
if target_col != "new_revenue":
    st.warning(
        "### Model must be fit on New Customer Net Sales\n\n"
        "The Spend-aMER model optimizes for **GP3 from new customers**. "
        "The current model was fit on "
        f"**{'Total Revenue' if target_col == 'revenue' else target_col.replace('_', ' ').title()}**, "
        "which includes returning customer revenue and would overstate the "
        "effect of paid media.\n\n"
        "Go to **Client Overview**, select **New Customer Net Sales** as the "
        "revenue target, and re-run the model."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL_DF + EVENTS
# ═══════════════════════════════════════════════════════════════

# Try session state first, then persisted pickle
model_df = st.session_state.get("model_df")
if model_df is None and results_dir is not None:
    model_df = MMMResults.load_model_df(str(results_dir))
    if model_df is not None:
        st.session_state["model_df"] = model_df

# Load events
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


# ═══════════════════════════════════════════════════════════════
# COMPUTE CONTEXT (SEASONAL, EVENTS, ORGANIC, HISTORICAL)
# ═══════════════════════════════════════════════════════════════

seasonal_indices = compute_seasonal_indices(results, model_df)
event_boosts = compute_event_boosts(results, model_df, events_df)
monthly_organic = compute_monthly_organic(results, model_df)
historical = compute_historical_backcheck(results, model_df)

# ── Model calibration ──────────────────────────────────────
calibration = compute_calibration_factor(results, model_df)
cal_factor = calibration["factor"]

historical_max_monthly_spend = 0
if historical is not None and len(historical) > 0:
    historical_max_monthly_spend = historical["total_spend"].max()

# ── Check for missing forward-looking events ──────────────
today = datetime.date.today()
_months_missing_events = []
for m in range(12):
    _month_num = (today.month - 1 + m) % 12 + 1
    _year = today.year + (today.month - 1 + m) // 12
    has_events = False
    if events_df is not None and not events_df.empty:
        _edf = events_df.copy()
        _edf["week_start"] = pd.to_datetime(_edf["week_start"])
        _me = _edf[
            (_edf["week_start"].dt.month == _month_num) &
            (_edf["week_start"].dt.year == _year)
        ]
        has_events = len(_me) > 0
    if not has_events:
        _months_missing_events.append(datetime.date(_year, _month_num, 1).strftime("%b %Y"))


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT: per-section data | context pairs
# ═══════════════════════════════════════════════════════════════

# ── Section 1: Calibration diagnostics + Unit Economics ────────

_m1, _c1 = st.columns([4, 1])

with _m1:

    # ── Model calibration diagnostic ──────────────────────────
    if calibration["status"] == "ok" and abs(cal_factor - 1.0) > 0.1:
        cal_pred = calibration["predicted_weekly_channel_rev"]
        cal_actual = calibration["actual_weekly_channel_contrib"]
        if cal_factor > 1.05:
            st.info(
                f"**Model calibration applied ({cal_factor:.2f}x)**\n\n"
                f"The model's saturation curves predict **{cal_pred:,.0f} SEK/wk** in channel revenue "
                f"at historical spend, but the model's own decomposition shows **{cal_actual:,.0f} SEK/wk**. "
                f"A {cal_factor:.2f}x calibration factor is applied to correct the level while preserving "
                f"the saturation curve shape (diminishing returns)."
            )
        elif cal_factor < 0.95:
            st.info(
                f"**Model calibration applied ({cal_factor:.2f}x)**\n\n"
                f"The optimizer over-predicts channel revenue vs the model's decomposition. "
                f"A {cal_factor:.2f}x correction is applied."
            )

    if _months_missing_events:
        st.error(
            "**Event calendar missing for planned months**\n\n"
            "The following months have **no events** in the calendar: "
            f"**{', '.join(_months_missing_events)}**.\n\n"
            "Without a forward-looking event plan, the model assumes no campaigns — "
            "which significantly underestimates spend capacity during discount months "
            "(e.g. Black Week, Birthday Week). "
            "Go to **Event Calendar** to add planned events before locking in these months."
        )


    # ═══════════════════════════════════════════════════════════════
    # UNIT ECONOMICS INPUTS
    # ═══════════════════════════════════════════════════════════════

    st.subheader("Unit Economics")

    col1, col2, col3 = st.columns(3)

    with col1:
        gm2_pct = st.number_input(
            "GM2 % (margin before marketing)",
            min_value=5.0,
            max_value=95.0,
            value=50.0,
            step=1.0,
            help="Gross margin after COGS, shipping, logistics, and transaction costs — "
                 "but before marketing spend. This is the margin available to cover marketing.",
        )

    with col2:
        cltv_expansion = st.number_input(
            "365D CLTV expansion %",
            min_value=0.0,
            max_value=500.0,
            value=30.0,
            step=5.0,
            help="How much additional revenue a new customer generates over 12 months "
                 "beyond their first order. E.g., 30% means a customer with a 1000 SEK "
                 "first order spends an additional 300 SEK over the next year.",
        )

    with col3:
        # Compute observed YoY trend from data
        yoy_trend = compute_observed_yoy_trend(model_df)
        observed_growth_label = ""
        if yoy_trend is not None:
            obs = yoy_trend["observed_capacity_growth_pct"]
            observed_growth_label = f" (observed: {obs:+.0f}%)"

        yoy_growth = st.number_input(
            "YoY spend capacity growth %",
            min_value=-50.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            help="How much more you expect to be able to spend at the same aMER vs last year. "
                 "0% = assume same spend capacity as last year. "
                 "Set based on brand growth trajectory.",
        )

    cltv_mult = 1 + cltv_expansion / 100
    gm2_frac = gm2_pct / 100
    breakeven_amer_fo = 1 / gm2_frac
    breakeven_amer_365d = 1 / (cltv_mult * gm2_frac)

    # Show observed YoY trend from data
    if yoy_trend is not None:
        obs_spend = yoy_trend["median_spend_change_pct"]
        obs_amer = yoy_trend["median_amer_change_pct"]
        obs_cap = yoy_trend["observed_capacity_growth_pct"]
        n_months = yoy_trend["n_overlapping_months"]

        trend_col1, trend_col2, trend_col3 = st.columns(3)
        with trend_col1:
            st.metric(
                "Observed YoY spend change",
                f"{obs_spend:+.0f}%",
                help=f"Median spend change across {n_months} overlapping months.",
            )
        with trend_col2:
            st.metric(
                "Observed YoY aMER change",
                f"{obs_amer:+.0f}%",
                help=f"Median aMER change across {n_months} overlapping months. "
                     "Positive = more efficient. Negative = less efficient.",
            )
        with trend_col3:
            st.metric(
                "Observed capacity growth",
                f"{obs_cap:+.0f}%",
                help="Spend growth in months where aMER didn't drop >10%. "
                     "This is what the brand can spend more at similar efficiency.",
            )
        if obs_cap > 5 and yoy_growth == 0:
            st.caption(
                f"Data shows {obs_cap:+.0f}% capacity growth — consider setting "
                f"YoY growth to reflect this if you expect the trend to continue."
            )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "First-order breakeven aMER",
            f"{breakeven_amer_fo:.2f}x",
            help="Minimum aMER to break even on the first order alone (1 / GM2%).",
        )
        st.caption(f"1 / {gm2_frac:.0%} = {breakeven_amer_fo:.2f}x")

    with col2:
        st.metric(
            "365D breakeven aMER",
            f"{breakeven_amer_365d:.2f}x",
            help="Minimum aMER when accounting for 12-month repeat purchases.",
        )
        st.caption(f"1 / ({cltv_mult:.2f} × {gm2_frac:.0%}) = {breakeven_amer_365d:.2f}x")

with _c1:
    context_block(
        "Unit Economics",
        "These inputs define what profit looks like for this brand.\n\n"
        "**GM2%** = margin before marketing. At 50%, half of revenue "
        "covers COGS, shipping, and logistics.\n\n"
        "**CLTV expansion** = how much more a new customer spends "
        "over 12 months beyond their first order. Higher CLTV means "
        "you can afford higher acquisition costs."
    )

    context_block(
        "YoY Growth",
        "0% = assume the brand can spend the same as last year at "
        "the same efficiency. Set higher if the brand is growing "
        "(stronger organic, better conversion rates).\n\n"
        "The observed trend from data helps calibrate this input."
    )

# ── Section 2: Monthly Spend Planner ────────────────────────────

_m2, _c2 = st.columns([4, 1])

with _m2:

    st.markdown("---")
    st.subheader("Monthly Spend Planner")

    # Initialize locked plans in session state
    if "locked_plans" not in st.session_state:
        st.session_state["locked_plans"] = {}

    locked_plans = st.session_state["locked_plans"]

    # Build list of next 12 months
    today = datetime.date.today()
    month_options = []
    for m in range(12):
        month_num = (today.month - 1 + m) % 12 + 1
        year = today.year + (today.month - 1 + m) // 12
        dt = datetime.date(year, month_num, 1)
        label = dt.strftime("%B %Y")
        key = f"{year}-{month_num:02d}"
        is_locked = key in locked_plans
        prefix = "✓ " if is_locked else ""
        month_options.append({"label": f"{prefix}{label}", "key": key, "month": month_num, "year": year, "date": dt})

    # Month selector
    selected_idx = st.selectbox(
        "Plan for month",
        range(len(month_options)),
        format_func=lambda i: month_options[i]["label"],
        help="Select a month to plan. Locked months show ✓.",
    )

    sel = month_options[selected_idx]
    sel_key = sel["key"]
    sel_month = sel["month"]
    sel_year = sel["year"]
    sel_label = sel["date"].strftime("%B %Y")

    # ── Compute month context ──

    seasonal_mult = seasonal_indices.get(sel_month, 1.0)
    organic_weekly = monthly_organic.get(sel_month, np.mean(list(monthly_organic.values())))

    # Event info for this month
    event_boost = 1.0
    month_events_list = []
    heavy_mult = event_boosts.get("heavy_discount", 1.0)
    light_mult = event_boosts.get("light_discount", 1.0)
    drop_mult = event_boosts.get("product_drop", 1.0)
    has_heavy_discount = False
    has_light_discount = False
    has_product_drop = False

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
                has_heavy_discount = (month_events["discount_campaign"] == 2).any()
                has_light_discount = (
                    (month_events["discount_campaign"] == 1).any()
                    and not has_heavy_discount
                )
                if has_heavy_discount:
                    event_boost *= heavy_mult
                elif has_light_discount:
                    event_boost *= light_mult
            if "product_drop" in month_events.columns:
                has_product_drop = (month_events["product_drop"] > 0).any()
                if has_product_drop:
                    event_boost *= drop_mult

    effective_mult = seasonal_mult * event_boost

    # ── Month context cards ──

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Seasonal index", f"{seasonal_mult:.2f}x",
                  help="How efficiently this month converts spend vs average.")
    with col2:
        st.metric("Event boost", f"{event_boost:.2f}x",
                  help="Multiplier from planned events this month.")
    with col3:
        st.metric("Organic baseline", f"{organic_weekly:,.0f}/wk",
                  help="Expected organic (non-paid) new customer revenue per week.")

    if month_events_list:
        st.caption(f"Events: {' · '.join(month_events_list)}")
    elif sel_label in _months_missing_events:
        st.warning(
            f"**No events planned for {sel_label}.** "
            "If you're running campaigns this month, add them in the "
            "**Event Calendar** — otherwise the model assumes no campaign boost."
        )

with _c2:
    context_separator()

    context_block(
        "GP3 Curve",
        "The core decision tool. GP3 = gross profit after marketing.\n\n"
        "**Orange line** (GP3 365D) = profit including CLTV. "
        "This is what you optimize for.\n\n"
        "**Red dashed** (GP3 FO) = profit from first order only. "
        "Below zero means you're investing in future customer value.\n\n"
        "The **star** marks the optimal spend — where GP3 is maximized."
    )

    context_tip(
        "**GP3 FO negative but GP3 365D positive?** You're acquiring "
        "customers at a loss on the first order but profiting over their "
        "lifetime. This is healthy if your CLTV assumptions hold."
    )

# ── Section 3: GP3 curve chart + results ────────────────────────

_m3, _c3 = st.columns([4, 1])

with _m3:

    st.markdown("---")
    st.markdown(f"#### GP3 Curve — {sel_label}")

    # Find optimal spend for this month
    optimal = find_optimal_spend(
        results, gm2_pct, cltv_expansion,
        organic_weekly_revenue=organic_weekly,
        seasonal_multiplier=effective_mult,
        calibration_factor=cal_factor,
        yoy_growth_pct=yoy_growth,
    )

    # Generate GP3 curve — start from 20% of current to avoid misleading S-curve region
    gp3_df = compute_gp3_curve(
        results, gm2_pct, cltv_expansion,
        organic_weekly_revenue=organic_weekly,
        seasonal_multiplier=effective_mult,
        calibration_factor=cal_factor,
        n_points=200,
        max_spend_mult=5.0,
    )

    # Clip to actionable range (from 20% of current spend onward)
    min_actionable = optimal["current_weekly_spend"] * 0.2
    gp3_plot = gp3_df[gp3_df["weekly_spend"] >= min_actionable].copy()

    # ── Dual-axis chart ──

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=gp3_plot["monthly_spend"],
            y=gp3_plot["gp3_365d_monthly"],
            name="GP3 365D",
            line=dict(color=ORANGE, width=3),
            fill="tozeroy",
            fillcolor="rgba(245, 133, 24, 0.08)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=gp3_plot["monthly_spend"],
            y=gp3_plot["gp3_first_order_monthly"],
            name="GP3 first order",
            line=dict(color=RED, width=2, dash="dash"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=gp3_plot["monthly_spend"],
            y=gp3_plot["total_new_customer_revenue"] * 4.33,
            name="NC net sales",
            line=dict(color=TEAL, width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    # Mark optimal
    fig.add_trace(
        go.Scatter(
            x=[optimal["optimal_monthly_spend"]],
            y=[optimal["optimal_gp3_365d_monthly"]],
            mode="markers+text",
            marker=dict(size=14, color=ORANGE, symbol="star"),
            text=[f"Optimal: {optimal['optimal_monthly_spend']:,.0f}"],
            textposition="top right",
            textfont=dict(color=ORANGE, size=12),
            name="Optimal",
            showlegend=False,
        ),
        secondary_y=False,
    )

    # Mark current
    fig.add_trace(
        go.Scatter(
            x=[optimal["current_monthly_spend"]],
            y=[optimal["current_gp3_365d_monthly"]],
            mode="markers+text",
            marker=dict(size=12, color="#E6EDF3", symbol="circle"),
            text=[f"Current: {optimal['current_monthly_spend']:,.0f}"],
            textposition="bottom left",
            textfont=dict(color="#E6EDF3", size=11),
            name="Current",
            showlegend=False,
        ),
        secondary_y=False,
    )

    fig.add_hline(
        y=0, line_dash="dash", line_color="rgba(225, 87, 89, 0.4)",
        annotation_text="GP3 = 0", annotation_position="bottom right",
        secondary_y=False,
    )

    fig.update_layout(
        title="",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(title_text="Monthly marketing spend (SEK)")
    fig.update_yaxes(title_text="GP3 (SEK/month)", secondary_y=False)
    fig.update_yaxes(title_text="New customer net sales (SEK/month)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # ── Summary cards for this month ──

    col1, col2, col3, col4 = st.columns(4)

    spend_change = optimal["spend_change_pct"]

    with col1:
        st.metric(
            "Recommended spend",
            f"{optimal['optimal_monthly_spend']:,.0f}",
            f"{spend_change:+.0f}% vs avg",
        )

    with col2:
        st.metric("GP3 (first order)", f"{optimal['optimal_gp3_first_order_monthly']:,.0f}")

    with col3:
        st.metric("GP3 (365D)", f"{optimal['optimal_gp3_365d_monthly']:,.0f}")

    with col4:
        amer_display = optimal["amer_at_optimal"]
        if amer_display > 100 or optimal["optimal_monthly_spend"] < 100:
            st.metric("aMER", "N/A")
        else:
            st.metric("aMER", f"{amer_display:.2f}x")

    if optimal.get("at_upper_bound", False):
        st.warning(
            "**Recommendation hit the search boundary.** The model suggests GP3 is still "
            "positive at the maximum spend level explored. This likely means the model is "
            "extrapolating beyond observed data — treat the exact number with caution and "
            "scale up gradually rather than jumping to this level."
        )

with _c3:
    pass  # Context for GP3 continues in the next section

# ── Section 4: Historical Benchmark ─────────────────────────────

_m4, _c4 = st.columns([4, 1])

with _m4:

    benchmark = compute_same_month_benchmark(model_df, sel_month, sel_year, yoy_growth_pct=yoy_growth)

    if benchmark is not None:
        st.markdown("---")
        st.markdown(f"#### Historical Benchmark — {sel_label}")

        latest_bm = benchmark["latest_benchmark"]

        st.markdown(
            f"In **{latest_bm['label']}** you spent **{latest_bm['total_spend']:,.0f} SEK** "
            f"at **{latest_bm['amer']:.2f}x aMER**. "
            f"With ~{benchmark['yoy_growth_pct']:.0f}% YoY growth, a reasonable target "
            f"for {sel_label} at similar efficiency is "
            f"**{benchmark['suggested_spend_same_amer']:,.0f} SEK**."
        )

        bm_col1, bm_col2, bm_col3, bm_col4 = st.columns(4)
        with bm_col1:
            st.metric(
                f"{latest_bm['label']} spend",
                f"{latest_bm['total_spend']:,.0f}",
            )
        with bm_col2:
            st.metric(
                f"{latest_bm['label']} aMER",
                f"{latest_bm['amer']:.2f}x",
            )
        with bm_col3:
            st.metric(
                f"Growth-adjusted target",
                f"{benchmark['suggested_spend_same_amer']:,.0f}",
                f"+{benchmark['yoy_growth_pct'] * benchmark['years_gap']:.0f}% YoY",
            )
        with bm_col4:
            model_vs_benchmark = (
                (optimal["optimal_monthly_spend"] - benchmark["suggested_spend_same_amer"])
                / (benchmark["suggested_spend_same_amer"] + 1e-8) * 100
            )
            st.metric(
                "Model vs benchmark",
                f"{model_vs_benchmark:+.0f}%",
                help="How the model's recommendation compares to the growth-adjusted historical benchmark.",
            )

        # Per-channel breakdown from last year
        ch_breakdown = latest_bm.get("channel_breakdown", {})
        if ch_breakdown:
            with st.expander(f"Channel breakdown — {latest_bm['label']}"):
                bm_rows = [
                    {"Channel": ch, "Spend": f"{v:,.0f}",
                     "Share": f"{v / (latest_bm['total_spend'] + 1e-8) * 100:.1f}%"}
                    for ch, v in ch_breakdown.items() if v > 0
                ]
                if bm_rows:
                    st.dataframe(pd.DataFrame(bm_rows), hide_index=True, use_container_width=True)

        # Flag large divergence
        if abs(model_vs_benchmark) > 50:
            st.warning(
                f"**Large gap between model and historical benchmark.** "
                f"The model recommends {optimal['optimal_monthly_spend']:,.0f} SEK vs "
                f"the benchmark of {benchmark['suggested_spend_same_amer']:,.0f} SEK "
                f"({model_vs_benchmark:+.0f}%). Consider using the historical benchmark as "
                f"a more conservative guide and scaling gradually."
            )

with _c4:
    context_separator()

    context_block(
        "Historical Benchmark",
        "Compares the model's recommendation to what you actually "
        "spent in the same month last year, adjusted for growth.\n\n"
        "A large gap (> 50%) between model and benchmark is a warning "
        "sign — the model may be extrapolating beyond observed data. "
        "Use the benchmark as a conservative guide."
    )

# ── Section 5: Channel Allocation ───────────────────────────────

_m5, _c5 = st.columns([4, 1])

with _m5:

    st.markdown("---")
    st.markdown(f"#### Channel Allocation — {sel_label}")

    alloc_df = optimize_channel_allocation(
        results,
        total_weekly_spend=optimal["optimal_weekly_spend"],
        seasonal_multiplier=effective_mult,
        calibration_factor=cal_factor,
    )

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        fig_pie = go.Figure(go.Pie(
            labels=alloc_df["channel"].str.replace("_", " ").str.title(),
            values=alloc_df["monthly_spend"],
            hole=0.4,
            marker_colors=COLORS[:len(alloc_df)],
            textinfo="label+percent",
            textposition="outside",
        ))
        fig_pie.update_layout(
            title="",
            height=320,
            margin=dict(t=20, b=20),
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        alloc_display = alloc_df.copy()
        alloc_display["channel"] = alloc_display["channel"].str.replace("_", " ").str.title()
        alloc_display.columns = [c.replace("_", " ").title() for c in alloc_display.columns]
        alloc_display = alloc_display.rename(columns={
            "Monthly Revenue": "Monthly NC Net Sales",
            "Weekly Revenue": "Weekly NC Net Sales",
        })

        fmt_alloc = alloc_display.copy()
        for col in ["Monthly Spend", "Weekly Spend", "Monthly NC Net Sales", "Weekly NC Net Sales"]:
            if col in fmt_alloc.columns:
                fmt_alloc[col] = fmt_alloc[col].apply(lambda x: f"{x:,.0f}")
        if "Pct" in fmt_alloc.columns:
            fmt_alloc["Pct"] = fmt_alloc["Pct"].apply(lambda x: f"{x}%")

        st.dataframe(
            fmt_alloc[["Channel", "Monthly Spend", "Pct", "Monthly NC Net Sales"]].rename(columns={
                "Pct": "Share",
            }),
            hide_index=True,
            use_container_width=True,
        )

with _c5:
    context_separator()

    context_block(
        "Channel Allocation",
        "How to split the recommended total spend across channels. "
        "Based on each channel's saturation curve and marginal "
        "efficiency at the recommended spend level."
    )

# ── Section 6: Annual Overview ──────────────────────────────────

_m6, _c6 = st.columns([4, 1])

with _m6:

    st.markdown("---")

    # Lock-in mechanism (smaller section, inline)
    is_locked = sel_key in locked_plans

    if is_locked:
        lp = locked_plans[sel_key]
        st.success(
            f"**{sel_label}** is locked at **{lp['monthly_spend']:,.0f} SEK/month** "
            f"(aMER {lp['amer']:.2f}x, GP3 365D {lp['gp3_365d']:,.0f})"
        )

        if st.button(f"Unlock {sel_label}", key=f"unlock_{sel_key}"):
            del st.session_state["locked_plans"][sel_key]
            st.rerun()
    else:
        if st.button(f"Lock in plan for {sel_label}", type="primary", key=f"lock_{sel_key}"):
            st.session_state["locked_plans"][sel_key] = {
                "month_label": sel_label,
                "monthly_spend": round(optimal["optimal_monthly_spend"], 0),
                "weekly_spend": round(optimal["optimal_weekly_spend"], 0),
                "gp3_fo": round(optimal["optimal_gp3_first_order_monthly"], 0),
                "gp3_365d": round(optimal["optimal_gp3_365d_monthly"], 0),
                "amer": round(optimal["amer_at_optimal"], 2),
                "revenue": round(optimal["new_customer_revenue_monthly"], 0),
                "seasonal_mult": seasonal_mult,
                "event_boost": event_boost,
                "channel_allocation": alloc_df.to_dict("records"),
            }
            st.rerun()

    st.markdown("---")
    st.subheader("Annual Overview")

    st.markdown(
        "Locked months use your confirmed plan. Unlocked months show the "
        "model's recommendation based on seasonal indices and planned events."
    )

    # Build overview for all 12 months
    overview_rows = []
    for mo in month_options:
        mk = mo["key"]
        if mk in locked_plans:
            lp = locked_plans[mk]
            overview_rows.append({
                "month": mo["date"].strftime("%b %Y"),
                "status": "Locked ✓",
                "monthly_spend": lp["monthly_spend"],
                "gp3_fo": lp["gp3_fo"],
                "gp3_365d": lp["gp3_365d"],
                "amer": lp["amer"],
                "revenue": lp["revenue"],
            })
        else:
            # Compute recommendation for this month
            m_num = mo["month"]
            m_year = mo["year"]
            s_mult = seasonal_indices.get(m_num, 1.0)
            o_weekly = monthly_organic.get(m_num, np.mean(list(monthly_organic.values())))

            # Event boost for this month
            e_boost = 1.0
            if events_df is not None and not events_df.empty:
                edf_t = events_df.copy()
                edf_t["week_start"] = pd.to_datetime(edf_t["week_start"])
                me = edf_t[
                    (edf_t["week_start"].dt.month == m_num) &
                    (edf_t["week_start"].dt.year == m_year)
                ]
                if not me.empty:
                    if "discount_campaign" in me.columns:
                        if (me["discount_campaign"] == 2).any():
                            e_boost *= heavy_mult
                        elif (me["discount_campaign"] == 1).any():
                            e_boost *= light_mult
                    if "product_drop" in me.columns and (me["product_drop"] > 0).any():
                        e_boost *= drop_mult

            eff = s_mult * e_boost
            opt = find_optimal_spend(
                results, gm2_pct, cltv_expansion,
                organic_weekly_revenue=o_weekly,
                seasonal_multiplier=eff,
                calibration_factor=cal_factor,
                yoy_growth_pct=yoy_growth,
            )
            overview_rows.append({
                "month": mo["date"].strftime("%b %Y"),
                "status": "Projected",
                "monthly_spend": round(opt["optimal_monthly_spend"], 0),
                "gp3_fo": round(opt["optimal_gp3_first_order_monthly"], 0),
                "gp3_365d": round(opt["optimal_gp3_365d_monthly"], 0),
                "amer": round(opt["amer_at_optimal"], 2),
                "revenue": round(opt["new_customer_revenue_monthly"], 0),
            })

    overview_df = pd.DataFrame(overview_rows)

    # ── Bar chart: locked vs projected ──

    fig_overview = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = [
        ORANGE if row["status"] == "Locked ✓" else "rgba(245, 133, 24, 0.35)"
        for _, row in overview_df.iterrows()
    ]

    fig_overview.add_trace(
        go.Bar(
            x=overview_df["month"],
            y=overview_df["monthly_spend"],
            name="Monthly spend",
            marker_color=bar_colors,
            text=[f"{v:,.0f}" for v in overview_df["monthly_spend"]],
            textposition="outside",
            textfont=dict(size=9),
        ),
        secondary_y=False,
    )

    fig_overview.add_trace(
        go.Scatter(
            x=overview_df["month"],
            y=overview_df["gp3_365d"],
            name="GP3 365D",
            line=dict(color=GREEN, width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ),
        secondary_y=True,
    )

    fig_overview.add_trace(
        go.Scatter(
            x=overview_df["month"],
            y=overview_df["gp3_fo"],
            name="GP3 first order",
            line=dict(color=RED, width=1, dash="dash"),
            mode="lines",
        ),
        secondary_y=True,
    )

    fig_overview.update_layout(
        title="",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )
    fig_overview.update_xaxes(title_text="")
    fig_overview.update_yaxes(title_text="Monthly spend (SEK)", secondary_y=False)
    fig_overview.update_yaxes(title_text="Monthly GP3 (SEK)", secondary_y=True)

    st.plotly_chart(fig_overview, use_container_width=True)

    # ── Summary table ──

    overview_display = overview_df.copy()
    for col in ["monthly_spend", "gp3_fo", "gp3_365d", "revenue"]:
        overview_display[col] = overview_display[col].apply(lambda x: f"{x:,.0f}")
    overview_display["amer"] = overview_display["amer"].apply(lambda x: f"{x:.2f}x")

    overview_display.columns = ["Month", "Status", "Spend", "GP3 (FO)", "GP3 (365D)", "aMER", "NC Net Sales"]
    st.dataframe(overview_display, hide_index=True, use_container_width=True)

    # Annual totals
    total_spend = overview_df["monthly_spend"].sum()
    total_gp3_365d = overview_df["gp3_365d"].sum()
    total_gp3_fo = overview_df["gp3_fo"].sum()
    total_revenue = overview_df["revenue"].sum()
    n_locked = sum(1 for r in overview_rows if r["status"] == "Locked ✓")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual spend", f"{total_spend:,.0f}")
    with col2:
        st.metric("Annual GP3 (365D)", f"{total_gp3_365d:,.0f}")
    with col3:
        st.metric("Annual GP3 (FO)", f"{total_gp3_fo:,.0f}")
    with col4:
        st.metric("Months locked", f"{n_locked} / 12")

    # Warn about months exceeding historical max
    if historical_max_monthly_spend > 0:
        flagged = overview_df[overview_df["monthly_spend"] > historical_max_monthly_spend * 1.5]["month"].tolist()
        if flagged:
            st.warning(
                f"**{', '.join(flagged)}** exceed 150% of historical max monthly spend "
                f"({historical_max_monthly_spend:,.0f} SEK). Scale gradually and measure real aMER."
            )

with _c6:
    context_separator()

    context_block(
        "Annual Overview",
        "Solid orange bars = locked months (your confirmed plan). "
        "Faded bars = projected (model recommendations).\n\n"
        "The GP3 lines show profitability per month. "
        "Lock each month as you confirm the plan."
    )

# ── Section 7: aMER vs Spend ────────────────────────────────────

_m7, _c7 = st.columns([4, 1])

with _m7:

    st.markdown("---")
    st.subheader("aMER vs. Spend")

    st.markdown(
        "As spend increases, aMER decreases due to saturation. "
        "Two breakeven thresholds shown: first-order and 365D."
    )

    amer_df = gp3_plot[gp3_plot["weekly_spend"] > 0].copy()

    fig_amer = go.Figure()

    fig_amer.add_trace(go.Scatter(
        x=amer_df["monthly_spend"],
        y=amer_df["amer"],
        name="aMER",
        line=dict(color=TEAL, width=3),
    ))

    fig_amer.add_hline(
        y=breakeven_amer_fo, line_dash="dash", line_color=RED,
        annotation_text=f"FO breakeven ({breakeven_amer_fo:.1f}x)",
        annotation_position="top right", annotation_font_color=RED,
    )
    fig_amer.add_hline(
        y=breakeven_amer_365d, line_dash="dot", line_color=GREEN,
        annotation_text=f"365D breakeven ({breakeven_amer_365d:.1f}x)",
        annotation_position="bottom right", annotation_font_color=GREEN,
    )

    fig_amer.add_trace(go.Scatter(
        x=[optimal["current_monthly_spend"]],
        y=[optimal["current_amer"]],
        mode="markers",
        marker=dict(size=12, color="#E6EDF3"),
        name="Current",
    ))
    fig_amer.add_trace(go.Scatter(
        x=[optimal["optimal_monthly_spend"]],
        y=[optimal["amer_at_optimal"]],
        mode="markers",
        marker=dict(size=14, color=ORANGE, symbol="star"),
        name="Optimal",
    ))

    fig_amer.update_layout(
        title="",
        xaxis_title="Monthly marketing spend (SEK)",
        yaxis_title="aMER (revenue / spend)",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )

    st.plotly_chart(fig_amer, use_container_width=True)

with _c7:
    context_separator()

    context_block(
        "aMER vs Spend",
        "As you spend more, aMER drops due to saturation. "
        "The red line is first-order breakeven, green is 365D breakeven.\n\n"
        "The optimal spend is where GP3 is maximized — "
        "not where aMER is highest (that would mean spending very little)."
    )

# ── Section 8: Historical Backcheck ────────────────────────────

_m8, _c8 = st.columns([4, 1])

with _m8:

    st.markdown("---")
    st.subheader("Historical Backcheck")

    if historical is not None and len(historical) > 0:
        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])

        fig_hist.add_trace(
            go.Bar(
                x=historical["month_name"],
                y=historical["total_spend"],
                name="Actual spend",
                marker_color="rgba(245, 133, 24, 0.5)",
            ),
            secondary_y=False,
        )
        fig_hist.add_trace(
            go.Scatter(
                x=historical["month_name"],
                y=historical["amer"],
                name="Actual aMER",
                line=dict(color=TEAL, width=2),
                mode="lines+markers",
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

        fig_hist.add_hline(
            y=breakeven_amer_fo, line_dash="dash", line_color=RED,
            annotation_text=f"FO breakeven ({breakeven_amer_fo:.1f}x)",
            annotation_position="top right", annotation_font_color=RED,
            secondary_y=True,
        )
        fig_hist.add_hline(
            y=breakeven_amer_365d, line_dash="dot", line_color=GREEN,
            annotation_text=f"365D breakeven ({breakeven_amer_365d:.1f}x)",
            annotation_position="bottom right", annotation_font_color=GREEN,
            secondary_y=True,
        )

        fig_hist.update_layout(
            title="",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PLOTLY_LAYOUT,
        )
        fig_hist.update_xaxes(title_text="")
        fig_hist.update_yaxes(title_text="Monthly spend (SEK)", secondary_y=False)
        fig_hist.update_yaxes(title_text="aMER", secondary_y=True)

        st.plotly_chart(fig_hist, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. monthly spend", f"{historical['total_spend'].mean():,.0f}")
        with col2:
            st.metric("Max monthly spend", f"{historical_max_monthly_spend:,.0f}")
        with col3:
            st.metric("Avg. aMER", f"{historical['amer'].mean():.2f}x")
    else:
        st.info(
            "Historical backcheck requires model data. "
            "Re-run the model on **Client Overview** to enable."
        )

with _c8:
    context_separator()

    context_block(
        "Historical Backcheck",
        "Actual spend and aMER from historical data. "
        "Use this to validate the model's recommendations against "
        "real-world outcomes. The model should roughly predict what "
        "actually happened."
    )

# ══════════════════════════════════════════════════════════════
# EXPANDABLE SECTIONS (full-width, outside columns)
# ══════════════════════════════════════════════════════════════

with st.expander("Event efficiency boosts (data-driven)"):
    st.markdown(
        "Multipliers computed from historical data: channel efficiency during "
        "event weeks vs non-event baseline."
    )
    boost_data = [
            {"Event Type": "Heavy discount", "Multiplier": f"{event_boosts['heavy_discount']:.2f}x",
         "Interpretation": f"{(event_boosts['heavy_discount']-1)*100:+.0f}% efficiency"},
            {"Event Type": "Light discount", "Multiplier": f"{event_boosts['light_discount']:.2f}x",
         "Interpretation": f"{(event_boosts['light_discount']-1)*100:+.0f}% efficiency"},
            {"Event Type": "Product drop", "Multiplier": f"{event_boosts['product_drop']:.2f}x",
         "Interpretation": f"{(event_boosts['product_drop']-1)*100:+.0f}% efficiency"},
    ]
    st.dataframe(pd.DataFrame(boost_data), hide_index=True, use_container_width=True)
    if model_df is None:
        st.caption(
            "Event boosts are 1.0x (neutral) because model_df was not loaded. "
            "Re-run the model on Client Overview to enable data-driven boosts."
        )

with st.expander("Seasonal efficiency indices"):
    idx_df = pd.DataFrame([
            {"Month": pd.Timestamp(f"2024-{m:02d}-01").strftime("%B"),
         "Index": f"{seasonal_indices.get(m, 1.0):.2f}"}
        for m in range(1, 13)
    ])
    st.dataframe(idx_df, hide_index=True, use_container_width=True)

with st.expander("About this model"):
    st.markdown("""
**GP3 = (New Customer Net Sales × (1 + CLTV expansion) × GM2%) − Marketing Spend**

The MMM gives us saturation curves per channel. This model layers unit economics
on top to find the monthly spend level that maximizes profit. Each month gets a
different recommendation based on seasonal efficiency and planned events.

**Two breakeven thresholds:**
- **First-order breakeven aMER** = 1 / GM2%. At 50% GM2, this is 2.0x.
- **365D breakeven aMER** = 1 / (GM2% × (1 + CLTV)). At 50% GM2 and 30% CLTV, this is 1.54x.

**Workflow:** Pick a month → review the GP3 curve and channel split → lock it in → move to the next month.
Locked months persist in your session and appear in the Annual Overview.
""")

with st.expander("Export"):
    csv_overview = overview_df.to_csv(index=False)
    st.download_button(
        "Download annual plan (CSV)",
        csv_overview,
        file_name=f"spend_amer_plan_{selected_client}.csv",
        mime="text/csv",
    )

    csv_curve = gp3_df.to_csv(index=False)
    st.download_button(
        "Download GP3 curve data (CSV)",
        csv_curve,
        file_name=f"gp3_curve_{selected_client}_{sel_key}.csv",
        mime="text/csv",
    )

    if locked_plans:
        st.download_button(
            "Download locked plans (JSON)",
            json.dumps(locked_plans, indent=2, default=str),
            file_name=f"locked_plans_{selected_client}.json",
            mime="application/json",
        )




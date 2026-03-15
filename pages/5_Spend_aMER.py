"""
Page 5: Spend-aMER Model

Optimal marketing spend planning based on GP3 maximization.
Uses MMM saturation curves + unit economics to find the spend level
that maximizes gross profit after all variable costs (including marketing).

Key concept: the "spending power" of the brand varies by month.
In November (Black Week), channel efficiency is higher and organic demand
is stronger, so the brand can spend more while staying GP3-positive.
In January, the opposite — spend less.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mmm import MMMResults
from data.events import load_events
from optimize.spend_amer import (
    compute_gp3_curve,
    find_optimal_spend,
    compute_seasonal_indices,
    compute_monthly_organic,
    monthly_spend_plan,
    optimize_channel_allocation,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)
ORANGE = "#F58518"
TEAL = "#76B7B2"
RED = "#E15759"
GREEN = "#59A14F"
COLORS = [ORANGE, TEAL, RED, GREEN, "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]

st.title("Spend-aMER Model")

# ── Load Results ─────────────────────────────────────────────

results = st.session_state.get("mmm_results")
if results is None:
    selected_client = st.session_state.get("selected_client", "juniper")
    for base in [Path("."), Path(__file__).parent.parent]:
        results_dir = base / "results" / selected_client
        if (results_dir / "results.pkl").exists():
            results = MMMResults.load(str(results_dir))
            if results is not None:
                st.session_state["mmm_results"] = results
                break

if results is None:
    st.markdown("### Model results required")
    st.markdown(
        "The Spend-aMER model uses the fitted MMM saturation curves to determine "
        "optimal spend levels. Run the model first on the **Client Overview** page."
    )
    st.markdown(
        "For best results, fit the model with **New Customer Revenue** as the target — "
        "this gives the most accurate spend → new customer acquisition curves."
    )
    st.stop()

# ── Load events for forward planning ────────────────────────

selected_client = st.session_state.get("selected_client", "juniper")
client_cfg = st.session_state.get("client_config", {})
events_path = client_cfg.get("events_csv", "")
events_df = None
for candidate in [events_path, f"events/{selected_client}_events.csv"]:
    if candidate:
        for base in [Path("."), Path(__file__).parent.parent]:
            full = base / candidate
            if full.exists():
                events_df = load_events(str(full))
                break
    if events_df is not None:
        break

# ── Compute seasonal context ────────────────────────────────

model_df = st.session_state.get("model_df")
seasonal_indices = compute_seasonal_indices(results, model_df)
monthly_organic = compute_monthly_organic(results, model_df)
avg_organic = np.mean(list(monthly_organic.values()))

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
    # Breakeven aMER calculation
    cltv_mult = 1 + cltv_expansion / 100
    gm2_frac = gm2_pct / 100
    breakeven_amer = 1 / (cltv_mult * gm2_frac)
    st.metric(
        "Breakeven aMER",
        f"{breakeven_amer:.2f}x",
        help="Below this aMER, each SEK of marketing spend destroys GP3.",
    )
    st.caption(
        f"1 / ({cltv_mult:.2f} × {gm2_frac:.0%}) = {breakeven_amer:.2f}x"
    )


# ═══════════════════════════════════════════════════════════════
# GP3 OPTIMIZATION CURVE
# ═══════════════════════════════════════════════════════════════

st.subheader("GP3 Optimization")

# Find optimal spend
optimal = find_optimal_spend(
    results, gm2_pct, cltv_expansion,
    organic_weekly_revenue=avg_organic,
    seasonal_multiplier=1.0,  # average month
)

# Generate GP3 curve
gp3_df = compute_gp3_curve(
    results, gm2_pct, cltv_expansion,
    organic_weekly_revenue=avg_organic,
    seasonal_multiplier=1.0,
    n_points=200,
    max_spend_mult=3.0,
)

# ── Dual-axis chart: GP3 + New Customer Revenue ──

fig = make_subplots(specs=[[{"secondary_y": True}]])

# GP3 curve (left axis) — the parabola
fig.add_trace(
    go.Scatter(
        x=gp3_df["monthly_spend"],
        y=gp3_df["gp3_monthly"],
        name="GP3 (monthly)",
        line=dict(color=ORANGE, width=3),
        fill="tozeroy",
        fillcolor="rgba(245, 133, 24, 0.08)",
    ),
    secondary_y=False,
)

# New customer revenue (right axis)
fig.add_trace(
    go.Scatter(
        x=gp3_df["monthly_spend"],
        y=gp3_df["total_new_customer_revenue"] * 4.33,
        name="New customer revenue (monthly)",
        line=dict(color=TEAL, width=2, dash="dot"),
    ),
    secondary_y=True,
)

# Mark optimal spend (GP3 peak)
fig.add_trace(
    go.Scatter(
        x=[optimal["optimal_monthly_spend"]],
        y=[optimal["optimal_gp3_monthly"]],
        mode="markers+text",
        marker=dict(size=14, color=ORANGE, symbol="star"),
        text=[f"Optimal: {optimal['optimal_monthly_spend']:,.0f}/mo"],
        textposition="top right",
        textfont=dict(color=ORANGE, size=12),
        name="Optimal spend",
        showlegend=False,
    ),
    secondary_y=False,
)

# Mark current spend
fig.add_trace(
    go.Scatter(
        x=[optimal["current_monthly_spend"]],
        y=[optimal["current_gp3_monthly"]],
        mode="markers+text",
        marker=dict(size=12, color="#E6EDF3", symbol="circle"),
        text=[f"Current: {optimal['current_monthly_spend']:,.0f}/mo"],
        textposition="bottom right",
        textfont=dict(color="#E6EDF3", size=11),
        name="Current spend",
        showlegend=False,
    ),
    secondary_y=False,
)

# GP3 = 0 breakeven line
fig.add_hline(
    y=0, line_dash="dash", line_color="rgba(225, 87, 89, 0.5)",
    annotation_text="GP3 breakeven", annotation_position="bottom right",
    secondary_y=False,
)

fig.update_layout(
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    **PLOTLY_LAYOUT,
)
fig.update_xaxes(title_text="Monthly marketing spend (SEK)")
fig.update_yaxes(title_text="GP3 (SEK/month)", secondary_y=False)
fig.update_yaxes(title_text="New customer revenue (SEK/month)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# ── Summary Cards ────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

spend_change = optimal["spend_change_pct"]
gp3_change = (
    (optimal["optimal_gp3_monthly"] - optimal["current_gp3_monthly"])
    / (abs(optimal["current_gp3_monthly"]) + 1e-8) * 100
)

with col1:
    st.metric(
        "Optimal monthly spend",
        f"{optimal['optimal_monthly_spend']:,.0f}",
        f"{spend_change:+.0f}% vs current",
    )

with col2:
    st.metric(
        "Projected monthly GP3",
        f"{optimal['optimal_gp3_monthly']:,.0f}",
        f"{gp3_change:+.0f}% vs current",
    )

with col3:
    st.metric(
        "aMER at optimal",
        f"{optimal['amer_at_optimal']:.2f}x",
        f"Breakeven: {breakeven_amer:.2f}x",
    )

with col4:
    st.metric(
        "Monthly new customer revenue",
        f"{optimal['new_customer_revenue_monthly']:,.0f}",
    )

# Interpretation
if spend_change > 10:
    st.info(
        f"The model suggests you can **increase spend by {spend_change:.0f}%** while improving GP3. "
        f"The saturation curves show room to scale — Meta in particular is only ~37% saturated. "
        f"At the optimal level, each additional SEK still generates more than {breakeven_amer:.2f} SEK "
        f"in 365-day customer value (after variable costs)."
    )
elif spend_change < -10:
    st.warning(
        f"The model suggests **reducing spend by {abs(spend_change):.0f}%** to maximize GP3. "
        f"Current spend levels are past the saturation point where marginal returns cover costs."
    )
else:
    st.success(
        f"Current spend is **near-optimal** for GP3 maximization. "
        f"The marginal return at this level is close to the breakeven aMER of {breakeven_amer:.2f}x."
    )


# ═══════════════════════════════════════════════════════════════
# MONTHLY SPEND PLAN
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Monthly Spend Plan")

st.markdown(
    "The model adjusts recommended spend by month based on **seasonal efficiency** "
    "(how effectively spend converts in each calendar month) and **planned events** "
    "(discount campaigns and product drops boost channel responsiveness)."
)

# Generate plan
plan_df = monthly_spend_plan(
    results, gm2_pct, cltv_expansion,
    seasonal_indices, monthly_organic,
    months_ahead=12,
    events_df=events_df,
)

# ── Monthly spend bar chart ──

fig_plan = make_subplots(specs=[[{"secondary_y": True}]])

fig_plan.add_trace(
    go.Bar(
        x=plan_df["month_name"],
        y=plan_df["recommended_monthly_spend"],
        name="Recommended spend",
        marker_color=ORANGE,
        opacity=0.85,
        text=[f"{v:,.0f}" for v in plan_df["recommended_monthly_spend"]],
        textposition="outside",
        textfont=dict(size=10),
    ),
    secondary_y=False,
)

fig_plan.add_trace(
    go.Scatter(
        x=plan_df["month_name"],
        y=plan_df["estimated_monthly_gp3"],
        name="Projected GP3",
        line=dict(color=GREEN, width=2),
        mode="lines+markers",
        marker=dict(size=6),
    ),
    secondary_y=True,
)

fig_plan.update_layout(
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    barmode="group",
    **PLOTLY_LAYOUT,
)
fig_plan.update_xaxes(title_text="")
fig_plan.update_yaxes(title_text="Monthly spend (SEK)", secondary_y=False)
fig_plan.update_yaxes(title_text="Monthly GP3 (SEK)", secondary_y=True)

st.plotly_chart(fig_plan, use_container_width=True)

# ── Plan table ──

plan_display = plan_df[[
    "month_name", "seasonal_index", "events",
    "recommended_monthly_spend", "estimated_monthly_revenue",
    "estimated_monthly_gp3", "amer",
]].copy()

plan_display.columns = [
    "Month", "Seasonal Index", "Events",
    "Rec. Spend", "Est. Revenue", "Est. GP3", "aMER",
]

# Format numbers
for col in ["Rec. Spend", "Est. Revenue", "Est. GP3"]:
    plan_display[col] = plan_display[col].apply(lambda x: f"{x:,.0f}")
plan_display["aMER"] = plan_display["aMER"].apply(lambda x: f"{x:.2f}x")

st.dataframe(plan_display, hide_index=True, use_container_width=True)

# ── Seasonal efficiency explanation ──

with st.expander("Seasonal efficiency indices"):
    st.markdown(
        "These indices are computed from historical data — how efficiently each "
        "calendar month converts ad spend into revenue, relative to the yearly average. "
        "A value of 1.20 means that month is historically 20% more efficient per SEK spent."
    )

    idx_df = pd.DataFrame([
        {"Month": pd.Timestamp(f"2024-{m:02d}-01").strftime("%B"), "Index": f"{seasonal_indices.get(m, 1.0):.2f}"}
        for m in range(1, 13)
    ])
    st.dataframe(idx_df, hide_index=True, use_container_width=True)

    if events_df is not None:
        st.caption(
            "Event boosts are applied on top of seasonal indices: "
            "heavy discount campaigns add +30%, light discounts +15%, product drops +10%."
        )


# ═══════════════════════════════════════════════════════════════
# CHANNEL ALLOCATION
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Channel Allocation at Optimal Spend")

# Optimize channel split at the average-month optimal spend
alloc_df = optimize_channel_allocation(
    results,
    total_weekly_spend=optimal["optimal_weekly_spend"],
    seasonal_multiplier=1.0,
)

# Pie chart
col1, col2 = st.columns([1, 1])

with col1:
    fig_pie = go.Figure(go.Pie(
        labels=alloc_df["channel"].str.replace("_", " ").str.title(),
        values=alloc_df["monthly_spend"],
        hole=0.4,
        marker_colors=COLORS[:len(alloc_df)],
        textinfo="label+percent",
        textposition="outside",
    ))
    fig_pie.update_layout(
        height=350,
        margin=dict(t=20, b=20),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    alloc_display = alloc_df.copy()
    alloc_display["channel"] = alloc_display["channel"].str.replace("_", " ").str.title()
    alloc_display["monthly_spend"] = alloc_display["monthly_spend"].apply(lambda x: f"{x:,.0f}")
    alloc_display["weekly_spend"] = alloc_display["weekly_spend"].apply(lambda x: f"{x:,.0f}")
    alloc_display["monthly_revenue"] = alloc_display["monthly_revenue"].apply(lambda x: f"{x:,.0f}")
    alloc_display["pct"] = alloc_display["pct"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        alloc_display[["channel", "monthly_spend", "pct", "monthly_revenue"]].rename(columns={
            "channel": "Channel",
            "monthly_spend": "Monthly Spend",
            "pct": "Share",
            "monthly_revenue": "Monthly Revenue",
        }),
        hide_index=True,
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════
# aMER CURVE
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("aMER vs. Spend")

st.markdown(
    "aMER (acquisition Marketing Efficiency Ratio) = New Customer Revenue ÷ Marketing Spend. "
    "As spend increases, aMER decreases due to saturation. The breakeven aMER is the level "
    "below which each additional SEK of spend destroys GP3."
)

# Filter to non-zero spend
amer_df = gp3_df[gp3_df["weekly_spend"] > 0].copy()

fig_amer = go.Figure()

fig_amer.add_trace(go.Scatter(
    x=amer_df["monthly_spend"],
    y=amer_df["amer"],
    name="aMER",
    line=dict(color=TEAL, width=3),
))

# Breakeven aMER line
fig_amer.add_hline(
    y=breakeven_amer,
    line_dash="dash",
    line_color=RED,
    annotation_text=f"Breakeven aMER ({breakeven_amer:.2f}x)",
    annotation_position="top right",
    annotation_font_color=RED,
)

# Mark current aMER
current_amer_row = amer_df.iloc[(amer_df["monthly_spend"] - optimal["current_monthly_spend"]).abs().idxmin()]
fig_amer.add_trace(go.Scatter(
    x=[optimal["current_monthly_spend"]],
    y=[current_amer_row["amer"]],
    mode="markers",
    marker=dict(size=12, color="#E6EDF3"),
    name="Current",
))

# Mark optimal aMER
fig_amer.add_trace(go.Scatter(
    x=[optimal["optimal_monthly_spend"]],
    y=[optimal["amer_at_optimal"]],
    mode="markers",
    marker=dict(size=14, color=ORANGE, symbol="star"),
    name="Optimal",
))

fig_amer.update_layout(
    xaxis_title="Monthly marketing spend (SEK)",
    yaxis_title="aMER (revenue / spend)",
    height=350,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    **PLOTLY_LAYOUT,
)

st.plotly_chart(fig_amer, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# MODEL NOTES
# ═══════════════════════════════════════════════════════════════

with st.expander("About this model"):
    st.markdown("""
**What is GP3?**

GP3 is gross profit after all variable costs including: product cost (COGS),
inbound shipping, outbound shipping & logistics, transaction costs, and marketing.
It's the profit left after every variable cost is covered.

**How the optimization works**

The MMM gives us saturation curves: how much incremental revenue each SEK of
ad spend generates. These curves flatten out (diminishing returns) as spend
increases. The Spend-aMER model layers unit economics on top:

1. **First order revenue** — what the MMM predicts from a given spend level
2. **365D CLTV expansion** — additional revenue from repeat purchases over 12 months
3. **GM2%** — the margin after all variable costs except marketing

GP3 = (Revenue × (1 + CLTV expansion) × GM2%) − Marketing Spend

The optimal spend is where the marginal GP3 from the next SEK of spend equals zero.
Below this point, you're leaving profit on the table. Above it, you're spending
past the point of diminishing returns.

**Seasonal adjustment**

The model computes monthly efficiency indices from historical data — how
effectively each calendar month converts spend into revenue. November (Black Week)
is typically the most efficient; January the least. Planned events (discount
campaigns, product drops) provide additional efficiency boosts.

**Limitations**

- The model assumes the current channel mix ratio when varying total spend.
  In practice, the optimal channel mix may shift at different spend levels.
- Seasonal efficiency indices are backward-looking — future months may differ.
- CLTV expansion is modeled as a flat multiplier; in reality it varies by
  acquisition channel, campaign type, and whether the customer was acquired
  during a discount period.
- For best results, fit the MMM with **New Customer Revenue** as the target.
""")

# ── Export ────────────────────────────────────────────────────

with st.expander("Export"):
    csv_plan = plan_df.to_csv(index=False)
    st.download_button(
        "Download monthly plan (CSV)",
        csv_plan,
        file_name=f"spend_amer_plan_{selected_client}.csv",
        mime="text/csv",
    )

    csv_curve = gp3_df.to_csv(index=False)
    st.download_button(
        "Download GP3 curve data (CSV)",
        csv_curve,
        file_name=f"gp3_curve_{selected_client}.csv",
        mime="text/csv",
    )

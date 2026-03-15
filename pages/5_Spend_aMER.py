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
import datetime

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

# ── Load events ─────────────────────────────────────────────

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

# ── Compute historical context ──────────────────────────────

model_df = st.session_state.get("model_df")
seasonal_indices = compute_seasonal_indices(results, model_df)
event_boosts = compute_event_boosts(results, model_df, events_df)
monthly_organic = compute_monthly_organic(results, model_df)
avg_organic = np.mean(list(monthly_organic.values()))
historical = compute_historical_backcheck(results, model_df)

historical_max_monthly_spend = 0
if historical is not None and len(historical) > 0:
    historical_max_monthly_spend = historical["total_spend"].max()


# ═══════════════════════════════════════════════════════════════
# UNIT ECONOMICS INPUTS
# ═══════════════════════════════════════════════════════════════

st.subheader("Unit Economics")

col1, col2 = st.columns(2)

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

# Breakeven aMER calculations
cltv_mult = 1 + cltv_expansion / 100
gm2_frac = gm2_pct / 100
breakeven_amer_fo = 1 / gm2_frac
breakeven_amer_365d = 1 / (cltv_mult * gm2_frac)

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


# ═══════════════════════════════════════════════════════════════
# GP3 OPTIMIZATION CURVE
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("GP3 Optimization")

# Find optimal spend
optimal = find_optimal_spend(
    results, gm2_pct, cltv_expansion,
    organic_weekly_revenue=avg_organic,
    seasonal_multiplier=1.0,
)

# Generate GP3 curve
gp3_df = compute_gp3_curve(
    results, gm2_pct, cltv_expansion,
    organic_weekly_revenue=avg_organic,
    seasonal_multiplier=1.0,
    n_points=200,
    max_spend_mult=3.0,
)

# ── Dual-axis chart: GP3 (first-order + 365D) + New Customer Revenue ──

fig = make_subplots(specs=[[{"secondary_y": True}]])

# 365D GP3 curve (primary — the optimization target)
fig.add_trace(
    go.Scatter(
        x=gp3_df["monthly_spend"],
        y=gp3_df["gp3_365d_monthly"],
        name="GP3 365D (monthly)",
        line=dict(color=ORANGE, width=3),
        fill="tozeroy",
        fillcolor="rgba(245, 133, 24, 0.08)",
    ),
    secondary_y=False,
)

# First-order GP3 curve
fig.add_trace(
    go.Scatter(
        x=gp3_df["monthly_spend"],
        y=gp3_df["gp3_first_order_monthly"],
        name="GP3 first order (monthly)",
        line=dict(color=RED, width=2, dash="dash"),
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

# Mark optimal spend (365D GP3 peak)
fig.add_trace(
    go.Scatter(
        x=[optimal["optimal_monthly_spend"]],
        y=[optimal["optimal_gp3_365d_monthly"]],
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
        y=[optimal["current_gp3_365d_monthly"]],
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
    y=0, line_dash="dash", line_color="rgba(225, 87, 89, 0.4)",
    annotation_text="GP3 = 0", annotation_position="bottom right",
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

with col1:
    st.metric(
        "Optimal monthly spend",
        f"{optimal['optimal_monthly_spend']:,.0f}",
        f"{spend_change:+.0f}% vs current",
    )

with col2:
    st.metric(
        "First-order GP3 /mo",
        f"{optimal['optimal_gp3_first_order_monthly']:,.0f}",
    )

with col3:
    st.metric(
        "365D GP3 /mo",
        f"{optimal['optimal_gp3_365d_monthly']:,.0f}",
    )

with col4:
    st.metric(
        "aMER at optimal",
        f"{optimal['amer_at_optimal']:.2f}x",
        f"FO break-even: {breakeven_amer_fo:.1f}x / 365D: {breakeven_amer_365d:.1f}x",
    )


# ═══════════════════════════════════════════════════════════════
# FORWARD EVENT CALENDAR
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Forward Event Calendar")

st.markdown(
    "Review and edit planned events for the next 12 months. These feed into "
    "the monthly spend plan — discount periods and product drops boost "
    "channel efficiency, allowing higher spend."
)

# Show upcoming events from events_df
today = datetime.date.today()
if events_df is not None and not events_df.empty:
    edf = events_df.copy()
    edf["week_start"] = pd.to_datetime(edf["week_start"])
    future_events = edf[edf["week_start"] >= pd.Timestamp(today)].sort_values("week_start")

    if not future_events.empty:
        display_events = future_events[["week_start", "discount_campaign", "product_drop", "holiday", "notes"]].copy()
        display_events["week_start"] = display_events["week_start"].dt.strftime("%Y-%m-%d")
        display_events.columns = ["Week", "Discount (0/1/2)", "Product Drop", "Holiday", "Notes"]

        edited_events = st.data_editor(
            display_events,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "Discount (0/1/2)": st.column_config.NumberColumn(
                    min_value=0, max_value=2, step=1,
                    help="0 = none, 1 = light discount, 2 = heavy discount"
                ),
                "Product Drop": st.column_config.NumberColumn(min_value=0, max_value=1, step=1),
                "Holiday": st.column_config.NumberColumn(min_value=0, max_value=1, step=1),
            },
        )
        st.caption("Edit events above and they'll be reflected in the monthly spend plan below.")
    else:
        st.info("No future events found in the event calendar. Add events on the **Event Calendar** page.")
else:
    st.info("No event calendar loaded. Add events on the **Event Calendar** page to enable event-adjusted planning.")

# Show data-driven event boosts
with st.expander("Event efficiency boosts (data-driven)"):
    st.markdown(
        "These multipliers are computed from historical data by comparing channel efficiency "
        "(revenue per SEK of spend) during event weeks vs non-event baseline weeks."
    )
    boost_data = [
        {"Event Type": "Heavy discount campaign", "Efficiency Multiplier": f"{event_boosts['heavy_discount']:.2f}x",
         "Interpretation": f"Spend converts {(event_boosts['heavy_discount']-1)*100:+.0f}% {'better' if event_boosts['heavy_discount'] > 1 else 'worse'} during heavy discounts"},
        {"Event Type": "Light discount campaign", "Efficiency Multiplier": f"{event_boosts['light_discount']:.2f}x",
         "Interpretation": f"Spend converts {(event_boosts['light_discount']-1)*100:+.0f}% {'better' if event_boosts['light_discount'] > 1 else 'worse'} during light discounts"},
        {"Event Type": "Product drop", "Efficiency Multiplier": f"{event_boosts['product_drop']:.2f}x",
         "Interpretation": f"Spend converts {(event_boosts['product_drop']-1)*100:+.0f}% {'better' if event_boosts['product_drop'] > 1 else 'worse'} around product launches"},
    ]
    st.dataframe(pd.DataFrame(boost_data), hide_index=True, use_container_width=True)

    if model_df is None:
        st.caption(
            "Event boosts are currently 1.0x (neutral) because the model was loaded from saved results. "
            "Re-run the model on Client Overview to enable data-driven event boosts."
        )


# ═══════════════════════════════════════════════════════════════
# MONTHLY SPEND PLAN
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Monthly Spend Plan")

st.markdown(
    "Recommended spend by month, adjusted for **seasonal efficiency** "
    "and **planned events**."
)

plan_df = monthly_spend_plan(
    results, gm2_pct, cltv_expansion,
    seasonal_indices, monthly_organic, event_boosts,
    months_ahead=12,
    events_df=events_df,
    historical_max_monthly_spend=historical_max_monthly_spend,
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
        y=plan_df["estimated_monthly_gp3_365d"],
        name="Projected 365D GP3",
        line=dict(color=GREEN, width=2),
        mode="lines+markers",
        marker=dict(size=6),
    ),
    secondary_y=True,
)

fig_plan.add_trace(
    go.Scatter(
        x=plan_df["month_name"],
        y=plan_df["estimated_monthly_gp3_fo"],
        name="Projected first-order GP3",
        line=dict(color=RED, width=1, dash="dash"),
        mode="lines",
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
    "estimated_monthly_gp3_fo", "estimated_monthly_gp3_365d", "amer",
]].copy()

plan_display.columns = [
    "Month", "Seasonal", "Events",
    "Rec. Spend", "Est. Revenue", "GP3 (FO)", "GP3 (365D)", "aMER",
]

for col in ["Rec. Spend", "Est. Revenue", "GP3 (FO)", "GP3 (365D)"]:
    plan_display[col] = plan_display[col].apply(lambda x: f"{x:,.0f}")
plan_display["aMER"] = plan_display["aMER"].apply(lambda x: f"{x:.2f}x")

st.dataframe(plan_display, hide_index=True, use_container_width=True)

# Flag months exceeding historical range
if plan_df["exceeds_historical"].any():
    flagged = plan_df[plan_df["exceeds_historical"]]["month_name"].tolist()
    st.warning(
        f"The following months exceed 150% of the highest historical monthly spend "
        f"({historical_max_monthly_spend:,.0f} SEK): **{', '.join(flagged)}**. "
        f"Projections at these levels are extrapolations — proceed with caution and "
        f"scale gradually."
    )


# ═══════════════════════════════════════════════════════════════
# HISTORICAL BACKCHECK
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Historical Backcheck")

st.markdown(
    "Compare recommended spend levels against actual historical data. "
    "If you spent 500K/month at a 1.8x aMER last year, projecting 1.5M/month "
    "at the same aMER is unlikely without a strong growth trend."
)

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

    # Breakeven lines
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
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )
    fig_hist.update_xaxes(title_text="")
    fig_hist.update_yaxes(title_text="Monthly spend (SEK)", secondary_y=False)
    fig_hist.update_yaxes(title_text="aMER", secondary_y=True)

    st.plotly_chart(fig_hist, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Historical avg. monthly spend", f"{historical['total_spend'].mean():,.0f}")
    with col2:
        st.metric("Historical max monthly spend", f"{historical_max_monthly_spend:,.0f}")
    with col3:
        st.metric("Historical avg. aMER", f"{historical['amer'].mean():.2f}x")

    # Trend check: is spend trending up or down?
    if len(historical) >= 6:
        recent_spend = historical.tail(3)["total_spend"].mean()
        earlier_spend = historical.head(3)["total_spend"].mean()
        trend_pct = (recent_spend - earlier_spend) / (earlier_spend + 1e-8) * 100
        if trend_pct > 20:
            st.info(f"Spend trend is **growing** ({trend_pct:+.0f}% recent vs earlier). This supports scaling up.")
        elif trend_pct < -20:
            st.warning(f"Spend trend is **declining** ({trend_pct:+.0f}% recent vs earlier). Scaling projections may be aggressive.")
        else:
            st.caption(f"Spend has been relatively stable ({trend_pct:+.0f}% change).")

else:
    st.info(
        "Historical backcheck requires the model data (model_df) in session state. "
        "Re-run the model on **Client Overview** to enable."
    )


# ═══════════════════════════════════════════════════════════════
# CHANNEL ALLOCATION
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Channel Allocation at Optimal Spend")

alloc_df = optimize_channel_allocation(
    results,
    total_weekly_spend=optimal["optimal_weekly_spend"],
    seasonal_multiplier=1.0,
)

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
    "As spend increases, aMER decreases due to saturation. "
    "Two breakeven thresholds are shown: first-order (covers costs on the first "
    "transaction) and 365D (covers costs when accounting for repeat purchases)."
)

amer_df = gp3_df[gp3_df["weekly_spend"] > 0].copy()

fig_amer = go.Figure()

fig_amer.add_trace(go.Scatter(
    x=amer_df["monthly_spend"],
    y=amer_df["amer"],
    name="aMER",
    line=dict(color=TEAL, width=3),
))

# First-order breakeven
fig_amer.add_hline(
    y=breakeven_amer_fo,
    line_dash="dash",
    line_color=RED,
    annotation_text=f"FO breakeven ({breakeven_amer_fo:.1f}x)",
    annotation_position="top right",
    annotation_font_color=RED,
)

# 365D breakeven
fig_amer.add_hline(
    y=breakeven_amer_365d,
    line_dash="dot",
    line_color=GREEN,
    annotation_text=f"365D breakeven ({breakeven_amer_365d:.1f}x)",
    annotation_position="bottom right",
    annotation_font_color=GREEN,
)

# Mark current aMER
fig_amer.add_trace(go.Scatter(
    x=[optimal["current_monthly_spend"]],
    y=[optimal["current_amer"]],
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
# SEASONAL INDICES
# ═══════════════════════════════════════════════════════════════

with st.expander("Seasonal efficiency indices"):
    st.markdown(
        "Computed from historical data — how efficiently each calendar month "
        "converts ad spend into revenue relative to the yearly average."
    )

    idx_df = pd.DataFrame([
        {"Month": pd.Timestamp(f"2024-{m:02d}-01").strftime("%B"),
         "Index": f"{seasonal_indices.get(m, 1.0):.2f}"}
        for m in range(1, 13)
    ])
    st.dataframe(idx_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# ABOUT + EXPORT
# ═══════════════════════════════════════════════════════════════

with st.expander("About this model"):
    st.markdown("""
**GP3 = (Revenue × (1 + CLTV expansion) × GM2%) − Marketing Spend**

The MMM gives us saturation curves that show diminishing returns on ad spend.
This model layers unit economics on top to find the spend level that maximizes
profit — the point where the marginal return from one more SEK of spend exactly
covers its cost.

**Two breakeven thresholds:**
- **First-order breakeven aMER** = 1 / GM2%. At 50% GM2, this is 2.0x.
  Below this, you lose money on the first transaction.
- **365D breakeven aMER** = 1 / (GM2% × (1 + CLTV)). At 50% GM2 and 30% CLTV,
  this is 1.54x. Below this, you lose money even accounting for repeat purchases.

**Historical backcheck:**
Projections are validated against actual spend history. If the model suggests
significantly higher spend than you've ever done, it's extrapolating beyond
observed data — scale gradually and measure real aMER at each level.
""")

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

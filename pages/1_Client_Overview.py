"""
Page 1: Client Overview

Revenue decomposition, model fit, and high-level channel contributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import sys
import logging

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import fetch_client_data, load_config, process_revenue_csvs
from data.process import merge_channel_data, prepare_model_input, get_spend_columns
from data.events import load_events, generate_event_template
from model.mmm import create_model, MMMResults
from model.diagnostics import assess_model_quality

logging.basicConfig(level=logging.INFO)

st.title("Client Overview")

# ── Get selected client ──────────────────────────────────────

selected_client = st.session_state.get("selected_client", "juniper")
client_cfg = st.session_state.get("client_config", {})
config = st.session_state.get("config", load_config())

st.header(client_cfg.get("display_name", selected_client))

# ── Plotly theme helper ──────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)
ORANGE = "#F58518"
TEAL = "#76B7B2"

# ── Data Controls ────────────────────────────────────────────

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    date_from = st.date_input("Data from", value=pd.Timestamp("2024-01-01"))
with col2:
    date_to = st.date_input("Data to", value=pd.Timestamp.now().date())
with col3:
    target_type = st.selectbox(
        "Revenue target",
        ["Total Revenue", "New Customer Revenue", "Returning Customer Revenue"],
        index=0,
        help="Which revenue metric to model. 'New Customer Revenue' is recommended "
             "for evaluating paid media effectiveness."
    )

target_col_map = {
    "Total Revenue": "revenue",
    "New Customer Revenue": "new_revenue",
    "Returning Customer Revenue": "returning_revenue",
}

# ── Revenue Data Upload ─────────────────────────────────────

st.markdown("#### Revenue Data")
st.markdown(
    "Upload weekly Shopify analytics exports (**Sales over time**) for "
    "**new customers** and **returning customers** separately. "
    "Expected columns: `Week`, `Total returns`, `Net sales`."
)

rev_col1, rev_col2 = st.columns(2)
with rev_col1:
    new_cust_file = st.file_uploader("New customers CSV", type="csv", key="new_cust_csv")
with rev_col2:
    ret_cust_file = st.file_uploader("Returning customers CSV", type="csv", key="ret_cust_csv")

# Process uploaded revenue CSVs and save to data/ for persistence
revenue_updated = False
if new_cust_file and ret_cust_file:
    try:
        new_df = pd.read_csv(new_cust_file)
        ret_df = pd.read_csv(ret_cust_file)
        revenue_df = process_revenue_csvs(new_df, ret_df)
        # Save to data/ so the model run can use it
        rev_csv_path = Path(__file__).parent.parent / "data" / f"{selected_client}_shopify_weekly.csv"
        revenue_df.to_csv(rev_csv_path, index=False)
        st.success(f"Revenue data processed: {len(revenue_df)} weeks, total revenue {revenue_df['revenue'].sum():,.0f} SEK")
        revenue_updated = True
    except Exception as e:
        st.error(f"Error processing revenue CSVs: {e}")
        st.info("Make sure each file has columns: Week, Total returns, Net sales")

# Check if pre-built revenue CSV exists
rev_csv_path = Path(__file__).parent.parent / "data" / f"{selected_client}_shopify_weekly.csv"
if not revenue_updated and rev_csv_path.exists():
    st.caption(f"Using existing revenue data from `{rev_csv_path.name}`")
elif not revenue_updated:
    st.warning("No revenue data available. Upload Shopify analytics CSVs above.")

# ── Run Model ────────────────────────────────────────────────

run_clicked = st.button("Fetch Data & Run Model", type="primary", use_container_width=True)

results_dir = Path(f"results/{selected_client}")

if run_clicked:
    with st.spinner("Fetching channel data..."):
        try:
            raw_data = fetch_client_data(
                selected_client, config,
                date_from=str(date_from),
                date_to=str(date_to),
            )
            st.success(f"Data fetched: {len(raw_data)} sources")

            # Show raw data summary
            for source, df in raw_data.items():
                status = f"{len(df)} rows" if not df.empty else "empty"
                st.caption(f"  {source}: {status}")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.info("Make sure WINDSOR_API_KEY is set. You can also upload CSV files below.")
            raw_data = None

    if raw_data:
        with st.spinner("Processing data..."):
            merged = merge_channel_data(raw_data, str(date_from), str(date_to))

            # Load events
            events_path = client_cfg.get("events_csv", "")
            if events_path and Path(events_path).exists():
                events_df = load_events(events_path)
            else:
                events_df = None

            model_df = prepare_model_input(merged, events_df)
            st.session_state["model_df"] = model_df

        # Check for revenue data
        target = target_col_map.get(target_type, "revenue")
        if target not in model_df.columns:
            target = "revenue"
        if target not in model_df.columns or model_df[target].sum() == 0:
            st.error("No revenue data available. Check that the Shopify CSV exists in the data/ folder.")
            st.stop()

        # Check for active spend channels before attempting model fit
        active_spend_cols = [c for c in model_df.columns if c.endswith("_spend") and model_df[c].sum() > 0]
        if "email_opens" in model_df.columns and model_df["email_opens"].sum() > 0:
            active_spend_cols.append("email_opens")

        if not active_spend_cols:
            st.warning("No ad spend data available. Connect your ad platforms in Windsor.ai to run the full MMM.")
            st.session_state["mmm_results"] = None
            # Store revenue-only data for the overview below
            st.session_state["revenue_only"] = True
        else:
            with st.spinner("Fitting MMM (this may take a minute)..."):
                model = create_model(config.get("model", {}))
                results = model.fit(model_df, target_col=target)

                # Save results
                results_dir.mkdir(parents=True, exist_ok=True)
                try:
                    results.save(str(results_dir), model_df=model_df)
                except TypeError:
                    # Fallback if save() doesn't accept model_df yet (stale cache)
                    results.save(str(results_dir))
                    # Persist model_df separately
                    import pickle as _pkl
                    with open(f"{results_dir}/model_df.pkl", "wb") as _f:
                        _pkl.dump(model_df, _f)
                st.session_state["mmm_results"] = results
                st.session_state["revenue_only"] = False

            st.success("Model fitted successfully!")

# ── Load existing results ────────────────────────────────────

results = st.session_state.get("mmm_results")
if results is None:
    # Try relative path first, then absolute path from project root
    for base in [Path("."), Path(__file__).parent.parent]:
        candidate = base / "results" / selected_client
        if (candidate / "results.pkl").exists():
            results = MMMResults.load(str(candidate))
            if results is not None:
                results_dir = candidate
                st.session_state["mmm_results"] = results
                break

# ── Revenue-only dashboard (when no ad data is available) ────

model_df = st.session_state.get("model_df")
revenue_only = st.session_state.get("revenue_only", False)

if results is None and model_df is not None and "revenue" in model_df.columns and model_df["revenue"].sum() > 0:
    # Show revenue overview even without the full MMM
    st.markdown("---")
    st.subheader("Revenue Overview")

    total_rev = model_df["revenue"].sum()
    total_orders = model_df["orders"].sum() if "orders" in model_df.columns else 0
    avg_weekly_rev = model_df["revenue"].mean()
    n_weeks = len(model_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"{total_rev:,.0f}")
    with col2:
        st.metric("Total Orders", f"{total_orders:,.0f}")
    with col3:
        st.metric("Avg Weekly Revenue", f"{avg_weekly_rev:,.0f}")
    with col4:
        st.metric("Weeks of Data", f"{n_weeks}")

    # Revenue time series
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=model_df["week_start"],
        y=model_df["revenue"],
        name="Weekly Revenue",
        line=dict(color=ORANGE, width=2),
        fill="tozeroy",
        fillcolor="rgba(245, 133, 24, 0.1)",
    ))
    # 4-week moving average
    if len(model_df) >= 4:
        ma4 = model_df["revenue"].rolling(4).mean()
        fig_rev.add_trace(go.Scatter(
            x=model_df["week_start"],
            y=ma4,
            name="4-Week MA",
            line=dict(color=TEAL, width=2, dash="dot"),
        ))
    fig_rev.update_layout(
        yaxis_title="Revenue",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Orders time series
    if "orders" in model_df.columns:
        fig_orders = go.Figure()
        fig_orders.add_trace(go.Scatter(
            x=model_df["week_start"],
            y=model_df["orders"],
            name="Weekly Orders",
            line=dict(color=TEAL, width=2),
        ))
        fig_orders.update_layout(
            yaxis_title="Orders",
            height=300,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_orders, use_container_width=True)

    if revenue_only:
        st.info("Connect ad platform accounts in Windsor.ai to enable the full Marketing Mix Model with channel ROAS and budget optimization.")
    st.stop()

if results is None:
    st.info("No results available. Click 'Fetch Data & Run Model' above, or upload data below.")

    # CSV upload fallback
    st.markdown("### Or upload CSV data")
    st.markdown("Upload a CSV with columns: `week_start`, `revenue`, `meta_spend`, `google_ads_spend`, etc.")
    uploaded = st.file_uploader("Upload model data CSV", type="csv")
    if uploaded:
        model_df = pd.read_csv(uploaded, parse_dates=["week_start"])
        st.session_state["model_df"] = model_df
        st.dataframe(model_df.head())

        if st.button("Run Model on Uploaded Data"):
            model = create_model(config.get("model", {}))
            spend_cols = get_spend_columns(model_df)
            if not spend_cols:
                st.error("No active spend columns found in the uploaded CSV.")
            else:
                results = model.fit(model_df, target_col="revenue", spend_cols=spend_cols)
                results_dir.mkdir(parents=True, exist_ok=True)
                results.save(str(results_dir))
                st.session_state["mmm_results"] = results
                st.rerun()

    st.stop()

# ═══════════════════════════════════════════════════════════════
# RESULTS DASHBOARD (full MMM results available)
# ═══════════════════════════════════════════════════════════════

# ── Summary Cards ────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

total_revenue = results.actual.sum()
# Only sum actual paid ad spend (exclude email opens which aren't spend)
paid_roas = results.channel_roas[results.channel_roas["channel"] != "email"]
total_spend = paid_roas["total_spend"].sum()
blended_roas = total_revenue / (total_spend + 1e-8)

with col1:
    st.metric("Total Revenue", f"{total_revenue:,.0f}")
with col2:
    st.metric("Total Ad Spend", f"{total_spend:,.0f}")
with col3:
    st.metric("Blended ROAS", f"{blended_roas:.2f}x")
with col4:
    st.metric("Model Fit (R²)", f"{results.r_squared:.3f}")

# ── Historical Spend Build-Up by Channel ─────────────────────

st.subheader("Weekly Spend by Channel")

# Load model_df from session or disk if needed
if model_df is None:
    for base in [Path("."), Path(__file__).parent.parent]:
        mdf_path = base / "results" / selected_client / "model_df.pkl"
        if mdf_path.exists():
            with open(mdf_path, "rb") as _f:
                model_df = pickle.load(_f)
                st.session_state["model_df"] = model_df
            break

if model_df is not None:
    spend_cols = [c for c in model_df.columns if c.endswith("_spend") and model_df[c].sum() > 0]
    if spend_cols:
        # Build stacked area chart
        CHANNEL_COLORS = ["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]
        fig_spend = go.Figure()
        for i, col in enumerate(spend_cols):
            ch_name = col.replace("_spend", "").replace("_", " ").title()
            fig_spend.add_trace(go.Scatter(
                x=model_df["week_start"],
                y=model_df[col],
                name=ch_name,
                stackgroup="spend",
                line=dict(width=0.5, color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)]),
                fillcolor=CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
            ))
        fig_spend.update_layout(
            yaxis_title="Spend (SEK)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.caption("No spend data available for build-up chart.")
else:
    st.caption("Run the model to see spend build-up by channel.")

# ── Marketing Spend Waterfall ─────────────────────────────────

st.subheader("Marketing Spend by Channel")

spend_waterfall_data = []
if model_df is not None:
    spend_cols_wf = [c for c in model_df.columns if c.endswith("_spend") and model_df[c].sum() > 0]
    for col in spend_cols_wf:
        ch_name = col.replace("_spend", "").replace("_", " ").title()
        spend_waterfall_data.append({"component": ch_name, "value": model_df[col].sum()})
else:
    # Fallback to ROAS table totals
    for _, row in results.channel_roas.iterrows():
        if row["channel"] == "email":
            continue
        ch_name = row["channel"].replace("_", " ").title()
        spend_waterfall_data.append({"component": ch_name, "value": row["total_spend"]})

if spend_waterfall_data:
    sw_df = pd.DataFrame(spend_waterfall_data)
    sw_df = pd.concat([sw_df, pd.DataFrame([{"component": "Total", "value": sw_df["value"].sum()}])], ignore_index=True)
    sw_measures = ["relative"] * (len(sw_df) - 1) + ["total"]

    CHANNEL_COLORS_WF = ["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]
    bar_colors = CHANNEL_COLORS_WF[:len(sw_df) - 1] + ["#59A14F"]

    fig_spend_wf = go.Figure(go.Waterfall(
        x=sw_df["component"],
        y=sw_df["value"],
        measure=sw_measures,
        textposition="outside",
        text=[f"{v:,.0f}" for v in sw_df["value"]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": ORANGE}},
        decreasing={"marker": {"color": "#E15759"}},
        totals={"marker": {"color": "#59A14F"}},
    ))
    fig_spend_wf.update_layout(
        title="Total Spend by Channel",
        yaxis_title="Spend (SEK)",
        showlegend=False,
        height=400,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_spend_wf, use_container_width=True)

# ── Revenue Decomposition Waterfall ──────────────────────────

st.subheader("Revenue Decomposition")

contrib_df = results.channel_contributions
total_rev = results.actual.sum()
baseline_total = results.baseline_contribution.sum()

# Build waterfall data
waterfall_data = [{"component": "Baseline (Organic)", "value": baseline_total}]
for col in contrib_df.columns:
    if col == "week_start":
        continue
    waterfall_data.append({"component": col.replace("_", " ").title(), "value": contrib_df[col].sum()})

# Add controls
ctrl_df = results.control_contributions
for col in ctrl_df.columns:
    if col == "week_start":
        continue
    val = ctrl_df[col].sum()
    if abs(val) > 0.01:
        waterfall_data.append({"component": col.replace("_", " ").title(), "value": val})

wf_df = pd.DataFrame(waterfall_data)

# Add total bar at the end
wf_df = pd.concat([wf_df, pd.DataFrame([{"component": "Total", "value": wf_df["value"].sum()}])], ignore_index=True)
measures = ["relative"] * (len(wf_df) - 1) + ["total"]

fig_waterfall = go.Figure(go.Waterfall(
    x=wf_df["component"],
    y=wf_df["value"],
    measure=measures,
    textposition="outside",
    text=[f"{v:,.0f}" for v in wf_df["value"]],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": ORANGE}},
    decreasing={"marker": {"color": "#E15759"}},
    totals={"marker": {"color": "#59A14F"}},
))
fig_waterfall.update_layout(
    title="Revenue Attribution by Component",
    yaxis_title="Revenue",
    showlegend=False,
    height=400,
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_waterfall, use_container_width=True)

# ── Actual vs Predicted Time Series ──────────────────────────

st.subheader("Actual vs. Model Predicted Revenue")

if model_df is not None:
    weeks = model_df["week_start"]
else:
    weeks = pd.date_range(results.date_range[0], periods=results.n_weeks, freq="W-MON")

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=weeks, y=results.actual, name="Actual", line=dict(color=ORANGE, width=2)))
fig_ts.add_trace(go.Scatter(x=weeks, y=results.predicted, name="Predicted", line=dict(color=TEAL, width=2, dash="dot")))
fig_ts.update_layout(
    yaxis_title="Revenue",
    height=350,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_ts, use_container_width=True)

# ── ROAS Table ───────────────────────────────────────────────

st.subheader("Channel ROAS")

roas_display = results.channel_roas.copy()

# Separate email (uses opens, not spend) from paid channels
is_email = roas_display["channel"] == "email"
paid_display = roas_display[~is_email].copy()
email_display = roas_display[is_email].copy()

total_paid_spend = paid_display["total_spend"].sum()

# Flag low-spend channels (< 5% of total paid spend)
paid_display["spend_share"] = paid_display["total_spend"] / (total_paid_spend + 1e-8) * 100
paid_display["is_low_spend"] = paid_display["spend_share"] < 5

# Format display columns for paid channels
paid_display["channel_display"] = paid_display.apply(
    lambda r: r["channel"].replace("_", " ").title() + " *" if r["is_low_spend"] else r["channel"].replace("_", " ").title(),
    axis=1,
)
paid_display["spend_display"] = paid_display.apply(
    lambda r: f"{r['total_spend']:,.0f} ({r['spend_share']:.1f}%)", axis=1
)
paid_display["contribution_display"] = paid_display["total_contribution"].apply(lambda x: f"{x:,.0f}")
paid_display["90% CI"] = paid_display.apply(lambda r: f"{r['roas_5']:.2f} – {r['roas_95']:.2f}", axis=1)
paid_display["roas_display"] = paid_display["roas_mean"].apply(lambda x: f"{x:.2f}x")

st.dataframe(
    paid_display[["channel_display", "spend_display", "contribution_display", "roas_display", "90% CI"]].rename(columns={
        "channel_display": "Channel",
        "spend_display": "Total Spend",
        "contribution_display": "Attributed Revenue",
        "roas_display": "ROAS",
        "90% CI": "90% Confidence Interval",
    }),
    hide_index=True,
    use_container_width=True,
)

# Show warning for low-spend channels
low_spend_channels = paid_display[paid_display["is_low_spend"]]
if not low_spend_channels.empty:
    channel_names = ", ".join(low_spend_channels["channel"].str.replace("_", " ").str.title())
    st.caption(
        f"\\* **Low-spend channels ({channel_names})**: These channels represent less than "
        f"5% of total ad spend. Their ROAS estimates are unreliable due to limited data — "
        f"the model cannot confidently separate their effect from noise at this scale."
    )

# Show email separately (no ROAS since opens ≠ spend)
if not email_display.empty:
    st.markdown("**Email (Klaviyo)**")
    email_row = email_display.iloc[0]
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.metric("Total Opens (model input)", f"{email_row['total_spend']:,.0f}")
    with ecol2:
        st.metric("Attributed Revenue", f"{email_row['total_contribution']:,.0f}")
    st.caption(
        "Email uses weekly opens as the media variable (not spend), so ROAS is not applicable. "
        "The attributed revenue reflects the model's estimate of revenue driven by email activity."
    )

# ── Actionable Insights ──────────────────────────────────────

st.subheader("Insights & Recommendations")

def generate_insights(results):
    """Generate actionable insights from model results, flagging uncertainties."""
    roas_df = results.channel_roas.copy()
    params = results.channel_params
    total_revenue = results.actual.sum()
    baseline_total = results.baseline_contribution.sum()
    baseline_pct = baseline_total / total_revenue * 100

    insights = []  # list of (emoji, title, body, severity)

    # ── Baseline dominance check ──
    if baseline_pct > 80:
        insights.append((
            "organic",
            "Organic revenue dominates",
            f"**{baseline_pct:.0f}%** of revenue comes from baseline (organic) demand. "
            f"Paid media accounts for a relatively small slice — which may mean the brand "
            f"is already well established, or that the model can't confidently attribute "
            f"revenue to ad spend at current levels.",
            "info",
        ))
    elif baseline_pct > 60:
        insights.append((
            "organic",
            "Strong organic foundation",
            f"**{baseline_pct:.0f}%** of revenue is organic. Paid channels are meaningful "
            f"contributors but the brand doesn't depend on them — a healthy position.",
            "success",
        ))

    # ── Per-channel insights ──
    paid_channels = roas_df[roas_df["channel"] != "email"]
    total_paid_spend = paid_channels["total_spend"].sum()

    for _, row in roas_df.iterrows():
        ch = row["channel"]
        ch_display = ch.replace("_", " ").title()
        is_email = ch == "email"

        if is_email:
            contrib = row["total_contribution"]
            contrib_pct = contrib / total_revenue * 100
            insights.append((
                "email",
                f"Email — {contrib_pct:.1f}% of revenue attributed",
                f"The model attributes **{contrib:,.0f} SEK** in revenue to email activity. "
                f"Since the model uses opens (not spend) as the input, this reflects correlation "
                f"between email engagement and sales — some of which may be driven by already-intent "
                f"customers rather than email itself. Treat this as an upper bound.",
                "info",
            ))
            continue

        roas_mean = row["roas_mean"]
        roas_lo = row["roas_5"]
        roas_hi = row["roas_95"]
        ci_width = roas_hi - roas_lo
        spend_share = row["total_spend"] / (total_paid_spend + 1e-8) * 100
        contrib = row["total_contribution"]

        # Saturation level
        ch_params = params.get(ch, {})
        sat_alpha = ch_params.get("saturation_alpha", 1)
        sat_lam = ch_params.get("saturation_lam", 1)

        # Is the confidence interval too wide to be actionable?
        ci_is_wide = ci_width > 3.0 or (roas_mean > 0 and ci_width / (roas_mean + 1e-8) > 1.5)

        # Low spend → noisy
        is_low_spend = spend_share < 5

        if is_low_spend:
            insights.append((
                "grey",
                f"{ch_display} — too little spend to read clearly",
                f"Only **{spend_share:.1f}%** of total ad budget. The model estimates "
                f"**{roas_mean:.1f}x ROAS** but the 90% CI is {roas_lo:.1f}x – {roas_hi:.1f}x. "
                f"At this scale, the signal is buried in noise. Either scale up to test properly "
                f"or reallocate this budget to higher-confidence channels.",
                "warning",
            ))
        elif ci_is_wide:
            insights.append((
                "grey",
                f"{ch_display} — wide uncertainty range",
                f"ROAS is estimated at **{roas_mean:.1f}x** but the 90% CI spans "
                f"**{roas_lo:.1f}x – {roas_hi:.1f}x** — too wide to act on with high confidence. "
                f"This often means the channel's week-to-week spend variation is low, making it hard "
                f"for the model to isolate its effect. Consider running deliberate spend tests "
                f"(scale up/down for 4–6 weeks) to sharpen the estimate.",
                "warning",
            ))
        elif roas_mean < 1.0 and roas_hi < 1.5:
            insights.append((
                "action",
                f"{ch_display} — below breakeven ({roas_mean:.2f}x)",
                f"ROAS is **{roas_mean:.2f}x** (90% CI: {roas_lo:.2f}x – {roas_hi:.2f}x). "
                f"Even the optimistic end of the range is marginal. Consider reducing spend "
                f"and reinvesting in better-performing channels — unless this channel serves "
                f"a top-of-funnel awareness goal that the model can't fully capture.",
                "error",
            ))
        elif roas_mean < 1.0 and roas_hi >= 1.5:
            insights.append((
                "grey",
                f"{ch_display} — below breakeven but uncertain ({roas_mean:.2f}x)",
                f"Point estimate is **{roas_mean:.2f}x** but the CI reaches up to **{roas_hi:.1f}x**. "
                f"The true ROAS might be above breakeven. Don't cut spend drastically — instead "
                f"gather more data or run controlled spend tests to narrow the range.",
                "warning",
            ))
        elif roas_mean >= 1.0 and roas_mean < 2.0:
            insights.append((
                "ok",
                f"{ch_display} — modest return ({roas_mean:.2f}x)",
                f"ROAS of **{roas_mean:.2f}x** (90% CI: {roas_lo:.2f}x – {roas_hi:.2f}x). "
                f"Positive but not exceptional. Hold current spend. If the saturation curve "
                f"shows room to grow, a modest increase could be worthwhile as a test.",
                "info",
            ))
        elif roas_mean >= 2.0:
            insights.append((
                "action",
                f"{ch_display} — strong return ({roas_mean:.2f}x)",
                f"ROAS of **{roas_mean:.2f}x** (90% CI: {roas_lo:.2f}x – {roas_hi:.2f}x). "
                f"{'The confidence interval is tight — this is a reliable signal. ' if not ci_is_wide else ''}"
                f"Check the saturation curve in Channel Analysis — if the channel isn't heavily "
                f"saturated, scaling spend here is likely the highest-leverage move.",
                "success",
            ))

    # ── Model fit caveat ──
    if results.r_squared < 0.7:
        insights.append((
            "grey",
            "Model fit is moderate — interpret with caution",
            f"R² = {results.r_squared:.2f} means the model explains {results.r_squared*100:.0f}% of "
            f"revenue variance. The remaining {(1-results.r_squared)*100:.0f}% is driven by factors "
            f"not in the model (e.g. PR, word-of-mouth, competitor activity, weather). "
            f"All ROAS estimates should be treated as directional rather than precise.",
            "warning",
        ))
    elif results.r_squared < 0.85:
        insights.append((
            "fit",
            "Model fit is good but not airtight",
            f"R² = {results.r_squared:.2f}. The model captures most revenue patterns but "
            f"there are unexplained fluctuations. ROAS estimates are useful for relative "
            f"comparison between channels, but exact numbers carry some uncertainty.",
            "info",
        ))

    return insights

insights = generate_insights(results)

for emoji_key, title, body, severity in insights:
    icon = {
        "action": ":material/trending_up:",
        "organic": ":material/spa:",
        "email": ":material/mail:",
        "grey": ":material/help:",
        "ok": ":material/check_circle:",
        "fit": ":material/analytics:",
    }.get(emoji_key, ":material/info:")

    if severity == "error":
        st.error(f"**{title}**\n\n{body}", icon=icon)
    elif severity == "warning":
        st.warning(f"**{title}**\n\n{body}", icon=icon)
    elif severity == "success":
        st.success(f"**{title}**\n\n{body}", icon=icon)
    else:
        st.info(f"**{title}**\n\n{body}", icon=icon)

st.caption(
    "These insights are generated from the model's fitted parameters and uncertainty estimates. "
    "They are meant to inform decisions, not replace judgement. MMMs measure correlation-based "
    "attribution — not true incrementality. For high-stakes budget changes, validate with "
    "controlled geo-lift or holdout experiments."
)

# ── Model Quality ────────────────────────────────────────────

with st.expander("Model Diagnostics"):
    checks = assess_model_quality(results)
    for check_name, check_data in checks.items():
        status_icon = {"good": "[OK]", "ok": "[WARN]", "warning": "[!]"}.get(check_data["status"], "[i]")
        value_str = f" ({check_data['value']:.3f})" if "value" in check_data else ""
        st.markdown(f"{status_icon} **{check_name}**{value_str}: {check_data['note']}")

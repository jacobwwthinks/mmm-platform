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

from data.ingest import fetch_client_data, load_config
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

# ── Run Model ────────────────────────────────────────────────

run_clicked = st.button("Fetch Data & Run Model", type="primary", use_container_width=True)

results_dir = Path(f"results/{selected_client}")

if run_clicked:
    with st.spinner("Fetching data from Windsor.ai..."):
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
                results.save(str(results_dir))
                st.session_state["mmm_results"] = results
                st.session_state["revenue_only"] = False

            st.success("Model fitted successfully!")

# ── Load existing results ────────────────────────────────────

results = st.session_state.get("mmm_results")
if results is None and (results_dir / "results.pkl").exists():
    results = MMMResults.load(str(results_dir))
    st.session_state["mmm_results"] = results

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
total_spend = results.channel_roas["total_spend"].sum()
blended_roas = total_revenue / (total_spend + 1e-8)

with col1:
    st.metric("Total Revenue", f"{total_revenue:,.0f}")
with col2:
    st.metric("Total Ad Spend", f"{total_spend:,.0f}")
with col3:
    st.metric("Blended ROAS", f"{blended_roas:.2f}x")
with col4:
    st.metric("Model Fit (R²)", f"{results.r_squared:.3f}")

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

fig_waterfall = go.Figure(go.Waterfall(
    x=wf_df["component"],
    y=wf_df["value"],
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
roas_display["channel"] = roas_display["channel"].str.replace("_", " ").str.title()
roas_display["total_spend"] = roas_display["total_spend"].apply(lambda x: f"{x:,.0f}")
roas_display["total_contribution"] = roas_display["total_contribution"].apply(lambda x: f"{x:,.0f}")
roas_display["90% CI"] = roas_display.apply(lambda r: f"{r['roas_5']:.2f} – {r['roas_95']:.2f}", axis=1)
roas_display["roas_mean"] = roas_display["roas_mean"].apply(lambda x: f"{x:.2f}x")

st.dataframe(
    roas_display[["channel", "total_spend", "total_contribution", "roas_mean", "90% CI"]].rename(columns={
        "channel": "Channel",
        "total_spend": "Total Spend",
        "total_contribution": "Attributed Revenue",
        "roas_mean": "ROAS",
        "90% CI": "90% Confidence Interval",
    }),
    hide_index=True,
    use_container_width=True,
)

# ── Model Quality ────────────────────────────────────────────

with st.expander("Model Diagnostics"):
    checks = assess_model_quality(results)
    for check_name, check_data in checks.items():
        status_icon = {"good": "[OK]", "ok": "[WARN]", "warning": "[!]"}.get(check_data["status"], "[i]")
        value_str = f" ({check_data['value']:.3f})" if "value" in check_data else ""
        st.markdown(f"{status_icon} **{check_name}**{value_str}: {check_data['note']}")

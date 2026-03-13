"""
Page 2: Channel Analysis

Deep dive into individual channel performance:
- Saturation curves
- Adstock visualization
- ROAS with uncertainty
- Contribution over time
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

from model.mmm import MMMResults, geometric_adstock, hill_saturation

st.title("Channel Analysis")

# ── Load Results ─────────────────────────────────────────────

results = st.session_state.get("mmm_results")
if results is None:
    results_dir = Path(f"results/{st.session_state.get('selected_client', 'juniper')}")
    if (results_dir / "results.pkl").exists():
        results = MMMResults.load(str(results_dir))
    else:
        st.info("No model results available. Run the model from the Client Overview page first.")
        st.stop()

# ── ROAS Comparison ──────────────────────────────────────────

st.subheader("Channel ROAS Comparison")

roas_df = results.channel_roas.copy()
roas_df["channel_display"] = roas_df["channel"].str.replace("_", " ").str.title()

fig_roas = go.Figure()

# Error bars showing 90% CI
fig_roas.add_trace(go.Bar(
    x=roas_df["channel_display"],
    y=roas_df["roas_mean"],
    error_y=dict(
        type="data",
        symmetric=False,
        array=roas_df["roas_95"] - roas_df["roas_mean"],
        arrayminus=roas_df["roas_mean"] - roas_df["roas_5"],
        color="rgba(230,237,243,0.3)",
    ),
    marker_color=["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"][:len(roas_df)],
    text=[f"{r:.2f}x" for r in roas_df["roas_mean"]],
    textposition="outside",
))

fig_roas.add_hline(y=1.0, line_dash="dash", line_color="#E15759", annotation_text="Breakeven (1.0x)")
fig_roas.update_layout(
    yaxis_title="ROAS (Return on Ad Spend)",
    height=400,
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)
st.plotly_chart(fig_roas, use_container_width=True)

# ── Channel Selector ─────────────────────────────────────────

st.markdown("---")
channels = list(results.channel_params.keys())
channel_display = {ch: ch.replace("_", " ").title() for ch in channels}

selected = st.selectbox("Select channel for detailed analysis", list(channel_display.values()))
selected_key = [k for k, v in channel_display.items() if v == selected][0]

params = results.channel_params[selected_key]
contrib = results.channel_contributions[selected_key].values

# ── Saturation Curve ─────────────────────────────────────────

st.subheader(f"Saturation Curve — {selected}")

# Generate saturation curve across range of spend levels
spend_range = np.linspace(0, contrib.max() * 3, 200)
saturated = hill_saturation(spend_range, params["saturation_alpha"], params["saturation_lam"])

fig_sat = go.Figure()
fig_sat.add_trace(go.Scatter(
    x=spend_range,
    y=saturated,
    mode="lines",
    line=dict(color="#F58518", width=3),
    name="Saturation curve",
))

# Mark current average spend level
current_avg = contrib.mean()
current_sat = hill_saturation(np.array([current_avg]), params["saturation_alpha"], params["saturation_lam"])[0]
fig_sat.add_trace(go.Scatter(
    x=[current_avg],
    y=[current_sat],
    mode="markers+text",
    marker=dict(size=15, color="#E15759"),
    text=[f"Current ({current_sat:.0%} saturated)"],
    textposition="top right",
    name="Current spend level",
))

fig_sat.update_layout(
    xaxis_title="Spend Level (adstocked)",
    yaxis_title="Effect (0 = none, 1 = fully saturated)",
    height=350,
    yaxis_range=[0, 1.05],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)

st.plotly_chart(fig_sat, use_container_width=True)

saturation_pct = current_sat * 100
if saturation_pct < 40:
    st.success(f"**{selected}** is at {saturation_pct:.0f}% saturation — significant room for growth")
elif saturation_pct < 70:
    st.warning(f"**{selected}** is at {saturation_pct:.0f}% saturation — moderate room for growth")
else:
    st.error(f"**{selected}** is at {saturation_pct:.0f}% saturation — diminishing returns, consider reducing spend")

# ── Adstock Decay ────────────────────────────────────────────

st.subheader(f"Adstock Decay — {selected}")

decay = params["adstock_decay"]
max_lag = 8
lag_weights = [decay ** i for i in range(max_lag + 1)]
lag_weeks = list(range(max_lag + 1))

fig_adstock = go.Figure()
fig_adstock.add_trace(go.Bar(
    x=lag_weeks,
    y=lag_weights,
    marker_color="#F58518",
    text=[f"{w:.1%}" for w in lag_weights],
    textposition="outside",
))
fig_adstock.update_layout(
    xaxis_title="Weeks After Ad Exposure",
    yaxis_title="Remaining Effect",
    height=300,
    yaxis_range=[0, 1.1],
    xaxis=dict(tickmode="linear"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)
st.plotly_chart(fig_adstock, use_container_width=True)

half_life = np.log(0.5) / np.log(decay + 1e-8)
st.info(f"**{selected}** has a decay rate of {decay:.2f} — half-life of approximately **{half_life:.1f} weeks**. "
        f"This means {'ads have lasting brand effects' if half_life > 2 else 'most effect happens in the week of exposure'}.")

# ── Contribution Over Time ───────────────────────────────────

st.subheader(f"Weekly Contribution — {selected}")

model_df = st.session_state.get("model_df")
if model_df is not None:
    weeks = model_df["week_start"]
else:
    weeks = pd.date_range(results.date_range[0], periods=results.n_weeks, freq="W-MON")

fig_contrib = go.Figure()
fig_contrib.add_trace(go.Scatter(
    x=weeks,
    y=contrib,
    fill="tozeroy",
    fillcolor="rgba(245, 133, 24, 0.2)",
    line=dict(color="#F58518", width=2),
    name="Channel contribution to revenue",
))
fig_contrib.update_layout(
    yaxis_title="Revenue Contribution",
    height=300,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
)
st.plotly_chart(fig_contrib, use_container_width=True)

# ── Channel Parameters Summary ───────────────────────────────

with st.expander("Fitted Parameters"):
    param_df = pd.DataFrame([{
        "Parameter": "Adstock Decay Rate",
        "Value": f"{params['adstock_decay']:.3f}",
        "Interpretation": f"{'Long memory' if params['adstock_decay'] > 0.4 else 'Short memory'} — "
                         f"{'ads keep working for weeks' if params['adstock_decay'] > 0.4 else 'effect fades quickly'}",
    }, {
        "Parameter": "Saturation Alpha",
        "Value": f"{params['saturation_alpha']:.3f}",
        "Interpretation": f"{'Sharp' if params['saturation_alpha'] > 2 else 'Gradual'} saturation onset",
    }, {
        "Parameter": "Saturation Lambda",
        "Value": f"{params['saturation_lam']:.3f}",
        "Interpretation": "Scale of diminishing returns",
    }, {
        "Parameter": "Beta (Effect Size)",
        "Value": f"{params['beta']:.4f}",
        "Interpretation": "Coefficient linking spend to revenue",
    }])
    st.dataframe(param_df, hide_index=True, use_container_width=True)

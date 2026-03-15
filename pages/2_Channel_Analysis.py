"""
Page 2: Channel Analysis

Deep dive into individual channel performance:
- Saturation curves
- Adstock visualization
- ROAS with uncertainty
- Contribution over time

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mmm import MMMResults, geometric_adstock, hill_saturation
from ui.layout import inject_context_css, render_sidebar, context_block, context_tip, context_separator

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E6EDF3",
    font_family="Inter, sans-serif",
    title_font=dict(size=13, color="#C9D1D9"),
)
ORANGE = "#F58518"

inject_context_css()
render_sidebar()

st.title("Channel Analysis")

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
        "Channel Analysis shows saturation curves, adstock decay, and ROAS for each "
        "media channel. This requires running the Marketing Mix Model, which needs "
        "active ad spend data from at least one platform."
    )
    st.markdown("**To enable this page:**")
    st.markdown(
        "1. Ensure your Windsor.ai API key is set in Streamlit secrets\n"
        "2. Connect at least one ad platform (Meta, Google Ads, Pinterest, etc.)\n"
        "3. Go to **Client Overview** and click **Fetch Data & Run Model**"
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT: data | context
# ═══════════════════════════════════════════════════════════════

# ── Section 1: ROAS Comparison ────────────────────────────────
_m1, _c1 = st.columns([4, 1])
with _m1:
    roas_df = results.channel_roas.copy()
    paid_roas_df = roas_df[roas_df["channel"] != "email"].copy()
    paid_roas_df["channel_display"] = paid_roas_df["channel"].str.replace("_", " ").str.title()

    fig_roas = go.Figure()
    fig_roas.add_trace(go.Bar(
        x=paid_roas_df["channel_display"],
        y=paid_roas_df["roas_mean"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=paid_roas_df["roas_95"] - paid_roas_df["roas_mean"],
            arrayminus=paid_roas_df["roas_mean"] - paid_roas_df["roas_5"],
            color="rgba(230,237,243,0.3)",
        ),
        marker_color=["#F58518", "#76B7B2", "#E15759", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"][:len(paid_roas_df)],
        text=[f"{r:.2f}x" for r in paid_roas_df["roas_mean"]],
        textposition="outside",
    ))
    fig_roas.add_hline(y=1.0, line_dash="dash", line_color="#E15759", annotation_text="Breakeven (1.0x)")
    fig_roas.update_layout(title="Channel ROAS Comparison", yaxis_title="ROAS (Return on Ad Spend)", height=400, showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_roas, use_container_width=True)

    email_roas = roas_df[roas_df["channel"] == "email"]
    if not email_roas.empty:
        email_row = email_roas.iloc[0]
        st.caption(
            f"Email (Klaviyo) is excluded from the ROAS chart — it uses opens as the media variable, not spend. "
            f"Attributed revenue from email: **{email_row['total_contribution']:,.0f}**."
        )

with _c1:
    context_block(
        "ROAS Comparison",
        "Bars show the model's best estimate of return on ad spend "
        "for each paid channel. Error bars are the **90% confidence interval**.\n\n"
        "The red dashed line is breakeven (1.0x) — below this line, the "
        "channel costs more than it returns in attributed revenue."
    )

# ── Channel Selector (full-width) ─────────────────────────────
st.markdown("---")
channels = list(results.channel_params.keys())
channel_display = {ch: ch.replace("_", " ").title() for ch in channels}
selected = st.selectbox("Select channel for detailed analysis", list(channel_display.values()))
selected_key = [k for k, v in channel_display.items() if v == selected][0]
params = results.channel_params[selected_key]
contrib = results.channel_contributions[selected_key].values

# ── Section 2: Saturation Curve ───────────────────────────────
_m2, _c2 = st.columns([4, 1])
with _m2:
    spend_range = np.linspace(0, contrib.max() * 3, 200)
    saturated = hill_saturation(spend_range, params["saturation_alpha"], params["saturation_lam"])

    fig_sat = go.Figure()
    fig_sat.add_trace(go.Scatter(x=spend_range, y=saturated, mode="lines", line=dict(color=ORANGE, width=3), name="Saturation curve"))

    current_avg = contrib.mean()
    current_sat = hill_saturation(np.array([current_avg]), params["saturation_alpha"], params["saturation_lam"])[0]
    fig_sat.add_trace(go.Scatter(
        x=[current_avg], y=[current_sat], mode="markers+text",
        marker=dict(size=15, color="#E15759"),
        text=[f"Current ({current_sat:.0%} saturated)"], textposition="top right",
        name="Current spend level",
    ))
    fig_sat.update_layout(title=f"Saturation Curve — {selected}",
                          xaxis_title="Spend Level (adstocked)", yaxis_title="Effect (0 = none, 1 = fully saturated)",
                          height=350, yaxis_range=[0, 1.05], **PLOTLY_LAYOUT)
    st.plotly_chart(fig_sat, use_container_width=True)

    saturation_pct = current_sat * 100
    if saturation_pct < 40:
        st.success(f"**{selected}** is at {saturation_pct:.0f}% saturation — significant room for growth")
    elif saturation_pct < 70:
        st.warning(f"**{selected}** is at {saturation_pct:.0f}% saturation — moderate room for growth")
    else:
        st.error(f"**{selected}** is at {saturation_pct:.0f}% saturation — diminishing returns, consider reducing spend")

with _c2:
    context_block(
        "Saturation Curve",
        "Shows how additional spend translates into incremental effect. "
        "The S-shaped curve flattens as the channel gets saturated.\n\n"
        "**Low saturation** (< 40%) = room to scale spend.\n"
        "**High saturation** (> 70%) = diminishing returns, budget is "
        "better moved elsewhere.\n\n"
        "The red dot marks where the channel is currently operating."
    )

# ── Section 3: Adstock Decay ─────────────────────────────────
_m3, _c3 = st.columns([4, 1])
with _m3:
    decay = params["adstock_decay"]
    max_lag = 8
    lag_weights = [decay ** i for i in range(max_lag + 1)]
    lag_weeks = list(range(max_lag + 1))

    fig_adstock = go.Figure()
    fig_adstock.add_trace(go.Bar(x=lag_weeks, y=lag_weights, marker_color=ORANGE,
                                  text=[f"{w:.1%}" for w in lag_weights], textposition="outside"))
    fig_adstock.update_layout(title=f"Adstock Decay — {selected}",
                               xaxis_title="Weeks After Ad Exposure", yaxis_title="Remaining Effect",
                               height=350, yaxis_range=[0, 1.3], xaxis=dict(tickmode="linear"),
                               margin=dict(t=60), **PLOTLY_LAYOUT)
    st.plotly_chart(fig_adstock, use_container_width=True)

    half_life = np.log(0.5) / np.log(decay + 1e-8)
    st.info(f"**{selected}** has a decay rate of {decay:.2f} — half-life of approximately **{half_life:.1f} weeks**. "
            f"This means {'ads have lasting brand effects' if half_life > 2 else 'most effect happens in the week of exposure'}.")

with _c3:
    context_block(
        "Adstock Decay",
        "Measures how long an ad's effect lasts after exposure.\n\n"
        "**High decay** (> 0.4) = long memory. Ads keep driving sales for "
        "weeks — typical for brand/awareness campaigns.\n\n"
        "**Low decay** (< 0.2) = short memory. Effect happens immediately "
        "and fades fast — typical for performance/direct response."
    )

# ── Section 4: Weekly Contribution ────────────────────────────
_m4, _c4 = st.columns([4, 1])
with _m4:
    model_df = st.session_state.get("model_df")
    if model_df is not None:
        weeks = model_df["week_start"]
    else:
        weeks = pd.date_range(results.date_range[0], periods=results.n_weeks, freq="W-MON")

    fig_contrib = go.Figure()
    fig_contrib.add_trace(go.Scatter(x=weeks, y=contrib, fill="tozeroy",
                                      fillcolor="rgba(245, 133, 24, 0.2)", line=dict(color=ORANGE, width=2),
                                      name="Channel contribution to revenue"))
    fig_contrib.update_layout(title=f"Weekly Contribution — {selected}", yaxis_title="Revenue Contribution", height=300, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_contrib, use_container_width=True)

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

with _c4:
    context_block(
        "Weekly Contribution",
        "Revenue attributed to this channel over time. "
        "Look for patterns: does the channel contribute more during "
        "campaigns? Are there periods of zero contribution?\n\n"
        "Flat lines may indicate the model struggled to separate "
        "this channel's effect from others."
    )
    context_tip(
        "**Key question:** Is this channel saturated or does it "
        "have room to grow? Cross-reference the saturation curve "
        "with the ROAS estimate to decide where to invest more."
    )

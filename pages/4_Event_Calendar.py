"""
Page 4: Event Calendar

Manage promotional events, product drops, and holidays.
Upload and edit the event calendar CSV that feeds into the model.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.events import load_events, generate_event_template

st.title("Event Calendar")

selected_client = st.session_state.get("selected_client", "juniper")
client_cfg = st.session_state.get("client_config", {})

st.header(f"Events — {client_cfg.get('display_name', selected_client)}")

# ── Current Calendar ─────────────────────────────────────────

events_path = client_cfg.get("events_csv", f"events/{selected_client}_events.csv")
events_dir = Path(events_path).parent
events_dir.mkdir(parents=True, exist_ok=True)

if Path(events_path).exists():
    events_df = load_events(events_path)
    st.success(f"Loaded {len(events_df)} event weeks from {events_path}")
else:
    st.info("No event calendar found. Generate a template or upload one below.")
    events_df = pd.DataFrame(columns=["week_start", "discount_campaign", "product_drop", "holiday", "notes"])

# ── Generate Template ────────────────────────────────────────

with st.expander("Generate Event Template"):
    col1, col2 = st.columns(2)
    with col1:
        template_from = st.date_input("From", value=pd.Timestamp("2024-01-01"), key="tmpl_from")
    with col2:
        template_to = st.date_input("To", value=pd.Timestamp.now().date(), key="tmpl_to")

    auto_holidays = st.checkbox("Auto-detect holidays (Black Friday, Christmas, etc.)", value=True)

    if st.button("Generate Template"):
        events_df = generate_event_template(
            str(template_from), str(template_to), events_path, auto_holidays=auto_holidays
        )
        st.success(f"Template generated with {len(events_df)} weeks")
        st.rerun()

# ── Upload Calendar ──────────────────────────────────────────

with st.expander("Upload Event Calendar CSV"):
    st.markdown("""
    Upload a CSV with columns:
    - `week_start` — Monday date (YYYY-MM-DD)
    - `discount_campaign` — 1 if sale/discount is active, 0 otherwise
    - `product_drop` — 1 if new product launched, 0 otherwise
    - `holiday` — 1 if major shopping holiday, 0 otherwise
    - `notes` — free text description (optional)
    """)

    uploaded = st.file_uploader("Upload CSV", type="csv", key="events_upload")
    if uploaded:
        new_events = pd.read_csv(uploaded, parse_dates=["week_start"])
        new_events.to_csv(events_path, index=False)
        st.success("Calendar uploaded and saved!")
        events_df = new_events
        st.rerun()

# ── SMS Spend Upload ─────────────────────────────────────────

with st.expander("Upload SMS Spend Data"):
    st.markdown("""
    Upload weekly SMS spend as CSV:
    - `week_start` — Monday date (YYYY-MM-DD)
    - `spend` — SMS campaign spend for that week
    """)

    sms_uploaded = st.file_uploader("Upload SMS CSV", type="csv", key="sms_upload")
    if sms_uploaded:
        sms_path = client_cfg.get("sms_csv", f"events/{selected_client}_sms_spend.csv")
        sms_df = pd.read_csv(sms_uploaded, parse_dates=["week_start"])
        sms_df.to_csv(sms_path, index=False)
        st.success(f"SMS spend data saved ({len(sms_df)} weeks)")

# ── Edit Calendar ────────────────────────────────────────────

if not events_df.empty:
    st.subheader("Edit Calendar")

    st.markdown("Mark weeks where discount campaigns, product drops, or holidays occurred.")

    # Only show weeks with events + allow adding new ones
    tab_events, tab_all = st.tabs(["Event Weeks Only", "Full Calendar"])

    with tab_events:
        event_weeks = events_df[
            (events_df["discount_campaign"] == 1) |
            (events_df["product_drop"] == 1) |
            (events_df["holiday"] == 1)
        ].copy()

        if event_weeks.empty:
            st.info("No events marked yet. Switch to 'Full Calendar' to add events.")
        else:
            st.dataframe(
                event_weeks.style.apply(
                    lambda row: ["background-color: #fff3cd" if row["discount_campaign"] == 1
                                else "background-color: #d4edda" if row["product_drop"] == 1
                                else "background-color: #cce5ff" if row["holiday"] == 1
                                else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )

    with tab_all:
        edited_df = st.data_editor(
            events_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "week_start": st.column_config.DateColumn("Week Start", format="YYYY-MM-DD"),
                "discount_campaign": st.column_config.CheckboxColumn("Discount"),
                "product_drop": st.column_config.CheckboxColumn("Drop"),
                "holiday": st.column_config.CheckboxColumn("Holiday"),
                "notes": st.column_config.TextColumn("Notes"),
            },
            height=400,
        )

        if st.button("Save Changes"):
            edited_df.to_csv(events_path, index=False)
            st.success("Calendar saved!")

    # ── Calendar Visualization ───────────────────────────────

    st.subheader("Event Timeline")

    import plotly.graph_objects as go

    fig = go.Figure()

    for event_type, color, label in [
        ("discount_campaign", "#F58518", "Discount Campaigns"),
        ("product_drop", "#59A14F", "Product Drops"),
        ("holiday", "#76B7B2", "Holidays"),
    ]:
        mask = events_df[event_type] == 1
        if mask.any():
            fig.add_trace(go.Scatter(
                x=events_df.loc[mask, "week_start"],
                y=[label] * mask.sum(),
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=label,
                text=events_df.loc[mask, "notes"],
                hovertemplate="%{x}<br>%{text}<extra></extra>",
            ))

    fig.update_layout(
        height=200,
        yaxis_title="",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E6EDF3",
        font_family="Inter, sans-serif",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Summary Stats ────────────────────────────────────────────

if not events_df.empty:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Weeks", len(events_df))
    with col2:
        st.metric("Discount Weeks", int(events_df["discount_campaign"].sum()))
    with col3:
        st.metric("Product Drops", int(events_df["product_drop"].sum()))
    with col4:
        st.metric("Holiday Weeks", int(events_df["holiday"].sum()))

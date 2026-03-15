"""
Page 4: Event Calendar

Manage promotional events, product drops, and holidays.
Split into:
  - Historical events (used for model fitting — read-only view)
  - Forward-looking events (used for spend planning — editable)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.events import load_events, generate_event_template

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

st.title("Event Calendar")

selected_client = st.session_state.get("selected_client", "juniper")
client_cfg = st.session_state.get("client_config", {})
config = st.session_state.get("config")

# Try to get client config from main config if not in session state
if not client_cfg and config:
    client_cfg = config.get("clients", {}).get(selected_client, {})

st.header(f"Events — {client_cfg.get('display_name', selected_client)}")

# ── Load existing calendar ────────────────────────────────

events_path = client_cfg.get("events_csv", f"events/{selected_client}_events.csv")
events_dir = Path(events_path).parent
events_dir.mkdir(parents=True, exist_ok=True)

if Path(events_path).exists():
    events_df = load_events(events_path)
else:
    st.info("No event calendar found. Generate a template or upload one below.")
    events_df = pd.DataFrame(columns=[
        "week_start", "discount_campaign", "product_drop",
        "product_offering", "holiday", "notes",
    ])


# ── Download / Upload controls ────────────────────────────

col_dl, col_ul = st.columns(2)

with col_dl:
    if not events_df.empty:
        csv_data = events_df.to_csv(index=False)
        st.download_button(
            "Download current calendar (CSV)",
            csv_data,
            file_name=f"{selected_client}_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("Download current calendar (CSV)", disabled=True, use_container_width=True)

with col_ul:
    uploaded = st.file_uploader(
        "Upload replacement CSV",
        type="csv",
        key="events_upload",
        label_visibility="collapsed",
    )
    if uploaded:
        new_events = pd.read_csv(uploaded, parse_dates=["week_start"])
        # Validate required columns
        required = {"week_start", "discount_campaign", "product_drop", "holiday"}
        missing = required - set(new_events.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            new_events.to_csv(events_path, index=False)
            st.success("Calendar uploaded and saved!")
            events_df = load_events(events_path)
            st.rerun()


# ── Split into historical vs forward ─────────────────────

today = pd.Timestamp(datetime.date.today())
# "Historical" = weeks before the start of this month
cutoff = pd.Timestamp(datetime.date(today.year, today.month, 1))

if not events_df.empty:
    events_df["week_start"] = pd.to_datetime(events_df["week_start"])
    hist_df = events_df[events_df["week_start"] < cutoff].copy()
    fwd_df = events_df[events_df["week_start"] >= cutoff].copy()
else:
    hist_df = events_df.copy()
    fwd_df = events_df.copy()


# ═══════════════════════════════════════════════════════════════
# FORWARD-LOOKING CALENDAR
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Forward-Looking Events")
st.markdown(
    "Plan your upcoming campaigns, product launches, and promotions. "
    "The **Spend-aMER** model uses these to estimate spend capacity per month. "
    "Months without events assume **no campaign boost** — which significantly "
    "underestimates spend capacity during discount periods."
)

# Check which of next 12 months have events
months_with_events = set()
months_without_events = []
for m in range(12):
    month_num = (today.month - 1 + m) % 12 + 1
    year = today.year + (today.month - 1 + m) // 12
    dt = datetime.date(year, month_num, 1)
    label = dt.strftime("%b %Y")
    if not fwd_df.empty:
        me = fwd_df[
            (fwd_df["week_start"].dt.month == month_num) &
            (fwd_df["week_start"].dt.year == year)
        ]
        if len(me) > 0:
            months_with_events.add(label)
            continue
    months_without_events.append(label)

if months_without_events:
    st.warning(
        f"**Months missing events:** {', '.join(months_without_events)}. "
        "Add planned campaigns below or the Spend-aMER plan will assume no event boost."
    )

# ── Forward event timeline ────────────────────────────────

if not fwd_df.empty:
    import plotly.graph_objects as go

    fig_fwd = go.Figure()

    for event_type, color, label, marker_size in [
        ("discount_campaign", ORANGE, "Discount Campaigns", 14),
        ("product_drop", GREEN, "Product Drops", 12),
        ("holiday", TEAL, "Holidays", 12),
    ]:
        if event_type == "discount_campaign":
            # Show heavy (2) vs light (1) differently
            heavy_mask = fwd_df[event_type] == 2
            light_mask = fwd_df[event_type] == 1
            if heavy_mask.any():
                fig_fwd.add_trace(go.Scatter(
                    x=fwd_df.loc[heavy_mask, "week_start"],
                    y=["Heavy discount"] * heavy_mask.sum(),
                    mode="markers",
                    marker=dict(size=marker_size, color=ORANGE, symbol="square"),
                    name="Heavy discount (2)",
                    text=fwd_df.loc[heavy_mask, "notes"],
                    hovertemplate="%{x|%b %d, %Y}<br>%{text}<extra>Heavy discount</extra>",
                ))
            if light_mask.any():
                fig_fwd.add_trace(go.Scatter(
                    x=fwd_df.loc[light_mask, "week_start"],
                    y=["Light discount"] * light_mask.sum(),
                    mode="markers",
                    marker=dict(size=marker_size - 2, color="rgba(245, 133, 24, 0.5)", symbol="square"),
                    name="Light discount (1)",
                    text=fwd_df.loc[light_mask, "notes"],
                    hovertemplate="%{x|%b %d, %Y}<br>%{text}<extra>Light discount</extra>",
                ))
        else:
            mask = fwd_df[event_type] > 0
            if mask.any():
                fig_fwd.add_trace(go.Scatter(
                    x=fwd_df.loc[mask, "week_start"],
                    y=[label] * mask.sum(),
                    mode="markers",
                    marker=dict(size=marker_size - 2, color=color, symbol="square"),
                    name=label,
                    text=fwd_df.loc[mask, "notes"],
                    hovertemplate="%{x|%b %d, %Y}<br>%{text}<extra></extra>",
                ))

    if "product_offering" in fwd_df.columns:
        offer_mask = fwd_df["product_offering"] > 0
        if offer_mask.any():
            fig_fwd.add_trace(go.Scatter(
                x=fwd_df.loc[offer_mask, "week_start"],
                y=["Product Offerings"] * offer_mask.sum(),
                mode="markers",
                marker=dict(size=10, color="#EDC948", symbol="diamond"),
                name="Product Offerings",
                text=fwd_df.loc[offer_mask, "notes"],
                hovertemplate="%{x|%b %d, %Y}<br>%{text}<extra></extra>",
            ))

    fig_fwd.update_layout(
        height=220,
        yaxis_title="",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_fwd, use_container_width=True)

# ── Editable forward calendar ─────────────────────────────

st.markdown("##### Edit forward events")
st.caption(
    "**discount_campaign**: 0 = none, 1 = light, 2 = heavy (Black Week, Birthday Week). "
    "Add rows for new weeks. Click **Save** when done."
)

# Ensure the editor has columns even if fwd_df is empty
if fwd_df.empty:
    fwd_df = pd.DataFrame({
        "week_start": pd.Series(dtype="datetime64[ns]"),
        "discount_campaign": pd.Series(dtype="int"),
        "product_drop": pd.Series(dtype="int"),
        "product_offering": pd.Series(dtype="int"),
        "holiday": pd.Series(dtype="int"),
        "notes": pd.Series(dtype="str"),
    })

# Make sure product_offering exists
if "product_offering" not in fwd_df.columns:
    fwd_df["product_offering"] = 0

# Column order for editor
edit_cols = ["week_start", "discount_campaign", "product_drop", "product_offering", "holiday", "notes"]
fwd_edit = fwd_df[edit_cols].copy()

edited_fwd = st.data_editor(
    fwd_edit,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "week_start": st.column_config.DateColumn("Week Start", format="YYYY-MM-DD"),
        "discount_campaign": st.column_config.NumberColumn(
            "Discount", min_value=0, max_value=2,
            help="0=none, 1=light, 2=heavy (Black Week/Birthday Week)",
        ),
        "product_drop": st.column_config.CheckboxColumn("Product Drop"),
        "product_offering": st.column_config.CheckboxColumn("Product Offering"),
        "holiday": st.column_config.CheckboxColumn("Holiday"),
        "notes": st.column_config.TextColumn("Notes", width="large"),
    },
    height=min(400, max(150, 35 * (len(fwd_edit) + 2))),
    key="fwd_events_editor",
)

if st.button("Save forward events", type="primary"):
    # Combine historical + edited forward events
    edited_fwd["week_start"] = pd.to_datetime(edited_fwd["week_start"])
    # Drop rows with no week_start (empty added rows)
    edited_fwd = edited_fwd.dropna(subset=["week_start"])

    if "product_offering" not in hist_df.columns:
        hist_df["product_offering"] = 0

    combined = pd.concat([hist_df[edit_cols], edited_fwd[edit_cols]], ignore_index=True)
    combined = combined.sort_values("week_start").reset_index(drop=True)
    combined.to_csv(events_path, index=False)
    st.success(f"Saved {len(combined)} event weeks ({len(hist_df)} historical + {len(edited_fwd)} forward)")
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# HISTORICAL CALENDAR (read-only)
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Historical Events")
st.markdown(
    "Past events used for model fitting. These are read-only here — "
    "edit the CSV directly if corrections are needed."
)

if not hist_df.empty:
    # Timeline chart
    import plotly.graph_objects as go

    fig_hist = go.Figure()

    for event_type, color, label in [
        ("discount_campaign", ORANGE, "Discount Campaigns"),
        ("product_drop", GREEN, "Product Drops"),
        ("holiday", TEAL, "Holidays"),
    ]:
        mask = hist_df[event_type] > 0
        if mask.any():
            fig_hist.add_trace(go.Scatter(
                x=hist_df.loc[mask, "week_start"],
                y=[label] * mask.sum(),
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=label,
                text=hist_df.loc[mask, "notes"],
                hovertemplate="%{x|%b %d, %Y}<br>%{text}<extra></extra>",
            ))

    if "product_offering" in hist_df.columns:
        offer_mask = hist_df["product_offering"] > 0
        if offer_mask.any():
            fig_hist.add_trace(go.Scatter(
                x=hist_df.loc[offer_mask, "week_start"],
                y=["Product Offerings"] * offer_mask.sum(),
                mode="markers",
                marker=dict(size=10, color="#EDC948", symbol="diamond"),
                name="Product Offerings",
                text=hist_df.loc[offer_mask, "notes"],
                hovertemplate="%{x|%b %d, %Y}<br>%{text}<extra></extra>",
            ))

    fig_hist.update_layout(
        height=200,
        yaxis_title="",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Read-only table
    with st.expander(f"View historical events ({len(hist_df)} weeks)"):
        hist_display = hist_df.copy()
        hist_display["week_start"] = hist_display["week_start"].dt.strftime("%Y-%m-%d")
        st.dataframe(hist_display, hide_index=True, use_container_width=True)
else:
    st.info("No historical events found.")


# ═══════════════════════════════════════════════════════════════
# SMS SPEND UPLOAD
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# GENERATE TEMPLATE
# ═══════════════════════════════════════════════════════════════

with st.expander("Generate blank event template"):
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


# ═══════════════════════════════════════════════════════════════
# SUMMARY STATS
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
all_events = events_df if not events_df.empty else pd.DataFrame()

if not all_events.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total weeks", len(all_events))
    with col2:
        st.metric("Historical", len(hist_df))
    with col3:
        st.metric("Forward", len(fwd_df) if not fwd_df.empty else 0)
    with col4:
        heavy_count = int((all_events["discount_campaign"] == 2).sum()) if "discount_campaign" in all_events.columns else 0
        st.metric("Heavy campaigns", heavy_count)
    with col5:
        drop_count = int((all_events["product_drop"] > 0).sum()) if "product_drop" in all_events.columns else 0
        st.metric("Product drops", drop_count)

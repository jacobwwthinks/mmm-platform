"""
Page 4: Event Calendar

Manage promotional events, product drops, and holidays.
Split into:
  - Historical events (used for model fitting — read-only view)
  - Forward-looking events (used for spend planning — editable)

Three-column layout: sidebar (nav) | main data | context panel.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.events import load_events, generate_event_template
from data.github_persist import save_file_to_github
from ui.layout import inject_context_css, context_block, context_tip, context_separator

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

def _save_events_csv(df: pd.DataFrame, local_path: str, repo_path: str, action: str = "Update"):
    """Save events CSV locally and persist to GitHub."""
    df.to_csv(local_path, index=False)
    csv_content = df.to_csv(index=False)
    gh_result = save_file_to_github(
        repo_path, csv_content,
        commit_message=f"{action} event calendar via Streamlit",
    )
    return gh_result


inject_context_css()

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


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT: data | context
# ═══════════════════════════════════════════════════════════════

main, ctx = st.columns([4, 1])

with main:
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
                gh = _save_events_csv(new_events, events_path, events_path, "Upload replacement")
                if gh["success"]:
                    st.success(f"Calendar uploaded and saved — {gh['message']}")
                else:
                    st.warning(f"Calendar uploaded locally. {gh['message']}")
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

    # Build list of Monday dates for the next 12 months (forecast period)
    _fwd_start = cutoff - pd.Timedelta(days=cutoff.weekday())  # Monday of current week
    _fwd_mondays = pd.date_range(start=_fwd_start, periods=52, freq="W-MON")
    _existing_mondays = set()
    if not fwd_df.empty:
        _existing_mondays = set(pd.to_datetime(fwd_df["week_start"]).dt.date)
    _available_mondays = [d.date() for d in _fwd_mondays if d.date() not in _existing_mondays]

    # Quick-add row
    if _available_mondays:
        st.markdown("###### Add a week")
        add_cols = st.columns([2, 1, 1, 1, 1, 2, 1])
        with add_cols[0]:
            _add_date = st.selectbox(
                "Week",
                options=_available_mondays,
                format_func=lambda d: f"{d.strftime('%b %d, %Y')}",
                label_visibility="collapsed",
            )
        with add_cols[1]:
            _add_discount = st.selectbox("Discount", options=[0, 1, 2], index=0,
                                          format_func=lambda x: ["None", "Light", "Heavy"][x],
                                          label_visibility="collapsed")
        with add_cols[2]:
            _add_drop = st.checkbox("Drop", key="add_drop")
        with add_cols[3]:
            _add_offering = st.checkbox("Offering", key="add_offering")
        with add_cols[4]:
            _add_holiday = st.checkbox("Holiday", key="add_holiday")
        with add_cols[5]:
            _add_notes = st.text_input("Notes", key="add_notes", label_visibility="collapsed",
                                        placeholder="Notes...")
        with add_cols[6]:
            if st.button("Add", type="primary", key="add_week_btn"):
                new_row = pd.DataFrame([{
                    "week_start": pd.Timestamp(_add_date),
                    "discount_campaign": _add_discount,
                    "product_drop": int(_add_drop),
                    "product_offering": int(_add_offering),
                    "holiday": int(_add_holiday),
                    "notes": _add_notes,
                }])
                # Combine with existing forward events and save
                if "product_offering" not in hist_df.columns:
                    hist_df["product_offering"] = 0
                edit_cols_save = ["week_start", "discount_campaign", "product_drop",
                                  "product_offering", "holiday", "notes"]
                combined = pd.concat([hist_df[edit_cols_save], fwd_df[edit_cols_save], new_row],
                                      ignore_index=True)
                combined = combined.sort_values("week_start").reset_index(drop=True)
                gh = _save_events_csv(combined, events_path, events_path, "Add event week to")
                if not gh["success"]:
                    st.warning(gh["message"])
                st.rerun()

    # Monday dropdown options for the selectbox column in the editor
    _all_monday_strs = [d.strftime("%Y-%m-%d") for d in _fwd_mondays]

    # Column order for editor
    edit_cols = ["week_start", "discount_campaign", "product_drop", "product_offering", "holiday", "notes"]
    fwd_edit = fwd_df[edit_cols].copy().reset_index(drop=True)
    # Convert week_start to string for selectbox display
    fwd_edit["week_start"] = pd.to_datetime(fwd_edit["week_start"]).dt.strftime("%Y-%m-%d")

    edited_fwd = st.data_editor(
        fwd_edit,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "week_start": st.column_config.SelectboxColumn(
                "Week Start",
                options=_all_monday_strs,
                help="Select a Monday from the forecast period",
            ),
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

    if st.button("Save forward events", type="primary", key="save_fwd_btn"):
        # Combine historical + edited forward events
        edited_fwd["week_start"] = pd.to_datetime(edited_fwd["week_start"], errors="coerce")
        # Drop rows with no week_start (empty added rows)
        edited_fwd = edited_fwd.dropna(subset=["week_start"])

        if "product_offering" not in hist_df.columns:
            hist_df["product_offering"] = 0

        combined = pd.concat([hist_df[edit_cols], edited_fwd[edit_cols]], ignore_index=True)
        combined = combined.sort_values("week_start").reset_index(drop=True)
        gh = _save_events_csv(combined, events_path, events_path, "Update forward events in")
        n_fwd = len(edited_fwd)
        n_hist = len(hist_df)
        if gh["success"]:
            st.success(f"Saved {len(combined)} event weeks ({n_hist} historical + {n_fwd} forward) — {gh['message']}")
        else:
            st.warning(
                f"Saved locally ({n_hist} historical + {n_fwd} forward) but **not persisted to repository**. "
                f"{gh['message']}"
            )
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


# ── Context Panel (right column) ────────────────────────────

with ctx:
    context_block(
        "Why Events Matter",
        "The MMM and Spend-aMER models use events to understand "
        "**why** certain weeks had higher efficiency. Without forward events, "
        "the model assumes no campaign boost — significantly underestimating "
        "spend capacity during discount months."
    )

    context_separator()

    context_block(
        "Discount Levels",
        "**0** = no discount that week\n"
        "**1** = light discount (10-20% off, limited scope)\n"
        "**2** = heavy discount (Black Week, Birthday Week, sitewide)\n\n"
        "Heavy discounts typically boost channel efficiency by 30-50% — "
        "meaning you can spend more while maintaining aMER."
    )

    context_separator()

    context_block(
        "Product Drops",
        "New product launches create organic demand spikes and "
        "improve ad conversion rates. Mark weeks with significant "
        "new product releases.\n\n"
        "The model learns how product drops affect revenue and "
        "accounts for this in future planning."
    )

    context_separator()

    context_block(
        "Forward Planning",
        "Add your planned campaigns, product launches, and "
        "key dates for the next 12 months. The more complete "
        "this calendar is, the better the Spend-aMER recommendations."
    )

    context_tip(
        "**Common mistake:** Leaving future months empty. "
        "Even if you don't have exact dates, mark approximate "
        "weeks for known campaigns (Black Week, summer sale, etc.)."
    )

    context_block(
        "Historical Events",
        "Past events are read-only here — they were used when "
        "fitting the model. The model learned event efficiency "
        "multipliers from this data (visible in Spend-aMER under "
        "'Event efficiency boosts')."
    )

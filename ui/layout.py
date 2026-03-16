"""
Layout helpers for the three-column design.

Left:   Streamlit sidebar (navigation + client selection) — handled by app.py
Middle: Primary data content (charts, tables, controls)
Right:  Contextual guidance — instructions, interpretation, what to look for

Usage in a page:

    from ui.layout import inject_context_css, context_block

    inject_context_css()

    main, ctx = st.columns([3, 1])

    with main:
        st.subheader("Revenue Decomposition")
        st.plotly_chart(fig, use_container_width=True)

    with ctx:
        context_block(
            "Revenue Decomposition",
            "The waterfall shows where revenue comes from. The baseline is revenue "
            "the brand would earn without any paid media."
        )
"""

import streamlit as st


def inject_global_css():
    """Inject all global CSS — fonts, sidebar, headings, metrics, layout.

    Must be called on every page (including app.py) to ensure consistent styling.
    Called automatically by render_sidebar().
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame,
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, textarea, select, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Plotly charts: transparent background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ── Hide default Streamlit auto-generated sidebar nav ── */
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    nav[data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavSeparator"],
    [data-testid="stSidebarNavLink"],
    ul[data-testid="stSidebarNavItems"] {
        display: none !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Narrower left sidebar */
    [data-testid="stSidebar"] {
        min-width: 180px !important;
        max-width: 180px !important;
        width: 180px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 180px !important;
    }

    /* ── Collapse button: hide icon-name text leak ── */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[kind="headerNoPadding"] {
        overflow: hidden !important;
    }
    [data-testid="collapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    button[kind="headerNoPadding"] span,
    [data-testid="stSidebar"] button[kind="header"] span,
    [data-testid="stSidebar"] [data-testid="stBaseButton-header"] span {
        font-size: 0 !important;
        line-height: 0 !important;
        overflow: hidden !important;
        display: inline-block !important;
        width: 1.2em !important;
        height: 1.2em !important;
    }

    /* ── All headings: smaller, lighter ── */
    h1, h1 span { font-size: 1.1rem !important; font-weight: 300 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
    h2, h2 span { font-size: 0.92rem !important; font-weight: 500 !important; }
    h3, h3 span { font-size: 0.85rem !important; font-weight: 500 !important; }
    h4, h4 span, h5, h5 span, h6, h6 span { font-size: 0.8rem !important; font-weight: 500 !important; }

    /* ── Body text ── */
    p, li, span, label, div {
        font-size: 0.82rem !important;
    }

    /* ── Sidebar overrides: even smaller, no uppercase ── */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h1 span,
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h2 span,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h3 span,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h4 span {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        font-size: 0.76rem !important;
    }
    [data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
    [data-testid="stSidebar"] .stPageLink a,
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        font-size: 0.78rem !important;
    }

    /* ── Date inputs: match body font size ── */
    [data-testid="stDateInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] input {
        font-size: 0.82rem !important;
    }

    /* ── Metric cards ── */
    /* Force equal height for tiles in the same row.
       Streamlit wraps metrics in several layers of divs:
       stHorizontalBlock > stColumn > div > div > ... > stMetric
       We need flex to propagate through every level. */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stColumn"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stColumn"] > div > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stColumn"] > div > div > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 16px 16px 14px 16px;
        text-align: left;
        flex: 1 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.45) !important;
        justify-content: flex-start !important;
        margin-bottom: 2px !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.92) !important;
        justify-content: flex-start !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        font-size: 0.7rem !important;
        justify-content: flex-start !important;
    }

    /* ── Tables/dataframes ── */
    [data-testid="stDataFrame"] {
        font-size: 0.78rem !important;
    }

    /* Full-width layout — reduce default padding */
    .stMainBlockContainer, .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the shared sidebar navigation on every page.

    This ensures the sidebar looks identical whether you're on the
    home page or any sub-page.  Must be called after session state
    has been initialised (selected_client, client_config, config).
    """
    inject_global_css()

    # Logo
    st.sidebar.markdown(
        '<div style="padding: 4px 0 10px 0; margin-bottom: 2px;">'
        '<span style="font-family: \'Inter\', sans-serif; font-weight: 900; '
        'font-size: 1.05rem; letter-spacing: 0.04em; color: white; '
        'line-height: 1.1;">ORANGE JUICE.</span></div>',
        unsafe_allow_html=True,
    )

    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/1_Client_Overview.py", label="Client Overview")
    st.sidebar.page_link("pages/2_Channel_Analysis.py", label="Channel Analysis")
    st.sidebar.page_link("pages/3_Budget_Optimizer.py", label="Budget Optimizer")
    st.sidebar.page_link("pages/4_Event_Calendar.py", label="Event Calendar")
    st.sidebar.page_link("pages/5_Spend_aMER.py", label="Spend-aMER")

    st.sidebar.markdown("---")

    # Client selector (only if config is available)
    config = st.session_state.get("config")
    if config:
        clients = config.get("clients", {})
        client_options = {v.get("display_name", k): k for k, v in clients.items()}

        selected_display = st.sidebar.selectbox(
            "Select Client",
            options=list(client_options.keys()),
            index=0,
            key="sidebar_client_select",
        )
        selected_client = client_options[selected_display]

        st.session_state["selected_client"] = selected_client
        st.session_state["client_config"] = clients[selected_client]

        st.sidebar.markdown("---")

        # Connected channels
        client_cfg = clients[selected_client]
        channels = client_cfg.get("channels", {})
        st.sidebar.markdown("**Connected Channels:**")
        for ch_name, ch_cfg in channels.items():
            if ch_cfg and ch_cfg.get("windsor_account"):
                st.sidebar.markdown(f"  + {ch_name.replace('_', ' ').title()}")
            else:
                st.sidebar.markdown(f"  - {ch_name.replace('_', ' ').title()}")

        email_cfg = client_cfg.get("email_source", {})
        if email_cfg.get("windsor_account"):
            st.sidebar.markdown("  + Email (Klaviyo)")
        else:
            st.sidebar.markdown("  - Email (Klaviyo)")

        rev_source = client_cfg.get("revenue_source", {})
        if rev_source.get("windsor_account"):
            st.sidebar.markdown("  + Shopify (Revenue)")
        else:
            st.sidebar.markdown("  - Shopify (Revenue)")

    # Log out + footer — consistent on every page
    st.sidebar.markdown("---")
    if st.sidebar.button("Log out", key="sidebar_logout"):
        st.session_state["authenticated"] = False
        st.rerun()
    st.sidebar.caption("Built for DTC e-commerce brands")


def inject_context_css():
    """Inject CSS for the right-column context panel styling."""
    st.markdown("""
    <style>
    /* Context panel blocks */
    .ctx-block {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #F58518;
        border-radius: 0 8px 8px 0;
        padding: 12px 14px;
        margin-bottom: 16px;
        font-size: 0.82em;
        line-height: 1.55;
        color: #9CA3AF;
    }
    .ctx-block h4 {
        color: #D1D5DB !important;
        font-size: 0.92em !important;
        font-weight: 600 !important;
        margin: 0 0 6px 0 !important;
        padding: 0 !important;
    }
    .ctx-block p {
        margin: 0 0 8px 0;
        color: #9CA3AF;
    }
    .ctx-block p:last-child {
        margin-bottom: 0;
    }
    .ctx-block strong {
        color: #D1D5DB;
    }
    .ctx-block code {
        background: rgba(255, 255, 255, 0.06);
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .ctx-separator {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 16px 0;
    }
    .ctx-tip {
        background: rgba(245, 133, 24, 0.06);
        border-left: 3px solid rgba(245, 133, 24, 0.4);
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 16px;
        font-size: 0.8em;
        line-height: 1.5;
        color: #D1A55A;
    }
    .ctx-tip strong {
        color: #F5B85A;
    }
    </style>
    """, unsafe_allow_html=True)


def context_block(title: str, body: str):
    """Render a styled context block in the right panel.

    Args:
        title: Short heading (e.g. "Revenue Decomposition")
        body: Markdown-compatible explanation text.
              Supports <p>, <strong>, <code> tags.
              Newlines are converted to paragraph breaks.
    """
    # Convert markdown-style bold to HTML
    import re
    html_body = body.replace("\n\n", "</p><p>").replace("\n", "</p><p>")
    html_body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_body)
    html_body = re.sub(r'`(.+?)`', r'<code>\1</code>', html_body)
    html_body = f"<p>{html_body}</p>"

    st.markdown(
        f'<div class="ctx-block"><h4>{title}</h4>{html_body}</div>',
        unsafe_allow_html=True,
    )


def context_tip(text: str):
    """Render a highlighted tip in the right panel."""
    import re
    html = text.replace("\n", "<br>")
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    st.markdown(f'<div class="ctx-tip">{html}</div>', unsafe_allow_html=True)


def context_separator():
    """Render a subtle horizontal rule in the context panel."""
    st.markdown('<hr class="ctx-separator">', unsafe_allow_html=True)

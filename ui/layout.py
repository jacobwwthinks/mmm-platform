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


def render_sidebar():
    """Render the shared sidebar navigation on every page.

    This ensures the sidebar looks identical whether you're on the
    home page or any sub-page.  Must be called after session state
    has been initialised (selected_client, client_config, config).
    """
    from pathlib import Path

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

"""
MMM Platform — Streamlit Application

Multi-client Marketing Mix Modeling dashboard.
Run with: streamlit run app.py
"""

import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(
    page_title="MMM Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication ──────────────────────────────────────────

def check_password():
    """Simple password gate for the app."""
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        return True

    st.title("📊 MMM Platform")
    st.markdown("Enter the team password to continue.")

    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Log in", type="primary"):
        correct_password = st.secrets.get("auth", {}).get("password", "")
        if password == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

# ── Load config ──────────────────────────────────────────────

@st.cache_data
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Inject Windsor API key from Streamlit secrets
    windsor_key = st.secrets.get("windsor", {}).get("api_key", "")
    if windsor_key:
        cfg["windsor"]["api_key"] = windsor_key

    return cfg

config = load_config()

# ── Sidebar: Client Selection ────────────────────────────────

st.sidebar.title("📊 MMM Platform")
st.sidebar.markdown("---")

clients = config.get("clients", {})
client_options = {v.get("display_name", k): k for k, v in clients.items()}

selected_display = st.sidebar.selectbox(
    "Select Client",
    options=list(client_options.keys()),
    index=0,
)
selected_client = client_options[selected_display]

# Store in session state for pages to access
st.session_state["selected_client"] = selected_client
st.session_state["client_config"] = clients[selected_client]
st.session_state["config"] = config

st.sidebar.markdown("---")

# Show connected channels
client_cfg = clients[selected_client]
channels = client_cfg.get("channels", {})
st.sidebar.markdown("**Connected Channels:**")
for ch_name, ch_cfg in channels.items():
    if ch_cfg and ch_cfg.get("windsor_account"):
        st.sidebar.markdown(f"  ✅ {ch_name.replace('_', ' ').title()}")
    else:
        st.sidebar.markdown(f"  ⬜ {ch_name.replace('_', ' ').title()}")

# Show email source
email_cfg = client_cfg.get("email_source", {})
if email_cfg.get("windsor_account"):
    st.sidebar.markdown(f"  ✅ Email (Klaviyo)")
else:
    st.sidebar.markdown(f"  ⬜ Email (Klaviyo)")

rev_source = client_cfg.get("revenue_source", {})
if rev_source.get("windsor_account"):
    st.sidebar.markdown(f"  ✅ Shopify (Revenue)")
else:
    st.sidebar.markdown(f"  ⬜ Shopify (Revenue)")

st.sidebar.markdown("---")

# Logout button
if st.sidebar.button("Log out"):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.caption("Built for DTC e-commerce brands")

# ── Main Page ────────────────────────────────────────────────

st.title(f"Marketing Mix Model — {selected_display}")

st.markdown("""
Welcome to the MMM Platform. Use the pages in the sidebar to:

1. **Client Overview** — Revenue decomposition, channel contributions, model fit
2. **Channel Analysis** — Deep-dive per channel: ROAS, saturation, adstock
3. **Budget Optimizer** — What-if scenarios and optimal budget allocation
4. **Event Calendar** — Manage promotional events, product drops, holidays

### Getting Started

To run the model for a client:
1. Ensure all channels are connected in Windsor.ai (see sidebar)
2. Go to **Client Overview** and click "Fetch Data & Run Model"
3. Review the results across all pages
4. Use the **Budget Optimizer** for allocation recommendations
""")

# Quick stats if results exist
results_path = Path(f"results/{selected_client}")
if results_path.exists():
    st.success("✅ Model results available — navigate to pages to explore")
else:
    st.info("ℹ️ No model results yet for this client. Go to Client Overview to run the model.")

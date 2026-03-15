"""
MMM Platform — Streamlit Application

Multi-client Marketing Mix Modeling dashboard.
Run with: streamlit run app.py
"""

import streamlit as st
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ui.layout import render_sidebar, inject_global_css

st.set_page_config(
    page_title="MMM Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# ── Authentication ──────────────────────────────────────────

def check_password():
    """Simple password gate for the app."""
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        return True

    st.title("MMM Platform")
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

# ── Sidebar: Navigation + Client Selection ──────────────────

st.session_state["config"] = config
render_sidebar()

selected_client = st.session_state.get("selected_client", "juniper")
selected_display = st.session_state.get("client_config", {}).get("display_name", selected_client)

# ── Main Page ────────────────────────────────────────────────

st.title(f"Marketing Mix Model — {selected_display}")

st.markdown("""
Welcome to the MMM Platform. Use the pages in the sidebar to:

1. **Client Overview** — Revenue decomposition, channel contributions, model fit
2. **Channel Analysis** — Deep-dive per channel: ROAS, saturation, adstock
3. **Budget Optimizer** — What-if scenarios and optimal budget allocation
4. **Event Calendar** — Manage promotional events, product drops, holidays
5. **Spend-aMER** — GP3-optimized spend planning by month with seasonal adjustments

### Getting Started

To run the model for a client:
1. Ensure all channels are connected in Windsor.ai (see sidebar)
2. Go to **Client Overview** and click \"Fetch Data & Run Model\"
3. Review the results across all pages
4. Use **Spend-aMER** for GP3-optimized monthly spend recommendations
""")

# Quick stats if results exist
results_path = Path(f"results/{selected_client}")
if results_path.exists():
    st.success("Model results available — navigate to pages to explore.")
else:
    st.info("No model results yet for this client. Go to Client Overview to run the model.")

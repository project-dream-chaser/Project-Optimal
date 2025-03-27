import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils.client_management import load_clients, save_clients
from utils.monte_carlo import run_monte_carlo_simulation
from utils.optimization import optimize_glidepath
from utils.market_assumptions import get_market_assumptions

# Configure the Streamlit page
st.set_page_config(
    page_title="Financial Planning",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'clients' not in st.session_state:
    st.session_state.clients = load_clients()
    
if 'current_client' not in st.session_state:
    st.session_state.current_client = None
    
if 'market_assumptions' not in st.session_state:
    # Default market assumptions
    st.session_state.market_assumptions = get_market_assumptions()

# Sidebar navigation
st.sidebar.image("https://cdn.jsdelivr.net/npm/feather-icons@4.29.0/dist/icons/bar-chart-2.svg", width=100)

menu = st.sidebar.radio(
    "Navigation",
    ["Clients", "Investments", "Plan"]
)

# Main content based on navigation selection
if menu == "Clients":
    from pages.clients import show_clients_page
    show_clients_page()
    
elif menu == "Investments":
    from pages.investments import show_investments_page
    show_investments_page()
    
elif menu == "Plan":
    from pages.plan import show_plan_page
    show_plan_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2025")

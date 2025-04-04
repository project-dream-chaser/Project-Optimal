import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.market_assumptions import (
    get_market_assumptions, update_market_assumptions, optimize_sub_asset_classes
)

def show_investments_page():
    """Display the Investments page with market assumptions and asset allocation models."""
    st.title("Investment Management")
    
    # Create tabs for the different investment sections
    tabs = st.tabs(["Capital Market Assumptions", "Sub-Asset Class Models"])
    
    with tabs[0]:
        show_market_assumptions()
    
    with tabs[1]:
        show_sub_asset_class_models()

def show_market_assumptions():
    """Display and allow editing of capital market assumptions."""
    st.header("Capital Market Assumptions")
    
    # Get market assumptions from session state
    market_assumptions = st.session_state.market_assumptions
    
    # Create tabs for short-term and long-term views
    view_tabs = st.tabs(["Short-Term View", "Long-Term View"])
    
    with view_tabs[0]:
        st.subheader("Short-Term Market Assumptions")
        short_term_assumptions = edit_assumptions(market_assumptions, "short_term")
    
    with view_tabs[1]:
        st.subheader("Long-Term Market Assumptions (Equilibrium)")
        long_term_assumptions = edit_assumptions(market_assumptions, "long_term")
    
    # Button to save changes
    if st.button("Save All Market Assumptions"):
        # Update the market assumptions in session state
        market_assumptions["short_term"] = short_term_assumptions
        market_assumptions["long_term"] = long_term_assumptions
        
        # Add debugging information
        st.info("Saving market assumptions...")
        st.write("Short-term expected return for Global Equity:", 
                market_assumptions["short_term"]["expected_returns"]["Global Equity"])
        st.write("Long-term expected return for Global Equity:", 
                market_assumptions["long_term"]["expected_returns"]["Global Equity"])
                
        update_market_assumptions(market_assumptions)
        
        # Verify data was saved by checking if the file exists
        import os
        if os.path.exists('data/market_assumptions.json'):
            file_size = os.path.getsize('data/market_assumptions.json')
            st.success(f"Market assumptions updated successfully! File size: {file_size} bytes")

def edit_assumptions(market_assumptions, view):
    """
    Edit market assumptions for a given view (short-term or long-term).
    
    Parameters:
    -----------
    market_assumptions : dict
        Market assumptions data
    view : str
        'short_term' or 'long_term'
        
    Returns:
    --------
    dict
        Updated assumptions for the view
    """
    asset_classes = market_assumptions["asset_classes"]
    assumptions = market_assumptions[view].copy()
    
    st.markdown("### Expected Returns")
    
    # Create a DataFrame for editing returns
    returns_data = []
    for asset in asset_classes:
        returns_data.append({
            "Asset Class": asset,
            "Expected Return (%)": assumptions["expected_returns"][asset] * 100
        })
    
    returns_df = pd.DataFrame(returns_data)
    
    # Display editable returns table
    edited_returns = st.data_editor(
        returns_df,
        column_config={
            "Asset Class": st.column_config.TextColumn("Asset Class", disabled=True),
            "Expected Return (%)": st.column_config.NumberColumn(
                "Expected Return (%)",
                format="%.2f",
                min_value=0.0,
                max_value=20.0,
                step=0.1
            )
        },
        use_container_width=True,
        hide_index=True,
        key=f"returns_editor_{view}"
    )
    
    st.markdown("### Volatility")
    
    # Create a DataFrame for editing volatilities
    vol_data = []
    for asset in asset_classes:
        vol_data.append({
            "Asset Class": asset,
            "Volatility (%)": assumptions["volatilities"][asset] * 100
        })
    
    vol_df = pd.DataFrame(vol_data)
    
    # Display editable volatility table
    edited_vols = st.data_editor(
        vol_df,
        column_config={
            "Asset Class": st.column_config.TextColumn("Asset Class", disabled=True),
            "Volatility (%)": st.column_config.NumberColumn(
                "Volatility (%)",
                format="%.2f",
                min_value=0.0,
                max_value=50.0,
                step=0.1
            )
        },
        use_container_width=True,
        hide_index=True,
        key=f"volatility_editor_{view}"
    )
    
    st.markdown("### Correlations")
    
    # Create a correlation DataFrame for editing
    corr_df = assumptions["correlations"].copy()
    
    # Display editable correlation matrix
    edited_corr = st.data_editor(
        corr_df,
        column_config={asset: st.column_config.NumberColumn(
            asset,
            format="%.2f",
            min_value=-1.0,
            max_value=1.0,
            step=0.01
        ) for asset in asset_classes},
        use_container_width=True,
        key=f"correlation_editor_{view}"
    )
    
    # Update the assumptions with edited values
    updated_assumptions = assumptions.copy()
    updated_assumptions["expected_returns"] = {
        asset: row["Expected Return (%)"] / 100
        for asset, row in zip(asset_classes, edited_returns.to_dict("records"))
    }
    
    updated_assumptions["volatilities"] = {
        asset: row["Volatility (%)"] / 100
        for asset, row in zip(asset_classes, edited_vols.to_dict("records"))
    }
    
    updated_assumptions["correlations"] = edited_corr
    
    return updated_assumptions

def show_sub_asset_class_models():
    """Display and optimize sub-asset class models."""
    st.header("Sub-Asset Class Models")
    
    # Get market assumptions from session state
    market_assumptions = st.session_state.market_assumptions
    
    # Select asset class to view/edit sub-asset classes
    asset_class = st.selectbox(
        "Select Asset Class",
        market_assumptions["asset_classes"]
    )
    
    # Get sub-asset classes for the selected asset class
    sub_asset_classes = market_assumptions["sub_asset_classes"][asset_class]
    
    st.subheader(f"Sub-Asset Classes for {asset_class}")
    
    # Create a DataFrame for editing sub-asset class returns and volatilities
    sub_data = []
    for sub in sub_asset_classes:
        sub_data.append({
            "Sub-Asset Class": sub,
            "Expected Return (%)": market_assumptions["sub_asset_class_returns"][sub] * 100,
            "Volatility (%)": market_assumptions["sub_asset_class_volatilities"][sub] * 100
        })
    
    sub_df = pd.DataFrame(sub_data)
    
    # Display editable sub-asset class table
    edited_sub = st.data_editor(
        sub_df,
        column_config={
            "Sub-Asset Class": st.column_config.TextColumn("Sub-Asset Class", disabled=True),
            "Expected Return (%)": st.column_config.NumberColumn(
                "Expected Return (%)",
                format="%.2f",
                min_value=0.0,
                max_value=20.0,
                step=0.1
            ),
            "Volatility (%)": st.column_config.NumberColumn(
                "Volatility (%)",
                format="%.2f",
                min_value=0.0,
                max_value=50.0,
                step=0.1
            )
        },
        use_container_width=True,
        hide_index=True,
        key=f"sub_asset_editor_{asset_class}"
    )
    
    # Save button for sub-asset class changes
    if st.button("Save Sub-Asset Class Changes"):
        # Update market assumptions
        for row in edited_sub.to_dict("records"):
            sub = row["Sub-Asset Class"]
            market_assumptions["sub_asset_class_returns"][sub] = row["Expected Return (%)"] / 100
            market_assumptions["sub_asset_class_volatilities"][sub] = row["Volatility (%)"] / 100
        
        update_market_assumptions(market_assumptions)
        st.success("Sub-asset class assumptions updated successfully!")
    
    # Add a section for constraints
    st.subheader("Sub-Asset Class Constraints")
    
    # Initialize sub-asset constraints if not already in session state
    if 'sub_asset_constraints' not in st.session_state:
        st.session_state.sub_asset_constraints = {}
    
    if asset_class not in st.session_state.sub_asset_constraints:
        st.session_state.sub_asset_constraints[asset_class] = {}
        
    # Create a DataFrame for constraints
    constraints_data = []
    for sub in sub_asset_classes:
        if sub in st.session_state.sub_asset_constraints[asset_class]:
            constraints = st.session_state.sub_asset_constraints[asset_class][sub]
            min_weight = constraints['min']
            max_weight = constraints['max']
        else:
            min_weight = 0.0
            max_weight = 100.0
        
        constraints_data.append({
            "Sub-Asset Class": sub,
            "Minimum (%)": min_weight,
            "Maximum (%)": max_weight
        })
    
    constraints_df = pd.DataFrame(constraints_data)
    
    # Display editable constraints table
    edited_constraints = st.data_editor(
        constraints_df,
        column_config={
            "Sub-Asset Class": st.column_config.TextColumn("Sub-Asset Class", disabled=True),
            "Minimum (%)": st.column_config.NumberColumn(
                "Minimum (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
                step=1.0
            ),
            "Maximum (%)": st.column_config.NumberColumn(
                "Maximum (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
                step=1.0
            )
        },
        use_container_width=True,
        hide_index=True,
        key=f"sub_asset_constraints_{asset_class}"
    )
    
    # Save button for constraints
    if st.button("Save Constraints"):
        # Update sub-asset constraints
        for row in edited_constraints.to_dict("records"):
            sub = row["Sub-Asset Class"]
            st.session_state.sub_asset_constraints[asset_class][sub] = {
                'min': row["Minimum (%)"],
                'max': row["Maximum (%)"]
            }
        st.success("Sub-asset class constraints updated successfully!")
    
    # Add a section for optimization
    st.subheader("Optimize Sub-Asset Class Weights")
    
    # Use fixed risk aversion parameter of 6.0 for all sub-asset class optimizations
    risk_aversion = 6.0
    
    st.write("Optimizing for risk-adjusted returns with fixed risk parameter")
    
    if st.button("Run Optimization"):
        with st.spinner("Optimizing sub-asset class weights..."):
            # Run the optimization with fixed risk aversion parameter
            optimized_allocations = optimize_sub_asset_classes(market_assumptions, risk_aversion=risk_aversion)
            
            # Display the results for the selected asset class
            if asset_class in optimized_allocations:
                result = optimized_allocations[asset_class]
                
                # Create a DataFrame for the results
                result_data = []
                for i, sub in enumerate(result["sub_assets"]):
                    result_data.append({
                        "Sub-Asset Class": sub,
                        "Optimal Weight (%)": result["weights"][i] * 100
                    })
                
                result_df = pd.DataFrame(result_data)
                
                st.write("Optimized Sub-Asset Class Allocation:")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                
                # Create a pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(
                    result["weights"],
                    labels=result["sub_assets"],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.axis('equal')
                ax.set_title(f'Optimized Allocation for {asset_class}')
                
                st.pyplot(fig)

def show_constraints():
    """Display and edit portfolio constraints."""
    st.header("Portfolio Constraints")
    
    # Get market assumptions to access asset classes
    market_assumptions = st.session_state.market_assumptions
    asset_classes = market_assumptions["asset_classes"]
    
    st.subheader("Asset Allocation Constraints")
    
    # Create a DataFrame for min/max constraints
    if "allocation_constraints" not in st.session_state:
        # Initialize with default constraints
        constraints_data = []
        for asset in asset_classes:
            constraints_data.append({
                "Asset Class": asset,
                "Minimum (%)": 5.0,
                "Maximum (%)": 60.0
            })
        st.session_state.allocation_constraints = pd.DataFrame(constraints_data)
    
    # Display editable constraints table
    edited_constraints = st.data_editor(
        st.session_state.allocation_constraints,
        column_config={
            "Asset Class": st.column_config.TextColumn("Asset Class", disabled=True),
            "Minimum (%)": st.column_config.NumberColumn(
                "Minimum (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
                step=1.0
            ),
            "Maximum (%)": st.column_config.NumberColumn(
                "Maximum (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
                step=1.0
            )
        },
        use_container_width=True,
        hide_index=True,
        key="constraints_editor"
    )
    
    # Save the edited constraints
    st.session_state.allocation_constraints = edited_constraints
    
    # Use fixed risk aversion parameter of 6.0 for all optimizations
    if "risk_aversion" not in st.session_state:
        st.session_state.risk_aversion = 6.0
    else:
        st.session_state.risk_aversion = 6.0  # Always set to 6.0
        
    st.info("Using fixed risk parameter of 6.0 for all optimizations")
    
    # Add mean reversion settings
    st.subheader("Mean Reversion Settings")
    
    if "mean_reversion_speed" not in st.session_state:
        st.session_state.mean_reversion_speed = 0.15
    
    mean_reversion_speed = st.slider(
        "Mean Reversion Speed",
        min_value=0.0,
        max_value=0.5,
        value=st.session_state.mean_reversion_speed,
        step=0.01,
        help="Speed at which returns revert to long-term means (0 = no mean reversion, 0.5 = fast mean reversion)"
    )
    
    # Save the mean reversion speed
    st.session_state.mean_reversion_speed = mean_reversion_speed
    
    # Button to save all constraint settings
    if st.button("Save All Constraints"):
        st.success("Constraints saved successfully!")

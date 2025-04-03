import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os
import json
import time

from models.client import Client
from models.plan import Plan, Goal, CashFlow, LiquidityGoal
from utils.monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_results, calculate_shortfall_risk
from utils.optimization import optimize_glidepath, plot_glidepath
from utils.ips_generator import generate_investment_policy_statement, display_ips_download_button
from utils.client_management import get_client_by_id

def show_plan_page():
    """Display the Plan page with financial planning tools."""
    # No title header as requested
    
    # Check if there are clients available
    if not st.session_state.clients:
        st.warning("Please add clients first before creating a financial plan.")
        return
    
    # Create vertical tabs for plan sections with the new order
    plan_tabs = st.tabs([
        "Liquidity",
        "Time Horizon",
        "Tax",
        "Risk Assessment",
        "Return Objective",
        "Glidepath Optimization",
        "Investment Policy Statement"
    ])
    
    # Initialize session state variables for planning
    if 'current_plan' not in st.session_state:
        st.session_state.current_plan = None
    
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    if 'glidepath_results' not in st.session_state:
        st.session_state.glidepath_results = None
    
    # Sidebar for client selection
    with st.sidebar:
        st.header("Select Client")
        
        client_options = [(client.id, f"{client.first_name} {client.last_name}") 
                         for client in st.session_state.clients]
        
        selected_client_id = st.selectbox(
            "Client",
            options=[c[0] for c in client_options],
            format_func=lambda x: next((c[1] for c in client_options if c[0] == x), x),
            key="plan_client_select"
        )
        
        selected_client = get_client_by_id(selected_client_id)
        
        if selected_client:
            # Load or create a plan for this client
            load_or_create_plan(selected_client)
    
    # Liquidity tab (includes Goals & Cash Flows)
    with plan_tabs[0]:
        if st.session_state.current_plan:
            show_liquidity(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Time Horizon tab
    with plan_tabs[1]:
        if st.session_state.current_plan:
            show_time_horizon(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Tax tab
    with plan_tabs[2]:
        if st.session_state.current_plan:
            show_tax_considerations(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Risk Assessment tab
    with plan_tabs[3]:
        if st.session_state.current_plan:
            show_risk_assessment(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Return Objective tab
    with plan_tabs[4]:
        if st.session_state.current_plan:
            show_return_objective(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Glidepath Optimization tab (includes Monte Carlo)
    with plan_tabs[5]:
        if st.session_state.current_plan:
            show_glidepath_optimization(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Investment Policy Statement tab
    with plan_tabs[6]:
        if st.session_state.current_plan and st.session_state.glidepath_results:
            show_ips_generation(selected_client)
        else:
            st.info("Optimize the glidepath first before generating an Investment Policy Statement.")

def load_or_create_plan(client):
    """
    Load an existing plan for the client or create a new one.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to load/create a plan
    """
    # Check if we have a plans directory
    os.makedirs('data/plans', exist_ok=True)
    
    # Try to load the client's plan
    plan_path = f'data/plans/{client.id}.json'
    
    if os.path.exists(plan_path):
        try:
            with open(plan_path, 'r') as f:
                plan_data = json.load(f)
                
            # Create Plan object from JSON data
            try:
                plan = Plan.from_dict(plan_data)
                st.session_state.current_plan = plan
                
                # Initialize session state variables if missing
                if 'simulation_results' not in st.session_state:
                    st.session_state.simulation_results = {}
                
                if 'glidepath_results' not in st.session_state:
                    st.session_state.glidepath_results = {}
                
                # If we have simulation results, try to load them
                sim_path = f'data/plans/{client.id}_simulation.json'
                if os.path.exists(sim_path):
                    try:
                        # First check file size to make sure it's not corrupted
                        file_size = os.path.getsize(sim_path)
                        if file_size < 100:  # Very small file likely indicates corruption
                            st.warning(f"Simulation results file appears to be corrupted (only {file_size} bytes). It will be regenerated when you run a simulation.")
                            # Rename the corrupted file but don't delete it
                            if os.path.exists(f"{sim_path}.corrupted"):
                                os.remove(f"{sim_path}.corrupted")
                            os.rename(sim_path, f"{sim_path}.corrupted")
                        else:
                            with open(sim_path, 'r') as f:
                                sim_data = json.load(f)
                            st.session_state.simulation_results = sim_data
                    except Exception as e:
                        st.warning(f"Could not load simulation results: {e}")
                        # Rename the corrupted file but don't delete it
                        if os.path.exists(sim_path):
                            if os.path.exists(f"{sim_path}.corrupted"):
                                os.remove(f"{sim_path}.corrupted")
                            os.rename(sim_path, f"{sim_path}.corrupted")
                
                # If we have glidepath results, try to load them
                glidepath_path = f'data/plans/{client.id}_glidepath.json'
                if os.path.exists(glidepath_path):
                    try:
                        # First check file size to make sure it's not corrupted
                        file_size = os.path.getsize(glidepath_path)
                        if file_size < 100:  # Very small file likely indicates corruption
                            st.warning(f"Glidepath results file appears to be corrupted (only {file_size} bytes). It will be regenerated when you run an optimization.")
                            # Rename the corrupted file but don't delete it
                            if os.path.exists(f"{glidepath_path}.corrupted"):
                                os.remove(f"{glidepath_path}.corrupted")
                            os.rename(glidepath_path, f"{glidepath_path}.corrupted")
                        else:
                            with open(glidepath_path, 'r') as f:
                                glidepath_data = json.load(f)
                                
                            # Convert lists back to numpy arrays where needed
                            if 'glidepath' in glidepath_data:
                                glidepath_data['glidepath'] = np.array(glidepath_data['glidepath'])
                                
                            st.session_state.glidepath_results = glidepath_data
                    except Exception as e:
                        st.warning(f"Could not load glidepath results: {e}")
                        # Rename the corrupted file but don't delete it
                        if os.path.exists(glidepath_path):
                            if os.path.exists(f"{glidepath_path}.corrupted"):
                                os.remove(f"{glidepath_path}.corrupted")
                            os.rename(glidepath_path, f"{glidepath_path}.corrupted")
                
            except Exception as e:
                st.error(f"Error processing plan data: {e}")
                # Create a new plan if loading fails
                create_new_plan(client)
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing plan JSON: {e}")
            # Create a new plan if loading fails
            create_new_plan(client)
        except Exception as e:
            st.error(f"Error loading plan: {e}")
            # Create a new plan if loading fails
            create_new_plan(client)
    else:
        # Create a new plan
        create_new_plan(client)

def create_new_plan(client):
    """
    Create a new financial plan for a client.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to create a plan
    """
    # Define default asset classes
    asset_classes = [
        'Global Equity',
        'Core Bond',
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    
    # Create default allocation constraints
    allocation_constraints = {}
    for asset_class in asset_classes:
        allocation_constraints[asset_class] = {
            'min': 5.0,
            'max': 60.0
        }
    
    # Create a basic plan with default values
    plan = Plan(
        client_id=client.id,
        name=f"Financial Plan for {client.first_name} {client.last_name}",
        goals=[],
        cash_flows=[],
        liquidity_goals=[],  # Empty liquidity goals list
        initial_portfolio=100000,  # Default initial portfolio
        asset_allocation=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],  # Default allocation
        allocation_constraints=allocation_constraints,  # Default constraints
        risk_aversion=3.0,  # Default risk aversion
        mean_reversion_speed=0.15,  # Default mean reversion speed
        pre_restylement_return=7.0,  # Default pre-restylement return
        post_restylement_return=5.0,  # Default post-restylement return
        return_objective_scenario='Possibilities',  # Default scenario
        desired_spending=0,  # Default desired spending
        desired_legacy=0  # Default desired legacy
    )
    
    st.session_state.current_plan = plan
    st.session_state.simulation_results = None
    st.session_state.glidepath_results = None
    
    # Save the new plan
    save_plan(plan)

def save_plan(plan):
    """
    Save the current plan to a JSON file.
    
    Parameters:
    -----------
    plan : Plan object
        The plan to save
    """
    try:
        # Ensure the plans directory exists
        os.makedirs('data/plans', exist_ok=True)
        
        # Convert plan to dictionary and save as JSON
        plan_dict = plan.to_dict()
        
        # Ensure all values are JSON serializable
        def make_json_serializable(obj):
            """Convert numpy arrays and other non-serializable objects to serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        # Process the entire dictionary to ensure everything is serializable
        plan_dict = make_json_serializable(plan_dict)
        
        plan_path = f'data/plans/{plan.client_id}.json'
        
        with open(plan_path, 'w') as f:
            json.dump(plan_dict, f, indent=2)
            
        # If we have simulation results, save them too
        if 'simulation_results' in st.session_state and st.session_state.simulation_results:
            try:
                # Convert any numpy arrays to lists for JSON serialization
                sim_data = st.session_state.simulation_results.copy()
                sim_data = make_json_serializable(sim_data)
                
                with open(f'data/plans/{plan.client_id}_simulation.json', 'w') as f:
                    json.dump(sim_data, f, indent=2)
            except Exception as e:
                st.warning(f"Error saving simulation results: {e}")
                import traceback
                st.warning(f"Detailed error: {traceback.format_exc()}")
        
        # If we have glidepath results, save them too
        if 'glidepath_results' in st.session_state and st.session_state.glidepath_results:
            try:
                # Convert numpy arrays to lists for JSON serialization
                glidepath_dict = st.session_state.glidepath_results.copy()
                glidepath_dict = make_json_serializable(glidepath_dict)
                
                with open(f'data/plans/{plan.client_id}_glidepath.json', 'w') as f:
                    json.dump(glidepath_dict, f, indent=2)
            except Exception as e:
                st.warning(f"Error saving glidepath results: {e}")
                import traceback
                st.warning(f"Detailed error: {traceback.format_exc()}")
                
    except Exception as e:
        st.error(f"Error saving plan: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def show_risk_assessment(client):
    """
    Display and edit risk assessment for a client's plan.
    
    Parameters:
    -----------
    client : Client object
        The client whose risk is being assessed
    """
    st.header("Risk Assessment")
    
    plan = st.session_state.current_plan
    
    # Calculate client's current age
    from datetime import datetime
    
    def calculate_age(date_of_birth):
        """Calculate age from date of birth in YYYY-MM-DD format."""
        birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    
    current_age = calculate_age(client.date_of_birth)
    
    # Basic client info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Client: {client.full_name()}")
        st.write(f"Age: {current_age}")
        
        if client.spouse:
            # Calculate spouse age using the same calculate_age function
            spouse_age = calculate_age(client.spouse['date_of_birth'])
            st.write(f"Spouse: {client.spouse['first_name']} {client.spouse['last_name']}")
            st.write(f"Spouse Age: {spouse_age}")
    
    with col2:
        # Client's risk profile 
        risk_profile = client.get_risk_profile()
        st.write(f"Risk Profile: {risk_profile}")
        
        # Update max stock percentage
        max_stock_pct = client.max_stock_pct if client.max_stock_pct is not None else 60
        new_max_stock = st.slider(
            "Maximum Stock Percentage",
            min_value=0,
            max_value=100,
            value=max_stock_pct,
            step=5,
            help="The maximum percentage of the portfolio that should be allocated to stocks (higher = more aggressive)"
        )
        
        if new_max_stock != max_stock_pct:
            # Update client's risk profile
            client.max_stock_pct = new_max_stock
            # Update the client in the session state
            for i, c in enumerate(st.session_state.clients):
                if c.id == client.id:
                    st.session_state.clients[i] = client
                    break
    
    # Risk Parameters
    st.subheader("Risk Parameters")
    
    # Initialize risk parameters if not in the plan
    if not hasattr(plan, 'risk_aversion'):
        plan.risk_aversion = 3.0
    
    if not hasattr(plan, 'mean_reversion_speed'):
        plan.mean_reversion_speed = 0.15
    
    # Risk aversion parameter
    risk_aversion = st.slider(
        "Risk Aversion Parameter",
        min_value=1.0,
        max_value=10.0,
        value=plan.risk_aversion,
        step=0.1,
        help="Higher values lead to more conservative allocations"
    )
    
    # Mean reversion parameter
    mean_reversion_speed = st.slider(
        "Mean Reversion Speed",
        min_value=0.0,
        max_value=0.5,
        value=plan.mean_reversion_speed,
        step=0.01,
        help="Speed at which returns revert to long-term means (0 = no mean reversion, 0.5 = fast mean reversion)"
    )
    
    # Update risk parameters if changed
    if risk_aversion != plan.risk_aversion or mean_reversion_speed != plan.mean_reversion_speed:
        plan.risk_aversion = risk_aversion
        plan.mean_reversion_speed = mean_reversion_speed
        save_plan(plan)
    
    # Asset Allocation Constraints
    st.subheader("Asset Allocation Constraints")
    
    asset_classes = [
        'Global Equity',
        'Core Bond',
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    
    # Initialize allocation constraints if not in the plan
    if not hasattr(plan, 'allocation_constraints') or not plan.allocation_constraints:
        plan.allocation_constraints = {}
        # Default constraints for each asset class
        for asset_class in asset_classes:
            plan.allocation_constraints[asset_class] = {
                'min': 5.0,
                'max': 60.0
            }
    
    # Create a DataFrame for min/max constraints
    constraints_data = []
    for asset_class in asset_classes:
        if asset_class in plan.allocation_constraints:
            constraints = plan.allocation_constraints[asset_class]
            min_weight = constraints['min']
            max_weight = constraints['max']
        else:
            min_weight = 5.0
            max_weight = 60.0
            plan.allocation_constraints[asset_class] = {
                'min': min_weight,
                'max': max_weight
            }
        
        constraints_data.append({
            "Asset Class": asset_class,
            "Minimum (%)": min_weight,
            "Maximum (%)": max_weight
        })
    
    # Display editable constraints table
    edited_constraints = st.data_editor(
        pd.DataFrame(constraints_data),
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
        key="allocation_constraints_editor"
    )
    
    # Update constraints if changed
    if st.button("Save Allocation Constraints"):
        for row in edited_constraints.to_dict("records"):
            asset_class = row["Asset Class"]
            plan.allocation_constraints[asset_class] = {
                'min': row["Minimum (%)"],
                'max': row["Maximum (%)"]
            }
        save_plan(plan)
        st.success("Allocation constraints saved successfully!")

def show_return_objective(client):
    """
    Display and edit return objectives for a client's plan.
    
    Parameters:
    -----------
    client : Client object
        The client whose return objectives are being defined
    """
    st.header("Return Objective")
    
    plan = st.session_state.current_plan
    
    # Initialize return objective fields if they don't exist
    if not hasattr(plan, 'pre_restylement_return'):
        plan.pre_restylement_return = 7.0  # Default 7% pre-restylement return
        
    if not hasattr(plan, 'post_restylement_return'):
        plan.post_restylement_return = 5.0  # Default 5% post-restylement return
        
    if not hasattr(plan, 'return_objective_scenario'):
        plan.return_objective_scenario = 'Possibilities'
        
    if not hasattr(plan, 'desired_spending'):
        plan.desired_spending = 0
        
    if not hasattr(plan, 'desired_legacy'):
        plan.desired_legacy = 0
    
    # Scenario selection
    return_objective_scenario = st.selectbox(
        "Return Objective Scenario",
        ["Possibilities", "Specific Spending and Legacy Goal"],
        index=0 if plan.return_objective_scenario == 'Possibilities' else 1,
        help="Choose how to determine your return objectives"
    )
    
    plan.return_objective_scenario = return_objective_scenario
    
    if return_objective_scenario == 'Possibilities':
        # Manual entry for pre and post restylement returns
        st.subheader("Manual Return Targets")
        col1, col2 = st.columns(2)
        
        with col1:
            pre_restylement_return = st.number_input(
                "Pre-Restylement Return (%)",
                min_value=0.0,
                max_value=15.0,
                value=plan.pre_restylement_return,
                step=0.1,
                format="%.1f",
                help="Target annual return before restylement (accumulation phase)"
            )
        
        with col2:
            post_restylement_return = st.number_input(
                "Post-Restylement Return (%)",
                min_value=0.0,
                max_value=15.0,
                value=plan.post_restylement_return,
                step=0.1,
                format="%.1f",
                help="Target annual return after restylement (distribution phase)"
            )
            
        # Update plan if values changed
        if pre_restylement_return != plan.pre_restylement_return or post_restylement_return != plan.post_restylement_return:
            plan.pre_restylement_return = pre_restylement_return
            plan.post_restylement_return = post_restylement_return
            save_plan(plan)
    
    else:  # Specific Spending and Legacy Goal
        st.subheader("Spending and Legacy Goals")
        
        # Calculate client's current age
        from datetime import datetime
        
        def calculate_age(date_of_birth):
            """Calculate age from date of birth in YYYY-MM-DD format."""
            birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        
        current_age = calculate_age(client.date_of_birth)
        
        # Years to restylement and end of plan
        years_to_restylement = max(0, client.restylement_age - current_age)
        years_to_end = max(0, client.longevity_age - current_age)
        
        col1, col2 = st.columns(2)
        
        with col1:
            desired_spending = st.number_input(
                f"Desired Annual Spending at Restylement (Age {client.restylement_age}) ($)",
                min_value=0,
                value=int(plan.desired_spending) if plan.desired_spending > 0 else 0,
                step=1000,
                help="Your desired annual spending amount when you reach restylement age"
            )
        
        with col2:
            desired_legacy = st.number_input(
                f"Desired Legacy at End of Plan (Age {client.longevity_age}) ($)",
                min_value=0,
                value=int(plan.desired_legacy) if plan.desired_legacy > 0 else 0,
                step=10000,
                help="The legacy amount you wish to leave at the end of your plan"
            )
        
        # Update plan if values changed
        if desired_spending != plan.desired_spending or desired_legacy != plan.desired_legacy:
            plan.desired_spending = desired_spending
            plan.desired_legacy = desired_legacy
            
            # Calculate required returns using 4% rule
            if plan.initial_portfolio > 0:
                if years_to_restylement > 0:
                    # Calculate pre-restylement return needed
                    needed_at_restylement = desired_spending * 25  # Using 4% withdrawal rule (100/4 = 25)
                    growth_rate = ((needed_at_restylement / plan.initial_portfolio) ** (1/years_to_restylement)) - 1
                    plan.pre_restylement_return = min(15.0, max(0.0, growth_rate * 100))
                else:
                    # Already at or past restylement age
                    plan.pre_restylement_return = 0.0
                
                # Calculate post-restylement return needed
                if years_to_end > 0 and years_to_restylement >= 0:
                    # Determine what portfolio value will be at restylement
                    if years_to_restylement > 0:
                        portfolio_at_restylement = plan.initial_portfolio * ((1 + plan.pre_restylement_return/100) ** years_to_restylement)
                    else:
                        portfolio_at_restylement = plan.initial_portfolio
                    
                    # Calculate how much is needed for annual spending
                    spending_needs = desired_spending * ((1 - (1 + 0.03) ** -(years_to_end - years_to_restylement)) / 0.03)
                    
                    # Additional amount needed for legacy
                    remaining_for_legacy = max(0, portfolio_at_restylement - spending_needs)
                    if remaining_for_legacy < desired_legacy:
                        # Need growth to reach legacy goal
                        years_in_restylement = years_to_end - years_to_restylement
                        if years_in_restylement > 0:
                            growth_rate = ((desired_legacy / remaining_for_legacy) ** (1/years_in_restylement)) - 1
                            plan.post_restylement_return = min(15.0, max(0.0, growth_rate * 100))
                        else:
                            plan.post_restylement_return = 0.0
                    else:
                        # Already have enough for legacy goal, can be conservative
                        plan.post_restylement_return = 3.0  # Conservative default
                else:
                    plan.post_restylement_return = 0.0
            
            save_plan(plan)
        
        # Display calculated returns
        st.subheader("Calculated Required Returns")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Pre-Restylement Return (%)", f"{plan.pre_restylement_return:.1f}")
            
        with col2:
            st.metric("Post-Restylement Return (%)", f"{plan.post_restylement_return:.1f}")
        
        if plan.initial_portfolio == 0:
            st.warning("Please enter your Initial Portfolio Value in the Liquidity section to calculate required returns.")
    
    st.subheader("Portfolio Return Requirements")
    
    # Calculate required return based on goals and cash flows
    if plan.goals or plan.cash_flows:
        # Initial portfolio value
        initial_portfolio = plan.initial_portfolio
        
        # Basic calculation of money needed and time horizon
        st.write(f"Initial Portfolio Value: ${initial_portfolio:,}")
        
        if plan.goals:
            total_goals = sum(goal.amount for goal in plan.goals)
            st.write(f"Total Future Goals: ${total_goals:,}")
        
        if 'glidepath_results' in st.session_state and st.session_state.glidepath_results:
            expected_return = st.session_state.glidepath_results.get('expected_return', 0) * 100
            st.write(f"Optimized Expected Return: {expected_return:.2f}%")
        else:
            st.info("Run the Glidepath Optimization to calculate the expected return based on your goals and constraints.")
    else:
        st.info("Add goals and cash flows in the Liquidity section to calculate return requirements.")

def show_time_horizon(client):
    """
    Display and edit time horizon for a client's plan.
    
    Parameters:
    -----------
    client : Client object
        The client whose time horizon is being defined
    """
    st.header("Time Horizon")
    
    plan = st.session_state.current_plan
    
    # Calculate client's current age
    from datetime import datetime
    
    def calculate_age(date_of_birth):
        """Calculate age from date of birth in YYYY-MM-DD format."""
        birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    
    current_age = calculate_age(client.date_of_birth)
    
    # Restylement and longevity ages
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Age Milestones")
        
        # Use the client's restylement and longevity ages if available
        restylement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
        longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
        
        new_restylement_age = st.number_input(
            "Restylement Age",
            min_value=current_age,
            max_value=100,
            value=restylement_age,
            step=1,
            help="Age at which the client plans to enter restylement (retirement)"
        )
        
        new_longevity_age = st.number_input(
            "Longevity Age",
            min_value=new_restylement_age,
            max_value=120,
            value=longevity_age,
            step=1,
            help="Age for which the client plans financial longevity"
        )
        
        if new_restylement_age != restylement_age or new_longevity_age != longevity_age:
            # Update client's ages
            client.restylement_age = new_restylement_age
            client.longevity_age = new_longevity_age
            # Update the client in the session state
            for i, c in enumerate(st.session_state.clients):
                if c.id == client.id:
                    st.session_state.clients[i] = client
                    break
    
    with col2:
        st.subheader("Time Horizon Metrics")
        
        # Calculate years until key milestones
        years_to_restylement = new_restylement_age - current_age
        planning_time_horizon = new_longevity_age - current_age
        restylement_duration = new_longevity_age - new_restylement_age
        
        st.write(f"Current Age: {current_age}")
        st.write(f"Years Until Restylement: {years_to_restylement}")
        st.write(f"Total Planning Horizon: {planning_time_horizon} years")
        st.write(f"Expected Restylement Duration: {restylement_duration} years")
    
    # Glidepath visualization if available
    if 'glidepath_results' in st.session_state and st.session_state.glidepath_results:
        st.subheader("Glidepath Over Time")
        fig = plot_glidepath(st.session_state.glidepath_results)
        st.pyplot(fig)
    else:
        st.info("Run the Glidepath Optimization to see how your allocation will change over time.")
    
    # Goals timeline
    if plan.goals:
        st.subheader("Goals Timeline")
        
        # Create timeline data
        timeline_data = []
        for goal in plan.goals:
            years_from_now = goal.age - current_age
            timeline_data.append({
                "Goal": goal.name,
                "Age": goal.age,
                "Years From Now": years_from_now,
                "Amount": f"${goal.amount:,}",
                "Priority": goal.priority
            })
        
        # Sort by age
        timeline_data = sorted(timeline_data, key=lambda x: x["Age"])
        
        # Display timeline
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    else:
        st.info("Add goals in the Liquidity section to visualize your timeline.")

def show_tax_considerations(client):
    """
    Display and edit tax considerations for a client's plan.
    
    Parameters:
    -----------
    client : Client object
        The client whose tax considerations are being defined
    """
    st.header("Tax Considerations")
    
    plan = st.session_state.current_plan
    
    # Calculate client's current age
    from datetime import datetime
    
    def calculate_age(date_of_birth):
        """Calculate age from date of birth in YYYY-MM-DD format."""
        birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    
    current_age = calculate_age(client.date_of_birth)
    
    st.subheader("Tax Rates")
    
    # Tax rates (default or from plan)
    if not hasattr(plan, 'tax_rates'):
        plan.tax_rates = {
            'income_tax_rate': 25.0,
            'capital_gains_rate': 15.0,
            'estate_tax_rate': 40.0
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        income_tax_rate = st.slider(
            "Income Tax Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=plan.tax_rates['income_tax_rate'],
            step=0.5
        )
        
        capital_gains_rate = st.slider(
            "Capital Gains Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=plan.tax_rates['capital_gains_rate'],
            step=0.5
        )
    
    with col2:
        estate_tax_rate = st.slider(
            "Estate Tax Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=plan.tax_rates['estate_tax_rate'],
            step=0.5
        )
    
    # Update tax rates if changed
    if (income_tax_rate != plan.tax_rates['income_tax_rate'] or 
        capital_gains_rate != plan.tax_rates['capital_gains_rate'] or
        estate_tax_rate != plan.tax_rates['estate_tax_rate']):
        
        plan.tax_rates = {
            'income_tax_rate': income_tax_rate,
            'capital_gains_rate': capital_gains_rate,
            'estate_tax_rate': estate_tax_rate
        }
        save_plan(plan)
    
    st.subheader("Account Types")
    
    # Account type allocations (default or from plan)
    if not hasattr(plan, 'account_types'):
        plan.account_types = {
            'taxable': 50.0,
            'tax_deferred': 30.0,
            'tax_free': 20.0
        }
    
    # Display pie chart of account types
    account_types = ['Taxable', 'Tax-Deferred', 'Tax-Free']
    account_allocations = [
        plan.account_types['taxable'],
        plan.account_types['tax_deferred'],
        plan.account_types['tax_free']
    ]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(account_allocations, labels=account_types, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Portfolio by Account Type')
    st.pyplot(fig)
    
    # Edit account allocations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        taxable_pct = st.slider(
            "Taxable (%)",
            min_value=0.0,
            max_value=100.0,
            value=plan.account_types['taxable'],
            step=1.0
        )
    
    with col2:
        tax_deferred_pct = st.slider(
            "Tax-Deferred (%)",
            min_value=0.0,
            max_value=100.0,
            value=plan.account_types['tax_deferred'],
            step=1.0
        )
    
    with col3:
        tax_free_pct = st.slider(
            "Tax-Free (%)",
            min_value=0.0,
            max_value=100.0,
            value=plan.account_types['tax_free'],
            step=1.0
        )
    
    # Normalize to 100%
    total = taxable_pct + tax_deferred_pct + tax_free_pct
    if total > 0:
        normalized_taxable = (taxable_pct / total) * 100
        normalized_tax_deferred = (tax_deferred_pct / total) * 100
        normalized_tax_free = (tax_free_pct / total) * 100
    else:
        normalized_taxable = 33.33
        normalized_tax_deferred = 33.33
        normalized_tax_free = 33.34
    
    # Update if changed
    if (normalized_taxable != plan.account_types['taxable'] or
        normalized_tax_deferred != plan.account_types['tax_deferred'] or
        normalized_tax_free != plan.account_types['tax_free']):
        
        plan.account_types = {
            'taxable': normalized_taxable,
            'tax_deferred': normalized_tax_deferred,
            'tax_free': normalized_tax_free
        }
        save_plan(plan)
    
    # Tax considerations and notes
    st.subheader("Tax Efficiency Notes")
    
    # Asset location suggestions based on tax efficiency
    st.write("Suggested Asset Location:")
    
    tax_efficiency = pd.DataFrame([
        {"Asset Class": "Global Equity", "Taxable": "Low", "Tax-Deferred": "High", "Tax-Free": "High"},
        {"Asset Class": "Core Bond", "Taxable": "Low", "Tax-Deferred": "High", "Tax-Free": "Medium"},
        {"Asset Class": "Short-Term Bond", "Taxable": "Medium", "Tax-Deferred": "Medium", "Tax-Free": "Low"},
        {"Asset Class": "Global Credit", "Taxable": "Low", "Tax-Deferred": "High", "Tax-Free": "Medium"},
        {"Asset Class": "Real Assets", "Taxable": "Medium", "Tax-Deferred": "Medium", "Tax-Free": "High"},
        {"Asset Class": "Liquid Alternatives", "Taxable": "Low", "Tax-Deferred": "High", "Tax-Free": "High"}
    ])
    
    st.dataframe(tax_efficiency, use_container_width=True, hide_index=True)

def show_liquidity(client):
    """
    Display and edit liquidity needs, goals, and cash flows for a client's plan.
    
    Parameters:
    -----------
    client : Client object
        The client whose liquidity is being assessed
    """
    st.header("Liquidity")
    
    plan = st.session_state.current_plan
    
    if not plan or not client:
        st.warning("Please select a client to begin planning.")
        return
        
    # Import necessary models
    from models.plan import Goal, CashFlow, LiquidityGoal
    from datetime import datetime
    
    # Define a function to calculate age
    def calculate_age(date_of_birth):
        """Calculate age from date of birth in YYYY-MM-DD format."""
        birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    
    # Calculate client's current age
    current_age = calculate_age(client.date_of_birth)
    
    # Basic Plan Info
    st.subheader("Portfolio Value")
    
    # Initial portfolio value
    initial_portfolio = st.number_input(
        "Initial Portfolio Value ($)",
        min_value=0,
        value=int(plan.initial_portfolio),
        step=10000
    )
    
    if initial_portfolio != plan.initial_portfolio:
        plan.initial_portfolio = initial_portfolio
        save_plan(plan)
    
    # Financial Goals
    st.subheader("Financial Goals")
    st.info("Goals represent one-time or recurring withdrawals for specific purposes (e.g., college tuition, home purchase, etc.)")
    
    # Create columns for goal form
    goal_col1, goal_col2 = st.columns([2, 1])
    
    with goal_col1:
        new_goal_name = st.text_input("Goal Name", key="new_goal_name", placeholder="Ex: College Tuition, Home Purchase, etc.")
    
    with goal_col2:
        new_goal_amount = st.number_input("Amount ($)", key="new_goal_amount", min_value=0, step=5000, value=50000)
    
    goal_col3, goal_col4, goal_col5 = st.columns([1, 1, 1])
    
    with goal_col3:
        new_goal_age = st.number_input("Age to Achieve", key="new_goal_age", min_value=current_age, max_value=100, value=min(current_age + 5, 100))
    
    with goal_col4:
        new_goal_recurring = st.checkbox("Recurring Goal?", key="new_goal_recurring", help="If checked, this goal will repeat annually for the duration specified")
    
    with goal_col5:
        # Initialize the duration variable outside the conditional block
        new_goal_duration = 1
        if new_goal_recurring:
            new_goal_duration = st.number_input("Duration (Years)", key="new_goal_duration", min_value=1, max_value=30, value=4)
        new_goal_priority = st.selectbox("Priority", key="new_goal_priority", options=["High", "Medium", "Low"], index=1)
    
    col_add_goal1, col_add_goal2 = st.columns([3, 1])
    
    with col_add_goal1:
        button_add_goal = st.button("Add Goal")
    
    with col_add_goal2:
        # Debug toggle to show verbose logging
        debug_mode = st.checkbox("Show Debug Info", False)
    
    if button_add_goal:
        if new_goal_name and new_goal_amount > 0:
            if debug_mode:
                st.write(f"DEBUG: Before adding goals, plan has {len(plan.goals)} goals")
            
            # Create a deep copy of the current goals to prevent reference issues
            current_goals = plan.goals.copy()
            
            # Prepare new goals to add
            goals_to_add = []
            if new_goal_recurring and new_goal_duration > 1:
                # Add multiple yearly goals
                for i in range(new_goal_duration):
                    new_goal = Goal(
                        name=f"{new_goal_name} (Year {i+1})",
                        amount=new_goal_amount,
                        age=new_goal_age + i,
                        priority=new_goal_priority
                    )
                    goals_to_add.append(new_goal)
            else:
                # Add a single goal
                new_goal = Goal(
                    name=new_goal_name,
                    amount=new_goal_amount,
                    age=new_goal_age,
                    priority=new_goal_priority
                )
                goals_to_add.append(new_goal)
            
            # Combine current goals with new goals
            plan.goals = current_goals + goals_to_add
            
            if debug_mode:
                st.write(f"DEBUG: After adding goals, plan has {len(plan.goals)} goals")
                st.write(f"DEBUG: Added {len(goals_to_add)} new goals:")
                for i, goal in enumerate(goals_to_add):
                    st.write(f"DEBUG: {i+1}. {goal.name}: ${goal.amount} at age {goal.age}")
            
            # Make sure the session state has the updated plan
            st.session_state.current_plan = plan
            
            try:
                # Save the plan to disk
                save_plan(plan)
                
                if debug_mode:
                    st.write(f"DEBUG: After save, plan has {len(plan.goals)} goals")
                
                # Success message
                st.success(f"Added {len(goals_to_add)} new goal(s)")
                
                # Use a container to control when to rerun the app
                with st.container():
                    # Only show the continue button in debug mode
                    if debug_mode:
                        if st.button("Continue"):
                            st.rerun()
                    else:
                        # Rerun immediately if not in debug mode
                        time.sleep(0.5)  # Small delay to ensure save is complete
                        st.rerun()
            except Exception as e:
                st.error(f"Error saving goals: {e}")
                import traceback
                if debug_mode:
                    st.error(f"Detailed error: {traceback.format_exc()}")
        else:
            st.error("Please enter a goal name and amount greater than zero.")
    
    # Debug buttons to check and fix goals
    debug_col1, debug_col2 = st.columns(2)
    
    with debug_col1:
        if st.button("Debug: Check Goals"):
            st.write(f"Number of goals in plan: {len(plan.goals)}")
            if plan.goals:
                st.write("Goals found:")
                for i, goal in enumerate(plan.goals):
                    st.write(f"{i+1}. {goal.name}: ${goal.amount:,.0f} at age {goal.age}")
            else:
                st.write("No goals found in the plan.")
    
    with debug_col2:
        if st.button("Debug: Add Test Goal"):
            # Add a test goal directly
            test_goal = Goal(
                name="Test Goal",
                amount=100000,
                age=current_age + 10,
                priority="Medium"
            )
            plan.goals.append(test_goal)
            
            # Force update of session state plan
            st.session_state.current_plan = plan
            
            # Save and verify
            st.write("About to save...")
            save_plan(plan)
            st.write("Save completed!")
            
            # Check if goal was actually saved
            st.write(f"Checking goals after save: {len(plan.goals)} goals")
            for i, goal in enumerate(plan.goals):
                st.write(f"{i+1}. {goal.name}: ${goal.amount:,.0f}")
            
            st.success("Test goal added and saved!")
            # Don't rerun, so we can see messages
    
    # Display existing goals
    if plan.goals:
        st.write("Current Goals:")
        goals_data = []
        for i, goal in enumerate(plan.goals):
            goals_data.append({
                "Name": goal.name,
                "Amount ($)": goal.amount,
                "Age": goal.age,
                "Priority": goal.priority
            })
        
        # Edit goals table
        edited_goals = st.data_editor(
            pd.DataFrame(goals_data),
            column_config={
                "Name": st.column_config.TextColumn("Goal Name"),
                "Amount ($)": st.column_config.NumberColumn("Amount ($)", min_value=0, step=1000, format="$%d"),
                "Age": st.column_config.NumberColumn("Age", min_value=current_age, max_value=100, step=1),
                "Priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"])
            },
            use_container_width=True,
            num_rows="dynamic",
            key="goals_editor"
        )
        
        # Update goals in the plan
        new_goals = []
        for _, row in edited_goals.iterrows():
            # Only add goals with valid data
            if pd.notna(row["Name"]) and pd.notna(row["Amount ($)"]) and row["Amount ($)"] > 0:
                new_goals.append(Goal(
                    name=row["Name"],
                    amount=row["Amount ($)"],
                    age=row["Age"],
                    priority=row["Priority"]
                ))
        
        if st.button("Save Goal Changes"):
            plan.goals = new_goals
            save_plan(plan)
            st.success("Goals updated successfully!")
            st.rerun()
            
        # Goals Summary
        st.subheader("Goals Summary")
        
        # Group goals by age
        goals_by_age = {}
        for goal in plan.goals:
            age_key = goal.age
            if age_key not in goals_by_age:
                goals_by_age[age_key] = []
            goals_by_age[age_key].append(goal)
        
        # Create a timeline of goals
        goal_ages = sorted(goals_by_age.keys())
        if goal_ages:
            goal_timeline_data = []
            for age in goal_ages:
                age_goals = goals_by_age[age]
                total_amount = sum(goal.amount for goal in age_goals)
                years_away = age - current_age
                
                goal_timeline_data.append({
                    "Age": age,
                    "Years From Now": years_away,
                    "Number of Goals": len(age_goals),
                    "Total Amount ($)": total_amount,
                    "Goals": ", ".join([f"{goal.name} (${goal.amount:,.0f})" for goal in age_goals])
                })
            
            # Display timeline table
            st.dataframe(
                pd.DataFrame(goal_timeline_data),
                column_config={
                    "Age": st.column_config.NumberColumn("Age"),
                    "Years From Now": st.column_config.NumberColumn("Years From Now"),
                    "Number of Goals": st.column_config.NumberColumn("# Goals"),
                    "Total Amount ($)": st.column_config.NumberColumn("Total Amount", format="$%d"),
                    "Goals": st.column_config.TextColumn("Details", width="large")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Display total goals amount
            total_goals_amount = sum(goal.amount for goal in plan.goals)
            st.write(f"Total Goal Funding Needed: ${total_goals_amount:,.0f}")
            
            # Compare with portfolio value
            if initial_portfolio > 0:
                goals_to_portfolio_ratio = total_goals_amount / initial_portfolio
                st.write(f"Goals to Portfolio Ratio: {goals_to_portfolio_ratio:.2f}")
                
                if goals_to_portfolio_ratio > 2.0:
                    st.warning("Your goals may require significant portfolio growth or additional contributions.")
                elif goals_to_portfolio_ratio > 1.0:
                    st.info("Your goals require moderate portfolio growth to be fully funded.")
                else:
                    st.success("Your current portfolio appears adequate for your goals.")
        
        # Add a visual representation of goals
        if st.checkbox("Show Goals Timeline Chart", value=False):
            # Prepare data for chart
            goal_years = [goal.age - current_age for goal in plan.goals]
            goal_amounts = [goal.amount for goal in plan.goals]
            goal_names = [goal.name for goal in plan.goals]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(goal_years, goal_amounts)
            
            # Add labels
            ax.set_xlabel('Years From Now')
            ax.set_ylabel('Amount ($)')
            ax.set_title('Goals Timeline')
            
            # Add data labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                        f'${goal_amounts[i]:,.0f}',
                        ha='center', va='bottom', rotation=0)
            
            # Display chart
            st.pyplot(fig)
    
    # Cash Flows
    st.subheader("Cash Flows")
    st.info("Cash flows represent ongoing contributions to (positive) or withdrawals from (negative) the portfolio over a period of time.")
    
    # Create columns for cash flow form
    cf_col1, cf_col2 = st.columns([2, 1])
    
    with cf_col1:
        new_cf_name = st.text_input("Cash Flow Description", key="new_cf_name", 
                                  placeholder="Ex: Salary, Pension, Living Expenses, etc.")
    
    with cf_col2:
        new_cf_type = st.radio("Type", 
                              ["Contribution", "Withdrawal"], 
                              key="new_cf_type", 
                              horizontal=True,
                              help="Contributions add to the portfolio, withdrawals subtract from it")
    
    cf_col3, cf_col4 = st.columns([1, 1])
    
    with cf_col3:
        new_cf_amount = st.number_input("Annual Amount ($)", 
                                     key="new_cf_amount", 
                                     min_value=0, 
                                     step=5000, 
                                     value=50000)
    
    with cf_col4:
        new_cf_growth = st.number_input("Annual Growth/Inflation Rate (%)", 
                                      key="new_cf_growth", 
                                      min_value=-10.0, 
                                      max_value=10.0, 
                                      value=2.5, 
                                      step=0.5,
                                      help="Rate at which this cash flow will grow each year (for expenses, this is the inflation rate)")
    
    cf_col5, cf_col6 = st.columns([1, 1])
    
    with cf_col5:
        # Use client's longevity age if available
        restylement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
        longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
        
        new_cf_start_age = st.number_input("Start Age", 
                                        key="new_cf_start_age", 
                                        min_value=current_age, 
                                        max_value=100, 
                                        value=current_age)
    
    with cf_col6:
        new_cf_end_age = st.number_input("End Age", 
                                      key="new_cf_end_age", 
                                      min_value=new_cf_start_age, 
                                      max_value=100, 
                                      value=restylement_age if new_cf_type == "Contribution" else longevity_age)
    
    cf_button_col1, cf_button_col2 = st.columns([3, 1])
    
    with cf_button_col1:
        cf_button_add = st.button("Add Cash Flow")
    
    with cf_button_col2:
        # Debug toggle to show verbose logging
        cf_debug_mode = st.checkbox("Show Debug Info", False, key="cf_debug_toggle")
    
    if cf_button_add:
        if new_cf_name and new_cf_amount > 0 and new_cf_end_age >= new_cf_start_age:
            if cf_debug_mode:
                st.write(f"DEBUG: Before adding cash flow, plan has {len(plan.cash_flows)} cash flows")
            
            # Create a deep copy of the current cash flows to prevent reference issues
            current_cash_flows = plan.cash_flows.copy()
            
            # Set the sign based on the type (contribution is positive, withdrawal is negative)
            amount = new_cf_amount if new_cf_type == "Contribution" else -new_cf_amount
            
            # Create the new cash flow
            new_cf = CashFlow(
                name=new_cf_name,
                amount=amount,
                start_age=new_cf_start_age,
                end_age=new_cf_end_age,
                growth_rate=new_cf_growth / 100
            )
            
            # Add to the plan's cash flows
            current_cash_flows.append(new_cf)
            plan.cash_flows = current_cash_flows
            
            if cf_debug_mode:
                st.write(f"DEBUG: After adding cash flow, plan has {len(plan.cash_flows)} cash flows")
                st.write(f"DEBUG: Added new cash flow: {new_cf_name}, ${amount:,.2f} from age {new_cf_start_age} to {new_cf_end_age}")
            
            # Make sure the session state has the updated plan
            st.session_state.current_plan = plan
            
            try:
                # Save the plan to disk
                save_plan(plan)
                
                if cf_debug_mode:
                    st.write(f"DEBUG: After save, plan has {len(plan.cash_flows)} cash flows")
                
                # Success message
                st.success(f"Added new cash flow: {new_cf_name}")
                
                # Use a container to control when to rerun the app
                with st.container():
                    # Only show the continue button in debug mode
                    if cf_debug_mode:
                        if st.button("Continue", key="cf_continue_button"):
                            st.rerun()
                    else:
                        # Rerun immediately if not in debug mode
                        time.sleep(0.5)  # Small delay to ensure save is complete
                        st.rerun()
            except Exception as e:
                st.error(f"Error saving cash flow: {e}")
                import traceback
                if cf_debug_mode:
                    st.error(f"Detailed error: {traceback.format_exc()}")
        else:
            st.error("Please enter a valid name, amount, and ensure end age is equal to or greater than start age.")
    
    # Display existing cash flows
    if plan.cash_flows:
        st.write("Current Cash Flows:")
        
        # Split cash flows into contributions and withdrawals for better visualization
        contributions = [cf for cf in plan.cash_flows if cf.amount > 0]
        withdrawals = [cf for cf in plan.cash_flows if cf.amount < 0]
        
        # Create tabs for contributions and withdrawals
        cf_tabs = st.tabs(["All Cash Flows", "Contributions", "Withdrawals"])
        
        with cf_tabs[0]:
            # All cash flows
            cash_flows_data = []
            for cf in plan.cash_flows:
                cash_flows_data.append({
                    "Name": cf.name,
                    "Amount ($)": cf.amount,
                    "Type": "Contribution" if cf.amount > 0 else "Withdrawal",
                    "Start Age": cf.start_age,
                    "End Age": cf.end_age,
                    "Growth Rate (%)": cf.growth_rate * 100
                })
            
            # Edit cash flows table
            edited_cash_flows = st.data_editor(
                pd.DataFrame(cash_flows_data),
                column_config={
                    "Name": st.column_config.TextColumn("Description"),
                    "Amount ($)": st.column_config.NumberColumn("Amount ($)", step=1000, format="$%d"),
                    "Type": st.column_config.SelectboxColumn("Type", options=["Contribution", "Withdrawal"], disabled=True),
                    "Start Age": st.column_config.NumberColumn("Start Age", min_value=current_age, max_value=100, step=1),
                    "End Age": st.column_config.NumberColumn("End Age", min_value=current_age, max_value=100, step=1),
                    "Growth Rate (%)": st.column_config.NumberColumn("Growth Rate (%)", min_value=-10.0, max_value=10.0, step=0.1, format="%.1f%%")
                },
                use_container_width=True,
                num_rows="dynamic",
                key="all_cash_flows_editor"
            )
            
            # Update cash flows in the plan
            if st.button("Save Cash Flow Changes"):
                new_cash_flows = []
                for _, row in edited_cash_flows.iterrows():
                    # Only add cash flows with valid data
                    if pd.notna(row["Name"]) and pd.notna(row["Amount ($)"]):
                        # Ensure correct sign based on the selected type
                        amount = abs(row["Amount ($)"])
                        if row["Type"] == "Withdrawal":
                            amount = -amount
                        
                        new_cash_flows.append(CashFlow(
                            name=row["Name"],
                            amount=amount,
                            start_age=row["Start Age"],
                            end_age=row["End Age"],
                            growth_rate=row["Growth Rate (%)"] / 100
                        ))
                
                plan.cash_flows = new_cash_flows
                save_plan(plan)
                st.success("Cash flows updated successfully!")
                st.rerun()
        
        with cf_tabs[1]:
            # Contributions only
            if contributions:
                contrib_data = []
                for cf in contributions:
                    contrib_data.append({
                        "Name": cf.name,
                        "Amount ($)": cf.amount,
                        "Start Age": cf.start_age,
                        "End Age": cf.end_age,
                        "Growth Rate (%)": cf.growth_rate * 100
                    })
                
                st.dataframe(
                    pd.DataFrame(contrib_data),
                    column_config={
                        "Name": st.column_config.TextColumn("Description"),
                        "Amount ($)": st.column_config.NumberColumn("Amount ($)", format="$%d"),
                        "Start Age": st.column_config.NumberColumn("Start Age"),
                        "End Age": st.column_config.NumberColumn("End Age"),
                        "Growth Rate (%)": st.column_config.NumberColumn("Growth Rate (%)", format="%.1f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Calculate total contributions over time
                total_annual_contribution = sum(cf.amount for cf in contributions if cf.start_age <= current_age <= cf.end_age)
                st.write(f"Current Annual Contribution: ${total_annual_contribution:,.2f}")
                
                # Calculate contribution timespan
                contribution_years = max(cf.end_age for cf in contributions) - current_age
                st.write(f"Planned Contribution Timespan: {contribution_years} years")
            else:
                st.info("No contributions defined. Add a positive cash flow as a contribution.")
        
        with cf_tabs[2]:
            # Withdrawals only
            if withdrawals:
                withdr_data = []
                for cf in withdrawals:
                    withdr_data.append({
                        "Name": cf.name,
                        "Amount ($)": abs(cf.amount),  # Show absolute value for readability
                        "Start Age": cf.start_age,
                        "End Age": cf.end_age,
                        "Growth Rate (%)": cf.growth_rate * 100
                    })
                
                st.dataframe(
                    pd.DataFrame(withdr_data),
                    column_config={
                        "Name": st.column_config.TextColumn("Description"),
                        "Amount ($)": st.column_config.NumberColumn("Amount ($)", format="$%d"),
                        "Start Age": st.column_config.NumberColumn("Start Age"),
                        "End Age": st.column_config.NumberColumn("End Age"),
                        "Growth Rate (%)": st.column_config.NumberColumn("Growth Rate (%)", format="%.1f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Calculate total withdrawals over time
                total_annual_withdrawal = sum(abs(cf.amount) for cf in withdrawals if cf.start_age <= current_age <= cf.end_age)
                st.write(f"Current Annual Withdrawal: ${total_annual_withdrawal:,.2f}")
                
                # Calculate withdrawal rate
                if plan.initial_portfolio > 0:
                    current_withdrawal_rate = total_annual_withdrawal / plan.initial_portfolio * 100
                    st.write(f"Current Withdrawal Rate: {current_withdrawal_rate:.2f}%")
            else:
                st.info("No withdrawals defined. Add a negative cash flow as a withdrawal.")
                
        # Overall Cash Flow Summary
        st.subheader("Cash Flow Summary")
        
        # Create cash flow projection by age
        if plan.cash_flows:
            # Calculate years in plan
            max_age = max(cf.end_age for cf in plan.cash_flows)
            years_in_plan = max_age - current_age
            
            # Initialize arrays for projection
            ages = list(range(current_age, max_age + 1))
            net_cash_flows = []
            contrib_cash_flows = []
            withdraw_cash_flows = []
            
            # Calculate cash flow for each year
            for age in ages:
                # Calculate contributions for this age
                contribution = sum(cf.amount * (1 + cf.growth_rate)**(age - cf.start_age) 
                              for cf in plan.cash_flows 
                              if cf.amount > 0 and cf.start_age <= age <= cf.end_age)
                
                # Calculate withdrawals for this age
                withdrawal = sum(cf.amount * (1 + cf.growth_rate)**(age - cf.start_age) 
                            for cf in plan.cash_flows 
                            if cf.amount < 0 and cf.start_age <= age <= cf.end_age)
                
                # Net cash flow
                net_cash_flow = contribution + withdrawal  # withdrawal is already negative
                
                contrib_cash_flows.append(contribution)
                withdraw_cash_flows.append(abs(withdrawal))  # Use absolute value for display
                net_cash_flows.append(net_cash_flow)
            
            # Create projection table
            projection_data = []
            for i, age in enumerate(ages):
                projection_data.append({
                    "Age": age,
                    "Years From Now": age - current_age,
                    "Contributions ($)": contrib_cash_flows[i],
                    "Withdrawals ($)": withdraw_cash_flows[i],
                    "Net Cash Flow ($)": net_cash_flows[i]
                })
            
            # Display table
            st.write("Annual Cash Flow Projection:")
            st.dataframe(
                pd.DataFrame(projection_data),
                column_config={
                    "Age": st.column_config.NumberColumn("Age"),
                    "Years From Now": st.column_config.NumberColumn("Years From Now"),
                    "Contributions ($)": st.column_config.NumberColumn("Contributions", format="$%d"),
                    "Withdrawals ($)": st.column_config.NumberColumn("Withdrawals", format="$%d"),
                    "Net Cash Flow ($)": st.column_config.NumberColumn("Net Cash Flow", format="$%d")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Cash flow chart
            if st.checkbox("Show Cash Flow Chart", value=False):
                # Years from now for x-axis
                years = [age - current_age for age in ages]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot contributions as positive bars
                ax.bar(years, contrib_cash_flows, color='green', alpha=0.7, label='Contributions')
                
                # Plot withdrawals as negative bars
                ax.bar(years, [-w for w in withdraw_cash_flows], color='red', alpha=0.7, label='Withdrawals')
                
                # Plot net cash flow as a line
                ax.plot(years, net_cash_flows, color='blue', marker='o', label='Net Cash Flow')
                
                # Add a horizontal line at y=0
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add labels and title
                ax.set_xlabel('Years From Now')
                ax.set_ylabel('Amount ($)')
                ax.set_title('Cash Flow Projection')
                ax.legend()
                
                # Add retirement age vertical line if available
                restylement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
                restylement_year = restylement_age - current_age
                if restylement_year > 0:
                    ax.axvline(x=restylement_year, color='purple', linestyle='--', alpha=0.7)
                    ax.text(restylement_year, ax.get_ylim()[1]*0.9, 'Restylement', rotation=90, verticalalignment='top')
                
                # Format y-axis with dollar signs
                import matplotlib.ticker as mtick
                fmt = '${x:,.0f}'
                tick = mtick.StrMethodFormatter(fmt)
                ax.yaxis.set_major_formatter(tick)
                
                # Display chart
                st.pyplot(fig)
    else:
        # Add quick sample cash flows
        st.write("Quick Templates:")
        
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            if st.button("Add Salary Template"):
                # Add salary contribution
                plan.cash_flows.append(CashFlow(
                    name="Salary Contribution",
                    amount=75000,
                    start_age=current_age,
                    end_age=client.restylement_age if hasattr(client, 'restylement_age') else 65,
                    growth_rate=0.03  # 3% annual increase
                ))
                save_plan(plan)
                st.rerun()
        
        with sample_col2:
            if st.button("Add Restylement Template"):
                # Use client's restylement and longevity ages if available
                restylement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
                longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
                
                # Add living expenses
                plan.cash_flows.append(CashFlow(
                    name="Living Expenses",
                    amount=-50000,
                    start_age=restylement_age,
                    end_age=longevity_age,
                    growth_rate=0.025  # 2.5% annual increase for inflation
                ))
                save_plan(plan)
                st.rerun()
        
        with sample_col3:
            if st.button("Add Complete Example"):
                # Use client's restylement and longevity ages if available
                restylement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
                longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
                
                # Add salary contribution
                plan.cash_flows.append(CashFlow(
                    name="Salary",
                    amount=100000,
                    start_age=current_age,
                    end_age=restylement_age,
                    growth_rate=0.03  # 3% annual increase
                ))
                
                # Add 401k contribution
                plan.cash_flows.append(CashFlow(
                    name="401(k) Contribution",
                    amount=20000,
                    start_age=current_age,
                    end_age=restylement_age,
                    growth_rate=0.02  # 2% annual increase
                ))
                
                # Add living expenses (pre-restylement)
                plan.cash_flows.append(CashFlow(
                    name="Pre-Restylement Living Expenses",
                    amount=-70000,
                    start_age=current_age,
                    end_age=restylement_age,
                    growth_rate=0.025  # 2.5% annual increase for inflation
                ))
                
                # Add living expenses (post-restylement)
                plan.cash_flows.append(CashFlow(
                    name="Restylement Living Expenses",
                    amount=-60000,
                    start_age=restylement_age,
                    end_age=longevity_age,
                    growth_rate=0.025  # 2.5% annual increase for inflation
                ))
                
                save_plan(plan)
                st.rerun()
    
    # Liquidity needs section
    st.subheader("Liquidity Needs Assessment")
    
    # Liquidity Goals section
    st.write("Liquidity Goals:")
    st.info("Liquidity goals represent funds that need to be kept in cash or highly liquid investments for specific purposes or time frames.")
    
    # Create columns for liquidity goal form
    lg_col1, lg_col2 = st.columns([2, 1])
    
    with lg_col1:
        new_lg_name = st.text_input("Liquidity Goal Description", key="new_lg_name", 
                                  placeholder="Ex: Emergency Fund, Home Down Payment, etc.")
    
    with lg_col2:
        new_lg_amount = st.number_input("Amount ($)", key="new_lg_amount", 
                                     min_value=0, 
                                     step=5000, 
                                     value=20000)
    
    lg_col3, lg_col4, lg_col5 = st.columns([1, 1, 1])
    
    with lg_col3:
        new_lg_timeframe = st.selectbox("Timeframe", key="new_lg_timeframe",
                                       options=["Immediate", "Short-term", "Medium-term"],
                                       index=0,
                                       help="When these funds are needed: Immediate (now), Short-term (< 1 year), Medium-term (1-3 years)")
    
    with lg_col4:
        new_lg_priority = st.selectbox("Priority", key="new_lg_priority", 
                                      options=["High", "Medium", "Low"], 
                                      index=0)
    
    with lg_col5:
        new_lg_notes = st.text_input("Notes", key="new_lg_notes", 
                                    placeholder="Additional information...")
    
    if st.button("Add Liquidity Goal"):
        if new_lg_name and new_lg_amount > 0:
            # Import LiquidityGoal at the beginning of this file
            from models.plan import LiquidityGoal
            
            # Add the liquidity goal
            plan.liquidity_goals.append(LiquidityGoal(
                name=new_lg_name,
                amount=new_lg_amount,
                timeframe=new_lg_timeframe,
                priority=new_lg_priority,
                notes=new_lg_notes
            ))
            
            save_plan(plan)
            st.success(f"Added new liquidity goal: {new_lg_name}")
            st.rerun()
        else:
            st.error("Please enter a goal name and amount greater than zero.")
    
    # Display existing liquidity goals
    if plan.liquidity_goals:
        st.write("Current Liquidity Goals:")
        liquidity_goals_data = []
        for lg in plan.liquidity_goals:
            liquidity_goals_data.append({
                "Name": lg.name,
                "Amount ($)": lg.amount,
                "Timeframe": lg.timeframe,
                "Priority": lg.priority,
                "Notes": lg.notes
            })
        
        # Edit liquidity goals table
        edited_liquidity_goals = st.data_editor(
            pd.DataFrame(liquidity_goals_data),
            column_config={
                "Name": st.column_config.TextColumn("Description"),
                "Amount ($)": st.column_config.NumberColumn("Amount ($)", min_value=0, step=1000, format="$%d"),
                "Timeframe": st.column_config.SelectboxColumn("Timeframe", options=["Immediate", "Short-term", "Medium-term"]),
                "Priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"]),
                "Notes": st.column_config.TextColumn("Notes", width="large")
            },
            use_container_width=True,
            num_rows="dynamic",
            key="liquidity_goals_editor"
        )
        
        # Update liquidity goals in the plan
        if st.button("Save Liquidity Goal Changes"):
            new_liquidity_goals = []
            from models.plan import LiquidityGoal
            
            for _, row in edited_liquidity_goals.iterrows():
                # Only add goals with valid data
                if pd.notna(row["Name"]) and pd.notna(row["Amount ($)"]) and row["Amount ($)"] > 0:
                    new_liquidity_goals.append(LiquidityGoal(
                        name=row["Name"],
                        amount=row["Amount ($)"],
                        timeframe=row["Timeframe"],
                        priority=row["Priority"],
                        notes=row["Notes"] if pd.notna(row["Notes"]) else ""
                    ))
            
            plan.liquidity_goals = new_liquidity_goals
            save_plan(plan)
            st.success("Liquidity goals updated successfully!")
            st.rerun()
        
        # Calculate total liquidity needs
        total_liquidity = sum(lg.amount for lg in plan.liquidity_goals)
        st.write(f"Total Liquidity Goals: ${total_liquidity:,.0f}")
        
        # Calculate liquidity coverage ratio
        if initial_portfolio > 0:
            liquidity_coverage = total_liquidity / initial_portfolio
            st.write(f"Liquidity Coverage Ratio (Total Liquidity / Portfolio): {liquidity_coverage:.2f}")
            
            if liquidity_coverage > 0.5:
                st.warning("High liquidity ratio: Consider your asset allocation strategy carefully.")
            else:
                st.success("Healthy liquidity coverage ratio.")
    else:
        # Add quick templates
        st.write("Suggested Liquidity Goals:")
        
        template_col1, template_col2 = st.columns(2)
        
        with template_col1:
            if st.button("Add Emergency Fund"):
                from models.plan import LiquidityGoal
                
                # Calculate recommended emergency fund (if we have expense data)
                monthly_expenses = 0
                for cf in plan.cash_flows:
                    if cf.amount < 0:  # It's an expense
                        # Convert annual amount to monthly
                        monthly_amount = abs(cf.amount) / 12
                        if cf.start_age <= current_age <= cf.end_age:
                            monthly_expenses += monthly_amount
                
                # Use calculated amount or default
                if monthly_expenses > 0:
                    emergency_fund = monthly_expenses * 6  # 6 months of expenses
                else:
                    emergency_fund = 30000  # Default amount
                
                # Add emergency fund goal
                plan.liquidity_goals.append(LiquidityGoal(
                    name="Emergency Fund",
                    amount=emergency_fund,
                    timeframe="Immediate",
                    priority="High",
                    notes="6 months of living expenses for unexpected events"
                ))
                
                save_plan(plan)
                st.rerun()
        
        with template_col2:
            if st.button("Add Major Purchase Fund"):
                from models.plan import LiquidityGoal
                
                # Add major purchase fund
                plan.liquidity_goals.append(LiquidityGoal(
                    name="Major Purchase Fund",
                    amount=20000,
                    timeframe="Medium-term",
                    priority="Medium",
                    notes="Fund for upcoming major purchases (car, home renovations, etc.)"
                ))
                
                save_plan(plan)
                st.rerun()
    
    # Emergency fund calculation
    st.write("Emergency Fund Recommendation:")
    
    # Find monthly expenses from cash flows
    monthly_expenses = 0
    for cf in plan.cash_flows:
        if cf.amount < 0:  # It's an expense
            # Convert annual amount to monthly
            monthly_amount = abs(cf.amount) / 12
            if cf.start_age <= current_age <= cf.end_age:
                monthly_expenses += monthly_amount
    
    if monthly_expenses > 0:
        st.write(f"Estimated Monthly Expenses: ${monthly_expenses:,.2f}")
        
        # Calculate emergency fund recommendation (3-6 months of expenses)
        emergency_fund_min = monthly_expenses * 3
        emergency_fund_max = monthly_expenses * 6
        
        st.write(f"Recommended Emergency Fund: ${emergency_fund_min:,.0f} to ${emergency_fund_max:,.0f}")
    else:
        st.info("Add expense cash flows to calculate emergency fund recommendations.")
    
    # Liquidity needed for near-term goals
    if plan.goals:
        st.write("Liquidity Needed for Near-Term Goals:")
        
        # Find goals in the next 5 years
        near_term_goals = []
        for goal in plan.goals:
            if goal.age - current_age <= 5:
                near_term_goals.append({
                    "Goal": goal.name,
                    "Age": goal.age,
                    "Years From Now": goal.age - current_age,
                    "Amount": goal.amount
                })
        
        if near_term_goals:
            # Display near-term goals
            near_term_df = pd.DataFrame(near_term_goals)
            st.dataframe(near_term_df, use_container_width=True, hide_index=True)
            
            # Calculate total near-term needs
            total_near_term = sum(goal["Amount"] for goal in near_term_goals)
            st.write(f"Total Liquidity Needed for Near-Term Goals: ${total_near_term:,.0f}")
            
            # Liquidity ratio
            if initial_portfolio > 0:
                liquidity_ratio = total_near_term / initial_portfolio
                st.write(f"Liquidity Ratio (Near-Term Needs / Portfolio): {liquidity_ratio:.2f}")
                
                if liquidity_ratio > 0.5:
                    st.warning("High liquidity ratio: Consider keeping more assets in liquid investments.")
                else:
                    st.success("Healthy liquidity ratio: Near-term needs appear manageable.")
        else:
            st.info("No goals identified within the next 5 years.")

def show_monte_carlo_simulation(client):
    """
    Run and display advanced Monte Carlo simulation results with enhanced visualization
    and comprehensive risk metrics using modern portfolio theory techniques.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to run the simulation
    """
    st.header("Advanced Monte Carlo Simulation")
    
    plan = st.session_state.current_plan
    
    # Display a notice about the enhanced modeling
    st.info("""
    This advanced simulation implements several improvements over traditional Monte Carlo models:
    * Mean-reverting returns instead of random walk assumptions
    * Time-varying expected returns based on market valuations
    * Fat-tailed return distributions to better model market crashes
    * Target wealth path tracking to better evaluate retirement readiness
    """)
    
    # Display the current asset allocation used for simulation
    asset_classes = st.session_state.market_assumptions['asset_classes']
    
    if hasattr(plan, 'asset_allocation') and plan.asset_allocation is not None:
        st.subheader("Current Asset Allocation for Simulation")
        # Display current allocation as a pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(plan.asset_allocation, labels=asset_classes, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Asset Allocation Used in Simulation')
        st.pyplot(fig)
    elif hasattr(plan, 'glidepath') and plan.glidepath is not None:
        st.subheader("Using Dynamic Glidepath for Simulation")
        st.info("Using the optimized glidepath allocation that changes over time.")
    else:
        st.warning("Please set an initial asset allocation or run glidepath optimization before simulation.")
    
    # Simulation parameters with enhanced options
    st.subheader("Simulation Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More simulations give more accurate results but take longer to run"
        )
    
    with col2:
        # Use client's longevity age as default value
        longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
        max_age = st.number_input(
            "Maximum Age",
            min_value=70,
            max_value=120,
            value=longevity_age,
            step=1,
            help="Maximum age to simulate to (typically life expectancy plus a buffer)"
        )
    
    with col3:
        view = st.selectbox(
            "Market View",
            options=["long_term", "short_term"],
            format_func=lambda x: "Long-Term Equilibrium" if x == "long_term" else "Current Conditions",
            index=0,
            help="Use long-term for strategic planning, short-term for tactical adjustments"
        )
    
    # Additional parameters
    col1, col2 = st.columns(2)
    
    with col1:
        mean_reversion = st.slider(
            "Mean Reversion Speed",
            min_value=0.05,
            max_value=0.30,
            value=plan.mean_reversion_speed if hasattr(plan, 'mean_reversion_speed') else 0.15,
            step=0.01,
            help="How quickly returns revert to long-term averages (higher = faster)"
        )
        plan.mean_reversion_speed = mean_reversion
        
    with col2:
        risk_aversion = st.slider(
            "Risk Aversion",
            min_value=1.0,
            max_value=10.0,
            value=plan.risk_aversion if hasattr(plan, 'risk_aversion') else 3.0,
            step=0.5,
            help="Higher values = more conservative allocations"
        )
        plan.risk_aversion = risk_aversion
    
    # Check if plan has necessary data for simulation
    if not hasattr(plan, 'initial_portfolio') or plan.initial_portfolio <= 0:
        st.warning("Please set an initial portfolio value in the Liquidity section before running simulations.")
        return
    
    if (not hasattr(plan, 'asset_allocation') or plan.asset_allocation is None) and (not hasattr(plan, 'glidepath') or plan.glidepath is None):
        st.warning("Please set an initial asset allocation or run glidepath optimization before running simulations.")
        return
    
    # Run simulation button
    if st.button("Run Advanced Monte Carlo Simulation"):
        with st.spinner("Running advanced simulation - this may take a moment..."):
            # Get market assumptions
            market_assumptions = st.session_state.market_assumptions
            
            # Run simulation with enhanced modeling
            simulation_results = run_monte_carlo_simulation(
                client,
                plan,
                market_assumptions,
                num_simulations=num_simulations,
                max_age=max_age
            )
            
            # Store the results
            st.session_state.simulation_results = simulation_results
            
            # Save the plan with simulation results
            save_plan(plan)
    
    # Display simulation results if available
    if 'simulation_results' in st.session_state and st.session_state.simulation_results:
        results = st.session_state.simulation_results
        
        st.subheader("Comprehensive Simulation Results")
        
        # Success probability
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_prob = results['success_probability']
            st.metric(
                "Probability of Overall Success",
                f"{success_prob:.1%}",
                help="Percentage of simulations where the portfolio lasts until maximum age"
            )
            
            # Create color-coded success indicator
            if success_prob >= 0.75:
                st.success("High probability of success")
            elif success_prob >= 0.5:
                st.warning("Moderate probability of success")
            else:
                st.error("Low probability of success")
        
        with col2:
            if 'retirement_success_probability' in results and not pd.isna(results['retirement_success_probability']):
                st.metric(
                    "Probability of Meeting Retirement Target",
                    f"{results['retirement_success_probability']:.1%}",
                    help="Percentage of simulations where the target retirement wealth is reached"
                )
        
        with col3:
            # Calculate target retirement wealth
            if 'target_wealth_paths' in results and 'target_percentiles' in results:
                retirement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
                current_year = pd.Timestamp.now().year
                birth_year = pd.Timestamp(client.date_of_birth).year
                current_age = current_year - birth_year
                
                # Find index for retirement age
                retirement_idx = retirement_age - current_age
                if retirement_idx > 0 and retirement_idx < len(results['target_percentiles']['median']):
                    target_wealth = results['target_percentiles']['median'][retirement_idx]
                    st.metric(
                        "Target Restylement Wealth",
                        f"${target_wealth:,.0f}",
                        help="Estimated wealth needed at restylement age"
                    )
        
        # Plot the enhanced simulation results
        fig = plot_monte_carlo_results(results, client.full_name())
        st.pyplot(fig)
        
        # Calculate comprehensive risk metrics
        risk_metrics = calculate_shortfall_risk(results)
        
        # Display enhanced risk metrics in tabs
        risk_tabs = st.tabs(["Portfolio Statistics", "Shortfall Risk", "Withdrawal Analysis"])
        
        with risk_tabs[0]:
            col1, col2 = st.columns(2)
            
            # Calculate and display statistics from the simulation
            final_values = results['portfolio_paths'][:, -1]
            
            with col1:
                st.metric("Median Final Portfolio", f"${np.median(final_values):,.0f}")
                st.metric("90th Percentile", f"${np.percentile(final_values, 90):,.0f}")
            
            with col2:
                st.metric("10th Percentile", f"${np.percentile(final_values, 10):,.0f}")
                st.metric("Average Maximum Drawdown", f"{risk_metrics['average_max_drawdown']:.1%}")
        
        with risk_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Shortfall Probability", f"{risk_metrics['shortfall_probability']:.1%}")
                st.metric("Expected Shortfall", f"${risk_metrics.get('expected_shortfall', 0):,.0f}")
            
            with col2:
                st.metric("Conditional Value at Risk (CVaR)", f"${risk_metrics['conditional_value_at_risk']:,.0f}")
                st.metric("Lifestyle Reduction Risk", f"{risk_metrics.get('lifestyle_reduction', 0):.1%}")
            
            st.info("""
            **Understanding Shortfall Risk Metrics:**
            - **Shortfall Probability**: Chance of running out of money before the end of plan
            - **Expected Shortfall**: Average amount below your target wealth across all simulations
            - **Conditional Value at Risk**: Average portfolio value in the worst 5% of scenarios
            - **Lifestyle Reduction Risk**: Potential reduction in spending needed in poor scenarios
            """)
        
        with risk_tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sustainable_withdrawal_rate' in risk_metrics:
                    st.metric(
                        "Sustainable Withdrawal Rate",
                        f"{risk_metrics['sustainable_withdrawal_rate']*100:.2f}%",
                        help="Maximum withdrawal rate with 90% probability of success"
                    )
            
            with col2:
                # Calculate dollar amount of sustainable withdrawal based on current portfolio
                if 'sustainable_withdrawal_rate' in risk_metrics:
                    initial_portfolio = plan.initial_portfolio
                    annual_withdrawal = initial_portfolio * risk_metrics['sustainable_withdrawal_rate']
                    
                    st.metric(
                        "Annual Sustainable Withdrawal",
                        f"${annual_withdrawal:,.0f}",
                        help="Dollar amount that can be withdrawn annually with 90% success probability"
                    )
                
            # Add explanation
            st.info("""
            The sustainable withdrawal rate is the percentage of your portfolio that you can withdraw
            each year with a 90% probability of not running out of money before your maximum age.
            This is a dynamic measure that should be recalculated regularly as markets change.
            """)
        
        # Recommendation for optimization if needed
        if success_prob < 0.75:
            st.warning("""
            The current asset allocation may not be optimal. Consider running the Glidepath Optimization 
            to find a dynamic asset allocation strategy that minimizes shortfall risk.
            """)
    else:
        st.info("Run a Monte Carlo simulation to see the results.")

def show_glidepath_optimization(client):
    """
    Run and display glidepath optimization results.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to optimize the glidepath
    """
    st.header("Glidepath Optimization")
    
    # Description of what this does
    st.markdown("""
    The Glidepath Optimization feature determines the optimal asset allocation strategy that **minimizes shortfall risk**
    (the probability of running out of money) throughout your lifetime. It creates a multi-period allocation strategy 
    that adjusts over time based on your changing financial needs and risk profile.
    
    The optimization uses two key time periods:
    - **7-year mean reversion**: The optimization starts with short-term capital market assumptions that gradually blend to long-term assumptions over a 7-year period.
    - **7-year pre-restylement protection**: Within 7 years of restylement, the optimization increasingly prioritizes capital preservation by putting greater weight on shortfall risk reduction.
    """)
    
    plan = st.session_state.current_plan
    
    # Optimization parameters
    st.subheader("Optimization Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            key="glidepath_simulations"
        )
    
    with col2:
        # Use client's longevity age as default value
        longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
        max_age = st.number_input(
            "Maximum Age",
            min_value=70,
            max_value=120,
            value=longevity_age,
            step=1,
            key="glidepath_max_age"
        )
    
    # Calculate years until restylement and set periods equal to that
    # Get client's current age
    current_year = datetime.now().year
    birth_year = datetime.strptime(client.date_of_birth, '%Y-%m-%d').year
    current_age = current_year - birth_year
    
    # Calculate years until restylement
    years_to_restylement = max(1, client.restylement_age - current_age)
    
    # Display calculated number of periods
    with col3:
        st.info(f"Number of glidepath periods: {years_to_restylement} (automatically set to match years until restylement)")
    
    # Button to run the optimization
    if st.button("Optimize Glidepath"):
        with st.spinner("Optimizing glidepath to minimize shortfall risk... (this may take a few minutes)"):
            # Get market assumptions
            market_assumptions = st.session_state.market_assumptions
            
            # Run the optimization
            glidepath_results = optimize_glidepath(
                client,
                plan,
                market_assumptions,
                num_simulations=num_simulations,
                max_age=max_age,
                num_periods=years_to_restylement
            )
            
            # Store the results
            st.session_state.glidepath_results = glidepath_results
            
            # Save the plan with glidepath results
            save_plan(plan)
    
    # Display glidepath results if available
    if st.session_state.glidepath_results:
        results = st.session_state.glidepath_results
        
        # Success probability
        success_prob = results['success_probability']
        st.subheader(f"Optimized Probability of Success: {success_prob:.1%}")
        st.markdown("*Probability of portfolio not going below $0 before the end of plan*")
        
        # Compare before and after optimization if we have simulation results
        if 'simulation_results' in st.session_state and st.session_state.simulation_results:
            original_prob = st.session_state.simulation_results['success_probability']
            improvement = success_prob - original_prob
            
            if improvement > 0:
                st.success(f"Optimization improved success probability by {improvement:.1%} compared to initial allocation")
            else:
                st.info(f"Optimization resulted in same success probability as initial allocation")
        
        # Create color-coded success indicator
        if success_prob >= 0.75:
            st.success("High probability of meeting financial goals with optimized glidepath")
        elif success_prob >= 0.5:
            st.warning("Moderate probability of meeting financial goals with optimized glidepath")
        else:
            st.error("Low probability of meeting financial goals with optimized glidepath")
        
        # Plot the glidepath
        fig = plot_glidepath(results)
        st.pyplot(fig)
        
        # Display glidepath data table
        st.subheader("Optimized Glidepath Data")
        
        # Create a DataFrame with the glidepath data
        glidepath_data = []
        for i, age in enumerate(results['ages']):
            row = {'Age': age}
            for j, asset_class in enumerate(results['asset_classes']):
                row[asset_class] = f"{results['glidepath'][i][j] * 100:.1f}%"
            glidepath_data.append(row)
        
        glidepath_df = pd.DataFrame(glidepath_data)
        st.dataframe(glidepath_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the glidepath optimization to see the results.")

def show_ips_generation(client):
    """
    Generate and display an Investment Policy Statement.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to generate the IPS
    """
    st.header("Investment Policy Statement")
    
    plan = st.session_state.current_plan
    glidepath_result = st.session_state.glidepath_results
    
    # Get the client's risk profile
    risk_profile = client.get_risk_profile()
    
    # Button to generate the IPS
    if st.button("Generate Investment Policy Statement"):
        # Check if we have valid glidepath results
        if not glidepath_result or not isinstance(glidepath_result, dict) or 'glidepath' not in glidepath_result:
            st.error("Cannot generate IPS: Missing or invalid glidepath optimization results. Please run the Glidepath Optimization first.")
            return
            
        # If necessary, initialize any missing data with defaults to prevent errors
        if 'success_probability' not in glidepath_result:
            glidepath_result['success_probability'] = 0.85  # Default 85% success probability
            
        if 'asset_classes' not in glidepath_result:
            glidepath_result['asset_classes'] = [
                'Global Equity', 'Core Bond', 'Short-Term Bond', 
                'Global Credit', 'Real Assets', 'Liquid Alternatives'
            ]
            
        if 'ages' not in glidepath_result:
            # Create age sequence from current age to longevity age
            current_age = datetime.now().year - datetime.strptime(client.date_of_birth, '%Y-%m-%d').year
            glidepath_result['ages'] = list(range(current_age, client.longevity_age + 1))
        
        with st.spinner("Generating Investment Policy Statement..."):
            try:
                # Generate the IPS
                ips_pdf = generate_investment_policy_statement(
                    client,
                    plan,
                    glidepath_result,
                    risk_profile
                )
                
                # Store the PDF in session state
                st.session_state.ips_pdf = ips_pdf
            except Exception as e:
                import traceback
                st.error(f"Error generating IPS: {e}")
                st.error(f"Detailed error: {traceback.format_exc()}")
                return
    
    # Display IPS if generated
    if hasattr(st.session_state, 'ips_pdf') and st.session_state.ips_pdf:
        st.success("Investment Policy Statement generated successfully!")
        
        # Display download button
        display_ips_download_button(st.session_state.ips_pdf)
        
        # Display IPS sections preview
        st.subheader("Investment Policy Statement Sections")
        
        sections = [
            "Introduction",
            "Risk Profile",
            "Return Expectations",
            "Time Horizon",
            "Tax Considerations",
            "Liquidity Requirements",
            "Unique Circumstances and Constraints",
            "Strategic Asset Allocation",
            "Glidepath Strategy",
            "Monitoring and Review"
        ]
        
        for section in sections:
            with st.expander(section):
                st.write(f"This section contains details about {section.lower()} for {client.full_name()}.")
    else:
        st.info("Generate an Investment Policy Statement to see the results.")

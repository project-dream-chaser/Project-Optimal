import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os
import json

from models.client import Client
from models.plan import Plan, Goal, CashFlow
from utils.monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_results
from utils.optimization import optimize_glidepath, plot_glidepath
from utils.ips_generator import generate_investment_policy_statement, display_ips_download_button
from utils.client_management import get_client_by_id

def show_plan_page():
    """Display the Plan page with financial planning tools."""
    st.title("Financial Planning")
    
    # Check if there are clients available
    if not st.session_state.clients:
        st.warning("Please add clients first before creating a financial plan.")
        return
    
    # Create vertical tabs for plan sections
    plan_tabs = st.tabs([
        "Goals & Cash Flows", 
        "Monte Carlo Simulation", 
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
    
    # Goals & Cash Flows tab
    with plan_tabs[0]:
        if st.session_state.current_plan:
            show_goals_cash_flows(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Monte Carlo Simulation tab
    with plan_tabs[1]:
        if st.session_state.current_plan:
            show_monte_carlo_simulation(selected_client)
        else:
            st.info("Select a client to begin planning.")
    
    # Glidepath Optimization tab
    with plan_tabs[2]:
        if st.session_state.current_plan and st.session_state.simulation_results:
            show_glidepath_optimization(selected_client)
        else:
            st.info("Run a Monte Carlo simulation first before optimizing the glidepath.")
    
    # Investment Policy Statement tab
    with plan_tabs[3]:
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
            plan = Plan.from_dict(plan_data)
            st.session_state.current_plan = plan
            
            # If we have simulation results, try to load them
            sim_path = f'data/plans/{client.id}_simulation.json'
            if os.path.exists(sim_path):
                with open(sim_path, 'r') as f:
                    sim_data = json.load(f)
                st.session_state.simulation_results = sim_data
            
            # If we have glidepath results, try to load them
            glidepath_path = f'data/plans/{client.id}_glidepath.json'
            if os.path.exists(glidepath_path):
                with open(glidepath_path, 'r') as f:
                    glidepath_data = json.load(f)
                    
                # Convert lists back to numpy arrays where needed
                if 'glidepath' in glidepath_data:
                    glidepath_data['glidepath'] = np.array(glidepath_data['glidepath'])
                    
                st.session_state.glidepath_results = glidepath_data
            
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
        initial_portfolio=100000,  # Default initial portfolio
        asset_allocation=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],  # Default allocation
        allocation_constraints=allocation_constraints,  # Default constraints
        risk_aversion=3.0,  # Default risk aversion
        mean_reversion_speed=0.15  # Default mean reversion speed
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
        
        with open(f'data/plans/{plan.client_id}.json', 'w') as f:
            json.dump(plan_dict, f)
            
        # If we have simulation results, save them too
        if st.session_state.simulation_results:
            with open(f'data/plans/{plan.client_id}_simulation.json', 'w') as f:
                json.dump(st.session_state.simulation_results, f)
        
        # If we have glidepath results, save them too
        if st.session_state.glidepath_results:
            # Convert numpy arrays to lists for JSON serialization
            glidepath_dict = st.session_state.glidepath_results.copy()
            if 'glidepath' in glidepath_dict:
                glidepath_dict['glidepath'] = glidepath_dict['glidepath'].tolist()
                
            with open(f'data/plans/{plan.client_id}_glidepath.json', 'w') as f:
                json.dump(glidepath_dict, f)
                
    except Exception as e:
        st.error(f"Error saving plan: {e}")

def show_goals_cash_flows(client):
    """
    Display and edit goals and cash flows for a client's plan.
    
    Parameters:
    -----------
    client : Client object
        The client whose plan is being edited
    """
    st.header("Goals & Cash Flows")
    
    plan = st.session_state.current_plan
    
    # Basic Plan Info
    st.subheader("Basic Plan Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate client's current age
        birth_year = datetime.strptime(client.date_of_birth, '%Y-%m-%d').year
        current_year = datetime.now().year
        current_age = current_year - birth_year
        
        st.write(f"Client: {client.full_name()}")
        st.write(f"Age: {current_age}")
        
        if client.spouse:
            spouse_birth_year = datetime.strptime(client.spouse['date_of_birth'], '%Y-%m-%d').year
            spouse_age = current_year - spouse_birth_year
            st.write(f"Spouse: {client.spouse['first_name']} {client.spouse['last_name']}")
            st.write(f"Spouse Age: {spouse_age}")
    
    with col2:
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
    
    # Asset Allocation Note
    st.subheader("Asset Allocation")
    st.info("Asset allocation will be determined through optimization to minimize shortfall risk. Run the Glidepath Optimization to generate the optimal asset allocation that maximizes your probability of success.")
    
    asset_classes = [
        'Global Equity',
        'Core Bond',
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    
    # If we have glidepath results, show the initial allocation from the glidepath
    if 'glidepath_results' in st.session_state and st.session_state.glidepath_results:
        glidepath = st.session_state.glidepath_results['glidepath']
        if glidepath is not None and len(glidepath) > 0:
            initial_allocation = glidepath[0]
            
            # Display optimized initial allocation as a pie chart
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(initial_allocation, labels=asset_classes, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Optimized Initial Asset Allocation')
            st.pyplot(fig)
            
            # Update the plan's asset allocation with the optimized initial allocation
            plan.asset_allocation = initial_allocation.tolist()
            save_plan(plan)
    # If no glidepath results yet, show a placeholder or message
    elif hasattr(plan, 'asset_allocation') and plan.asset_allocation:
        st.write("Current allocation (will be optimized when you run the Glidepath Optimization):")
        # Display current allocation as a pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(plan.asset_allocation, labels=asset_classes, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Current Asset Allocation (Not Optimized)')
        st.pyplot(fig)
    
    # Asset Allocation Constraints
    st.subheader("Asset Allocation Constraints")
    
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
    
    # Financial Goals
    st.subheader("Financial Goals")
    
    # Display existing goals
    if plan.goals:
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
                "Amount ($)": st.column_config.NumberColumn("Amount ($)", min_value=0, step=1000),
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
            new_goals.append(Goal(
                name=row["Name"],
                amount=row["Amount ($)"],
                age=row["Age"],
                priority=row["Priority"]
            ))
        
        plan.goals = new_goals
        save_plan(plan)
    else:
        # Add a new goal button
        if st.button("Add First Goal"):
            plan.goals.append(Goal(
                name="Retirement",
                amount=1000000,
                age=65,
                priority="High"
            ))
            save_plan(plan)
            st.rerun()
    
    # Cash Flows
    st.subheader("Cash Flows")
    
    # Display existing cash flows
    if plan.cash_flows:
        cash_flows_data = []
        for i, cf in enumerate(plan.cash_flows):
            cash_flows_data.append({
                "Name": cf.name,
                "Amount ($)": cf.amount,
                "Start Age": cf.start_age,
                "End Age": cf.end_age,
                "Growth Rate (%)": cf.growth_rate * 100
            })
        
        # Edit cash flows table
        edited_cash_flows = st.data_editor(
            pd.DataFrame(cash_flows_data),
            column_config={
                "Name": st.column_config.TextColumn("Description"),
                "Amount ($)": st.column_config.NumberColumn("Amount ($)", step=100),
                "Start Age": st.column_config.NumberColumn("Start Age", min_value=current_age, max_value=100, step=1),
                "End Age": st.column_config.NumberColumn("End Age", min_value=current_age, max_value=100, step=1),
                "Growth Rate (%)": st.column_config.NumberColumn("Growth Rate (%)", min_value=-10.0, max_value=10.0, step=0.1)
            },
            use_container_width=True,
            num_rows="dynamic",
            key="cash_flows_editor"
        )
        
        # Update cash flows in the plan
        new_cash_flows = []
        for _, row in edited_cash_flows.iterrows():
            new_cash_flows.append(CashFlow(
                name=row["Name"],
                amount=row["Amount ($)"],
                start_age=row["Start Age"],
                end_age=row["End Age"],
                growth_rate=row["Growth Rate (%)"] / 100
            ))
        
        plan.cash_flows = new_cash_flows
        save_plan(plan)
    else:
        # Add default cash flows
        if st.button("Add Sample Cash Flows"):
            # Use the client's restylement and longevity ages if available
            restylement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
            longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
            
            # Add restylement contributions
            plan.cash_flows.append(CashFlow(
                name="Restylement Contributions",
                amount=20000,
                start_age=current_age,
                end_age=restylement_age,
                growth_rate=0.02  # 2% annual increase
            ))
            
            # Add restylement withdrawals
            plan.cash_flows.append(CashFlow(
                name="Restylement Withdrawals",
                amount=-70000,
                start_age=restylement_age,
                end_age=longevity_age,
                growth_rate=0.025  # 2.5% annual increase for inflation
            ))
            
            save_plan(plan)
            st.rerun()

def show_monte_carlo_simulation(client):
    """
    Run and display Monte Carlo simulation results.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to run the simulation
    """
    st.header("Monte Carlo Simulation")
    
    plan = st.session_state.current_plan
    
    # Display a notice about asset allocation optimization
    st.info("Monte Carlo simulation uses the current asset allocation. For best results, run the Glidepath Optimization first to determine the optimal asset allocation that minimizes shortfall risk.")
    
    # Display the current asset allocation used for simulation
    asset_classes = [
        'Global Equity',
        'Core Bond', 
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    
    if hasattr(plan, 'asset_allocation') and plan.asset_allocation:
        st.subheader("Current Asset Allocation for Simulation")
        # Display current allocation as a pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(plan.asset_allocation, labels=asset_classes, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Asset Allocation Used in Simulation')
        st.pyplot(fig)
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
    
    with col2:
        # Use client's longevity age as default value
        longevity_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
        max_age = st.number_input(
            "Maximum Age",
            min_value=70,
            max_value=120,
            value=longevity_age,
            step=1
        )
    
    with col3:
        view = st.selectbox(
            "Market View",
            options=["long_term", "short_term"],
            format_func=lambda x: "Long-Term (Equilibrium)" if x == "long_term" else "Short-Term"
        )
    
    # Button to run the simulation
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Get market assumptions
            market_assumptions = st.session_state.market_assumptions
            
            # Run the simulation
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
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results
        
        # Success probability
        success_prob = results['success_probability']
        st.subheader(f"Probability of Success: {success_prob:.1%}")
        st.markdown("*Probability of portfolio not going below $0 before the end of plan*")
        
        # Create color-coded success indicator
        if success_prob >= 0.75:
            st.success("High probability of meeting financial goals")
        elif success_prob >= 0.5:
            st.warning("Moderate probability of meeting financial goals")
        else:
            st.error("Low probability of meeting financial goals")
        
        # Plot the simulation results
        fig = plot_monte_carlo_results(results, client.full_name())
        st.pyplot(fig)
        
        # Additional statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Statistics")
            
            # Calculate and display statistics from the simulation
            final_values = results['portfolio_paths'][:, -1]
            st.write(f"Median Final Portfolio: ${np.median(final_values):,.0f}")
            st.write(f"90th Percentile: ${np.percentile(final_values, 90):,.0f}")
            st.write(f"10th Percentile: ${np.percentile(final_values, 10):,.0f}")
        
        with col2:
            st.subheader("Risk Metrics")
            
            # Calculate probability of ruin (portfolio value going to zero)
            ruin_prob = np.sum(final_values <= 0) / len(final_values)
            st.write(f"Probability of Shortfall: {ruin_prob:.1%}")
            st.write("(Probability of running out of money)")
            
            # Calculate average shortfall
            shortfall_values = final_values[final_values <= 0]
            avg_shortfall = np.mean(shortfall_values) if len(shortfall_values) > 0 else 0
            st.write(f"Average Shortfall: ${abs(avg_shortfall):,.0f}")
            
        # Recommendation for optimization
        if success_prob < 0.75:
            st.warning("The current asset allocation may not be optimal. Consider running the Glidepath Optimization to find an asset allocation that minimizes shortfall risk.")
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
    
    with col3:
        num_periods = st.number_input(
            "Number of Glidepath Periods",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="How many allocation adjustments over the lifetime of the plan"
        )
    
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
                num_periods=num_periods
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
        with st.spinner("Generating Investment Policy Statement..."):
            # Generate the IPS
            ips_pdf = generate_investment_policy_statement(
                client,
                plan,
                glidepath_result,
                risk_profile
            )
            
            # Store the PDF in session state
            st.session_state.ips_pdf = ips_pdf
    
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

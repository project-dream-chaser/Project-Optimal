import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils.monte_carlo import run_monte_carlo_simulation, calculate_shortfall_risk

def optimize_glidepath(client, plan, market_assumptions, num_simulations=500, max_age=95, num_periods=10):
    """
    Optimize the glidepath to minimize shortfall risk.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to optimize the glidepath
    plan : Plan object
        Contains goals, cash flows, and initial asset allocation
    market_assumptions : dict
        Market assumptions including expected returns, volatility, and correlations
    num_simulations : int
        Number of simulations to run for each glidepath evaluation
    max_age : int
        Maximum age to simulate to
    num_periods : int
        Number of periods in the glidepath
        
    Returns:
    --------
    dict
        Optimized glidepath and related metrics
    """
    # Calculate client's current age from date of birth
    current_year = pd.Timestamp.now().year
    birth_year = pd.Timestamp(client.date_of_birth).year
    current_age = current_year - birth_year
    
    # Asset classes
    asset_classes = [
        'Global Equity',
        'Core Bond',
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    num_assets = len(asset_classes)
    
    # Initial asset allocation
    initial_allocation = plan.asset_allocation
    
    # Calculate years to simulate
    years_to_simulate = max_age - current_age
    period_length = max(1, years_to_simulate // num_periods)
    
    # Define constraints for optimization
    # Sum of weights must equal 1 for each period
    constraints = []
    for i in range(num_periods):
        start_idx = i * num_assets
        end_idx = (i + 1) * num_assets
        constraints.append({
            'type': 'eq',
            'fun': lambda x, s=start_idx, e=end_idx: np.sum(x[s:e]) - 1.0
        })
    
    # Asset allocation bounds (0-100%)
    bounds = [(0.0, 1.0) for _ in range(num_assets * num_periods)]
    
    # Define the objective function to minimize shortfall risk
    def objective(weights):
        # Reshape weights into periods x assets
        reshaped_weights = np.reshape(weights, (num_periods, num_assets))
        
        # Create a glidepath by interpolating between the period weights
        glidepath = []
        for year in range(years_to_simulate + 1):
            period_idx = min(int(year / period_length), num_periods - 1)
            glidepath.append(reshaped_weights[period_idx])
        
        # Set the glidepath in the plan
        plan.glidepath = np.array(glidepath)
        
        # Run monte carlo simulation with this glidepath
        sim_results = run_monte_carlo_simulation(
            client, 
            plan, 
            market_assumptions, 
            num_simulations=num_simulations, 
            max_age=max_age
        )
        
        # Calculate shortfall risk
        risk_metrics = calculate_shortfall_risk(sim_results)
        
        # Objective: minimize shortfall probability
        return risk_metrics['shortfall_probability']
    
    # Initial guess: linear transition from current allocation to conservative allocation
    conservative_allocation = np.array([0.2, 0.4, 0.2, 0.1, 0.05, 0.05])  # Conservative allocation
    
    initial_guess = []
    for i in range(num_periods):
        t = i / (num_periods - 1)  # Interpolation parameter
        period_allocation = (1 - t) * initial_allocation + t * conservative_allocation
        initial_guess.extend(period_allocation)
    
    # Run the optimization
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100, 'disp': True}
    )
    
    # Extract the optimized glidepath
    optimized_weights = np.reshape(result.x, (num_periods, num_assets))
    
    # Interpolate to get a full glidepath
    glidepath = []
    for year in range(years_to_simulate + 1):
        period_idx = min(int(year / period_length), num_periods - 1)
        glidepath.append(optimized_weights[period_idx])
    
    # Run a final simulation with the optimized glidepath
    plan.glidepath = np.array(glidepath)
    final_sim_results = run_monte_carlo_simulation(
        client, 
        plan, 
        market_assumptions, 
        num_simulations=num_simulations * 2,  # More simulations for final result
        max_age=max_age
    )
    
    # Calculate final risk metrics
    final_risk_metrics = calculate_shortfall_risk(final_sim_results)
    
    return {
        'glidepath': np.array(glidepath),
        'asset_classes': asset_classes,
        'success_probability': 1 - final_risk_metrics['shortfall_probability'],
        'shortfall_risk': final_risk_metrics,
        'simulation_results': final_sim_results,
        'ages': list(range(current_age, max_age + 1))
    }

def mean_variance_optimize(expected_returns, cov_matrix, risk_aversion=3.0):
    """
    Perform mean-variance optimization for sub-asset classes.
    
    Parameters:
    -----------
    expected_returns : array-like
        Expected returns for each sub-asset class
    cov_matrix : array-like
        Covariance matrix of returns
    risk_aversion : float
        Risk aversion parameter (higher = more conservative)
        
    Returns:
    --------
    array
        Optimal weights for each sub-asset class
    """
    num_assets = len(expected_returns)
    
    # Define the objective function to maximize utility (return - risk_aversion * variance)
    def objective(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        utility = portfolio_return - (risk_aversion * portfolio_variance)
        return -utility  # Minimize negative utility
    
    # Constraints: sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    # Bounds: weights between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    
    # Initial guess: equal weight
    initial_guess = np.ones(num_assets) / num_assets
    
    # Run the optimization
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def plot_glidepath(glidepath_result):
    """
    Generate a visualization of the optimized glidepath.
    
    Parameters:
    -----------
    glidepath_result : dict
        Results from the glidepath optimization
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure with the glidepath visualization
    """
    glidepath = glidepath_result['glidepath']
    asset_classes = glidepath_result['asset_classes']
    ages = glidepath_result['ages']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked area chart
    bottom = np.zeros(len(glidepath))
    for i, asset in enumerate(asset_classes):
        values = [allocation[i] for allocation in glidepath]
        ax.fill_between(ages, bottom, bottom + values, label=asset)
        bottom += values
    
    # Add labels and title
    ax.set_xlabel('Age')
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Optimized Glidepath Allocation')
    ax.legend(loc='upper right')
    
    # Add success probability text
    success_text = f"Probability of Success: {glidepath_result['success_probability']:.1%}"
    ax.text(0.05, 0.95, success_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(ages[0], ages[-1])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    return fig

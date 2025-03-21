import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.plan import CashFlow, Goal
from utils.market_assumptions import get_asset_returns_covariance

def run_monte_carlo_simulation(client, plan, market_assumptions, num_simulations=1000, max_age=95):
    """
    Run a Monte Carlo simulation for a client's financial plan.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to run the simulation
    plan : Plan object
        Contains goals, cash flows, and initial asset allocation
    market_assumptions : dict
        Market assumptions including expected returns, volatility, and correlations
    num_simulations : int
        Number of simulations to run
    max_age : int
        Maximum age to simulate to
        
    Returns:
    --------
    dict
        Results of the simulation including success probability and portfolio paths
    """
    # Calculate client's current age from date of birth
    current_year = pd.Timestamp.now().year
    birth_year = pd.Timestamp(client.date_of_birth).year
    current_age = current_year - birth_year
    
    # Duration of simulation
    years_to_simulate = max_age - current_age
    
    # Get initial portfolio value and asset allocation
    initial_portfolio = plan.initial_portfolio
    asset_allocation = plan.asset_allocation
    
    # Extract returns and volatilities from market assumptions
    asset_returns, asset_vols, correlations = get_asset_returns_covariance(market_assumptions, 'long_term')
    
    # Prepare array for simulation results
    portfolio_paths = np.zeros((num_simulations, years_to_simulate + 1))
    portfolio_paths[:, 0] = initial_portfolio  # Set initial portfolio value
    
    # Simulate portfolio returns with mean reversion
    mean_reversion_speed = 0.15  # Mean reversion parameter
    long_term_mean = np.dot(asset_allocation, asset_returns)
    
    # Extract expected returns from dictionary
    short_term_returns = np.array([market_assumptions['short_term']['expected_returns'][asset] 
                                 for asset in market_assumptions['asset_classes']])
    current_return = np.dot(asset_allocation, short_term_returns)
    
    # Create covariance matrix from volatilities and correlations
    cov_matrix = np.zeros((len(asset_vols), len(asset_vols)))
    for i in range(len(asset_vols)):
        for j in range(len(asset_vols)):
            if i == j:
                cov_matrix[i, j] = asset_vols[i]**2
            else:
                cov_matrix[i, j] = asset_vols[i] * asset_vols[j] * correlations[i, j]
    
    # Run simulations
    for sim in range(num_simulations):
        current_portfolio = initial_portfolio
        
        for year in range(1, years_to_simulate + 1):
            # Apply mean reversion to expected returns
            current_return = current_return + mean_reversion_speed * (long_term_mean - current_return)
            
            # Get this year's cash flows (contributions and withdrawals)
            cash_flows = sum(cf.amount for cf in plan.cash_flows if cf.start_age <= current_age + year <= cf.end_age)
            
            # Get this year's goal withdrawals
            goal_withdrawals = sum(goal.amount for goal in plan.goals if goal.age == current_age + year)
            
            # Generate random return for this year
            yearly_return = np.random.normal(current_return, np.sqrt(np.dot(asset_allocation, np.dot(cov_matrix, asset_allocation))))
            
            # Update portfolio value
            current_portfolio = current_portfolio * (1 + yearly_return) + cash_flows - goal_withdrawals
            current_portfolio = max(0, current_portfolio)  # Portfolio can't go negative
            
            portfolio_paths[sim, year] = current_portfolio
    
    # Calculate success probability
    # Success defined as having assets remaining at max age
    success_count = np.sum(portfolio_paths[:, -1] > 0)
    success_probability = success_count / num_simulations
    
    # Calculate percentiles for confidence bands
    percentiles = {
        'lower': np.percentile(portfolio_paths, 10, axis=0),
        'median': np.percentile(portfolio_paths, 50, axis=0),
        'upper': np.percentile(portfolio_paths, 90, axis=0)
    }
    
    return {
        'success_probability': success_probability,
        'portfolio_paths': portfolio_paths,
        'percentiles': percentiles,
        'years': list(range(years_to_simulate + 1)),
        'ages': list(range(current_age, max_age + 1))
    }

def plot_monte_carlo_results(results, client_name):
    """
    Create visualizations for Monte Carlo simulation results.
    
    Parameters:
    -----------
    results : dict
        Results from the Monte Carlo simulation
    client_name : str
        Name of the client for titling
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure with the simulation visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot sample of portfolio paths
    sample_paths = 100
    for i in range(min(sample_paths, results['portfolio_paths'].shape[0])):
        ax.plot(results['years'], results['portfolio_paths'][i], 'k-', alpha=0.05)
    
    # Plot percentile bands
    ax.plot(results['years'], results['percentiles']['median'], 'b-', linewidth=2, label='Median')
    ax.plot(results['years'], results['percentiles']['upper'], 'g-', linewidth=2, label='90th Percentile')
    ax.plot(results['years'], results['percentiles']['lower'], 'r-', linewidth=2, label='10th Percentile')
    
    # Add labels and title
    ax.set_xlabel('Years')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'Monte Carlo Simulation: {client_name}')
    ax.legend()
    
    # Add success probability text
    success_text = f"Probability of Success: {results['success_probability']:.1%}"
    ax.text(0.05, 0.95, success_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

def calculate_shortfall_risk(results):
    """
    Calculate shortfall risk metrics from simulation results.
    
    Parameters:
    -----------
    results : dict
        Results from the Monte Carlo simulation
        
    Returns:
    --------
    dict
        Shortfall risk metrics
    """
    # Calculate probability of shortfall (portfolio value reaching zero)
    shortfall_prob = 1 - results['success_probability']
    
    # Calculate conditional value at risk (CVaR) - average of worst outcomes
    final_values = results['portfolio_paths'][:, -1]
    sorted_values = np.sort(final_values)
    
    # Calculate the 5% worst outcomes
    tail_size = int(len(sorted_values) * 0.05)
    tail_values = sorted_values[:tail_size]
    cvar = np.mean(tail_values)
    
    # Calculate maximum drawdown
    max_drawdowns = []
    for path in results['portfolio_paths']:
        cummax = np.maximum.accumulate(path)
        drawdown = (path - cummax) / cummax
        max_drawdowns.append(np.min(drawdown))
    
    avg_max_drawdown = np.mean(max_drawdowns)
    
    return {
        'shortfall_probability': shortfall_prob,
        'conditional_value_at_risk': cvar,
        'average_max_drawdown': avg_max_drawdown
    }

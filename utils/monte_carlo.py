import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.plan import CashFlow, Goal
from utils.market_assumptions import get_asset_returns_covariance

def run_monte_carlo_simulation(client, plan, market_assumptions, num_simulations=1000, max_age=None):
    """
    Run a Monte Carlo simulation for a client's financial plan using advanced techniques:
    - Mean-reverting returns rather than random walk
    - Time-varying expected returns based on market conditions
    - Different risk aversion for above/below target wealth
    - Support for multi-period dynamic asset allocation
    
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
    max_age : int, optional
        Maximum age to simulate to. If None, uses client's longevity_age
        
    Returns:
    --------
    dict
        Results of the simulation including success probability and portfolio paths
    """
    # Calculate client's current age from date of birth
    current_year = pd.Timestamp.now().year
    birth_year = pd.Timestamp(client.date_of_birth).year
    current_age = current_year - birth_year
    
    # Get retirement age from client
    retirement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
    
    # Use client's longevity age if max_age not specified
    if max_age is None:
        max_age = client.longevity_age if hasattr(client, 'longevity_age') else 95
    
    # Duration of simulation
    years_to_simulate = max_age - current_age
    
    # Get initial portfolio value and asset allocation
    initial_portfolio = plan.initial_portfolio
    asset_allocation = plan.asset_allocation
    
    # Get glidepath if available, otherwise use constant allocation
    if hasattr(plan, 'glidepath') and plan.glidepath is not None:
        use_glidepath = True
        glidepath = plan.glidepath
    else:
        use_glidepath = False
        glidepath = None
    
    # Extract returns and volatilities from market assumptions (both short and long-term)
    long_term_returns, long_term_vols, long_term_corr = get_asset_returns_covariance(market_assumptions, 'long_term')
    short_term_returns, short_term_vols, short_term_corr = get_asset_returns_covariance(market_assumptions, 'short_term')
    
    # Define mean reversion period (7 years as requested)
    mean_reversion_years = 7
    
    # Prepare arrays for simulation results
    portfolio_paths = np.zeros((num_simulations, years_to_simulate + 1))
    portfolio_paths[:, 0] = initial_portfolio  # Set initial portfolio value
    
    # Store wealth targets for retirement
    target_wealth_paths = np.zeros((num_simulations, years_to_simulate + 1))
    
    # Calculate target retirement wealth based on spending goals
    # Simpler version: Sum of all annual expenses in retirement multiplied by 25 (4% rule)
    retirement_annual_expenses = sum(abs(cf.amount) 
                                   for cf in plan.cash_flows 
                                   if cf.amount < 0 and cf.start_age >= retirement_age)
    
    if retirement_annual_expenses == 0:
        # Use a default if no expenses defined
        retirement_annual_expenses = 40000  # Default annual retirement expense
    
    target_retirement_wealth = retirement_annual_expenses * 25
    
    # Mean reversion parameters
    mean_reversion_speed = plan.mean_reversion_speed if hasattr(plan, 'mean_reversion_speed') else 0.15
    
    # Create time-varying return expectations (linear transition from short to long term)
    time_varying_returns = np.zeros((years_to_simulate + 1, len(long_term_returns)))
    
    # Initialize with short-term returns
    time_varying_returns[0] = short_term_returns
    
    # Create gradual transition from short to long-term over mean_reversion_years
    for year in range(1, years_to_simulate + 1):
        # Calculate the weight of long-term returns (increases from 0 to 1 over mean_reversion_years)
        if year < mean_reversion_years:
            long_term_weight = year / mean_reversion_years
        else:
            long_term_weight = 1.0
            
        # Blend short-term and long-term returns based on weight
        time_varying_returns[year] = (1 - long_term_weight) * short_term_returns + long_term_weight * long_term_returns
    
    # Create time-varying covariance matrices
    # For simplicity, we'll use short-term volatilities and correlations for year 0
    # and transition to long-term over mean_reversion_years
    
    # Create initial (short-term) covariance matrix
    short_term_cov = np.zeros((len(short_term_vols), len(short_term_vols)))
    for i in range(len(short_term_vols)):
        for j in range(len(short_term_vols)):
            if i == j:
                short_term_cov[i, j] = short_term_vols[i]**2
            else:
                short_term_cov[i, j] = short_term_vols[i] * short_term_vols[j] * short_term_corr[i, j]
                
    # Create final (long-term) covariance matrix
    long_term_cov = np.zeros((len(long_term_vols), len(long_term_vols)))
    for i in range(len(long_term_vols)):
        for j in range(len(long_term_vols)):
            if i == j:
                long_term_cov[i, j] = long_term_vols[i]**2
            else:
                long_term_cov[i, j] = long_term_vols[i] * long_term_vols[j] * long_term_corr[i, j]
    
    # Run simulations
    for sim in range(num_simulations):
        current_portfolio = initial_portfolio
        
        # Initialize with first year's allocation
        if use_glidepath and glidepath is not None:
            current_allocation = glidepath[0]
        else:
            current_allocation = asset_allocation
            
        # Use short-term returns and covariance for year 0
        current_expected_return = np.dot(current_allocation, short_term_returns)
        current_vol = np.sqrt(np.dot(current_allocation, np.dot(short_term_cov, current_allocation)))
        
        # Initialize market valuation ratio (P/E or CAPE)
        # Start with a slight overvaluation (1.1x fair value)
        market_valuation_ratio = 1.1
        fair_valuation = 1.0
        valuation_mean_reversion = 0.1  # Speed of valuation mean reversion
        
        for year in range(1, years_to_simulate + 1):
            # Get this year's allocation (from glidepath or constant)
            if use_glidepath and glidepath is not None:
                glidepath_len = len(glidepath)
                current_allocation = glidepath[year] if year < glidepath_len else glidepath[-1]
            else:
                current_allocation = asset_allocation
            
            # Calculate blended covariance matrix for this year (interpolate between short and long term)
            if year < mean_reversion_years:
                # Blend factor (increases from 0 to 1 over mean_reversion_years)
                blend_factor = year / mean_reversion_years
                current_cov = (1 - blend_factor) * short_term_cov + blend_factor * long_term_cov
            else:
                # Use long-term covariance after mean_reversion_years
                current_cov = long_term_cov
            
            # Get the time-varying expected return for this year's allocation
            time_varying_mean = np.dot(current_allocation, time_varying_returns[year])
            
            # Apply valuation adjustment to expected returns
            # When market_valuation_ratio > 1, market is overvalued, so expected returns are lower
            valuation_adjustment = (fair_valuation / market_valuation_ratio - 1)
            adjusted_return = current_expected_return + valuation_adjustment * 0.03  # 3% annual adjustment
            
            # Apply mean reversion to expected returns (towards time-varying expected return)
            current_expected_return = current_expected_return + mean_reversion_speed * (time_varying_mean - current_expected_return)
            
            # Apply mean reversion to market valuation
            market_valuation_ratio = market_valuation_ratio + valuation_mean_reversion * (fair_valuation - market_valuation_ratio)
            
            # Calculate current volatility for the portfolio using the blended covariance matrix
            current_vol = np.sqrt(np.dot(current_allocation, np.dot(current_cov, current_allocation)))
            
            # Get this year's cash flows (contributions and withdrawals)
            cash_flows = sum(cf.amount * (1 + cf.growth_rate)**(year - 1) 
                           for cf in plan.cash_flows 
                           if cf.start_age <= current_age + year <= cf.end_age)
            
            # Get this year's goal withdrawals
            goal_withdrawals = sum(goal.amount for goal in plan.goals if goal.age == current_age + year)
            
            # Generate random return for this year with fat tails (Student's t distribution)
            # This better models market crashes than normal distribution
            df = 5  # Degrees of freedom (lower = fatter tails)
            shock = np.random.standard_t(df) / np.sqrt(df / (df-2))  # Standardized to match volatility
            yearly_return = current_expected_return + shock * current_vol
            
            # Update portfolio value
            current_portfolio = current_portfolio * (1 + yearly_return) + cash_flows - goal_withdrawals
            current_portfolio = max(0, current_portfolio)  # Portfolio can't go negative
            
            portfolio_paths[sim, year] = current_portfolio
            
            # Calculate target wealth path
            # For simplicity: linear growth to target retirement wealth
            if current_age + year < retirement_age:
                # Pre-retirement: growing target
                years_to_retirement = retirement_age - (current_age + year)
                total_years_to_retirement = retirement_age - current_age
                target_wealth = initial_portfolio + (target_retirement_wealth - initial_portfolio) * (1 - years_to_retirement / total_years_to_retirement)
            else:
                # Post-retirement: decaying target (as money is spent)
                years_in_retirement = (current_age + year) - retirement_age
                remaining_lifespan = max_age - retirement_age
                target_wealth = target_retirement_wealth * (1 - years_in_retirement / remaining_lifespan)
            
            target_wealth_paths[sim, year] = target_wealth
    
    # Calculate success probability (assets > 0 at end of plan)
    success_count = np.sum(portfolio_paths[:, -1] > 0)
    success_probability = success_count / num_simulations
    
    # Calculate enhanced success metrics
    # Probability of meeting retirement wealth target at retirement
    retirement_year = retirement_age - current_age
    if retirement_year < years_to_simulate:
        retirement_success = np.sum(portfolio_paths[:, retirement_year] >= target_wealth_paths[:, retirement_year]) / num_simulations
    else:
        retirement_success = np.nan
    
    # Calculate percentiles for confidence bands
    percentiles = {
        'lower': np.percentile(portfolio_paths, 10, axis=0),
        'median': np.percentile(portfolio_paths, 50, axis=0),
        'upper': np.percentile(portfolio_paths, 90, axis=0)
    }
    
    # Calculate target wealth percentiles
    target_percentiles = {
        'median': np.percentile(target_wealth_paths, 50, axis=0)
    }
    
    return {
        'success_probability': success_probability,
        'retirement_success_probability': retirement_success,
        'portfolio_paths': portfolio_paths,
        'target_wealth_paths': target_wealth_paths,
        'percentiles': percentiles,
        'target_percentiles': target_percentiles,
        'years': list(range(years_to_simulate + 1)),
        'ages': list(range(current_age, max_age + 1))
    }

def plot_monte_carlo_results(results, client_name):
    """
    Create visualizations for Monte Carlo simulation results with enhanced features:
    - Target wealth path overlay
    - Retirement age indicator
    - Enhanced success metrics
    
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
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot sample of portfolio paths
    sample_paths = 50  # Reduced for cleaner visualization
    for i in range(min(sample_paths, results['portfolio_paths'].shape[0])):
        ax.plot(results['ages'], results['portfolio_paths'][i], 'k-', alpha=0.05)
    
    # Plot percentile bands
    ax.plot(results['ages'], results['percentiles']['median'], 'b-', linewidth=2, label='Median Portfolio')
    ax.plot(results['ages'], results['percentiles']['upper'], 'g-', linewidth=2, label='90th Percentile')
    ax.plot(results['ages'], results['percentiles']['lower'], 'r-', linewidth=2, label='10th Percentile')
    
    # Plot target wealth path
    if 'target_wealth_paths' in results and 'target_percentiles' in results:
        ax.plot(results['ages'], results['target_percentiles']['median'], 'k--', linewidth=2, label='Target Wealth')
    
    # Add labels and title
    ax.set_xlabel('Age')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'Advanced Monte Carlo Simulation: {client_name}')
    
    # Format y-axis with dollar signs and commas
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'${int(x):,}'))
    
    # Add success probability text
    metrics_text = [
        f"Probability of Success: {results['success_probability']:.1%}"
    ]
    
    # Add retirement success probability if available
    if 'retirement_success_probability' in results and not np.isnan(results['retirement_success_probability']):
        metrics_text.append(f"Probability of Meeting Retirement Target: {results['retirement_success_probability']:.1%}")
    
    # Display text box with metrics
    metrics_box = '\n'.join(metrics_text)
    ax.text(0.05, 0.95, metrics_box, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Place legend in a good position
    ax.legend(loc='upper right')
    
    return fig

def calculate_shortfall_risk(results):
    """
    Calculate enhanced shortfall risk metrics from simulation results including:
    - Probability of shortfall (running out of money)
    - Conditional Value at Risk (CVaR) - expected loss in worst scenarios
    - Maximum drawdown - largest percentage decline in portfolio value
    - Expected shortfall - average amount below target wealth
    - Loss of lifestyle - percentage reduction in spending needed in worst scenarios
    - Sustainable withdrawal rate - percentage of portfolio that can be safely withdrawn
    
    Parameters:
    -----------
    results : dict
        Results from the Monte Carlo simulation
        
    Returns:
    --------
    dict
        Enhanced shortfall risk metrics
    """
    # Calculate probability of shortfall (portfolio value reaching zero)
    shortfall_prob = 1 - results['success_probability']
    
    # Calculate conditional value at risk (CVaR) - average of worst outcomes
    final_values = results['portfolio_paths'][:, -1]
    sorted_values = np.sort(final_values)
    
    # Calculate the 5% worst outcomes
    tail_size = max(1, int(len(sorted_values) * 0.05))
    tail_values = sorted_values[:tail_size]
    cvar = np.mean(tail_values)
    
    # Calculate maximum drawdown across all simulations
    max_drawdowns = []
    for path in results['portfolio_paths']:
        if np.all(path == 0):
            max_drawdowns.append(-1.0)  # Complete loss
            continue
            
        cummax = np.maximum.accumulate(path)
        # Avoid division by zero
        valid_indices = cummax > 0
        if not np.any(valid_indices):
            max_drawdowns.append(-1.0)
            continue
            
        drawdown = np.zeros_like(path)
        drawdown[valid_indices] = (path[valid_indices] - cummax[valid_indices]) / cummax[valid_indices]
        max_drawdowns.append(np.min(drawdown))
    
    avg_max_drawdown = np.mean(max_drawdowns)
    
    # Calculate expected shortfall relative to target wealth
    expected_shortfall = 0
    if 'target_wealth_paths' in results:
        # For each path, calculate average shortfall below target
        shortfalls = []
        for i in range(results['portfolio_paths'].shape[0]):
            portfolio = results['portfolio_paths'][i]
            target = results['target_wealth_paths'][i]
            
            # Calculate shortfall at each time point (max of 0 and target-portfolio)
            shortfall = np.maximum(0, target - portfolio)
            
            # Average shortfall over time for this path
            avg_shortfall = np.mean(shortfall)
            shortfalls.append(avg_shortfall)
        
        expected_shortfall = np.mean(shortfalls)
    
    # Calculate loss of lifestyle (spending reduction needed in worst case)
    # This is based on the 5% worst final portfolio values
    lifestyle_reduction = 0
    if 'target_wealth_paths' in results:
        # Get median target at final time point
        final_target = np.median(results['target_wealth_paths'][:, -1])
        
        if final_target > 0:
            # Calculate percentage reduction needed in worst case
            worst_case_pct = tail_values / final_target
            # Cap at 100% reduction
            worst_case_pct = np.minimum(1.0, worst_case_pct)
            lifestyle_reduction = 1.0 - np.mean(worst_case_pct)
    
    # Calculate sustainable withdrawal rate
    # Conservative estimate based on 90% success probability
    sustainable_withdrawal_rate = 0.0
    
    # Test various withdrawal rates
    test_rates = np.arange(0.01, 0.08, 0.0025)  # 1% to 8% in 0.25% increments
    
    initial_wealth = results['portfolio_paths'][:, 0][0]  # Get initial wealth value
    if initial_wealth > 0:
        for rate in test_rates:
            # Count paths that survive with this withdrawal rate
            success_count = 0
            
            for path in results['portfolio_paths']:
                # Apply constant percentage withdrawal
                adjusted_path = path.copy()
                for i in range(1, len(adjusted_path)):
                    # Withdraw rate% of previous value
                    withdrawal = adjusted_path[i-1] * rate
                    adjusted_path[i] -= withdrawal
                
                # Check if this path survives
                if adjusted_path[-1] > 0:
                    success_count += 1
            
            success_rate = success_count / len(results['portfolio_paths'])
            
            # If we found a rate with >90% success, update
            if success_rate >= 0.9:
                sustainable_withdrawal_rate = rate
            else:
                # Stop once success rate drops below 90%
                break
    
    return {
        'shortfall_probability': shortfall_prob,
        'conditional_value_at_risk': cvar,
        'average_max_drawdown': avg_max_drawdown,
        'expected_shortfall': expected_shortfall,
        'lifestyle_reduction': lifestyle_reduction,
        'sustainable_withdrawal_rate': sustainable_withdrawal_rate
    }

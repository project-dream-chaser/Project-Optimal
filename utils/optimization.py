import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils.monte_carlo import run_monte_carlo_simulation, calculate_shortfall_risk

def optimize_glidepath(client, plan, market_assumptions, num_simulations=500, max_age=95, num_periods=10):
    """
    Optimize the glidepath to minimize shortfall risk using the open-loop with recourse methodology.
    
    This advanced implementation:
    1. Uses dynamic programming principles to determine optimal allocations
    2. Accounts for different risk aversion above/below target wealth
    3. Considers time-varying expected returns based on market conditions
    4. Generates a complete glidepath that automatically adjusts to changing conditions
    
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
    
    # Get retirement age
    retirement_age = client.restylement_age if hasattr(client, 'restylement_age') else 65
    
    # Asset classes
    asset_classes = market_assumptions['asset_classes']
    num_assets = len(asset_classes)
    
    # Initial asset allocation
    initial_allocation = plan.asset_allocation
    
    # Calculate years to simulate
    years_to_simulate = max_age - current_age
    period_length = max(1, years_to_simulate // num_periods)
    
    # Calculate target retirement wealth based on spending goals
    # Get annual expenses in retirement from cash flows
    retirement_annual_expenses = sum(abs(cf.amount) 
                                   for cf in plan.cash_flows 
                                   if cf.amount < 0 and cf.start_age >= retirement_age)
    
    if retirement_annual_expenses == 0:
        # Default if no expenses defined
        retirement_annual_expenses = 40000  # Default annual retirement expense
    
    # Simple target: 25x annual expenses (4% rule)
    target_retirement_wealth = retirement_annual_expenses * 25
    
    # Get risk aversion parameters (different for above/below target)
    # Default: more risk averse below target (loss aversion)
    above_target_risk_aversion = plan.risk_aversion if hasattr(plan, 'risk_aversion') else 3.0
    below_target_risk_aversion = above_target_risk_aversion * 2.5  # Higher risk aversion below target
    
    # Client-specific stock constraint if available
    max_stock_pct = client.max_stock_pct if hasattr(client, 'max_stock_pct') else None
    
    # Account for client-specific allocation constraints
    allocation_bounds = []
    for asset_class in asset_classes:
        if asset_class == 'Global Equity' and max_stock_pct is not None:
            # Apply client's maximum stock constraint
            max_bound = max_stock_pct / 100.0
        else:
            max_bound = 1.0
            
        # Start with standard bounds
        allocation_bounds.append((0.0, max_bound))
    
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
    
    # Create full bounds array for all periods
    bounds = allocation_bounds * num_periods
    
    # Define the objective function to minimize shortfall risk
    # Use a composite objective that:
    # 1. Minimizes shortfall probability
    # 2. Maximizes expected final wealth
    # 3. Minimizes expected shortfall
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
        
        # Configure the plan to use time-varying returns with 7-year mean reversion
        # This will make the optimization use short-term CMA initially that blends to 
        # long-term CMA over 7 years
        plan.glidepath_info = {
            'description': 'This glidepath uses a 7-year mean reversion from short-term to long-term capital market assumptions',
            'short_term_source': 'Current market conditions (short-term capital market assumptions)',
            'long_term_source': 'Equilibrium expectations (long-term capital market assumptions)',
            'mean_reversion_years': 7,
            'time_varying_returns': True
        }
        
        # Run monte carlo simulation with this glidepath and time-varying returns
        sim_results = run_monte_carlo_simulation(
            client, 
            plan, 
            market_assumptions, 
            num_simulations=num_simulations, 
            max_age=max_age
        )
        
        # Calculate comprehensive risk metrics
        risk_metrics = calculate_shortfall_risk(sim_results)
        
        # Calculate expected final wealth
        expected_final_wealth = np.mean(sim_results['portfolio_paths'][:, -1])
        
        # Normalize expected wealth for balanced weighting in the objective
        # Use target retirement wealth as a reference point
        normalized_wealth = min(expected_final_wealth / target_retirement_wealth, 2.0)
        
        # Determine if we're within 7 years of restylement
        years_to_retirement = retirement_age - current_age
        
        # Get target returns from the plan
        pre_restylement_return_target = plan.pre_restylement_return / 100.0 if hasattr(plan, 'pre_restylement_return') else 0.07
        post_restylement_return_target = plan.post_restylement_return / 100.0 if hasattr(plan, 'post_restylement_return') else 0.05
        
        # Calculate the actual returns for pre and post restylement periods
        pre_restylement_actual_return = 0.0
        post_restylement_actual_return = 0.0
        pre_restylement_count = 0
        post_restylement_count = 0
        
        # Find the actual returns in the simulations
        for t, age in enumerate(range(current_age, max_age + 1)):
            if t < len(sim_results['avg_returns']):
                # Pre-restylement period
                if age < retirement_age:
                    pre_restylement_actual_return += sim_results['avg_returns'][t]
                    pre_restylement_count += 1
                # Post-restylement period
                else:
                    post_restylement_actual_return += sim_results['avg_returns'][t]
                    post_restylement_count += 1
        
        # Calculate average returns
        if pre_restylement_count > 0:
            pre_restylement_actual_return /= pre_restylement_count
        if post_restylement_count > 0:
            post_restylement_actual_return /= post_restylement_count
        
        # Calculate return deviations (squared to penalize both too high and too low)
        pre_restylement_return_deviation = (pre_restylement_actual_return - pre_restylement_return_target) ** 2
        post_restylement_return_deviation = (post_restylement_actual_return - post_restylement_return_target) ** 2
        
        # Different optimization hierarchy based on years to restylement
        if years_to_retirement <= 7:
            # Within 7 years of restylement:
            # 1. Prioritize shortfall risk (increasingly as we get closer)
            # 2. Then post-restylement return target
            
            # Gradually increase weight on shortfall risk in last 7 years before restylement
            # Weight increases from 1x to 3x as we approach restylement
            shortfall_weight_multiplier = 1.0 + (2.0 * (1.0 - years_to_retirement / 7.0))
            
            # Components with different risk aversion and weighting
            shortfall_component = risk_metrics['shortfall_probability'] * below_target_risk_aversion * shortfall_weight_multiplier
            lifestyle_component = risk_metrics['lifestyle_reduction'] * below_target_risk_aversion * shortfall_weight_multiplier
            
            # Post-restylement return component (secondary priority for near-restylement)
            # Deviation from target return (both too high and too low are penalized)
            post_return_component = post_restylement_return_deviation * 20.0  # Weight to balance with other components
            
            # Add penalty for asset classes where short-term returns differ significantly from long-term
            # The penalty is proportional to:
            # 1. How much the asset's allocation deviates from its "natural" allocation
            # 2. How much the short-term returns differ from long-term returns
            
            # Start with zero penalty
            market_condition_penalty = 0.0
            
            # Get first period allocation (we're penalizing early allocations that don't account for market conditions)
            first_period_allocation = reshaped_weights[0]
            
            # Loop through all asset classes
            for i, asset in enumerate(asset_classes):
                # Get the asset's allocation in the first period
                asset_allocation = first_period_allocation[i]
                
                # Get the adjustment factor for this asset from our pre-computed values
                adjustment_factor = return_adjustments.get(asset, 1.0)
                
                # Calculate a penalty based on the difference from 1.0 (neutral)
                # If adjustment_factor < 1.0, penalize high allocations
                # If adjustment_factor > 1.0, penalize low allocations
                if adjustment_factor < 0.8:  # Significantly worse short-term returns
                    # Penalize high allocations to this asset class
                    # Penalty increases as allocation increases and as adjustment factor decreases
                    penalty = asset_allocation * (1.0 - adjustment_factor) * 25.0
                    market_condition_penalty += penalty
                elif adjustment_factor > 1.2:  # Significantly better short-term returns
                    # Penalize low allocations to this asset class
                    # Penalty increases as allocation decreases and as adjustment factor increases
                    penalty = (1.0 - asset_allocation) * (adjustment_factor - 1.0) * 25.0
                    market_condition_penalty += penalty
            
            # Add the market condition penalty to the objective
            return shortfall_component + lifestyle_component + post_return_component + market_condition_penalty
        else:
            # More than 7 years from restylement:
            # 1. Prioritize pre-restylement return target
            # 2. Consider wealth component as secondary
            
            # Pre-restylement return component (primary priority for far-from-restylement)
            # Deviation from target return (both too high and too low are penalized)
            pre_return_component = pre_restylement_return_deviation * 30.0  # Higher weight to make this primary
            
            # Wealth component still considered but secondary
            wealth_component = -normalized_wealth * (1/above_target_risk_aversion)  # Negative because we're maximizing
            
            # Basic shortfall risk still included but with lower weight
            basic_shortfall = risk_metrics['shortfall_probability'] * below_target_risk_aversion * 0.5  # Half weight
            
            # Add penalty for asset classes where short-term returns differ significantly from long-term
            # The penalty is proportional to:
            # 1. How much the asset's allocation deviates from its "natural" allocation
            # 2. How much the short-term returns differ from long-term returns
            
            # Start with zero penalty
            market_condition_penalty = 0.0
            
            # Get first period allocation (we're penalizing early allocations that don't account for market conditions)
            first_period_allocation = reshaped_weights[0]
            
            # Loop through all asset classes
            for i, asset in enumerate(asset_classes):
                # Get the asset's allocation in the first period
                asset_allocation = first_period_allocation[i]
                
                # Get the adjustment factor for this asset from our pre-computed values
                adjustment_factor = return_adjustments.get(asset, 1.0)
                
                # Calculate a penalty based on the difference from 1.0 (neutral)
                # If adjustment_factor < 1.0, penalize high allocations
                # If adjustment_factor > 1.0, penalize low allocations
                if adjustment_factor < 0.8:  # Significantly worse short-term returns
                    # Penalize high allocations to this asset class
                    # Penalty increases as allocation increases and as adjustment factor decreases
                    penalty = asset_allocation * (1.0 - adjustment_factor) * 35.0
                    market_condition_penalty += penalty
                elif adjustment_factor > 1.2:  # Significantly better short-term returns
                    # Penalize low allocations to this asset class
                    # Penalty increases as allocation decreases and as adjustment factor increases
                    penalty = (1.0 - asset_allocation) * (adjustment_factor - 1.0) * 35.0
                    market_condition_penalty += penalty
            
            # Add the market condition penalty to the objective
            return pre_return_component + basic_shortfall + wealth_component + market_condition_penalty
    
    # Generate smarter initial guess based on both age and short-term vs. long-term return expectations
    # Extract short-term and long-term returns for all asset classes
    from utils.market_assumptions import get_asset_returns_covariance
    
    short_term_returns, _, _ = get_asset_returns_covariance(market_assumptions, 'short_term')
    long_term_returns, _, _ = get_asset_returns_covariance(market_assumptions, 'long_term')
    
    # Create adjustment factors for each asset class based on short-term vs. long-term return differentials
    return_adjustments = {}
    
    # Loop through all asset classes to compute return adjustments
    for i, asset_class in enumerate(asset_classes):
        short_term_return = short_term_returns[i]
        long_term_return = long_term_returns[i]
        
        # Calculate adjustment factor (ratio of short-term to long-term return)
        # Values < 1.0 mean short-term returns are lower than long-term
        if long_term_return > 0:
            adjustment = min(1.5, max(0.5, short_term_return / max(0.01, long_term_return)))
        else:
            # If long-term return is zero or negative, use a neutral adjustment
            adjustment = 1.0
            
        return_adjustments[asset_class] = adjustment
        print(f"{asset_class} - Short-term: {short_term_return:.2%}, Long-term: {long_term_return:.2%}, Adjustment: {adjustment:.2f}")
    
    # Determine client risk profile based on max_stock_pct
    is_aggressive = max_stock_pct > 70 if max_stock_pct is not None else False
    
    initial_guess = []
    for i in range(num_periods):
        # Calculate age at this period
        period_age = current_age + (i * period_length)
        
        # Base stock percentage on age
        if is_aggressive:
            base_stock_pct = max(0.15, min(0.90, (110 - period_age) / 100))
        else:
            base_stock_pct = max(0.10, min(0.80, (100 - period_age) / 100))
        
        # Start with standard asset class allocation model (simplified to stocks/bonds/alternatives)
        stock_pct = base_stock_pct
        
        # Get the equity adjustment factor (for consistency with older code)
        equity_adjustment = return_adjustments.get('Global Equity', 1.0)
                
        # Adjust if client has a specific max stock percentage
        if max_stock_pct is not None:
            stock_pct = min(stock_pct, max_stock_pct / 100.0)
            
        # Simplified allocation percentages
        if period_age < retirement_age:
            # Pre-retirement: more alternatives
            bonds_pct = (1 - stock_pct) * 0.6
            alts_pct = (1 - stock_pct) * 0.4
        else:
            # Post-retirement: more bonds
            bonds_pct = (1 - stock_pct) * 0.8
            alts_pct = (1 - stock_pct) * 0.2
        
        # Distribute among specific asset classes
        period_allocation = np.zeros(num_assets)
        for j, asset in enumerate(asset_classes):
            # Base allocation
            if asset == 'Global Equity':
                base_alloc = stock_pct
            elif asset == 'Core Bond':
                base_alloc = bonds_pct * 0.7
            elif asset == 'Short-Term Bond':
                base_alloc = bonds_pct * 0.3
            elif asset == 'Global Credit':
                base_alloc = alts_pct * 0.3
            elif asset == 'Real Assets':
                base_alloc = alts_pct * 0.4
            elif asset == 'Liquid Alternatives':
                base_alloc = alts_pct * 0.3
            else:
                base_alloc = 0
                
            # Apply adjustment for market conditions in first 7 years
            if i < 7:
                # Linear blend from full adjustment at year 0 to no adjustment at year 7
                adjustment_weight = 1.0 - (i / 7.0)
                
                # Get the adjustment factor for this asset class
                asset_adjustment = return_adjustments.get(asset, 1.0)
                
                # Apply the adjustment:
                # - If asset_adjustment < 1.0 (short-term returns worse than long-term), reduce allocation
                # - If asset_adjustment > 1.0 (short-term returns better than long-term), increase allocation
                # The adjustment factor is weighted by how far we are into the 7-year period
                
                # Scale adjustments to prevent extreme allocations
                asset_adjustment_scaled = 1.0 + (asset_adjustment - 1.0) * 0.5  # reduce effect of big deviations
                
                # Apply scaled adjustment, weighted by position in 7-year period
                period_allocation[j] = base_alloc * (1.0 + adjustment_weight * (asset_adjustment_scaled - 1.0))
            else:
                # Beyond 7 years, use the base allocation
                period_allocation[j] = base_alloc
        
        # Append to initial guess
        initial_guess.extend(period_allocation)
    
    # Run optimization with larger number of iterations to find better solution
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'disp': True, 'ftol': 1e-5}
    )
    
    # Extract the optimized glidepath
    optimized_weights = np.reshape(result.x, (num_periods, num_assets))
    
    # Interpolate to get a full glidepath
    glidepath = []
    for year in range(years_to_simulate + 1):
        period_idx = min(int(year / period_length), num_periods - 1)
        glidepath.append(optimized_weights[period_idx])
    
    # Create a conditional glidepath for recourse strategy
    # This adds an alternative path if market conditions change
    bearish_glidepath = []
    bullish_glidepath = []
    
    # Create market-condition dependent glidepaths
    for year, allocation in enumerate(glidepath):
        # For bearish markets: reduce equity by up to 15%, increase bonds
        # Bearish adjustments for different market conditions
        bearish_adjustment = np.zeros_like(allocation)
        
        # Determine stock index (Global Equity)
        stock_idx = asset_classes.index('Global Equity')
        bond_idx = asset_classes.index('Core Bond')
        
        # Adjust stock exposure down in bearish scenarios (up to 15%)
        equity_reduction = min(0.15, allocation[stock_idx] * 0.3)
        bearish_adjustment[stock_idx] = -equity_reduction
        bearish_adjustment[bond_idx] = equity_reduction  # Move to bonds
        
        # For bullish markets: increase equity by up to 10%, reduce bonds
        bullish_adjustment = np.zeros_like(allocation)
        
        # Calculate how much more equity exposure can be added without exceeding bounds
        max_equity_increase = min(0.10, 1.0 - allocation[stock_idx])
        bullish_adjustment[stock_idx] = max_equity_increase
        bullish_adjustment[bond_idx] = -max_equity_increase  # Take from bonds
        
        # Apply adjustments
        bearish_allocation = np.clip(allocation + bearish_adjustment, 0, 1)
        bullish_allocation = np.clip(allocation + bullish_adjustment, 0, 1)
        
        # Normalize to ensure sum is 1.0
        bearish_allocation = bearish_allocation / np.sum(bearish_allocation)
        bullish_allocation = bullish_allocation / np.sum(bullish_allocation)
        
        bearish_glidepath.append(bearish_allocation)
        bullish_glidepath.append(bullish_allocation)
    
    # Convert to numpy arrays
    glidepath = np.array(glidepath)
    bearish_glidepath = np.array(bearish_glidepath)
    bullish_glidepath = np.array(bullish_glidepath)
    
    # Define the mean reversion process info for the glidepath result
    mean_reversion_info = {
        'description': 'This glidepath uses a 7-year mean reversion from short-term to long-term capital market assumptions',
        'short_term_source': 'Current market conditions (short-term capital market assumptions)',
        'long_term_source': 'Equilibrium expectations (long-term capital market assumptions)',
        'mean_reversion_years': 7,
        'time_varying_returns': True,
        'shortfall_risk_years': 7,
        'shortfall_risk_description': 'Weight on shortfall risk increases in the 7 years before restylement to ensure capital preservation as the transition to restylement approaches'
    }
    
    # Run a final simulation with the optimized glidepath
    plan.glidepath = glidepath
    plan.glidepath_info = mean_reversion_info
    
    # Ensure time_varying_returns is explicitly set to True to apply the mean reversion
    # This will make the simulation start with short-term assumptions and blend to 
    # long-term assumptions over the 7-year period
    final_sim_results = run_monte_carlo_simulation(
        client, 
        plan, 
        market_assumptions, 
        num_simulations=num_simulations * 2,  # More simulations for final result
        max_age=max_age
    )
    
    # Calculate final risk metrics with more comprehensive measures
    final_risk_metrics = calculate_shortfall_risk(final_sim_results)
    
    # Create a more comprehensive result
    return {
        'glidepath': glidepath,
        'bearish_glidepath': bearish_glidepath,
        'bullish_glidepath': bullish_glidepath,
        'asset_classes': asset_classes,
        'success_probability': 1 - final_risk_metrics['shortfall_probability'],
        'shortfall_risk': final_risk_metrics,
        'simulation_results': final_sim_results,
        'ages': list(range(current_age, max_age + 1)),
        'retirement_age': retirement_age,
        'target_retirement_wealth': target_retirement_wealth,
        'sustainable_withdrawal_rate': final_risk_metrics.get('sustainable_withdrawal_rate', 0.04)
    }

def mean_variance_optimize(expected_returns, cov_matrix, risk_aversion=3.0, min_weights=None, max_weights=None):
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
    min_weights : array-like or None
        Minimum weights for each sub-asset class (if None, defaults to 0)
    max_weights : array-like or None
        Maximum weights for each sub-asset class (if None, defaults to 1)
        
    Returns:
    --------
    array
        Optimal weights for each sub-asset class
    """
    num_assets = len(expected_returns)
    
    # Set default min/max weights if not provided
    if min_weights is None:
        min_weights = np.zeros(num_assets)
    if max_weights is None:
        max_weights = np.ones(num_assets)
    
    # Define the objective function to maximize utility (return - risk_aversion * variance)
    def objective(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        utility = portfolio_return - (risk_aversion * portfolio_variance)
        return -utility  # Minimize negative utility
    
    # Constraints: sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    # Bounds: weights between min and max values
    bounds = [(min_weights[i], max_weights[i]) for i in range(num_assets)]
    
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
    Generate a visualization of the optimized glidepath allocation over time.
    
    Parameters:
    -----------
    glidepath_result : dict
        Results from the glidepath optimization
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure with the glidepath visualization
    """
    # Main figure with single plot (removed dynamic market adjustment charts)
    fig = plt.figure(figsize=(16, 8))
    
    # Extract data from results
    glidepath = glidepath_result['glidepath']
    asset_classes = glidepath_result['asset_classes']
    ages = glidepath_result['ages']
    retirement_age = glidepath_result.get('retirement_age', 65)
    
    # Create single plot for glidepath allocation
    ax1 = plt.subplot(1, 1, 1)
    
    # Create stacked area chart
    bottom = np.zeros(len(glidepath))
    for i, asset in enumerate(asset_classes):
        values = [allocation[i] for allocation in glidepath]
        ax1.fill_between(ages, bottom, bottom + values, label=asset)
        bottom += values
    
    # Add retirement age vertical line
    retirement_index = None
    for i, age in enumerate(ages):
        if age >= retirement_age:
            retirement_index = i
            break
            
    if retirement_index is not None:
        ax1.axvline(x=retirement_age, color='r', linestyle='--', alpha=0.7, label='Restylement Age')
        
    # Add shaded area for 7 years before restylement to show increased weight on shortfall risk
    if retirement_index is not None and retirement_index >= 7:
        # Get x-coordinates for shaded area (7 years before restylement)
        shortfall_risk_start = retirement_age - 7
        ax1.axvspan(shortfall_risk_start, retirement_age, alpha=0.15, color='red', 
                   label='Increased Shortfall Risk Weight')
    
    # Add labels and title
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Allocation (%)')
    ax1.set_title('Optimized Glidepath Allocation', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add metrics info box
    metrics_text = [
        f"Probability of Success: {glidepath_result['success_probability']:.1%}",
    ]
    
    # Add sustainable withdrawal rate if available
    if 'sustainable_withdrawal_rate' in glidepath_result:
        metrics_text.append(f"Sustainable Withdrawal Rate: {glidepath_result['sustainable_withdrawal_rate']*100:.2f}%")
        
    # Format text box
    metrics_box = '\n'.join(metrics_text)
    ax1.text(0.05, 0.95, metrics_box, transform=ax1.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_xlim(ages[0], ages[-1])
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig

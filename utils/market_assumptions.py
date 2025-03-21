import numpy as np
import pandas as pd
import streamlit as st

def get_market_assumptions():
    """
    Get default market assumptions if none are stored in session state.
    
    Returns:
    --------
    dict
        Market assumptions for both short-term and long-term views
    """
    asset_classes = [
        'Global Equity',
        'Core Bond',
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    
    sub_asset_classes = {
        'Global Equity': ['US Large Cap', 'US Small Cap', 'International Developed', 'Emerging Markets'],
        'Core Bond': ['US Treasuries', 'TIPS', 'Agency MBS'],
        'Short-Term Bond': ['Short Duration Treasuries', 'Short Duration Credit'],
        'Global Credit': ['US Investment Grade', 'US High Yield', 'International Credit', 'Emerging Market Debt'],
        'Real Assets': ['REITs', 'Commodities', 'Infrastructure'],
        'Liquid Alternatives': ['Market Neutral', 'Managed Futures', 'Global Macro']
    }
    
    # Default long-term (equilibrium) assumptions
    long_term = {
        'expected_returns': {
            'Global Equity': 0.067,
            'Core Bond': 0.023,
            'Short-Term Bond': 0.018,
            'Global Credit': 0.039,
            'Real Assets': 0.052,
            'Liquid Alternatives': 0.042
        },
        'volatilities': {
            'Global Equity': 0.18,
            'Core Bond': 0.046,
            'Short-Term Bond': 0.025,
            'Global Credit': 0.08,
            'Real Assets': 0.14,
            'Liquid Alternatives': 0.09
        },
        'correlations': pd.DataFrame(
            [
                [1.00, -0.10, -0.05, 0.50, 0.60, 0.40],
                [-0.10, 1.00, 0.80, 0.20, -0.10, -0.20],
                [-0.05, 0.80, 1.00, 0.15, -0.05, -0.10],
                [0.50, 0.20, 0.15, 1.00, 0.45, 0.35],
                [0.60, -0.10, -0.05, 0.45, 1.00, 0.25],
                [0.40, -0.20, -0.10, 0.35, 0.25, 1.00]
            ],
            index=asset_classes,
            columns=asset_classes
        )
    }
    
    # Default sub-asset class assumptions
    sub_asset_class_returns = {
        'US Large Cap': 0.065,
        'US Small Cap': 0.075,
        'International Developed': 0.063,
        'Emerging Markets': 0.082,
        'US Treasuries': 0.021,
        'TIPS': 0.023,
        'Agency MBS': 0.026,
        'Short Duration Treasuries': 0.017,
        'Short Duration Credit': 0.021,
        'US Investment Grade': 0.035,
        'US High Yield': 0.045,
        'International Credit': 0.037,
        'Emerging Market Debt': 0.048,
        'REITs': 0.056,
        'Commodities': 0.043,
        'Infrastructure': 0.058,
        'Market Neutral': 0.035,
        'Managed Futures': 0.043,
        'Global Macro': 0.048
    }
    
    sub_asset_class_volatilities = {
        'US Large Cap': 0.17,
        'US Small Cap': 0.21,
        'International Developed': 0.18,
        'Emerging Markets': 0.23,
        'US Treasuries': 0.04,
        'TIPS': 0.045,
        'Agency MBS': 0.05,
        'Short Duration Treasuries': 0.02,
        'Short Duration Credit': 0.03,
        'US Investment Grade': 0.065,
        'US High Yield': 0.12,
        'International Credit': 0.08,
        'Emerging Market Debt': 0.14,
        'REITs': 0.16,
        'Commodities': 0.19,
        'Infrastructure': 0.12,
        'Market Neutral': 0.06,
        'Managed Futures': 0.12,
        'Global Macro': 0.09
    }
    
    # Create default short-term assumptions with slightly different values
    # Typically short-term would differ based on market conditions
    short_term = {
        'expected_returns': {
            'Global Equity': 0.07,
            'Core Bond': 0.015,
            'Short-Term Bond': 0.02,
            'Global Credit': 0.035,
            'Real Assets': 0.05,
            'Liquid Alternatives': 0.04
        },
        'volatilities': {
            'Global Equity': 0.185,
            'Core Bond': 0.05,
            'Short-Term Bond': 0.028,
            'Global Credit': 0.075,
            'Real Assets': 0.15,
            'Liquid Alternatives': 0.085
        },
        'correlations': long_term['correlations'].copy()  # Use same correlations for simplicity
    }
    
    return {
        'asset_classes': asset_classes,
        'sub_asset_classes': sub_asset_classes,
        'long_term': long_term,
        'short_term': short_term,
        'sub_asset_class_returns': sub_asset_class_returns,
        'sub_asset_class_volatilities': sub_asset_class_volatilities
    }

def update_market_assumptions(new_assumptions):
    """
    Update market assumptions in the session state.
    
    Parameters:
    -----------
    new_assumptions : dict
        New market assumptions to store
    """
    st.session_state.market_assumptions = new_assumptions

def get_asset_returns_covariance(market_assumptions, view='long_term'):
    """
    Get returns and covariance matrix from market assumptions.
    
    Parameters:
    -----------
    market_assumptions : dict
        Market assumptions
    view : str
        'short_term' or 'long_term'
        
    Returns:
    --------
    tuple
        (returns, volatilities, correlation matrix)
    """
    # Extract the appropriate view
    assumptions = market_assumptions[view]
    
    # Get asset classes
    asset_classes = market_assumptions['asset_classes']
    
    # Extract returns and volatilities
    returns = np.array([assumptions['expected_returns'][asset] for asset in asset_classes])
    volatilities = np.array([assumptions['volatilities'][asset] for asset in asset_classes])
    
    # Extract correlation matrix
    correlations = assumptions['correlations'].values
    
    return returns, volatilities, correlations

def optimize_sub_asset_classes(market_assumptions):
    """
    Optimize sub-asset class allocations within each major asset class.
    
    Parameters:
    -----------
    market_assumptions : dict
        Market assumptions including sub-asset class data
        
    Returns:
    --------
    dict
        Optimized allocations for sub-asset classes within each major asset class
    """
    from utils.optimization import mean_variance_optimize
    
    optimized_allocations = {}
    
    for asset_class, sub_assets in market_assumptions['sub_asset_classes'].items():
        # Extract returns for this group of sub-assets
        returns = np.array([market_assumptions['sub_asset_class_returns'][sub] for sub in sub_assets])
        
        # Extract volatilities for this group of sub-assets
        vols = np.array([market_assumptions['sub_asset_class_volatilities'][sub] for sub in sub_assets])
        
        # Create a simple correlation matrix (could be more sophisticated in a real model)
        n = len(sub_assets)
        corr_matrix = np.zeros((n, n))
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Fill in off-diagonal elements with reasonable correlation values
        for i in range(n):
            for j in range(i+1, n):
                # Set a moderate correlation (0.5) between sub-assets in same class
                corr_matrix[i, j] = 0.5
                corr_matrix[j, i] = 0.5
        
        # Create covariance matrix
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = vols[i] * vols[j] * corr_matrix[i, j]
        
        # Run mean-variance optimization
        weights = mean_variance_optimize(returns, cov_matrix)
        
        # Store the optimized weights
        optimized_allocations[asset_class] = {
            'sub_assets': sub_assets,
            'weights': weights
        }
    
    return optimized_allocations

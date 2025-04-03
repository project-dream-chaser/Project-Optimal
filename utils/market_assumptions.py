import numpy as np
import pandas as pd
import streamlit as st

def get_market_assumptions():
    """
    Get market assumptions from file if it exists, otherwise create defaults.
    
    Returns:
    --------
    dict
        Market assumptions for both short-term and long-term views
    """
    import json
    import os
    import pandas as pd
    
    # Check if market assumptions file exists
    if os.path.exists('data/market_assumptions.json'):
        try:
            with open('data/market_assumptions.json', 'r') as f:
                assumptions = json.load(f)
                
                # Convert dictionary representations of DataFrames back to actual DataFrames
                if 'short_term' in assumptions and 'correlations' in assumptions['short_term']:
                    if isinstance(assumptions['short_term']['correlations'], dict):
                        assumptions['short_term']['correlations'] = pd.DataFrame(assumptions['short_term']['correlations'])
                        
                if 'long_term' in assumptions and 'correlations' in assumptions['long_term']:
                    if isinstance(assumptions['long_term']['correlations'], dict):
                        assumptions['long_term']['correlations'] = pd.DataFrame(assumptions['long_term']['correlations'])
                
                return assumptions
        except Exception as e:
            st.error(f"Error loading market assumptions: {e}")
            # If there's an error, fall back to defaults
            pass
    asset_classes = [
        'Global Equity',
        'Core Bond',
        'Short-Term Bond',
        'Global Credit',
        'Real Assets',
        'Liquid Alternatives'
    ]
    
    sub_asset_classes = {
        'Global Equity': ['US Value', 'US Quality', 'US Small Cap', 'International Developed', 'International Developed Small', 'Emerging Markets'],
        'Core Bond': ['1-5 Year Investment Grade', 'Absolute Return'],
        'Short-Term Bond': ['Short Duration Treasuries', 'Short Duration Credit'],
        'Global Credit': ['Quality High Yield', 'High Yield', 'Bank Loans', 'Securitized Credit', 'Emerging Market Debt'],
        'Real Assets': ['REITs', 'Commodities', 'Infrastructure', 'TIPS', 'Natural Resources'],
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
        'US Value': 0.069,
        'US Quality': 0.063,
        'US Small Cap': 0.075,
        'International Developed': 0.063,
        'International Developed Small': 0.072,
        'Emerging Markets': 0.082,
        '1-5 Year Investment Grade': 0.024,
        'Absolute Return': 0.028,
        'Short Duration Treasuries': 0.017,
        'Short Duration Credit': 0.021,
        'Quality High Yield': 0.040,
        'High Yield': 0.047,
        'Bank Loans': 0.043,
        'Securitized Credit': 0.038,
        'Emerging Market Debt': 0.048,
        'REITs': 0.056,
        'Commodities': 0.043,
        'Infrastructure': 0.058,
        'TIPS': 0.027,
        'Natural Resources': 0.053,
        'Market Neutral': 0.035,
        'Managed Futures': 0.043,
        'Global Macro': 0.048
    }
    
    sub_asset_class_volatilities = {
        'US Value': 0.17,
        'US Quality': 0.16,
        'US Small Cap': 0.21,
        'International Developed': 0.18,
        'International Developed Small': 0.20,
        'Emerging Markets': 0.23,
        '1-5 Year Investment Grade': 0.035,
        'Absolute Return': 0.042,
        'Short Duration Treasuries': 0.02,
        'Short Duration Credit': 0.03,
        'Quality High Yield': 0.08,
        'High Yield': 0.12,
        'Bank Loans': 0.09,
        'Securitized Credit': 0.07,
        'Emerging Market Debt': 0.14,
        'REITs': 0.16,
        'Commodities': 0.19,
        'Infrastructure': 0.12,
        'TIPS': 0.05,
        'Natural Resources': 0.16,
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
    Update market assumptions in the session state and save to file.
    
    Parameters:
    -----------
    new_assumptions : dict
        New market assumptions to store
    """
    import json
    import os
    import numpy as np
    import pandas as pd
    import streamlit as st
    
    # Update session state
    st.session_state.market_assumptions = new_assumptions
    
    # Make JSON serializable function
    def make_json_serializable(obj):
        """Convert numpy arrays, pandas DataFrames and other non-serializable objects to serializable types."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
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
    serializable_assumptions = make_json_serializable(new_assumptions)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to file
    with open('data/market_assumptions.json', 'w') as f:
        json.dump(serializable_assumptions, f, indent=4)
        
    # Debug - verify file was written correctly
    try:
        # Read the file back to verify it was written correctly
        with open('data/market_assumptions.json', 'r') as f:
            saved_data = json.load(f)
        
        # Debug info (this will only show up in the console logs)
        print("Market assumptions saved successfully:")
        print(f"File size: {os.path.getsize('data/market_assumptions.json')} bytes")
        print(f"Number of asset classes: {len(saved_data['asset_classes'])}")
        print(f"Sample return value (Global Equity short-term): {saved_data['short_term']['expected_returns']['Global Equity']}")
    except Exception as e:
        st.error(f"Error verifying saved data: {e}")

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

def optimize_sub_asset_classes(market_assumptions, risk_aversion=None):
    """
    Optimize sub-asset class allocations within each major asset class.
    
    Parameters:
    -----------
    market_assumptions : dict
        Market assumptions including sub-asset class data
    risk_aversion : float, optional
        Risk aversion parameter (higher = more conservative)
        If None, uses the value from session state or default 4.0
        
    Returns:
    --------
    dict
        Optimized allocations for sub-asset classes within each major asset class
    """
    from utils.optimization import mean_variance_optimize
    import streamlit as st
    
    optimized_allocations = {}
    
    # Get risk aversion parameter from input, session state, or default
    if risk_aversion is None:
        risk_aversion = 4.0  # Default is now 4.0
        if 'risk_aversion' in st.session_state:
            risk_aversion = st.session_state.risk_aversion
    
    # Check if we have sub-asset class constraints
    sub_asset_constraints = {}
    if 'sub_asset_constraints' in st.session_state:
        sub_asset_constraints = st.session_state.sub_asset_constraints
    
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
        
        # Set up min and max weights
        min_weights = np.zeros(n)
        max_weights = np.ones(n)
        
        # Apply constraints if available for this asset class
        if asset_class in sub_asset_constraints:
            constraints = sub_asset_constraints[asset_class]
            for i, sub in enumerate(sub_assets):
                if sub in constraints:
                    min_weights[i] = constraints[sub]['min'] / 100.0  # Convert from percentage
                    max_weights[i] = constraints[sub]['max'] / 100.0  # Convert from percentage
        
        # Run mean-variance optimization with constraints
        weights = mean_variance_optimize(
            returns, 
            cov_matrix, 
            risk_aversion=risk_aversion,
            min_weights=min_weights,
            max_weights=max_weights
        )
        
        # Store the optimized weights
        optimized_allocations[asset_class] = {
            'sub_assets': sub_assets,
            'weights': weights
        }
    
    return optimized_allocations

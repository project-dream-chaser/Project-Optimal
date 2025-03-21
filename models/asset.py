class Asset:
    """
    Asset model representing an investable asset class.
    
    Attributes:
    -----------
    name : str
        Name of the asset class
    category : str
        Top-level category (e.g., 'Global Equity', 'Core Bond')
    expected_return : float
        Expected annualized return
    volatility : float
        Expected annualized volatility (standard deviation)
    is_sub_asset : bool
        Whether this is a sub-asset class
    parent_asset : str or None
        Parent asset class name if this is a sub-asset
    """
    
    def __init__(self, name, category, expected_return, volatility, is_sub_asset=False, parent_asset=None):
        """
        Initialize an Asset object.
        
        Parameters:
        -----------
        name : str
            Name of the asset class
        category : str
            Top-level category (e.g., 'Global Equity', 'Core Bond')
        expected_return : float
            Expected annualized return
        volatility : float
            Expected annualized volatility (standard deviation)
        is_sub_asset : bool
            Whether this is a sub-asset class
        parent_asset : str or None
            Parent asset class name if this is a sub-asset
        """
        self.name = name
        self.category = category
        self.expected_return = expected_return
        self.volatility = volatility
        self.is_sub_asset = is_sub_asset
        self.parent_asset = parent_asset
    
    def to_dict(self):
        """
        Convert asset object to dictionary.
        
        Returns:
        --------
        dict
            Asset data as dictionary
        """
        return {
            'name': self.name,
            'category': self.category,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'is_sub_asset': self.is_sub_asset,
            'parent_asset': self.parent_asset
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create an Asset object from dictionary data.
        
        Parameters:
        -----------
        data : dict
            Asset data as dictionary
            
        Returns:
        --------
        Asset
            New Asset object
        """
        return cls(
            name=data.get('name'),
            category=data.get('category'),
            expected_return=data.get('expected_return'),
            volatility=data.get('volatility'),
            is_sub_asset=data.get('is_sub_asset', False),
            parent_asset=data.get('parent_asset')
        )


class AssetAllocation:
    """
    Asset allocation model representing a portfolio allocation.
    
    Attributes:
    -----------
    name : str
        Name of the allocation (e.g., 'Conservative', 'Aggressive')
    weights : dict
        Dictionary mapping asset class names to weights
    """
    
    def __init__(self, name, weights):
        """
        Initialize an AssetAllocation object.
        
        Parameters:
        -----------
        name : str
            Name of the allocation (e.g., 'Conservative', 'Aggressive')
        weights : dict
            Dictionary mapping asset class names to weights
        """
        self.name = name
        self.weights = weights
        
        # Validate weights sum to approximately 1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def expected_return(self, market_assumptions):
        """
        Calculate the expected return of the allocation.
        
        Parameters:
        -----------
        market_assumptions : dict
            Market assumptions with expected returns
            
        Returns:
        --------
        float
            Expected portfolio return
        """
        expected_return = 0.0
        for asset, weight in self.weights.items():
            expected_return += weight * market_assumptions['long_term']['expected_returns'].get(asset, 0)
        return expected_return
    
    def expected_volatility(self, market_assumptions):
        """
        Calculate the expected volatility of the allocation.
        
        Parameters:
        -----------
        market_assumptions : dict
            Market assumptions with volatilities and correlations
            
        Returns:
        --------
        float
            Expected portfolio volatility
        """
        assets = list(self.weights.keys())
        asset_weights = [self.weights[asset] for asset in assets]
        
        # Extract volatilities
        vols = [market_assumptions['long_term']['volatilities'].get(asset, 0) for asset in assets]
        
        # Extract correlation matrix
        corr_matrix = market_assumptions['long_term']['correlations']
        
        # Calculate portfolio variance
        portfolio_variance = 0.0
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                # Get correlation between assets i and j
                if i == j:
                    correlation = 1.0
                else:
                    correlation = corr_matrix.loc[asset_i, asset_j]
                
                portfolio_variance += (
                    asset_weights[i] * asset_weights[j] * 
                    vols[i] * vols[j] * 
                    correlation
                )
        
        return portfolio_variance ** 0.5  # Square root for standard deviation
    
    def to_dict(self):
        """
        Convert allocation object to dictionary.
        
        Returns:
        --------
        dict
            Allocation data as dictionary
        """
        return {
            'name': self.name,
            'weights': self.weights
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create an AssetAllocation object from dictionary data.
        
        Parameters:
        -----------
        data : dict
            Allocation data as dictionary
            
        Returns:
        --------
        AssetAllocation
            New AssetAllocation object
        """
        return cls(
            name=data.get('name'),
            weights=data.get('weights', {})
        )

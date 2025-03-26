class CashFlow:
    """
    CashFlow model representing a recurring cash flow in a financial plan.
    
    Attributes:
    -----------
    name : str
        Description of the cash flow
    amount : float
        Annual amount (positive for income, negative for expenses)
    start_age : int
        Age when the cash flow starts
    end_age : int
        Age when the cash flow ends
    growth_rate : float
        Annual growth rate (inflation rate for expenses)
    """
    
    def __init__(self, name, amount, start_age, end_age, growth_rate=0.0):
        """
        Initialize a CashFlow object.
        
        Parameters:
        -----------
        name : str
            Description of the cash flow
        amount : float
            Annual amount (positive for income, negative for expenses)
        start_age : int
            Age when the cash flow starts
        end_age : int
            Age when the cash flow ends
        growth_rate : float
            Annual growth rate (inflation rate for expenses)
        """
        self.name = name
        self.amount = amount
        self.start_age = start_age
        self.end_age = end_age
        self.growth_rate = growth_rate
    
    def to_dict(self):
        """
        Convert cash flow object to dictionary.
        
        Returns:
        --------
        dict
            Cash flow data as dictionary
        """
        return {
            'name': self.name,
            'amount': self.amount,
            'start_age': self.start_age,
            'end_age': self.end_age,
            'growth_rate': self.growth_rate
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a CashFlow object from dictionary data.
        
        Parameters:
        -----------
        data : dict
            Cash flow data as dictionary
            
        Returns:
        --------
        CashFlow
            New CashFlow object
        """
        return cls(
            name=data.get('name'),
            amount=data.get('amount'),
            start_age=data.get('start_age'),
            end_age=data.get('end_age'),
            growth_rate=data.get('growth_rate', 0.0)
        )


class Goal:
    """
    Goal model representing a financial goal in a plan.
    
    Attributes:
    -----------
    name : str
        Description of the goal
    amount : float
        Cost of the goal
    age : int
        Age when the goal occurs
    priority : str
        Priority of the goal ('High', 'Medium', 'Low')
    """
    
    def __init__(self, name, amount, age, priority='Medium'):
        """
        Initialize a Goal object.
        
        Parameters:
        -----------
        name : str
            Description of the goal
        amount : float
            Cost of the goal
        age : int
            Age when the goal occurs
        priority : str
            Priority of the goal ('High', 'Medium', 'Low')
        """
        self.name = name
        self.amount = amount
        self.age = age
        self.priority = priority
    
    def to_dict(self):
        """
        Convert goal object to dictionary.
        
        Returns:
        --------
        dict
            Goal data as dictionary
        """
        return {
            'name': self.name,
            'amount': self.amount,
            'age': self.age,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a Goal object from dictionary data.
        
        Parameters:
        -----------
        data : dict
            Goal data as dictionary
            
        Returns:
        --------
        Goal
            New Goal object
        """
        return cls(
            name=data.get('name'),
            amount=data.get('amount'),
            age=data.get('age'),
            priority=data.get('priority', 'Medium')
        )


class Plan:
    """
    Plan model representing a client's financial plan.
    
    Attributes:
    -----------
    client_id : str
        ID of the client this plan belongs to
    name : str
        Name of the plan
    goals : list
        List of Goal objects
    cash_flows : list
        List of CashFlow objects
    initial_portfolio : float
        Starting portfolio value
    asset_allocation : array-like
        Initial asset allocation weights
    glidepath : array-like or None
        Glidepath allocation over time if optimized
    glidepath_info : dict or None
        Additional information about the glidepath, including capital market assumption details
    pre_restylement_return : float
        Target annual return before restylement (accumulation phase)
    post_restylement_return : float
        Target annual return after restylement (distribution phase)
    """
    
    def __init__(self, client_id, name, goals=None, cash_flows=None, initial_portfolio=0, 
                 asset_allocation=None, allocation_constraints=None, risk_aversion=3.0, 
                 mean_reversion_speed=0.15, glidepath=None, glidepath_info=None,
                 pre_restylement_return=7.0, post_restylement_return=5.0):
        """
        Initialize a Plan object.
        
        Parameters:
        -----------
        client_id : str
            ID of the client this plan belongs to
        name : str
            Name of the plan
        goals : list
            List of Goal objects
        cash_flows : list
            List of CashFlow objects
        initial_portfolio : float
            Starting portfolio value
        asset_allocation : array-like
            Initial asset allocation weights
        allocation_constraints : dict
            Dictionary mapping asset class names to min/max constraints
        risk_aversion : float
            Risk aversion parameter (higher = more conservative)
        mean_reversion_speed : float
            Speed at which returns revert to long-term means
        glidepath : array-like or None
            Glidepath allocation over time if optimized
        glidepath_info : dict or None
            Additional information about the glidepath, including capital market assumption details
        pre_restylement_return : float
            Target annual return before restylement (accumulation phase)
        post_restylement_return : float
            Target annual return after restylement (distribution phase)
        """
        self.client_id = client_id
        self.name = name
        self.goals = goals or []
        self.cash_flows = cash_flows or []
        self.initial_portfolio = initial_portfolio
        self.asset_allocation = asset_allocation or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.allocation_constraints = allocation_constraints or {}
        self.risk_aversion = risk_aversion
        self.mean_reversion_speed = mean_reversion_speed
        self.glidepath = glidepath
        self.glidepath_info = glidepath_info
        self.pre_restylement_return = pre_restylement_return
        self.post_restylement_return = post_restylement_return
    
    def add_goal(self, goal):
        """
        Add a goal to the plan.
        
        Parameters:
        -----------
        goal : Goal
            Goal to add
        """
        self.goals.append(goal)
    
    def add_cash_flow(self, cash_flow):
        """
        Add a cash flow to the plan.
        
        Parameters:
        -----------
        cash_flow : CashFlow
            Cash flow to add
        """
        self.cash_flows.append(cash_flow)
    
    def to_dict(self):
        """
        Convert plan object to dictionary.
        
        Returns:
        --------
        dict
            Plan data as dictionary
        """
        return {
            'client_id': self.client_id,
            'name': self.name,
            'goals': [goal.to_dict() for goal in self.goals],
            'cash_flows': [cf.to_dict() for cf in self.cash_flows],
            'initial_portfolio': self.initial_portfolio,
            'asset_allocation': self.asset_allocation,
            'allocation_constraints': self.allocation_constraints,
            'risk_aversion': self.risk_aversion,
            'mean_reversion_speed': self.mean_reversion_speed,
            'glidepath': self.glidepath.tolist() if self.glidepath is not None else None,
            'glidepath_info': self.glidepath_info,
            'pre_restylement_return': self.pre_restylement_return,
            'post_restylement_return': self.post_restylement_return
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a Plan object from dictionary data.
        
        Parameters:
        -----------
        data : dict
            Plan data as dictionary
            
        Returns:
        --------
        Plan
            New Plan object
        """
        goals = [Goal.from_dict(goal_data) for goal_data in data.get('goals', [])]
        cash_flows = [CashFlow.from_dict(cf_data) for cf_data in data.get('cash_flows', [])]
        
        return cls(
            client_id=data.get('client_id'),
            name=data.get('name'),
            goals=goals,
            cash_flows=cash_flows,
            initial_portfolio=data.get('initial_portfolio', 0),
            asset_allocation=data.get('asset_allocation'),
            allocation_constraints=data.get('allocation_constraints', {}),
            risk_aversion=data.get('risk_aversion', 3.0),
            mean_reversion_speed=data.get('mean_reversion_speed', 0.15),
            glidepath=data.get('glidepath'),
            glidepath_info=data.get('glidepath_info'),
            pre_restylement_return=data.get('pre_restylement_return', 7.0),
            post_restylement_return=data.get('post_restylement_return', 5.0)
        )

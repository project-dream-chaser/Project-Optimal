class Client:
    """
    Client model representing a financial planning client.
    
    Attributes:
    -----------
    id : str
        Unique client identifier
    first_name : str
        Client's first name
    last_name : str
        Client's last name
    email : str
        Client's email address
    date_of_birth : str
        Client's date of birth in YYYY-MM-DD format
    max_stock_pct : int or None
        Client's maximum stock percentage (0-100%)
    spouse : dict or None
        Spouse information if applicable
    restylement_age : int or None
        Age at which the client plans to enter restylement (retirement)
    longevity_age : int or None
        Age to which the client plans for financial longevity
    """
    
    def __init__(self, id, first_name, last_name, email, date_of_birth, max_stock_pct=None, spouse=None, 
                 restylement_age=65, longevity_age=95):
        """
        Initialize a Client object.
        
        Parameters:
        -----------
        id : str
            Unique client identifier
        first_name : str
            Client's first name
        last_name : str
            Client's last name
        email : str
            Client's email address
        date_of_birth : str
            Client's date of birth in YYYY-MM-DD format
        max_stock_pct : int or None
            Client's maximum stock percentage (0-100%)
        spouse : dict or None
            Spouse information if applicable
        restylement_age : int
            Age at which the client plans to enter restylement (retirement), default is 65
        longevity_age : int
            Age to which the client plans for financial longevity, default is 95
        """
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.date_of_birth = date_of_birth
        self.max_stock_pct = max_stock_pct
        self.spouse = spouse
        self.restylement_age = restylement_age
        self.longevity_age = longevity_age
    
    def full_name(self):
        """Return the client's full name."""
        return f"{self.first_name} {self.last_name}"
    
    def get_risk_profile(self):
        """
        Get the client's risk profile based on maximum stock percentage.
        
        Returns:
        --------
        str
            Risk profile description
        """
        if self.max_stock_pct is None:
            return "Not assessed"
        
        if self.max_stock_pct <= 20:
            return "Very Conservative"
        elif self.max_stock_pct <= 40:
            return "Conservative"
        elif self.max_stock_pct <= 60:
            return "Moderate"
        elif self.max_stock_pct <= 80:
            return "Aggressive"
        else:
            return "Very Aggressive"
    
    def to_dict(self):
        """
        Convert client object to dictionary.
        
        Returns:
        --------
        dict
            Client data as dictionary
        """
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'date_of_birth': self.date_of_birth,
            'max_stock_pct': self.max_stock_pct,
            'spouse': self.spouse,
            'restylement_age': self.restylement_age,
            'longevity_age': self.longevity_age
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a Client object from dictionary data.
        
        Parameters:
        -----------
        data : dict
            Client data as dictionary
            
        Returns:
        --------
        Client
            New Client object
        """
        # Handle backward compatibility with risk_score
        max_stock_pct = data.get('max_stock_pct')
        if max_stock_pct is None and 'risk_score' in data:
            risk_score = data.get('risk_score')
            # Convert risk_score (1-10) to max_stock_pct (0-100%)
            if risk_score is not None:
                try:
                    risk_score = int(risk_score)
                    max_stock_pct = risk_score * 10  # Simple conversion
                except (ValueError, TypeError):
                    max_stock_pct = None
        
        return cls(
            id=data.get('id'),
            first_name=data.get('first_name'),
            last_name=data.get('last_name'),
            email=data.get('email'),
            date_of_birth=data.get('date_of_birth'),
            max_stock_pct=max_stock_pct,
            spouse=data.get('spouse'),
            restylement_age=data.get('restylement_age', 65),
            longevity_age=data.get('longevity_age', 95)
        )

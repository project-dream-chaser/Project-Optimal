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
    risk_score : int or None
        Client's risk tolerance score (1-10)
    spouse : dict or None
        Spouse information if applicable
    restylement_age : int or None
        Age at which the client plans to enter restylement (retirement)
    longevity_age : int or None
        Age to which the client plans for financial longevity
    """
    
    def __init__(self, id, first_name, last_name, email, date_of_birth, risk_score=None, spouse=None, 
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
        risk_score : int or None
            Client's risk tolerance score (1-10)
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
        self.risk_score = risk_score
        self.spouse = spouse
        self.restylement_age = restylement_age
        self.longevity_age = longevity_age
    
    def full_name(self):
        """Return the client's full name."""
        return f"{self.first_name} {self.last_name}"
    
    def get_risk_profile(self):
        """
        Get the client's risk profile based on risk score.
        
        Returns:
        --------
        str
            Risk profile description
        """
        if self.risk_score is None:
            return "Not assessed"
        
        if self.risk_score <= 2:
            return "Very Conservative"
        elif self.risk_score <= 4:
            return "Conservative"
        elif self.risk_score <= 6:
            return "Moderate"
        elif self.risk_score <= 8:
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
            'risk_score': self.risk_score,
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
        return cls(
            id=data.get('id'),
            first_name=data.get('first_name'),
            last_name=data.get('last_name'),
            email=data.get('email'),
            date_of_birth=data.get('date_of_birth'),
            risk_score=data.get('risk_score'),
            spouse=data.get('spouse'),
            restylement_age=data.get('restylement_age', 65),
            longevity_age=data.get('longevity_age', 95)
        )

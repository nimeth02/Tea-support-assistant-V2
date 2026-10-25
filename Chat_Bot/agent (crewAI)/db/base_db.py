from abc import ABC, abstractmethod

class BaseDatabase(ABC):
    """Abstract base class for database connections."""
    
    @abstractmethod
    def connect(self):
        """Initialize the database connection."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
    
    @abstractmethod
    def get_session(self):
        """Return a session/client for database operations."""
        pass
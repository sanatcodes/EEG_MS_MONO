from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any

class BaseClusterer(ABC):
    """Base class for all clustering implementations."""
    
    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        """Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters to find
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """Fit the clustering model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            self: The fitted clusterer
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            labels: Predicted cluster labels
        """
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and predict cluster labels.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            labels: Predicted cluster labels
        """
        return self.fit(X).predict(X)
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the clusterer.
        
        Returns:
            params: Dictionary of parameters
        """
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        } 
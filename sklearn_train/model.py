"""
model.py
--------
Contains:
 LogisticRegressionScratch  — pure NumPy implementation (sigmoid, gradient descent)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

class Logistic_regression_scratch:
      def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

      def _sigmoid(self, z: np.ndarray): 
          return 1/ (1 + np.exp(-z))
      
      def _loss(self, y:np.ndarray, y_pred:np.ndarray):
          eps = 1e-15
          y_pred = np.clip(y_pred, eps, 1-eps)
          return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))
      
      def fit(self, X:np.ndarray, y:np.ndarray):
          n_samples, n_features  = X.shape
          self.weights = np.zeros(n_features)
          self.bias = 0

          for i in range(self.n_iterations):
              linear_model = np.dot(X, self.weights) + self.bias
              y_pred = self._sigmoid(linear_model)

              dw = (1/ n_samples) * np.dot(X.T, (y_pred - y))
              db = (1/ n_samples) * np.sum(y_pred - y)

              self.weights -= self.learning_rate * dw
              self.bias -= self.learning_rate * db

              loss = self._loss(y, y_pred)
              self.loss_history.append(loss)

          return self

      def predict_proba(self, X:np.ndarray):
          linear_model = np.dot(X, self.weights) + self.bias
          return self._sigmoid(linear_model)
      
      def predict(self, X: np.ndarray, threshold: float = 0.5):
          proba = self.predict_proba(X)
          return (proba >= threshold).astype(int)
      
def get_model(
        penalty, l1_ratio = 0.5):
    
    solver_map = {
        "l1" : "liblinear",
        "l2" : "lbfgs",
        "elasticnet" : "saga",
        "none" : "lbfgs",
        }
    
    if penalty not in solver_map:
        raise ValueError(f"Unsupported penalty: {penalty}")
    
    solver = solver_map[penalty]

    kwargs = {
        "penalty": penalty,
        "solver": solver,
        "max_iter": 1000,
        "random_state": 42,
    }

    if penalty == "l1":
        kwargs["penalty"] = "l1"
        kwargs["l1_ratio"] = 1

    elif penalty == "l2":
        kwargs["penalty"] = "l2"
        kwargs["l1_ratio"] = 0

    elif penalty == "elastic_net":
        kwargs["penalty"] = "elasticnet"
        kwargs["l1_ratio"] = l1_ratio

    elif penalty == "none":
        kwargs["penalty"] = None

    return LogisticRegression(**kwargs)        
    
   
      

          
            
        

 
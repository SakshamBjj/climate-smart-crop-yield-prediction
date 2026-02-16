"""
Uncertainty quantification for crop yield predictions.

Provides confidence intervals via quantile regression (tree ensembles)
or dropout-based uncertainty (deep learning).

Usage:
    from uncertainty import ConfidencePrediction
    
    predictor = ConfidencePrediction(model, method='quantile')
    mean, lower, upper = predictor.predict_with_confidence(X_test)
"""

import numpy as np
from typing import Tuple, Literal, Optional
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings


class ConfidencePrediction:
    """
    Wrapper for predictions with confidence intervals.
    
    Methods:
    - 'quantile': Quantile regression for tree ensembles
    - 'bootstrap': Bootstrap resampling
    - 'dropout': MC dropout for neural networks (requires model support)
    """
    
    def __init__(self, 
                 base_model,
                 method: Literal['quantile', 'bootstrap', 'dropout'] = 'quantile',
                 confidence_level: float = 0.9):
        """
        Args:
            base_model: Trained prediction model
            method: Uncertainty estimation method
            confidence_level: Confidence interval coverage (0-1)
        """
        self.base_model = base_model
        self.method = method
        self.confidence_level = confidence_level
        self.quantile_models = None
    
    def fit_quantile_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train quantile regression models for lower/upper bounds.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        alpha = (1 - self.confidence_level) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha
        
        print(f"Training quantile models ({lower_quantile:.2f}, {upper_quantile:.2f})...")
        
        if isinstance(self.base_model, RandomForestRegressor):
            # RandomForest doesn't support native quantile regression like GradientBoosting
            # Proper implementation requires either:
            # 1. quantile-forest package (RandomForestQuantileRegressor)
            # 2. Manual extraction from tree leaf distributions
            #
            # The simple approach of training with criterion='absolute_error'
            # does NOT produce quantiles — it just produces MAE-optimized mean predictions
            
            raise NotImplementedError(
                "RandomForest quantile regression not implemented. "
                "Use GradientBoostingRegressor (supports native quantile loss) "
                "or switch to bootstrap method via method='bootstrap'. "
                "For production, consider quantile-forest package."
            )
            
        elif isinstance(self.base_model, GradientBoostingRegressor):
            # GradientBoosting supports native quantile regression
            lower_model = clone(self.base_model)
            lower_model.set_params(loss='quantile', alpha=lower_quantile)
            lower_model.fit(X_train, y_train)
            
            upper_model = clone(self.base_model)
            upper_model.set_params(loss='quantile', alpha=upper_quantile)
            upper_model.fit(X_train, y_train)
            
            self.quantile_models = {
                'lower': lower_model,
                'upper': upper_model,
                'lower_q': lower_quantile,
                'upper_q': upper_quantile
            }
        else:
            warnings.warn(f"Quantile regression not directly supported for {type(self.base_model)}. "
                         "Falling back to bootstrap method.")
            self.method = 'bootstrap'
    
    def predict_with_confidence(self, 
                                X: np.ndarray,
                                n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            X: Feature matrix for prediction
            n_bootstrap: Bootstrap iterations (if method='bootstrap')
        
        Returns:
            Tuple of (mean_predictions, lower_bound, upper_bound)
        """
        if self.method == 'quantile':
            return self._predict_quantile(X)
        elif self.method == 'bootstrap':
            return self._predict_bootstrap(X, n_bootstrap)
        elif self.method == 'dropout':
            return self._predict_dropout(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _predict_quantile(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantile regression prediction."""
        if self.quantile_models is None:
            raise ValueError("Call fit_quantile_models() first")
        
        mean_pred = self.base_model.predict(X)
        lower_pred = self.quantile_models['lower'].predict(X)
        upper_pred = self.quantile_models['upper'].predict(X)
        
        return mean_pred, lower_pred, upper_pred
    
    def _predict_bootstrap(self, X: np.ndarray, n_bootstrap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap uncertainty estimation."""
        predictions = []
        
        for _ in range(n_bootstrap):
            # RandomForest already does bootstrap - use tree predictions
            if hasattr(self.base_model, 'estimators_'):
                # Get predictions from random subset of trees
                tree_preds = np.array([
                    tree.predict(X) for tree in 
                    np.random.choice(self.base_model.estimators_, size=50, replace=True)
                ])
                predictions.append(tree_preds.mean(axis=0))
            else:
                # Fallback for other models
                warnings.warn("Bootstrap not optimized for this model type")
                predictions.append(self.base_model.predict(X))
        
        predictions = np.array(predictions)
        
        alpha = (1 - self.confidence_level) / 2
        mean_pred = predictions.mean(axis=0)
        lower_pred = np.percentile(predictions, alpha * 100, axis=0)
        upper_pred = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return mean_pred, lower_pred, upper_pred
    
    def _predict_dropout(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MC Dropout for neural networks (requires model support)."""
        if not hasattr(self.base_model, 'predict_with_dropout'):
            raise NotImplementedError("Model must implement predict_with_dropout()")
        
        predictions = []
        for _ in range(n_samples):
            predictions.append(self.base_model.predict_with_dropout(X))
        
        predictions = np.array(predictions)
        
        alpha = (1 - self.confidence_level) / 2
        mean_pred = predictions.mean(axis=0)
        lower_pred = np.percentile(predictions, alpha * 100, axis=0)
        upper_pred = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return mean_pred, lower_pred, upper_pred
    
    def coverage_score(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> float:
        """
        Calculate empirical coverage of confidence intervals.
        
        Should be close to self.confidence_level for well-calibrated intervals.
        
        Args:
            X_test: Test features
            y_test: True test values
        
        Returns:
            Fraction of test samples within confidence interval
        """
        _, lower, upper = self.predict_with_confidence(X_test)
        
        within_interval = (y_test >= lower) & (y_test <= upper)
        coverage = within_interval.mean()
        
        return coverage


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("Confidence Interval Demo")
    print("=" * 70)
    
    # Synthetic data
    X, y = make_regression(n_samples=1000, n_features=14, noise=50, random_state=42)
    y += 2500  # Shift to yield scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train base model
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("\n1. Training quantile models for 90% confidence...")
    predictor = ConfidencePrediction(model, method='quantile', confidence_level=0.90)
    predictor.fit_quantile_models(X_train, y_train)
    
    print("\n2. Making predictions with confidence intervals...")
    mean_pred, lower_pred, upper_pred = predictor.predict_with_confidence(X_test[:5])
    
    print("\nSample predictions:")
    print("-" * 70)
    for i in range(5):
        width = upper_pred[i] - lower_pred[i]
        print(f"Sample {i+1}: {mean_pred[i]:.0f} kg/ha  "
              f"[{lower_pred[i]:.0f}, {upper_pred[i]:.0f}]  "
              f"(width: {width:.0f} kg/ha)")
    
    print("\n3. Evaluating coverage...")
    coverage = predictor.coverage_score(X_test, y_test)
    print(f"Empirical coverage: {coverage:.2%}")
    print(f"Target coverage:    {predictor.confidence_level:.2%}")
    print(f"Calibration error:  {abs(coverage - predictor.confidence_level):.2%}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation for Deployment")
    print("=" * 70)
    print(f"Mean prediction width: {(upper_pred - lower_pred).mean():.0f} kg/ha")
    print(f"Relative uncertainty:  {((upper_pred - lower_pred).mean() / mean_pred.mean() * 100):.1f}%")
    print("\nUse case:")
    print("- Policy planning: Use mean prediction")
    print("- Risk assessment: Use lower bound (conservative)")
    print("- Subsidy allocation: Flag high-uncertainty districts (wide intervals)")
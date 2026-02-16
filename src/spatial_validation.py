"""
Spatial cross-validation for crop yield models.

Tests model generalization to unseen geographic regions via leave-one-state-out.
Validates that the model captures agricultural processes, not just memorized patterns.

Usage:
    from spatial_validation import SpatialCV
    
    cv = SpatialCV(states=['Punjab', 'Maharashtra', ...])
    results = cv.leave_one_state_out(X, y, state_labels, model)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Callable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone
import warnings


class SpatialCV:
    """
    Spatial cross-validation for agricultural yield prediction.
    
    Implements leave-one-state-out to test spatial generalization.
    Critical for validating models won't fail when deployed to new regions.
    """
    
    def __init__(self, states: List[str]):
        """
        Args:
            states: List of state names in dataset
        """
        self.states = states
        self.results = None
    
    def leave_one_state_out(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           state_labels: np.ndarray,
                           model: Any,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Perform leave-one-state-out cross-validation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            state_labels: State assignment for each sample (n_samples,)
            model: Scikit-learn compatible model
            verbose: Print progress
        
        Returns:
            DataFrame with per-state performance metrics
        """
        results = []
        
        for held_out_state in self.states:
            if verbose:
                print(f"Evaluating: Hold out {held_out_state}...", end=" ")
            
            # Split data
            train_mask = state_labels != held_out_state
            test_mask = state_labels == held_out_state
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            if len(X_test) < 10:
                warnings.warn(f"Skipping {held_out_state}: insufficient samples ({len(X_test)})")
                continue
            
            # Train fresh model
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            
            # Predict
            y_pred = model_copy.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Relative error
            mean_yield = y_test.mean()
            relative_rmse = (rmse / mean_yield) * 100
            
            results.append({
                'state': held_out_state,
                'n_test': len(y_test),
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mean_yield': mean_yield,
                'relative_rmse_%': relative_rmse
            })
            
            if verbose:
                print(f"RMSE={rmse:.1f}, R²={r2:.3f}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def summarize(self) -> Dict[str, float]:
        """
        Aggregate statistics across all held-out states.
        
        Returns:
            Dict with mean/median/std of metrics
        """
        if self.results is None:
            raise ValueError("Run leave_one_state_out() first")
        
        return {
            'mean_rmse': self.results['rmse'].mean(),
            'median_rmse': self.results['rmse'].median(),
            'std_rmse': self.results['rmse'].std(),
            'mean_r2': self.results['r2'].mean(),
            'min_r2': self.results['r2'].min(),
            'max_r2': self.results['r2'].max(),
            'worst_state': self.results.loc[self.results['rmse'].idxmax(), 'state'],
            'best_state': self.results.loc[self.results['rmse'].idxmin(), 'state']
        }
    
    def identify_vulnerable_regions(self, rmse_threshold: float = 700) -> List[str]:
        """
        Identify states where model performance degrades significantly.
        
        Default threshold of 700 kg/ha represents ~22% relative error on 3,200 kg/ha
        average yield, which exceeds typical policy planning tolerance (~15-18%).
        
        Threshold is configurable for different use cases:
        - Policy planning: 650-700 kg/ha (~20% relative)
        - Risk assessment: 800+ kg/ha (>25% relative)
        - Farm-level decisions: 500 kg/ha (~15% relative)
        
        Args:
            rmse_threshold: RMSE above which state is flagged (kg/ha, default 700)
        
        Returns:
            List of state names with poor generalization
        """
        if self.results is None:
            raise ValueError("Run leave_one_state_out() first")
        
        vulnerable = self.results[self.results['rmse'] > rmse_threshold]
        return vulnerable['state'].tolist()


def compare_temporal_vs_spatial(temporal_r2: float, 
                                spatial_r2: float,
                                verbose: bool = True) -> str:
    """
    Interpret difference between temporal and spatial validation performance.
    
    Thresholds based on learning curve analysis and practical deployment considerations:
    - <0.05 delta: Strong generalization (model learns universal processes)
    - 0.05-0.15 delta: Moderate dependence (acceptable for regional deployment)
    - >0.15 delta: Weak generalization (requires regional calibration)
    
    These are heuristic guidelines, not validated classification criteria.
    
    Args:
        temporal_r2: R² from temporal holdout (2016-17)
        spatial_r2: Mean R² from leave-one-state-out
        verbose: Print interpretation
    
    Returns:
        Interpretation string
    """
    delta = temporal_r2 - spatial_r2
    
    if delta < 0.05:
        interpretation = "Strong spatial generalization - model captures universal crop-climate processes"
    elif delta < 0.15:
        interpretation = "Moderate spatial dependence - acceptable for district-level policy tools with regional calibration"
    else:
        interpretation = "Weak spatial generalization - requires state-specific retraining before deployment"
    
    if verbose:
        print(f"Temporal R²: {temporal_r2:.3f}")
        print(f"Spatial R²:  {spatial_r2:.3f}")
        print(f"Delta:       {delta:.3f}")
        print(f"\n{interpretation}")
    
    return interpretation


# Example usage with synthetic data
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    
    print("=" * 70)
    print("Spatial Cross-Validation Demo")
    print("=" * 70)
    
    # Synthetic dataset simulating 300 districts across 5 states
    np.random.seed(42)
    n_samples = 1440  # 300 districts × 3 crops × 1.6 years avg
    n_features = 14
    
    states = ['Punjab', 'Maharashtra', 'Rajasthan', 'Karnataka', 'Tamil Nadu']
    state_labels = np.random.choice(states, size=n_samples)
    
    X = np.random.randn(n_samples, n_features)
    # Add state-specific bias to simulate regional patterns
    state_biases = {'Punjab': 500, 'Maharashtra': 0, 'Rajasthan': -300, 
                   'Karnataka': 100, 'Tamil Nadu': 200}
    y = 2500 + X[:, 0] * 300 + X[:, 1] * 200  # Base yield + GDD + NDVI
    for i, state in enumerate(state_labels):
        y[i] += state_biases[state]
    y += np.random.randn(n_samples) * 150  # Noise
    
    # Run spatial CV
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    cv = SpatialCV(states=states)
    
    print("\nRunning leave-one-state-out validation...")
    print("-" * 70)
    results = cv.leave_one_state_out(X, y, state_labels, model, verbose=True)
    
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(results.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Aggregate Statistics")
    print("=" * 70)
    summary = cv.summarize()
    for key, value in summary.items():
        if isinstance(value, str):
            print(f"{key:20s}: {value}")
        else:
            print(f"{key:20s}: {value:.2f}")
    
    # Compare with temporal validation
    print("\n" + "=" * 70)
    print("Generalization Analysis")
    print("=" * 70)
    temporal_r2 = 0.78  # From README results
    compare_temporal_vs_spatial(temporal_r2, summary['mean_r2'])
    
    # Vulnerable regions
    vulnerable = cv.identify_vulnerable_regions(rmse_threshold=650)
    if vulnerable:
        print(f"\nVulnerable regions (RMSE > 650): {', '.join(vulnerable)}")
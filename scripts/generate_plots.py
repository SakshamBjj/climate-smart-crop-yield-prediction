"""
Reproducible Visualization Generator
Reads metrics from metrics.json and generates publication-quality plots

Usage: python generate_plots.py
Output: PNG files in ../assets/ directory
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
METRICS_FILE = Path(__file__).parent / "metrics.json"
OUTPUT_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR.mkdir(exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

def load_metrics():
    """Load metrics from JSON file"""
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}\n"
            f"Please create metrics.json with model performance data"
        )
    
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)

def plot_model_comparison_rmse(data):
    """Plot RMSE comparison across models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data['model_metrics'])
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(df['Model'], df['RMSE'], color=colors, 
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 15,
                f'{height:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    ax.set_ylabel('RMSE (kg/ha)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Model Comparison: RMSE (Lower is Better)', 
                 fontweight='bold', pad=20)
    ax.axhline(y=600, color='gray', linestyle='--', alpha=0.5, 
               label='Target: 600 kg/ha')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_comparison_rmse.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_model_comparison_r2(data):
    """Plot R² comparison across models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data['model_metrics'])
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(df['Model'], df['R2'], color=colors, 
                   edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Model Comparison: R² Score (Higher is Better)', 
                 fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, 
               label='Target: 0.75')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_comparison_r2.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_learning_curves(data):
    """Plot DeepFusionNN learning curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    learning_data = data['learning_curves']
    samples = np.array(learning_data['training_samples'])
    rmse = np.array(learning_data['deepfusion_val_rmse'])
    r2 = np.array(learning_data['deepfusion_val_r2'])
    rf_rmse = learning_data['rf_baseline_rmse']
    rf_r2 = learning_data['rf_baseline_r2']
    
    # RMSE plot
    ax1.plot(samples, rmse, 'o-', color='#e74c3c', 
             linewidth=2.5, markersize=8, label='DeepFusionNN')
    ax1.axhline(y=rf_rmse, color='#2ecc71', linestyle='--', 
                linewidth=2, label='Random Forest')
    ax1.fill_between(samples, rmse - 30, rmse + 30, 
                      alpha=0.2, color='#e74c3c')
    
    ax1.annotate('Plateau at\n~3,000 samples', 
                 xy=(3000, 712), xytext=(4000, 850),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax1.set_xlabel('Training Samples', fontweight='bold')
    ax1.set_ylabel('Validation RMSE (kg/ha)', fontweight='bold')
    ax1.set_title('Learning Curve: RMSE', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # R² plot
    ax2.plot(samples, r2, 'o-', color='#e74c3c', 
             linewidth=2.5, markersize=8, label='DeepFusionNN')
    ax2.axhline(y=rf_r2, color='#2ecc71', linestyle='--', 
                linewidth=2, label='Random Forest')
    ax2.fill_between(samples, r2 - 0.03, r2 + 0.03, 
                      alpha=0.2, color='#e74c3c')
    
    ax2.set_xlabel('Training Samples', fontweight='bold')
    ax2.set_ylabel('Validation R²', fontweight='bold')
    ax2.set_title('Learning Curve: R²', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "learning_curves_deepfusion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_feature_importance(data):
    """Plot feature importance from Random Forest"""
    fig, ax = plt.subplots(figsize=(12, 7))
    df = pd.DataFrame(data['feature_importance'])
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(df)))
    bars = ax.barh(df['Feature'], df['Importance'], 
                   color=colors, edgecolor='black', linewidth=1.2)
    
    for i, (bar, imp) in enumerate(zip(bars, df['Importance'])):
        ax.text(imp + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{imp}%', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Feature Importance (%)', fontweight='bold')
    ax.set_title('Top 10 Predictive Features\n(GDD + NDVI = 50% of predictive power)', 
                 fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_importance_rf.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_temporal_performance(data):
    """Plot RMSE by year"""
    fig, ax = plt.subplots(figsize=(14, 6))
    df = pd.DataFrame(data['temporal_data'])
    
    colors = ['#2ecc71' if 'Normal' in c or 'Good' in c else '#e74c3c' 
              for c in df['Climate']]
    
    bars = ax.bar(df['Year'], df['RMSE'], color=colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    
    for bar, rmse, climate in zip(bars, df['RMSE'], df['Climate']):
        height = bar.get_height()
        label = f'{rmse:.0f}'
        if '*' in climate:
            label += '\n(Drought)'
        ax.text(bar.get_x() + bar.get_width()/2., height + 15,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('RMSE (kg/ha)', fontweight='bold')
    ax.set_title('Temporal Performance: RMSE by Year\n(Drought years: +25-30% error)', 
                 fontweight='bold', pad=20)
    ax.axhline(y=650, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Normal/Good'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Drought')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "temporal_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_crop_specific(data):
    """Plot crop-specific performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    df = pd.DataFrame(data['crop_performance'])
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    # RMSE
    bars1 = ax1.bar(df['Crop'], df['RMSE'], color=colors, 
                     edgecolor='black', linewidth=1.2)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 15,
                 f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('RMSE (kg/ha)', fontweight='bold')
    ax1.set_xlabel('Crop', fontweight='bold')
    ax1.set_title('Crop-Specific: RMSE', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # R²
    bars2 = ax2.bar(df['Crop'], df['R2'], color=colors, 
                     edgecolor='black', linewidth=1.2)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('R² Score', fontweight='bold')
    ax2.set_xlabel('Crop', fontweight='bold')
    ax2.set_title('Crop-Specific: R²', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "crop_specific_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    """Generate all plots from metrics.json"""
    print("\n" + "="*70)
    print("REPRODUCIBLE VISUALIZATION GENERATION")
    print("="*70 + "\n")
    
    try:
        data = load_metrics()
        print(f"✓ Loaded metrics from: {METRICS_FILE}")
        print(f"✓ Output directory: {OUTPUT_DIR}\n")
        
        plot_model_comparison_rmse(data)
        plot_model_comparison_r2(data)
        plot_learning_curves(data)
        plot_feature_importance(data)
        plot_temporal_performance(data)
        plot_crop_specific(data)
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"\nGenerated 6 plots in: {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure metrics.json exists with the correct structure.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

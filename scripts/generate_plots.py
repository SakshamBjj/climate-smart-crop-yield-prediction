"""
Reproducible Visualization Generator
Reads metrics from metrics.json and generates publication-quality plots

Usage: python generate_plots.py
Output: PNG files in ../assets/ directory
"""

import matplotlib
matplotlib.use('Agg')  

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
METRICS_FILE = Path(__file__).parent / "metrics.json"
OUTPUT_DIR = Path(__file__).parent / "assets"
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

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 15,
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
    print(f"  Saved: {output_path}")
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
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.015,
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
    print(f"  Saved: {output_path}")
    plt.close()


def plot_learning_curves(data):
    """Plot DeepFusionNN learning curves with data-derived annotation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    lc = data['learning_curves']
    samples = np.array(lc['training_samples'])
    rmse = np.array(lc['deepfusion_val_rmse'])
    r2 = np.array(lc['deepfusion_val_r2'])
    rf_rmse = lc['rf_baseline_rmse']
    rf_r2 = lc['rf_baseline_r2']

    # RMSE plot
    ax1.plot(samples, rmse, 'o-', color='#e74c3c',
             linewidth=2.5, markersize=8, label='DeepFusionNN')
    ax1.axhline(y=rf_rmse, color='#2ecc71', linestyle='--',
                linewidth=2, label=f'Random Forest ({rf_rmse:.0f} kg/ha)')
    ax1.fill_between(samples, rmse - 30, rmse + 30,
                     alpha=0.2, color='#e74c3c')

    # Derive plateau from data — smallest absolute gradient (excluding last point)
    gradients = np.abs(np.gradient(rmse, samples))
    plateau_idx = int(np.argmin(gradients[:-1]))
    ax1.annotate(
        f'Plateau at\n~{samples[plateau_idx]:,} samples',
        xy=(samples[plateau_idx], rmse[plateau_idx]),
        xytext=(samples[plateau_idx] + 800, rmse[plateau_idx] + 130),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    ax1.set_xlabel('Training Samples', fontweight='bold')
    ax1.set_ylabel('Validation RMSE (kg/ha)', fontweight='bold')
    ax1.set_title('Learning Curve: RMSE', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # R² plot
    ax2.plot(samples, r2, 'o-', color='#e74c3c',
             linewidth=2.5, markersize=8, label='DeepFusionNN')
    ax2.axhline(y=rf_r2, color='#2ecc71', linestyle='--',
                linewidth=2, label=f'Random Forest (R²={rf_r2:.2f})')
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
    print(f"  Saved: {output_path}")
    plt.close()


def plot_feature_importance(data):
    """Plot feature importance from Random Forest"""
    fig, ax = plt.subplots(figsize=(12, 7))
    df = pd.DataFrame(data['feature_importance'])

    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(df)))
    bars = ax.barh(df['Feature'], df['Importance'],
                   color=colors, edgecolor='black', linewidth=1.2)

    for bar, imp in zip(bars, df['Importance']):
        ax.text(imp + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{imp}%', va='center', fontweight='bold', fontsize=10)

    ax.set_xlabel('Feature Importance (%)', fontweight='bold')
    ax.set_title('Top 10 Predictive Features\n(GDD + NDVI = 50% of predictive power)',
                 fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "feature_importance_rf.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_temporal_performance(data):
    """
    Plot RMSE by year — Option 1: complete methodology story.

    Two evaluation approaches shown together:
    - Solid bars (2008-2015): OOB training-year evaluation
    - Hatched bars (2016-2017): OOB test-year context
    - Dotted line + annotation: strict temporal holdout headline RMSE (578)

    Per-year OOB and pooled holdout RMSE differ by design — different methodologies.
    Chart makes this explicit rather than implying a single consistent evaluation.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    df = pd.DataFrame(data['temporal_data'])

    # Draw bars — use is_drought boolean from JSON (not brittle string matching)
    bar_artists = []
    for _, row in df.iterrows():
        is_test = (row['eval_type'] == 'oob_test_context')
        color = '#e74c3c' if row['is_drought'] else '#2ecc71'
        bar = ax.bar(
            row['Year'], row['RMSE'],
            color=color,
            edgecolor='black', linewidth=1.4,
            alpha=0.85 if not is_test else 0.55,
            hatch='//' if is_test else '',
            width=0.7
        )
        bar_artists.append((bar[0], row))

    # Value labels
    for bar, row in bar_artists:
        height = bar.get_height()
        label = f"{int(height)}"
        if row['is_drought']:
            label += '\n[Drought]'
        ax.text(
            bar.get_x() + bar.get_width() / 2., height + 18,
            label, ha='center', va='bottom', fontsize=8.5, fontweight='bold'
        )

    # Train / Test divider
    ax.axvline(x=2015.5, color='#2c3e50', linestyle='--',
               linewidth=2.0, alpha=0.75, zorder=5)
    ymax = df['RMSE'].max() * 1.28
    ax.text(2013.8, ymax * 0.98, 'Training period\n(OOB evaluation)',
            ha='center', va='top', fontsize=9, color='#555',
            style='italic')
    ax.text(2016.5, ymax * 0.98, 'Test period\n(OOB context)',
            ha='center', va='top', fontsize=9, color='#555',
            style='italic')

    # Headline held-out RMSE — annotated separately and explicitly
    headline_rmse = data['learning_curves']['headline_held_out_rmse']
    ax.axhline(y=headline_rmse, color='#2c3e50', linestyle=':',
               linewidth=1.8, alpha=0.65, zorder=4)
    ax.annotate(
        f'Strict temporal holdout\n'
        f'(train 2008–2015 | test 2016–2017)\n'
        f'RF RMSE = {headline_rmse:.0f} kg/ha',
        xy=(2016.5, headline_rmse),
        xytext=(2012.5, headline_rmse + 155),
        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.8),
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                  edgecolor='#2c3e50', linewidth=1.5),
        zorder=10
    )

    # Axis labels and title
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('RMSE (kg/ha)', fontweight='bold')
    ax.set_title(
        'Temporal Performance: Per-Year RMSE (OOB Evaluation)\n'
        'Drought years: +25–30% error  |  '
        'Strict holdout RMSE annotated separately',
        fontweight='bold', pad=20
    )
    ax.set_xticks(df['Year'])
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black',
                       label='Normal / Good year (OOB)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black',
                       label='Drought year (OOB)'),
        mpatches.Patch(facecolor='#aab7b8', edgecolor='black', hatch='//',
                       label='Test years 2016–2017 (OOB context)'),
        mpatches.Patch(facecolor='#ecf0f1', edgecolor='#2c3e50',
                       label=f'Strict holdout RMSE: {headline_rmse:.0f} kg/ha'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.92)

    # Methodology footnote
    fig.text(
        0.5, -0.04,
        'Per-year bars use OOB evaluation: each sample predicted by trees that '
        'did not train on it — all years contribute to the forest.\n'
        'Headline RMSE uses strict temporal holdout: model trained on 2008–2015 '
        'only, zero exposure to 2016–2017 during training. These are different '
        'methodologies and produce different numbers by design.',
        ha='center', va='top', fontsize=8, color='#666',
        style='italic'
    )

    plt.tight_layout()
    output_path = OUTPUT_DIR / "temporal_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_crop_specific(data):
    """Plot crop-specific performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    df = pd.DataFrame(data['crop_performance'])
    colors = ['#3498db', '#2ecc71', '#f39c12']

    bars1 = ax1.bar(df['Crop'], df['RMSE'], color=colors,
                    edgecolor='black', linewidth=1.2)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 15,
                 f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('RMSE (kg/ha)', fontweight='bold')
    ax1.set_xlabel('Crop', fontweight='bold')
    ax1.set_title('Crop-Specific: RMSE', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(df['Crop'], df['R2'], color=colors,
                    edgecolor='black', linewidth=1.2)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.015,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('R² Score', fontweight='bold')
    ax2.set_xlabel('Crop', fontweight='bold')
    ax2.set_title('Crop-Specific: R²', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "crop_specific_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots from metrics.json"""
    print("\n" + "=" * 70)
    print("REPRODUCIBLE VISUALIZATION GENERATION")
    print("=" * 70 + "\n")

    try:
        data = load_metrics()
        print(f"Loaded: {METRICS_FILE}")
        print(f"Output: {OUTPUT_DIR}\n")

        plot_model_comparison_rmse(data)
        plot_model_comparison_r2(data)
        plot_learning_curves(data)
        plot_feature_importance(data)
        plot_temporal_performance(data)
        plot_crop_specific(data)

        print("\n" + "=" * 70)
        print("ALL 6 PLOTS GENERATED")
        print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
"""
Model Comparison Script for Stage 1
Compares Centralized vs Decentralized (Federated) models

Usage:
    python evaluation/compare_models.py

Requires both models to be evaluated first:
    python evaluation/evaluate_centralized.py
    python evaluation/evaluate_decentralized.py
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_metrics(results_dir: str, model_name: str) -> dict:
    """Load metrics from JSON file."""
    metrics_path = Path(results_dir) / f'{model_name}_metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        return json.load(f)


def create_comparison_table(centralized_metrics: dict, decentralized_metrics: dict) -> pd.DataFrame:
    """Create comparison DataFrame."""
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1-Score', 'f1'),
        ('Sensitivity (TPR)', 'sensitivity'),
        ('Specificity (TNR)', 'specificity'),
        ('AUC-ROC', 'auc_roc'),
        ('AUC-PR', 'auc_pr'),
        ('Precision (Macro)', 'precision_macro'),
        ('Recall (Macro)', 'recall_macro'),
        ('F1 (Macro)', 'f1_macro'),
        ('Precision (Weighted)', 'precision_weighted'),
        ('Recall (Weighted)', 'recall_weighted'),
        ('F1 (Weighted)', 'f1_weighted'),
    ]

    data = []
    for display_name, key in metrics_to_compare:
        cent_val = centralized_metrics.get(key, 0)
        decent_val = decentralized_metrics.get(key, 0)

        if cent_val is None:
            cent_val = 0
        if decent_val is None:
            decent_val = 0

        diff = decent_val - cent_val
        diff_pct = (diff / cent_val * 100) if cent_val != 0 else 0

        data.append({
            'Metric': display_name,
            'Centralized': cent_val,
            'Decentralized': decent_val,
            'Difference': diff,
            'Diff (%)': diff_pct
        })

    return pd.DataFrame(data)


def plot_metrics_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """Plot side-by-side comparison of key metrics."""
    # Select key metrics for visualization
    key_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity (TPR)',
                   'Specificity (TNR)', 'AUC-ROC', 'AUC-PR']

    plot_df = comparison_df[comparison_df['Metric'].isin(key_metrics)].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(plot_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, plot_df['Centralized'], width, label='Centralized',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, plot_df['Decentralized'], width, label='Decentralized (Federated)',
                   color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Centralized vs Decentralized Model Comparison\nCOVIDx CXR-4 Classification', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Metric'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_confusion_matrix_comparison(cent_cm: list, decent_cm: list,
                                     class_names: list, save_path: str = None):
    """Plot confusion matrices side by side."""
    import seaborn as sns

    cent_cm = np.array(cent_cm)
    decent_cm = np.array(decent_cm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Centralized
    sns.heatmap(cent_cm.astype('float') / cent_cm.sum(axis=1)[:, np.newaxis],
                annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('Centralized Model\nNormalized Confusion Matrix', fontsize=12)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Decentralized
    sns.heatmap(decent_cm.astype('float') / decent_cm.sum(axis=1)[:, np.newaxis],
                annot=True, fmt='.2%', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('Decentralized (Federated) Model\nNormalized Confusion Matrix', fontsize=12)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix comparison to {save_path}")

    return fig


def generate_comparison_report(comparison_df: pd.DataFrame,
                               cent_metrics: dict,
                               decent_metrics: dict,
                               output_path: str):
    """Generate text comparison report."""

    lines = [
        "=" * 80,
        "STAGE 1 MODEL COMPARISON REPORT",
        "Centralized vs Decentralized (Federated) Learning",
        "COVIDx CXR-4 Binary Classification",
        "=" * 80,
        f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nCentralized Model Evaluation: {cent_metrics.get('evaluation_timestamp', 'N/A')}",
        f"Decentralized Model Evaluation: {decent_metrics.get('evaluation_timestamp', 'N/A')}",
        f"\nTest Samples: {cent_metrics.get('num_samples', 'N/A')}",
        "",
        "-" * 80,
        "METRICS COMPARISON TABLE",
        "-" * 80,
        "",
    ]

    # Format table
    lines.append(f"{'Metric':<25} {'Centralized':>12} {'Decentralized':>14} {'Diff':>10} {'Diff %':>10}")
    lines.append("-" * 75)

    for _, row in comparison_df.iterrows():
        cent_val = row['Centralized']
        decent_val = row['Decentralized']
        diff = row['Difference']
        diff_pct = row['Diff (%)']

        # Format values
        if cent_val < 10:  # Likely a ratio
            cent_str = f"{cent_val:.4f}"
            decent_str = f"{decent_val:.4f}"
            diff_str = f"{diff:+.4f}"
        else:
            cent_str = f"{cent_val:.0f}"
            decent_str = f"{decent_val:.0f}"
            diff_str = f"{diff:+.0f}"

        lines.append(f"{row['Metric']:<25} {cent_str:>12} {decent_str:>14} {diff_str:>10} {diff_pct:>+9.2f}%")

    lines.extend([
        "",
        "-" * 80,
        "CONFUSION MATRIX BREAKDOWN",
        "-" * 80,
        "",
        "Centralized Model:",
        f"  True Positives (TP):   {cent_metrics.get('true_positives', 0)}",
        f"  True Negatives (TN):   {cent_metrics.get('true_negatives', 0)}",
        f"  False Positives (FP):  {cent_metrics.get('false_positives', 0)}",
        f"  False Negatives (FN):  {cent_metrics.get('false_negatives', 0)}",
        "",
        "Decentralized Model:",
        f"  True Positives (TP):   {decent_metrics.get('true_positives', 0)}",
        f"  True Negatives (TN):   {decent_metrics.get('true_negatives', 0)}",
        f"  False Positives (FP):  {decent_metrics.get('false_positives', 0)}",
        f"  False Negatives (FN):  {decent_metrics.get('false_negatives', 0)}",
        "",
        "-" * 80,
        "KEY OBSERVATIONS",
        "-" * 80,
        "",
    ])

    # Add observations
    acc_diff = decent_metrics.get('accuracy', 0) - cent_metrics.get('accuracy', 0)
    f1_diff = decent_metrics.get('f1', 0) - cent_metrics.get('f1', 0)
    auc_diff = (decent_metrics.get('auc_roc', 0) or 0) - (cent_metrics.get('auc_roc', 0) or 0)

    if acc_diff >= 0:
        lines.append(f"1. Accuracy: Decentralized model {'matches' if acc_diff == 0 else 'outperforms'} centralized by {abs(acc_diff)*100:.2f}%")
    else:
        lines.append(f"1. Accuracy: Centralized model outperforms decentralized by {abs(acc_diff)*100:.2f}%")

    if f1_diff >= 0:
        lines.append(f"2. F1-Score: Decentralized model {'matches' if f1_diff == 0 else 'outperforms'} centralized by {abs(f1_diff)*100:.2f}%")
    else:
        lines.append(f"2. F1-Score: Centralized model outperforms decentralized by {abs(f1_diff)*100:.2f}%")

    if auc_diff >= 0:
        lines.append(f"3. AUC-ROC: Decentralized model {'matches' if auc_diff == 0 else 'outperforms'} centralized by {abs(auc_diff):.4f}")
    else:
        lines.append(f"3. AUC-ROC: Centralized model outperforms decentralized by {abs(auc_diff):.4f}")

    # Add interpretation
    lines.extend([
        "",
        "-" * 80,
        "INTERPRETATION FOR STAGE 1",
        "-" * 80,
        "",
        "This comparison shows the performance trade-off between:",
        "  - Centralized training: All data aggregated on a single server",
        "  - Decentralized (Federated) training: Data remains on clients, only",
        "    model updates are shared",
        "",
        "For medical imaging (COVID detection), federated learning offers:",
        "  - Privacy preservation: Patient data never leaves the hospital",
        "  - Regulatory compliance: HIPAA/GDPR considerations",
        "  - Multi-center collaboration: Models trained across institutions",
        "",
        "Expected trade-offs:",
        "  - Federated models may show slight performance degradation due to",
        "    non-IID data distribution across clients",
        "  - Communication overhead and convergence considerations",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    report_text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"Saved comparison report to {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Compare centralized vs decentralized models')
    parser.add_argument('--centralized_dir', type=str,
                        default='evaluation/results/stage1/centralized',
                        help='Directory with centralized results')
    parser.add_argument('--decentralized_dir', type=str,
                        default='evaluation/results/stage1/decentralized',
                        help='Directory with decentralized results')
    parser.add_argument('--output_dir', type=str,
                        default='evaluation/results/stage1/comparison',
                        help='Output directory for comparison')
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")

    # Load metrics
    try:
        cent_metrics = load_metrics(args.centralized_dir, 'centralized')
        print(f"  Loaded centralized metrics from {args.centralized_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run: python evaluation/evaluate_centralized.py first")
        return

    try:
        decent_metrics = load_metrics(args.decentralized_dir, 'decentralized')
        print(f"  Loaded decentralized metrics from {args.decentralized_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run: python evaluation/evaluate_decentralized.py first")
        return

    print("\nGenerating comparison...")

    # Create comparison table
    comparison_df = create_comparison_table(cent_metrics, decent_metrics)

    # Save comparison table as CSV
    comparison_df.to_csv(output_path / 'metrics_comparison.csv', index=False)
    print(f"Saved comparison table to {output_path / 'metrics_comparison.csv'}")

    # Print table
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Plot comparison
    plot_metrics_comparison(comparison_df, str(output_path / 'metrics_comparison.png'))

    # Plot confusion matrices
    if 'confusion_matrix' in cent_metrics and 'confusion_matrix' in decent_metrics:
        plot_confusion_matrix_comparison(
            cent_metrics['confusion_matrix'],
            decent_metrics['confusion_matrix'],
            ['Negative', 'Positive'],
            str(output_path / 'confusion_matrix_comparison.png')
        )

    # Generate text report
    report = generate_comparison_report(
        comparison_df, cent_metrics, decent_metrics,
        str(output_path / 'comparison_report.txt')
    )

    print("\n" + report)

    # Save metrics as combined JSON
    combined = {
        'centralized': cent_metrics,
        'decentralized': decent_metrics,
        'comparison_timestamp': datetime.now().isoformat()
    }
    with open(output_path / 'combined_metrics.json', 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPARISON COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

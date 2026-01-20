"""
Evaluation Metrics Module for COVIDx CXR-4 Classification
Stage 1: Baseline Evaluation

Metrics included:
- Classification: Accuracy, Precision, Recall, F1-score
- Medical-specific: Sensitivity, Specificity, AUC-ROC, AUC-PR
- Per-class metrics for COVID positive/negative
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    auc
)
from typing import Dict, List, Tuple, Optional


class MetricsCalculator:
    """
    Comprehensive metrics calculator for binary COVID classification.
    """
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Negative', 'Positive']
        self.num_classes = len(self.class_names)

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: np.ndarray = None) -> Dict:
        """
        Calculate all relevant metrics for binary classification.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities for positive class (optional)

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Macro and weighted averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Sensitivity and Specificity (critical for medical)
        sens_spec = self.calculate_sensitivity_specificity(y_true, y_pred)
        metrics.update(sens_spec)

        # AUC metrics (if probabilities provided)
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            except Exception as e:
                print(f"Could not calculate AUC: {e}")
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)

            metrics[f'{class_name}_precision'] = precision_score(binary_true, binary_pred, zero_division=0)
            metrics[f'{class_name}_recall'] = recall_score(binary_true, binary_pred, zero_division=0)
            metrics[f'{class_name}_f1'] = f1_score(binary_true, binary_pred, zero_division=0)

        return metrics

    def calculate_sensitivity_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate sensitivity (recall) and specificity for binary classification.

        For COVID detection:
        - Sensitivity (True Positive Rate): Ability to detect COVID cases
        - Specificity (True Negative Rate): Ability to correctly identify non-COVID cases
        """
        cm = confusion_matrix(y_true, y_pred)

        # For binary: cm = [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            return {'sensitivity': 0, 'specificity': 0}

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'sensitivity': sensitivity,  # Same as recall for positive class
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get sklearn classification report as string."""
        return classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)


class MetricsVisualizer:
    """
    Visualization utilities for evaluation metrics.
    """
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Negative', 'Positive']

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None,
                               normalize: bool = True, title: str = None) -> plt.Figure:
        """Plot confusion matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            default_title = 'Normalized Confusion Matrix'
        else:
            cm_display = cm
            fmt = 'd'
            default_title = 'Confusion Matrix'

        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax, cbar=True)

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title or default_title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                        save_path: str = None, title: str = None) -> plt.Figure:
        """Plot ROC curve for binary classification."""
        fig, ax = plt.subplots(figsize=(8, 6))

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title(title or 'Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC curve to {save_path}")

        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                     save_path: str = None, title: str = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {pr_auc:.3f})')

        # Add baseline (random classifier)
        baseline = y_true.sum() / len(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title or 'Precision-Recall Curve', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved PR curve to {save_path}")

        return fig

    def plot_metrics_comparison(self, results_dict: Dict[str, Dict],
                                 metrics: List[str], save_path: str = None,
                                 title: str = None) -> plt.Figure:
        """
        Plot comparison of multiple models across metrics.

        Args:
            results_dict: {model_name: {metric: value}}
            metrics: List of metric names to compare
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics))
        width = 0.8 / len(results_dict)

        colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

        for i, (model_name, results) in enumerate(results_dict.items()):
            values = [results.get(m, 0) for m in metrics]
            offset = (i - len(results_dict)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name, color=colors[i])

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title or 'Model Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")

        return fig

    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray = None,
                                 save_path: str = None) -> plt.Figure:
        """Plot class distribution in ground truth and predictions."""
        fig, axes = plt.subplots(1, 2 if y_pred is not None else 1, figsize=(12, 5))

        if y_pred is None:
            axes = [axes]

        # Ground truth distribution
        unique, counts = np.unique(y_true, return_counts=True)
        colors = ['#2ecc71', '#e74c3c']  # Green for negative, red for positive
        axes[0].bar([self.class_names[i] for i in unique], counts, color=colors)
        axes[0].set_title('Ground Truth Distribution', fontsize=14)
        axes[0].set_ylabel('Count', fontsize=12)
        for i, (u, c) in enumerate(zip(unique, counts)):
            axes[0].text(i, c + max(counts)*0.02, str(c), ha='center', fontsize=11)

        if y_pred is not None:
            # Prediction distribution
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            axes[1].bar([self.class_names[i] for i in unique_pred], counts_pred, color=colors)
            axes[1].set_title('Prediction Distribution', fontsize=14)
            axes[1].set_ylabel('Count', fontsize=12)
            for i, (u, c) in enumerate(zip(unique_pred, counts_pred)):
                axes[1].text(i, c + max(counts_pred)*0.02, str(c), ha='center', fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved distribution plot to {save_path}")

        return fig


def format_metrics_table(metrics: Dict, title: str = "Evaluation Results") -> str:
    """Format metrics as a readable table string."""
    lines = [
        f"\n{'='*60}",
        f"{title:^60}",
        f"{'='*60}",
        "",
        "Classification Metrics:",
        f"  Accuracy:              {metrics.get('accuracy', 0)*100:.2f}%",
        f"  Precision:             {metrics.get('precision', 0)*100:.2f}%",
        f"  Recall:                {metrics.get('recall', 0)*100:.2f}%",
        f"  F1-Score:              {metrics.get('f1', 0)*100:.2f}%",
        "",
        "Medical Metrics:",
        f"  Sensitivity (TPR):     {metrics.get('sensitivity', 0)*100:.2f}%",
        f"  Specificity (TNR):     {metrics.get('specificity', 0)*100:.2f}%",
    ]

    if metrics.get('auc_roc') is not None:
        lines.extend([
            f"  AUC-ROC:               {metrics.get('auc_roc', 0):.4f}",
            f"  AUC-PR:                {metrics.get('auc_pr', 0):.4f}",
        ])

    lines.extend([
        "",
        "Confusion Matrix Breakdown:",
        f"  True Positives (TP):   {metrics.get('true_positives', 0)}",
        f"  True Negatives (TN):   {metrics.get('true_negatives', 0)}",
        f"  False Positives (FP):  {metrics.get('false_positives', 0)}",
        f"  False Negatives (FN):  {metrics.get('false_negatives', 0)}",
        "",
        f"{'='*60}",
    ])

    return "\n".join(lines)

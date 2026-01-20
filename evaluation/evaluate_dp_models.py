"""
Evaluate Differential Privacy Models for Report and Presentation
Tests models with epsilon = 5.0, 10.0, and ∞ (no DP)

Generates:
- Detailed metrics (accuracy, precision, recall, F1, AUC-ROC)
- Confusion matrices
- Classification reports  
- Privacy-utility trade-off analysis
- Visualizations for presentation
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model_side.models.cnn_model import SimpleCNN
from model_side.data.data_loader_enhanced import get_global_test_loader


class DPModelEvaluator:
    """Evaluates DP-trained models and generates comprehensive reports."""
    
    def __init__(self, output_dir='evaluation/results/dp_evaluation'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = ['Negative', 'Positive']
        
        print(f"Using device: {self.device}")
        print(f"Results will be saved to: {self.output_dir}")
    
    def load_model(self, model_path, epsilon):
        """Load a trained DP model."""
        model = SimpleCNN(num_classes=2)
        
        # For DP models (not inf), we need to fix the model architecture
        if epsilon != float('inf'):
            from opacus.validators import ModuleValidator
            model = ModuleValidator.fix(model)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, model, test_loader, desc="Evaluating"):
        """Run inference and return predictions."""
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=desc):
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_labels.extend(labels.numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Calculate sensitivity (recall) and specificity
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC
        try:
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            else:
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            metrics['auc_roc'] = None
            metrics['auc_pr'] = None
        
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, epsilon_str, ax=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=self.class_names, yticklabels=self.class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (ε = {epsilon_str})')
        
        return ax
    
    def plot_roc_curve(self, y_true, y_prob, epsilon_str, ax=None):
        """Plot ROC curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f'ε = {epsilon_str} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        
        return ax
    
    def evaluate_single_model(self, model_path, epsilon, test_loader):
        """Evaluate a single model."""
        epsilon_str = '∞' if epsilon == float('inf') else str(epsilon)
        print(f"\n{'='*60}")
        print(f"Evaluating model with ε = {epsilon_str}")
        print('='*60)
        
        # Load model
        model = self.load_model(model_path, epsilon)
        
        # Get predictions
        y_true, y_pred, y_prob = self.predict(model, test_loader, f"Testing ε={epsilon_str}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        # Print key metrics
        print(f"\nKey Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Sensitivity: {metrics.get('sensitivity', 'N/A'):.4f}" if metrics.get('sensitivity') else "  Sensitivity: N/A")
        print(f"  Specificity: {metrics.get('specificity', 'N/A'):.4f}" if metrics.get('specificity') else "  Specificity: N/A")
        if metrics['auc_roc']:
            print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        
        return {
            'epsilon': epsilon if epsilon != float('inf') else 'inf',
            'epsilon_display': epsilon_str,
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def evaluate_all_models(self, epsilons=[5.0, 10.0, float('inf')], batch_size=32):
        """Evaluate all specified DP models."""
        print("\n" + "="*70)
        print("DIFFERENTIAL PRIVACY MODEL EVALUATION")
        print("="*70)
        
        # Load test data
        print("\nLoading global test set...")
        test_loader = get_global_test_loader(batch_size=batch_size)
        
        results = []
        all_predictions = {}
        
        for epsilon in epsilons:
            epsilon_str = 'inf' if epsilon == float('inf') else str(epsilon)
            model_path = f'models/dp_epsilon_{epsilon_str}.pth'
            
            if not os.path.exists(model_path):
                print(f"\nWARNING: Model not found at {model_path}, skipping...")
                continue
            
            result = self.evaluate_single_model(model_path, epsilon, test_loader)
            results.append(result)
            all_predictions[result['epsilon_display']] = result
        
        # Generate comparison visualizations
        self.generate_comparison_plots(results)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_comparison_plots(self, results):
        """Generate comparison visualizations."""
        n_models = len(results)
        if n_models == 0:
            return
        
        # 1. Confusion matrices comparison
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for ax, result in zip(axes, results):
            self.plot_confusion_matrix(
                result['y_true'], result['y_pred'], 
                result['epsilon_display'], ax
            )
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved confusion matrices to {self.output_dir / 'confusion_matrices_comparison.png'}")
        
        # 2. ROC curves comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, result in enumerate(results):
            y_true, y_prob = result['y_true'], result['y_prob']
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = result['metrics'].get('auc_roc', 0)
                ax.plot(fpr, tpr, color=colors[i % len(colors)], 
                       label=f'ε = {result["epsilon_display"]} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Privacy-Utility Trade-off', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curves to {self.output_dir / 'roc_curves_comparison.png'}")
        
        # 3. Metrics bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prepare data
        epsilons = [r['epsilon_display'] for r in results]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(epsilons))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = [r['metrics'][metric] for r in results]
            axes[0].bar(x + i*width, values, width, label=metric.capitalize())
        
        axes[0].set_xlabel('Epsilon (ε)')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Classification Metrics by Privacy Level')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels([f'ε = {e}' for e in epsilons])
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Sensitivity/Specificity comparison
        sens_values = [r['metrics'].get('sensitivity', 0) for r in results]
        spec_values = [r['metrics'].get('specificity', 0) for r in results]
        
        axes[1].bar(x - width/2, sens_values, width, label='Sensitivity', color='#2ca02c')
        axes[1].bar(x + width/2, spec_values, width, label='Specificity', color='#d62728')
        
        axes[1].set_xlabel('Epsilon (ε)')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Sensitivity & Specificity by Privacy Level')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'ε = {e}' for e in epsilons])
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics comparison to {self.output_dir / 'metrics_comparison.png'}")
        
        # 4. Privacy-Utility Trade-off plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        eps_numeric = []
        accuracies = []
        for r in results:
            eps = r['epsilon']
            if eps == 'inf':
                eps_numeric.append(20)  # Use 20 for visualization of "infinity"
            else:
                eps_numeric.append(float(eps))
            accuracies.append(r['metrics']['accuracy'])
        
        ax.plot(eps_numeric, accuracies, 'bo-', markersize=12, linewidth=2)
        
        for i, (eps, acc) in enumerate(zip(eps_numeric, accuracies)):
            label = results[i]['epsilon_display']
            ax.annotate(f'ε={label}\n{acc:.3f}', (eps, acc), 
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=10)
        
        ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Privacy-Utility Trade-off', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(eps_numeric)
        ax.set_xticklabels([r['epsilon_display'] for r in results])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'privacy_utility_tradeoff.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved privacy-utility tradeoff to {self.output_dir / 'privacy_utility_tradeoff.png'}")
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save JSON report
        json_results = []
        for r in results:
            json_r = {
                'epsilon': r['epsilon'],
                'metrics': {k: v for k, v in r['metrics'].items() if not isinstance(v, np.ndarray)}
            }
            json_results.append(json_r)
        
        with open(self.output_dir / 'dp_evaluation_results.json', 'w') as f:
            json.dump({'timestamp': timestamp, 'results': json_results}, f, indent=2)
        
        # Generate markdown report
        report_lines = [
            "# Differential Privacy Model Evaluation Report",
            f"\n**Generated:** {timestamp}\n",
            "## Summary Table\n",
            "| Epsilon (ε) | Accuracy | Precision | Recall | F1 | Sensitivity | Specificity | AUC-ROC |",
            "|-------------|----------|-----------|--------|-----|-------------|-------------|---------|"
        ]
        
        for r in results:
            m = r['metrics']
            auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "N/A"
            sens = f"{m.get('sensitivity', 0):.4f}"
            spec = f"{m.get('specificity', 0):.4f}"
            
            report_lines.append(
                f"| {r['epsilon_display']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                f"{m['recall']:.4f} | {m['f1']:.4f} | {sens} | {spec} | {auc} |"
            )
        
        report_lines.extend([
            "\n## Privacy-Utility Trade-off Analysis\n",
            "### Key Observations:\n"
        ])
        
        # Add analysis
        if len(results) > 1:
            sorted_results = sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True)
            best = sorted_results[0]
            
            report_lines.append(f"- **Best Accuracy:** ε = {best['epsilon_display']} with {best['metrics']['accuracy']:.4f}")
            
            if len(results) >= 2:
                # Compare strongest vs weakest privacy
                strongest_dp = [r for r in results if r['epsilon'] != 'inf']
                no_dp = [r for r in results if r['epsilon'] == 'inf']
                
                if strongest_dp and no_dp:
                    dp_best = min(strongest_dp, key=lambda x: float(x['epsilon']))
                    no_dp_model = no_dp[0]
                    
                    acc_diff = no_dp_model['metrics']['accuracy'] - dp_best['metrics']['accuracy']
                    report_lines.append(
                        f"- **Privacy Cost:** Switching from ε=∞ to ε={dp_best['epsilon_display']} "
                        f"results in {acc_diff*100:.2f}% accuracy {'loss' if acc_diff > 0 else 'gain'}"
                    )
        
        report_lines.extend([
            "\n## Visualizations\n",
            "- [Confusion Matrices](confusion_matrices_comparison.png)",
            "- [ROC Curves](roc_curves_comparison.png)",
            "- [Metrics Comparison](metrics_comparison.png)",
            "- [Privacy-Utility Trade-off](privacy_utility_tradeoff.png)",
            "\n## Classification Reports\n"
        ])
        
        for r in results:
            report_lines.append(f"\n### ε = {r['epsilon_display']}\n")
            report_lines.append("```")
            report_lines.append(classification_report(
                r['y_true'], r['y_pred'], 
                target_names=self.class_names, 
                zero_division=0
            ))
            report_lines.append("```")
        
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nSaved evaluation report to {self.output_dir / 'evaluation_report.md'}")
        print(f"Saved JSON results to {self.output_dir / 'dp_evaluation_results.json'}")
        
        # Print final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE - SUMMARY")
        print("="*70)
        print(f"\n{'Epsilon (ε)':<15} {'Accuracy':<12} {'F1':<12} {'AUC-ROC':<12}")
        print("-"*51)
        for r in results:
            m = r['metrics']
            auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "N/A"
            print(f"{r['epsilon_display']:<15} {m['accuracy']:<12.4f} {m['f1']:<12.4f} {auc:<12}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DP models for report')
    parser.add_argument('--epsilons', nargs='+', type=str, default=['5.0', '10.0', 'inf'],
                       help='Epsilon values to evaluate (use "inf" for no DP)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation/results/dp_evaluation',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Convert epsilon strings to numbers
    epsilons = []
    for e in args.epsilons:
        if e.lower() == 'inf':
            epsilons.append(float('inf'))
        else:
            epsilons.append(float(e))
    
    evaluator = DPModelEvaluator(output_dir=args.output_dir)
    results = evaluator.evaluate_all_models(epsilons=epsilons, batch_size=args.batch_size)
    
    return results


if __name__ == '__main__':
    main()

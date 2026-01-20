"""
Decentralized (Federated) Model Evaluation Script for Stage 1
COVIDx CXR-4 Binary Classification (COVID Positive/Negative)

Usage:
    python evaluation/evaluate_decentralized.py --model_path models/best_decentralized_model.pth

Results are saved to: evaluation/results/stage1/decentralized/
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model_side.models.cnn_model import COVIDxCNN
from evaluation.data_loader import COVIDxDataset, get_standard_transform
from evaluation.metrics import MetricsCalculator, MetricsVisualizer, format_metrics_table


class ModelEvaluator:
    """
    Comprehensive model evaluation for COVIDx classification.
    """
    def __init__(self, model, device, class_names=None):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or ['Negative', 'Positive']
        self.metrics_calc = MetricsCalculator(self.class_names)
        self.visualizer = MetricsVisualizer(self.class_names)

    def predict(self, data_loader, show_progress=True):
        """
        Run inference on data loader.

        Returns:
            y_true, y_pred, y_prob (probabilities for positive class)
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        iterator = tqdm(data_loader, desc="Evaluating") if show_progress else data_loader

        with torch.no_grad():
            for images, labels in iterator:
                images = images.to(self.device)
                outputs = self.model(images)

                # Get probabilities via softmax
                probs = torch.softmax(outputs, dim=1)

                all_labels.extend(labels.numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                # Probability for positive class (index 1)
                all_probs.extend(probs[:, 1].cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def evaluate(self, data_loader, output_dir: str = None, model_name: str = "decentralized"):
        """
        Full evaluation with metrics and visualizations.

        Args:
            data_loader: PyTorch DataLoader with test data
            output_dir: Directory to save results
            model_name: Name for labeling outputs
        """
        print(f"\nRunning evaluation for: {model_name}")
        print("-" * 50)

        # Run inference
        y_true, y_pred, y_prob = self.predict(data_loader)

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(y_true, y_pred, y_prob)

        # Print classification report
        report = self.metrics_calc.get_classification_report(y_true, y_pred)
        print("\nClassification Report:")
        print(report)

        # Print formatted metrics
        print(format_metrics_table(metrics, f"{model_name.upper()} Model Evaluation"))

        # Generate visualizations if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Confusion matrix (normalized)
            self.visualizer.plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=str(output_path / f'{model_name}_confusion_matrix_normalized.png'),
                normalize=True,
                title=f'{model_name.title()} Model - Normalized Confusion Matrix'
            )

            # Confusion matrix (counts)
            self.visualizer.plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=str(output_path / f'{model_name}_confusion_matrix_counts.png'),
                normalize=False,
                title=f'{model_name.title()} Model - Confusion Matrix (Counts)'
            )

            # ROC curve
            if y_prob is not None:
                self.visualizer.plot_roc_curve(
                    y_true, y_prob,
                    save_path=str(output_path / f'{model_name}_roc_curve.png'),
                    title=f'{model_name.title()} Model - ROC Curve'
                )

                # Precision-Recall curve
                self.visualizer.plot_precision_recall_curve(
                    y_true, y_prob,
                    save_path=str(output_path / f'{model_name}_pr_curve.png'),
                    title=f'{model_name.title()} Model - Precision-Recall Curve'
                )

            # Class distribution
            self.visualizer.plot_class_distribution(
                y_true, y_pred,
                save_path=str(output_path / f'{model_name}_class_distribution.png')
            )

            # Save metrics to JSON
            metrics_to_save = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
            metrics_to_save['model_name'] = model_name
            metrics_to_save['evaluation_timestamp'] = datetime.now().isoformat()
            metrics_to_save['num_samples'] = len(y_true)

            with open(output_path / f'{model_name}_metrics.json', 'w') as f:
                json.dump(metrics_to_save, f, indent=2)

            # Save classification report
            with open(output_path / f'{model_name}_classification_report.txt', 'w') as f:
                f.write(f"Classification Report - {model_name.title()} Model\n")
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)
                f.write("\n\n")
                f.write(format_metrics_table(metrics, f"{model_name.upper()} Model Evaluation"))

            print(f"\nResults saved to: {output_path}")

        return metrics, y_true, y_pred, y_prob


def load_model(model_path: str, num_classes: int = 2, device: torch.device = None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = COVIDxCNN(num_classes=num_classes)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # Handle case where state_dict is wrapped
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Check if this is a raw ResNet (without backbone. prefix)
    # The decentralized model might be saved as raw ResNet18
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith('backbone.'):
        # Add backbone. prefix to all keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle fc layer renaming (raw resnet uses fc.weight, COVIDxCNN uses backbone.fc.1.weight)
            if k == 'fc.weight':
                new_state_dict['backbone.fc.1.weight'] = v
            elif k == 'fc.bias':
                new_state_dict['backbone.fc.1.bias'] = v
            else:
                new_state_dict[f'backbone.{k}'] = v
        state_dict = new_state_dict
        print("Converted raw ResNet state dict to COVIDxCNN format")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")
    print(f"Model parameters: {model.get_num_parameters():,}")

    return model


def get_test_dataloader(data_dir: str, batch_size: int = 32):
    """Create test data loader from extracted dataset."""
    transform = get_standard_transform()

    dataset = COVIDxDataset(
        data_dir=data_dir,
        split='test',
        source_filter=None,  # Use all sources for global test
        transform=transform
    )

    print(f"Test dataset size: {len(dataset)} samples")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate decentralized (federated) COVIDx model')
    parser.add_argument('--model_path', type=str, default='models/best_decentralized_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset/archive',
                        help='Path to extracted dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation/results/stage1/decentralized',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, args.num_classes, device)

    # Load test data
    test_loader = get_test_dataloader(args.data_dir, args.batch_size)

    # Create evaluator and run
    evaluator = ModelEvaluator(model, device, class_names=['Negative', 'Positive'])
    metrics, y_true, y_pred, y_prob = evaluator.evaluate(
        test_loader,
        output_dir=args.output_dir,
        model_name='decentralized'
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == '__main__':
    main()

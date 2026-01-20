# Member 3: Evaluation & Interpretability Lead
## Detailed Working Plan

---

## Role Summary

| Attribute | Details |
|-----------|---------|
| **Primary Role** | Evaluation & Robustness/Interpretability Lead |
| **Main Responsibilities** | Metrics, model evaluation, Grad-CAM, SHAP, robustness testing |
| **Key Tools** | scikit-learn, Captum, SHAP, pytorch-grad-cam, Matplotlib |
| **Trust Dimension (Stage 2)** | Interpretability (Grad-CAM, SHAP) |

---

## Stage 1: Baseline Evaluation (Week 1-2)

### Task 1.1: Define Evaluation Metrics
**Duration:** Day 1-2
**Priority:** HIGH

#### Actionables:
1. **Research appropriate metrics for medical imaging**

   **Classification Metrics:**
   | Metric | Formula | Why Important |
   |--------|---------|---------------|
   | Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
   | Precision | TP/(TP+FP) | Minimize false alarms |
   | Recall/Sensitivity | TP/(TP+FN) | Detect all COVID cases |
   | Specificity | TN/(TN+FP) | Avoid false COVID diagnoses |
   | F1-Score | 2*(P*R)/(P+R) | Balance precision/recall |
   | AUC-ROC | Area under ROC curve | Discriminative ability |
   | AUC-PR | Area under PR curve | Better for imbalanced data |

2. **Medical imaging specific considerations**
   - **Sensitivity is critical**: Missing COVID case is worse than false positive
   - **Per-class metrics**: Different importance for each class
   - **Confidence calibration**: Predicted probabilities should be reliable

3. **Document metric selection rationale**
   ```markdown
   ## Metric Selection Rationale

   Primary Metrics:
   - Sensitivity (COVID class): Critical for screening applications
   - Specificity: Avoid unnecessary burden on healthcare
   - AUC-ROC: Overall model quality

   Secondary Metrics:
   - Per-class F1: Balanced view of all classes
   - Confusion matrix: Understand error patterns
   ```

#### Deliverables:
- [ ] Metrics documentation
- [ ] Rationale for metric selection

---

### Task 1.2: Implement Evaluation Framework
**Duration:** Day 2-4
**Priority:** HIGH

#### Actionables:
1. **Create metrics module** `src/evaluation/metrics.py`

2. **Implement comprehensive metrics calculation**
   ```python
   import numpy as np
   import torch
   from sklearn.metrics import (
       accuracy_score, precision_score, recall_score, f1_score,
       confusion_matrix, classification_report, roc_auc_score,
       precision_recall_curve, average_precision_score, roc_curve
   )
   from typing import Dict, List, Tuple
   import matplotlib.pyplot as plt
   import seaborn as sns

   class MetricsCalculator:
       """
       Comprehensive metrics calculator for medical image classification.
       """
       def __init__(self, class_names: List[str] = None):
           self.class_names = class_names or ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
           self.num_classes = len(self.class_names)

       def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: np.ndarray = None) -> Dict:
           """
           Calculate all relevant metrics.

           Args:
               y_true: Ground truth labels
               y_pred: Predicted labels
               y_prob: Predicted probabilities (optional, for AUC)

           Returns:
               Dictionary with all metrics
           """
           metrics = {}

           # Basic metrics
           metrics['accuracy'] = accuracy_score(y_true, y_pred)
           metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
           metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
           metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

           metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
           metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
           metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

           # Per-class metrics
           for i, class_name in enumerate(self.class_names):
               binary_true = (y_true == i).astype(int)
               binary_pred = (y_pred == i).astype(int)

               metrics[f'{class_name}_precision'] = precision_score(binary_true, binary_pred, zero_division=0)
               metrics[f'{class_name}_recall'] = recall_score(binary_true, binary_pred, zero_division=0)
               metrics[f'{class_name}_f1'] = f1_score(binary_true, binary_pred, zero_division=0)

           # AUC metrics (if probabilities provided)
           if y_prob is not None:
               try:
                   # Multi-class AUC (one-vs-rest)
                   metrics['auc_roc_macro'] = roc_auc_score(
                       y_true, y_prob, multi_class='ovr', average='macro'
                   )
                   metrics['auc_roc_weighted'] = roc_auc_score(
                       y_true, y_prob, multi_class='ovr', average='weighted'
                   )

                   # Per-class AUC
                   for i, class_name in enumerate(self.class_names):
                       binary_true = (y_true == i).astype(int)
                       metrics[f'{class_name}_auc'] = roc_auc_score(binary_true, y_prob[:, i])

               except Exception as e:
                   print(f"Could not calculate AUC: {e}")

           # Confusion matrix
           metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

           return metrics

       def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
           """Get sklearn classification report as string."""
           return classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)

       def calculate_sensitivity_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
           """
           Calculate sensitivity and specificity per class.
           Critical for medical applications.
           """
           results = {}
           cm = confusion_matrix(y_true, y_pred)

           for i, class_name in enumerate(self.class_names):
               # True positives, false negatives, false positives, true negatives
               tp = cm[i, i]
               fn = cm[i, :].sum() - tp
               fp = cm[:, i].sum() - tp
               tn = cm.sum() - tp - fn - fp

               sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
               specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

               results[f'{class_name}_sensitivity'] = sensitivity
               results[f'{class_name}_specificity'] = specificity

           return results


   class MetricsVisualizer:
       """
       Visualization utilities for evaluation metrics.
       """
       def __init__(self, class_names: List[str]):
           self.class_names = class_names

       def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None,
                                  normalize: bool = True) -> plt.Figure:
           """Plot confusion matrix as heatmap."""
           fig, ax = plt.subplots(figsize=(10, 8))

           if normalize:
               cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
               fmt = '.2%'
               title = 'Normalized Confusion Matrix'
           else:
               cm_display = cm
               fmt = 'd'
               title = 'Confusion Matrix'

           sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax)

           ax.set_xlabel('Predicted Label', fontsize=12)
           ax.set_ylabel('True Label', fontsize=12)
           ax.set_title(title, fontsize=14)

           plt.tight_layout()

           if save_path:
               plt.savefig(save_path, dpi=150, bbox_inches='tight')

           return fig

       def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                           save_path: str = None) -> plt.Figure:
           """Plot ROC curves for each class."""
           fig, ax = plt.subplots(figsize=(10, 8))

           colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

           for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
               binary_true = (y_true == i).astype(int)
               fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
               auc = roc_auc_score(binary_true, y_prob[:, i])

               ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{class_name} (AUC = {auc:.3f})')

           ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

           ax.set_xlabel('False Positive Rate', fontsize=12)
           ax.set_ylabel('True Positive Rate', fontsize=12)
           ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
           ax.legend(loc='lower right')
           ax.grid(True, alpha=0.3)

           plt.tight_layout()

           if save_path:
               plt.savefig(save_path, dpi=150, bbox_inches='tight')

           return fig

       def plot_metrics_comparison(self, results_dict: Dict[str, Dict],
                                    metrics: List[str], save_path: str = None) -> plt.Figure:
           """
           Plot comparison of multiple models across metrics.

           Args:
               results_dict: {model_name: {metric: value}}
               metrics: List of metric names to compare
           """
           fig, ax = plt.subplots(figsize=(12, 6))

           x = np.arange(len(metrics))
           width = 0.8 / len(results_dict)

           for i, (model_name, results) in enumerate(results_dict.items()):
               values = [results.get(m, 0) for m in metrics]
               offset = (i - len(results_dict)/2 + 0.5) * width
               ax.bar(x + offset, values, width, label=model_name)

           ax.set_xlabel('Metric', fontsize=12)
           ax.set_ylabel('Value', fontsize=12)
           ax.set_title('Model Comparison', fontsize=14)
           ax.set_xticks(x)
           ax.set_xticklabels(metrics, rotation=45, ha='right')
           ax.legend()
           ax.grid(True, axis='y', alpha=0.3)

           plt.tight_layout()

           if save_path:
               plt.savefig(save_path, dpi=150, bbox_inches='tight')

           return fig
   ```

3. **Create main evaluation script** `src/evaluation/evaluate.py`
   ```python
   import torch
   import numpy as np
   import json
   import argparse
   from pathlib import Path
   from torch.utils.data import DataLoader

   from model_side.models.cnn_model import COVIDxCNN
   from model_side.data.dataset import COVIDxDataset
   from model_side.data.preprocessing import get_test_transforms
   from model_side.evaluation.metrics import MetricsCalculator, MetricsVisualizer


   class ModelEvaluator:
       """
       Comprehensive model evaluation.
       """
       def __init__(self, model, device, class_names):
           self.model = model.to(device)
           self.device = device
           self.class_names = class_names
           self.metrics_calc = MetricsCalculator(class_names)
           self.visualizer = MetricsVisualizer(class_names)

       def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
           """
           Run inference on data loader.

           Returns:
               y_true, y_pred, y_prob
           """
           self.model.eval()
           all_labels = []
           all_preds = []
           all_probs = []

           with torch.no_grad():
               for images, labels in data_loader:
                   images = images.to(self.device)
                   outputs = self.model(images)
                   probs = torch.softmax(outputs, dim=1)

                   all_labels.extend(labels.numpy())
                   all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                   all_probs.extend(probs.cpu().numpy())

           return np.array(all_labels), np.array(all_preds), np.array(all_probs)

       def evaluate(self, data_loader, output_dir: str = None) -> Dict:
           """
           Full evaluation with metrics and visualizations.
           """
           y_true, y_pred, y_prob = self.predict(data_loader)

           # Calculate metrics
           metrics = self.metrics_calc.calculate_all_metrics(y_true, y_pred, y_prob)
           sens_spec = self.metrics_calc.calculate_sensitivity_specificity(y_true, y_pred)
           metrics.update(sens_spec)

           # Get classification report
           report = self.metrics_calc.get_classification_report(y_true, y_pred)
           print("\nClassification Report:")
           print(report)

           # Generate visualizations if output directory provided
           if output_dir:
               output_path = Path(output_dir)
               output_path.mkdir(parents=True, exist_ok=True)

               # Confusion matrix
               self.visualizer.plot_confusion_matrix(
                   metrics['confusion_matrix'],
                   save_path=str(output_path / 'confusion_matrix.png')
               )

               # ROC curves
               self.visualizer.plot_roc_curves(
                   y_true, y_prob,
                   save_path=str(output_path / 'roc_curves.png')
               )

               # Save metrics to JSON
               metrics_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                   for k, v in metrics.items()}
               with open(output_path / 'metrics.json', 'w') as f:
                   json.dump(metrics_to_save, f, indent=2)

           return metrics


   def main(args):
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']

       # Load model
       model = COVIDxCNN(num_classes=4)
       model.load_state_dict(torch.load(args.model_path, map_location=device))
       print(f"Loaded model from {args.model_path}")

       # Load data
       test_dataset = COVIDxDataset(args.data_dir, 'test', get_test_transforms())
       test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

       # Evaluate
       evaluator = ModelEvaluator(model, device, class_names)
       metrics = evaluator.evaluate(test_loader, output_dir=args.output_dir)

       print(f"\nResults saved to {args.output_dir}")

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--model_path', type=str, required=True)
       parser.add_argument('--data_dir', type=str, default='data/processed')
       parser.add_argument('--output_dir', type=str, default='results/evaluation')
       parser.add_argument('--batch_size', type=int, default=32)
       args = parser.parse_args()
       main(args)
   ```

#### Deliverables:
- [ ] `src/evaluation/metrics.py`
- [ ] `src/evaluation/evaluate.py`

---

### Task 1.3: Evaluate Centralized Baseline
**Duration:** Day 5-6 (blocked by Member 2's model)
**Priority:** HIGH

#### Actionables:
1. **Receive centralized model from Member 2**

2. **Run full evaluation**
   ```bash
   python model_side/evaluation/evaluate.py \
       --model_path models/best_centralized.pth \
       --data_dir data/processed \
       --output_dir results/stage1/centralized
   ```

3. **Document results**
   ```markdown
   ## Centralized Baseline Results

   | Metric | Value |
   |--------|-------|
   | Accuracy | X% |
   | Macro F1 | X% |
   | Macro AUC | X |
   | COVID Sensitivity | X% |
   | COVID Specificity | X% |
   ```

4. **Analyze confusion matrix**
   - Which classes are confused most often?
   - Is COVID being detected reliably?

#### Deliverables:
- [ ] `results/stage1/centralized/` with all outputs
- [ ] Analysis notes

---

### Task 1.4: Evaluate Federated Model
**Duration:** Day 6-7
**Priority:** HIGH

#### Actionables:
1. **Receive federated model from Member 2**

2. **Evaluate global federated model**
   ```bash
   python model_side/evaluation/evaluate.py \
       --model_path models/federated_final.pth \
       --data_dir data/processed \
       --output_dir results/stage1/federated
   ```

3. **Evaluate on each client's data** (to see local performance)
   ```python
   for client_id in [1, 2, 3]:
       evaluate_on_client(model, f'data/federated/client_{client_id}_data.npz')
   ```

4. **Create comparison table**

   | Metric | Centralized | Federated | Difference |
   |--------|-------------|-----------|------------|
   | Accuracy | X% | X% | Δ% |
   | Macro F1 | X% | X% | Δ% |
   | COVID Recall | X% | X% | Δ% |

#### Deliverables:
- [ ] `results/stage1/federated/` with all outputs
- [ ] Comparison table (markdown/CSV)
- [ ] Analysis of centralized vs federated

---

### Task 1.5: Initial Robustness Assessment
**Duration:** Day 6-7 (parallel with evaluation)
**Priority:** MEDIUM

#### Actionables:
1. **Test with simple perturbations**
   ```python
   import torch
   import torchvision.transforms.functional as TF

   def add_gaussian_noise(image, std=0.1):
       noise = torch.randn_like(image) * std
       return torch.clamp(image + noise, 0, 1)

   def add_blur(image, kernel_size=5):
       return TF.gaussian_blur(image, kernel_size)

   def test_robustness(model, test_loader, perturbation_fn, device):
       """Test model accuracy under perturbation."""
       model.eval()
       correct = 0
       total = 0

       with torch.no_grad():
           for images, labels in test_loader:
               images = perturbation_fn(images)
               images, labels = images.to(device), labels.to(device)

               outputs = model(images)
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()

       return correct / total
   ```

2. **Run robustness tests**
   ```python
   perturbations = [
       ('Clean', lambda x: x),
       ('Gaussian σ=0.05', lambda x: add_gaussian_noise(x, 0.05)),
       ('Gaussian σ=0.1', lambda x: add_gaussian_noise(x, 0.1)),
       ('Blur k=3', lambda x: add_blur(x, 3)),
       ('Blur k=5', lambda x: add_blur(x, 5)),
   ]

   results = {}
   for name, perturbation in perturbations:
       acc = test_robustness(model, test_loader, perturbation, device)
       results[name] = acc
       print(f"{name}: {acc:.4f}")
   ```

3. **Document initial findings**
   - How sensitive is the model to noise?
   - This informs Stage 2 robustness testing

#### Deliverables:
- [ ] Initial robustness results
- [ ] Notes on areas to investigate in Stage 2

---

### Task 1.6: Prepare Stage 1 Presentation
**Duration:** Day 7 (Week 2 end)
**Priority:** HIGH

#### Actionables:
1. **Consolidate all team results**
   - Data statistics from Member 1
   - Model details from Member 2
   - Evaluation results from your work

2. **Create presentation slides**
   ```
   Stage 1 Presentation Outline:
   1. Introduction & Problem (1 slide)
   2. Dataset Overview (2 slides) - Member 1
   3. Model Architecture (1 slide) - Member 2
   4. Federated Learning Setup (2 slides) - Member 2
   5. Evaluation Results (3 slides) - Member 3
      - Centralized results
      - Federated results
      - Comparison
   6. Limitations & Stage 2 Plan (2 slides) - All
   7. Q&A
   ```

3. **Prepare speaking notes**

#### Deliverables:
- [ ] `presentations/stage1_presentation.pptx`
- [ ] Speaking notes

---

### Task 1.7: Stage 1 Report Contribution
**Duration:** Day 7 (Week 2 end)
**Priority:** HIGH

#### Sections to Write:
1. **Evaluation Methodology** (0.5 page)
   - Metrics used and rationale
   - Evaluation protocol

2. **Preliminary Results** (1 page)
   - Centralized baseline results
   - Federated results
   - Comparison analysis

3. **Limitations and Trust Dimensions** (0.5 page)
   - Observed weaknesses
   - Planned trust enhancements for Stage 2

#### Deliverables:
- [ ] ~2 pages of Stage 1 report content

---

## Stage 2: Interpretability & Robustness (Week 3-5)

### Task 2.1: Implement Grad-CAM
**Duration:** Week 3, Day 1-3
**Priority:** HIGH

#### Actionables:
1. **Create interpretability module** `src/interpretability/gradcam.py`

2. **Implement Grad-CAM visualization**
   ```python
   import torch
   import torch.nn.functional as F
   import numpy as np
   import matplotlib.pyplot as plt
   from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
   from pytorch_grad_cam.utils.image import show_cam_on_image
   from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
   from PIL import Image

   class GradCAMExplainer:
       """
       Grad-CAM visualization for CNN models.
       """
       def __init__(self, model, target_layers, device):
           """
           Args:
               model: PyTorch model
               target_layers: List of layers to visualize (typically last conv layer)
               device: torch device
           """
           self.model = model.to(device)
           self.device = device
           self.cam = GradCAM(model=model, target_layers=target_layers)

       def explain(self, image_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
           """
           Generate Grad-CAM explanation for an image.

           Args:
               image_tensor: Preprocessed image tensor [1, C, H, W]
               target_class: Class to explain (None = predicted class)

           Returns:
               CAM heatmap as numpy array
           """
           image_tensor = image_tensor.to(self.device)

           if target_class is not None:
               targets = [ClassifierOutputTarget(target_class)]
           else:
               targets = None

           grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
           return grayscale_cam[0, :]

       def visualize(self, image_tensor: torch.Tensor, original_image: np.ndarray,
                      target_class: int = None, save_path: str = None) -> plt.Figure:
           """
           Create Grad-CAM visualization overlaid on original image.

           Args:
               image_tensor: Preprocessed image tensor [1, C, H, W]
               original_image: Original image as numpy array [H, W, 3] normalized to [0, 1]
               target_class: Class to explain
               save_path: Path to save figure
           """
           cam = self.explain(image_tensor, target_class)

           # Overlay on original image
           visualization = show_cam_on_image(original_image, cam, use_rgb=True)

           fig, axes = plt.subplots(1, 3, figsize=(15, 5))

           axes[0].imshow(original_image)
           axes[0].set_title('Original Image')
           axes[0].axis('off')

           axes[1].imshow(cam, cmap='jet')
           axes[1].set_title('Grad-CAM Heatmap')
           axes[1].axis('off')

           axes[2].imshow(visualization)
           axes[2].set_title('Grad-CAM Overlay')
           axes[2].axis('off')

           plt.tight_layout()

           if save_path:
               plt.savefig(save_path, dpi=150, bbox_inches='tight')

           return fig

       def explain_batch(self, images: List[torch.Tensor], labels: List[int],
                          original_images: List[np.ndarray], save_dir: str):
           """
           Generate Grad-CAM for a batch of images.
           """
           import os
           os.makedirs(save_dir, exist_ok=True)

           for i, (img_tensor, label, orig_img) in enumerate(zip(images, labels, original_images)):
               self.visualize(
                   img_tensor.unsqueeze(0),
                   orig_img,
                   target_class=label,
                   save_path=os.path.join(save_dir, f'gradcam_{i}_class{label}.png')
               )


   def get_target_layer(model, model_type='resnet18'):
       """
       Get the appropriate target layer for Grad-CAM based on model architecture.
       """
       if model_type == 'resnet18':
           return [model.backbone.layer4[-1]]
       elif model_type == 'simple_cnn':
           # Last conv layer
           return [model.features[-3]]  # Adjust based on architecture
       else:
           raise ValueError(f"Unknown model type: {model_type}")
   ```

3. **Test Grad-CAM implementation**
   ```python
   from model_side.interpretability.gradcam import GradCAMExplainer, get_target_layer

   model = COVIDxCNN(num_classes=4)
   model.load_state_dict(torch.load('models/best_centralized.pth'))

   target_layers = get_target_layer(model, 'resnet18')
   explainer = GradCAMExplainer(model, target_layers, device)

   # Test on sample image
   sample_image, sample_label = test_dataset[0]
   explainer.visualize(
       sample_image.unsqueeze(0),
       get_original_image(test_dataset, 0),
       save_path='results/stage2/gradcam_sample.png'
   )
   ```

4. **Generate Grad-CAM gallery**
   - Select representative samples from each class
   - Generate explanations for correct and incorrect predictions
   - Analyze what model is "looking at"

#### Deliverables:
- [ ] `src/interpretability/gradcam.py`
- [ ] Grad-CAM visualizations gallery
- [ ] Analysis of model attention patterns

---

### Task 2.2: Implement SHAP Explanations
**Duration:** Week 3, Day 3-5
**Priority:** HIGH

#### Actionables:
1. **Create SHAP module** `src/interpretability/shap_explain.py`

2. **Implement SHAP explanations**
   ```python
   import shap
   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from typing import List

   class SHAPExplainer:
       """
       SHAP (SHapley Additive exPlanations) for model interpretability.
       """
       def __init__(self, model, background_data: torch.Tensor, device):
           """
           Args:
               model: PyTorch model
               background_data: Background dataset for SHAP (sample of training data)
               device: torch device
           """
           self.model = model.to(device)
           self.device = device

           # Wrap model for SHAP
           def model_predict(x):
               self.model.eval()
               with torch.no_grad():
                   x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                   outputs = self.model(x_tensor)
                   return outputs.cpu().numpy()

           self.model_predict = model_predict

           # Create SHAP explainer (use GradientExplainer for neural networks)
           self.explainer = shap.GradientExplainer(model, background_data.to(device))

       def explain(self, images: torch.Tensor) -> np.ndarray:
           """
           Calculate SHAP values for images.

           Args:
               images: Batch of images [N, C, H, W]

           Returns:
               SHAP values
           """
           images = images.to(self.device)
           shap_values = self.explainer.shap_values(images)
           return shap_values

       def visualize_single(self, image: torch.Tensor, class_names: List[str],
                             save_path: str = None) -> plt.Figure:
           """
           Visualize SHAP values for a single image.
           """
           shap_values = self.explain(image.unsqueeze(0))

           # Average across color channels for visualization
           if len(shap_values) > 0 and shap_values[0].shape[1] > 1:
               shap_display = [sv[0].mean(axis=0) for sv in shap_values]
           else:
               shap_display = [sv[0, 0] for sv in shap_values]

           fig, axes = plt.subplots(1, len(class_names) + 1, figsize=(20, 4))

           # Original image
           img_display = image.permute(1, 2, 0).cpu().numpy()
           if img_display.shape[2] == 1:
               img_display = img_display.squeeze()
           axes[0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
           axes[0].set_title('Original')
           axes[0].axis('off')

           # SHAP for each class
           for i, (shap_val, class_name) in enumerate(zip(shap_display, class_names)):
               im = axes[i+1].imshow(shap_val, cmap='RdBu_r', vmin=-np.abs(shap_val).max(),
                                     vmax=np.abs(shap_val).max())
               axes[i+1].set_title(f'{class_name}')
               axes[i+1].axis('off')

           plt.colorbar(im, ax=axes[-1], shrink=0.8)
           plt.tight_layout()

           if save_path:
               plt.savefig(save_path, dpi=150, bbox_inches='tight')

           return fig

       def summary_plot(self, images: torch.Tensor, save_path: str = None):
           """
           Create SHAP summary plot for batch of images.
           """
           shap_values = self.explain(images)

           # Flatten for summary plot
           shap_flat = np.array(shap_values).reshape(len(shap_values), -1, images.shape[0]).transpose(2, 1, 0)

           plt.figure(figsize=(12, 8))
           shap.summary_plot(shap_flat, show=False)

           if save_path:
               plt.savefig(save_path, dpi=150, bbox_inches='tight')


   class DeepSHAPExplainer:
       """
       Alternative: DeepSHAP explainer using Captum library.
       More stable for deep networks.
       """
       def __init__(self, model, device):
           from captum.attr import DeepLift, DeepLiftShap

           self.model = model.to(device).eval()
           self.device = device
           self.deep_lift = DeepLift(model)

       def explain(self, image: torch.Tensor, target_class: int,
                    baseline: torch.Tensor = None) -> np.ndarray:
           """
           Calculate attributions using DeepLIFT.
           """
           image = image.to(self.device).requires_grad_()

           if baseline is None:
               baseline = torch.zeros_like(image).to(self.device)

           attributions = self.deep_lift.attribute(image, baselines=baseline, target=target_class)
           return attributions.detach().cpu().numpy()
   ```

3. **Generate SHAP visualizations**
   - Background data: sample 100-200 training images
   - Explain predictions for test samples
   - Compare explanations across models (centralized, federated, DP)

#### Deliverables:
- [ ] `src/interpretability/shap_explain.py`
- [ ] SHAP visualizations
- [ ] Explanation comparison across models

---

### Task 2.3: Robustness Testing
**Duration:** Week 3-4
**Priority:** HIGH

#### Actionables:
1. **Create robustness module** `src/robustness/adversarial.py`

2. **Implement adversarial attacks**
   ```python
   import torch
   import torch.nn as nn
   import numpy as np
   from typing import Tuple

   class AdversarialAttacks:
       """
       Adversarial attack implementations for robustness testing.
       """
       def __init__(self, model, device, num_classes=4):
           self.model = model.to(device).eval()
           self.device = device
           self.num_classes = num_classes
           self.criterion = nn.CrossEntropyLoss()

       def fgsm_attack(self, images: torch.Tensor, labels: torch.Tensor,
                        epsilon: float = 0.03) -> torch.Tensor:
           """
           Fast Gradient Sign Method (FGSM) attack.

           Args:
               images: Input images [N, C, H, W]
               labels: True labels [N]
               epsilon: Perturbation magnitude

           Returns:
               Adversarial images
           """
           images = images.clone().detach().to(self.device).requires_grad_(True)
           labels = labels.to(self.device)

           outputs = self.model(images)
           loss = self.criterion(outputs, labels)
           loss.backward()

           # Get gradient sign
           perturbation = epsilon * images.grad.sign()

           # Create adversarial example
           adv_images = images + perturbation
           adv_images = torch.clamp(adv_images, 0, 1)

           return adv_images.detach()

       def pgd_attack(self, images: torch.Tensor, labels: torch.Tensor,
                       epsilon: float = 0.03, alpha: float = 0.01,
                       num_steps: int = 10) -> torch.Tensor:
           """
           Projected Gradient Descent (PGD) attack.
           Stronger iterative attack.

           Args:
               images: Input images
               labels: True labels
               epsilon: Maximum perturbation
               alpha: Step size
               num_steps: Number of iterations
           """
           images = images.clone().detach().to(self.device)
           labels = labels.to(self.device)

           # Start with random perturbation
           adv_images = images + torch.zeros_like(images).uniform_(-epsilon, epsilon)
           adv_images = torch.clamp(adv_images, 0, 1)

           for _ in range(num_steps):
               adv_images.requires_grad_(True)

               outputs = self.model(adv_images)
               loss = self.criterion(outputs, labels)
               loss.backward()

               # Take gradient step
               adv_images = adv_images + alpha * adv_images.grad.sign()

               # Project back to epsilon ball
               perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
               adv_images = torch.clamp(images + perturbation, 0, 1).detach()

           return adv_images

       def test_robustness(self, test_loader, attack_fn) -> Tuple[float, float]:
           """
           Test model robustness against an attack.

           Returns:
               clean_accuracy, adversarial_accuracy
           """
           clean_correct = 0
           adv_correct = 0
           total = 0

           for images, labels in test_loader:
               images, labels = images.to(self.device), labels.to(self.device)

               # Clean accuracy
               with torch.no_grad():
                   outputs = self.model(images)
                   clean_correct += (outputs.argmax(1) == labels).sum().item()

               # Adversarial accuracy
               adv_images = attack_fn(images, labels)
               with torch.no_grad():
                   adv_outputs = self.model(adv_images)
                   adv_correct += (adv_outputs.argmax(1) == labels).sum().item()

               total += labels.size(0)

           return clean_correct / total, adv_correct / total


   class RobustnessEvaluator:
       """
       Comprehensive robustness evaluation.
       """
       def __init__(self, model, device):
           self.model = model
           self.device = device
           self.attacks = AdversarialAttacks(model, device)

       def evaluate_all(self, test_loader) -> dict:
           """Run all robustness tests."""
           results = {}

           # Natural perturbations
           noise_levels = [0.05, 0.1, 0.15]
           for noise in noise_levels:
               acc = self._test_gaussian_noise(test_loader, noise)
               results[f'gaussian_noise_{noise}'] = acc

           # Adversarial attacks
           epsilons = [0.01, 0.03, 0.05]
           for eps in epsilons:
               # FGSM
               _, fgsm_acc = self.attacks.test_robustness(
                   test_loader,
                   lambda x, y: self.attacks.fgsm_attack(x, y, eps)
               )
               results[f'fgsm_eps_{eps}'] = fgsm_acc

               # PGD
               _, pgd_acc = self.attacks.test_robustness(
                   test_loader,
                   lambda x, y: self.attacks.pgd_attack(x, y, eps)
               )
               results[f'pgd_eps_{eps}'] = pgd_acc

           return results

       def _test_gaussian_noise(self, test_loader, std):
           self.model.eval()
           correct = 0
           total = 0

           with torch.no_grad():
               for images, labels in test_loader:
                   images = images.to(self.device)
                   labels = labels.to(self.device)

                   # Add noise
                   noisy_images = images + torch.randn_like(images) * std
                   noisy_images = torch.clamp(noisy_images, 0, 1)

                   outputs = self.model(noisy_images)
                   correct += (outputs.argmax(1) == labels).sum().item()
                   total += labels.size(0)

           return correct / total
   ```

3. **Test against data poisoning attacks**
   - Receive poisoned data from Member 1
   - Test model performance on clean vs poisoned data
   - Document attack success rates

4. **Create robustness report**

   | Attack Type | Clean Acc | Attacked Acc | Drop |
   |-------------|-----------|--------------|------|
   | Gaussian σ=0.05 | X% | X% | Δ% |
   | Gaussian σ=0.1 | X% | X% | Δ% |
   | FGSM ε=0.01 | X% | X% | Δ% |
   | FGSM ε=0.03 | X% | X% | Δ% |
   | PGD ε=0.03 | X% | X% | Δ% |

#### Deliverables:
- [ ] `src/robustness/adversarial.py`
- [ ] Robustness results for all models
- [ ] Comparison table

---

### Task 2.4: Trust Metrics and Calibration
**Duration:** Week 4
**Priority:** MEDIUM

#### Actionables:
1. **Implement calibration analysis**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.calibration import calibration_curve

   def plot_reliability_diagram(y_true, y_prob, n_bins=10, save_path=None):
       """
       Plot reliability diagram for model calibration.
       """
       fig, axes = plt.subplots(1, 2, figsize=(14, 5))

       # For multi-class, plot for each class
       for class_idx in range(y_prob.shape[1]):
           binary_true = (y_true == class_idx).astype(int)
           prob_pos, mean_predicted = calibration_curve(
               binary_true, y_prob[:, class_idx], n_bins=n_bins
           )
           axes[0].plot(mean_predicted, prob_pos, 's-', label=f'Class {class_idx}')

       axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
       axes[0].set_xlabel('Mean Predicted Probability')
       axes[0].set_ylabel('Fraction of Positives')
       axes[0].set_title('Reliability Diagram')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)

       # Confidence histogram
       max_probs = y_prob.max(axis=1)
       axes[1].hist(max_probs, bins=20, edgecolor='black')
       axes[1].set_xlabel('Confidence')
       axes[1].set_ylabel('Count')
       axes[1].set_title('Confidence Distribution')
       axes[1].grid(True, alpha=0.3)

       plt.tight_layout()

       if save_path:
           plt.savefig(save_path, dpi=150, bbox_inches='tight')

       return fig

   def calculate_ece(y_true, y_prob, n_bins=15):
       """
       Calculate Expected Calibration Error (ECE).
       Lower is better calibrated.
       """
       confidences = y_prob.max(axis=1)
       predictions = y_prob.argmax(axis=1)
       accuracies = (predictions == y_true).astype(float)

       bin_boundaries = np.linspace(0, 1, n_bins + 1)
       ece = 0

       for i in range(n_bins):
           in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
           prop_in_bin = in_bin.mean()

           if prop_in_bin > 0:
               avg_confidence = confidences[in_bin].mean()
               avg_accuracy = accuracies[in_bin].mean()
               ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

       return ece
   ```

2. **Calculate trust metrics**
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)
   - Brier score
   - Uncertainty quantification

#### Deliverables:
- [ ] Calibration analysis for all models
- [ ] Trust metrics comparison table

---

### Task 2.5: Final Cross-Model Evaluation
**Duration:** Week 4-5
**Priority:** HIGH

#### Actionables:
1. **Receive all models**
   - Centralized baseline (from Member 2)
   - Federated model (from Member 2)
   - DP models ε = 1, 5, 10 (from Member 2)
   - DP + FL model (from Member 2)

2. **Comprehensive evaluation of all models**
   - Classification metrics
   - Interpretability (Grad-CAM consistency)
   - Robustness testing
   - Calibration analysis

3. **Create final comparison table**

   | Model | Accuracy | F1 | AUC | COVID Recall | ECE | FGSM Robust |
   |-------|----------|----|----|--------------|-----|-------------|
   | Centralized | X% | X | X | X% | X | X% |
   | Federated | X% | X | X | X% | X | X% |
   | DP ε=1 | X% | X | X | X% | X | X% |
   | DP ε=5 | X% | X | X | X% | X | X% |
   | DP ε=10 | X% | X | X | X% | X | X% |
   | DP + FL | X% | X | X | X% | X | X% |

4. **Statistical significance testing**
   ```python
   from scipy import stats

   def compare_models(acc1, acc2, n_samples):
       """McNemar's test for comparing classifiers."""
       # Simplified - use proper bootstrap for rigorous testing
       se = np.sqrt((acc1 * (1-acc1) + acc2 * (1-acc2)) / n_samples)
       z = (acc1 - acc2) / se
       p_value = 2 * (1 - stats.norm.cdf(abs(z)))
       return z, p_value
   ```

#### Deliverables:
- [ ] Final comparison table (all metrics, all models)
- [ ] Statistical significance analysis
- [ ] Visualization of results

---

### Task 2.6: Prepare Final Presentation
**Duration:** Week 5
**Priority:** HIGH

#### Actionables:
1. **Consolidate all results from team**

2. **Create presentation structure**
   ```
   Final Presentation Outline (20-25 min):
   1. Introduction (2 slides) - Member 1 or 2
   2. Dataset & Preprocessing (2 slides) - Member 1
   3. Model Architecture & FL (3 slides) - Member 2
   4. Differential Privacy (3 slides) - Member 2
   5. Evaluation Results (4 slides) - Member 3
      - Classification performance
      - Interpretability insights
      - Robustness analysis
   6. Trust Trade-offs Discussion (2 slides) - Member 3
   7. Conclusions & Future Work (2 slides) - All
   8. Q&A
   ```

3. **Coordinate with team members**

#### Deliverables:
- [ ] `presentations/final_presentation.pptx`
- [ ] Coordinated speaking sections

---

### Task 2.7: Stage 2 Report Contribution
**Duration:** Week 5
**Priority:** HIGH

#### Sections to Write (~6 pages):
1. **Interpretability Methods** (2 pages)
   - Grad-CAM methodology and implementation
   - SHAP methodology and implementation
   - Sample visualizations and analysis
   - What the model "looks at" for COVID detection

2. **Robustness Evaluation** (2 pages)
   - Attack types and parameters
   - Results under different perturbations
   - Comparison across models

3. **Trust Metrics & Calibration** (1 page)
   - Calibration analysis
   - ECE and other trust metrics
   - Reliability diagrams

4. **Final Comparative Analysis** (1 page)
   - Comprehensive results table
   - Trade-off analysis
   - Recommendations

#### Deliverables:
- [ ] ~6 pages of final report content

---

## Communication Protocol

### With Member 1 (Data Lead):
- **Week 4:** Receive poisoned datasets for robustness testing
- **Week 4-5:** Receive cross-evaluation results on your metrics

### With Member 2 (Modeling Lead):
- **Week 2:** Receive centralized model
- **Week 2:** Receive federated model
- **Week 4:** Receive all DP models
- Coordinate on model architecture details for Grad-CAM

### Models to Receive and Evaluate:
| Model | When | From |
|-------|------|------|
| Centralized baseline | Week 2 | Member 2 |
| Federated model | Week 2 | Member 2 |
| DP ε=1 | Week 4 | Member 2 |
| DP ε=5 | Week 4 | Member 2 |
| DP ε=10 | Week 4 | Member 2 |
| DP + FL | Week 4 | Member 2 |

---

## Time Estimates

| Task | Estimated Hours | Priority |
|------|-----------------|----------|
| Define metrics | 2-3h | High |
| Evaluation framework | 6-8h | High |
| Evaluate centralized | 2-3h | High |
| Evaluate federated | 2-3h | High |
| Initial robustness | 2-3h | Medium |
| Stage 1 presentation | 4-6h | High |
| Stage 1 report | 4-5h | High |
| Grad-CAM implementation | 6-8h | High |
| SHAP implementation | 4-6h | High |
| Robustness testing | 6-8h | High |
| Calibration analysis | 3-4h | Medium |
| Final cross-evaluation | 6-8h | High |
| Final presentation | 4-6h | High |
| Stage 2 report | 8-10h | High |
| **Total** | **~60-75h** | |

---

## Checklist

### Stage 1
- [ ] Define evaluation metrics with rationale
- [ ] Implement MetricsCalculator class
- [ ] Implement MetricsVisualizer class
- [ ] Create main evaluation script
- [ ] Evaluate centralized baseline
- [ ] Evaluate federated model
- [ ] Create centralized vs federated comparison
- [ ] Initial robustness assessment
- [ ] Prepare Stage 1 presentation
- [ ] Write Stage 1 report sections

### Stage 2
- [ ] Implement Grad-CAM explainer
- [ ] Generate Grad-CAM visualizations for all models
- [ ] Implement SHAP explainer
- [ ] Generate SHAP visualizations
- [ ] Implement adversarial attacks (FGSM, PGD)
- [ ] Test robustness against data poisoning
- [ ] Implement calibration analysis
- [ ] Calculate trust metrics (ECE)
- [ ] Final evaluation of all models
- [ ] Create comprehensive comparison table
- [ ] Prepare final presentation
- [ ] Write Stage 2 report sections

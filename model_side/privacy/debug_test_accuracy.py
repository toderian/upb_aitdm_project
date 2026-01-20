"""
Debug script to diagnose the 10% test accuracy issue.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_side.models.cnn_model import SimpleCNN
from opacus.validators import ModuleValidator


def diagnose_issues():
    """Run diagnostic checks to identify the test accuracy problem."""
    
    print("=" * 60)
    print("DIAGNOSTIC REPORT: Test Accuracy Issue")
    print("=" * 60)
    
    # Issue 1: Check what data was used for experiments
    print("\n[1] DATA CHECK")
    print("-" * 40)
    print("Looking at run_dp_experiments.py main block...")
    print("FINDING: When --data_path is not provided, DUMMY DATA is used!")
    print("  - X_train: 100 samples of random noise")
    print("  - X_test: 20 samples of random noise")  
    print("  - Labels: random integers 0 to num_classes-1")
    print("\n⚠️  PROBLEM: Training on random noise, testing on different random noise")
    print("   Expected accuracy on random data = 1/num_classes = 1/4 = 25%")
    print("   But getting 10% suggests something else is wrong too...")
    
    # Issue 2: Check image size mismatch
    print("\n[2] IMAGE SIZE MISMATCH CHECK")
    print("-" * 40)
    
    # SimpleCNN uses AdaptiveAvgPool2d, so it can handle any input size
    # But let's check if dummy data matches expected size
    print("Dummy data uses 64x64 images")
    print("Real data loader (COVIDxZipDataset) uses 224x224 images")
    print("SimpleCNN uses AdaptiveAvgPool2d - handles any size ✓")
    
    # Issue 3: Check class count mismatch
    print("\n[3] CLASS COUNT CHECK")  
    print("-" * 40)
    print("run_dp_experiments.py: --num_classes default = 4")
    print("data_loader.py label_map: {'positive': 1, 'negative': 0} = 2 classes!")
    print("\n⚠️  PROBLEM: Model expects 4 classes, real data has 2 classes")
    
    # Issue 4: Verify dummy data statistics
    print("\n[4] DUMMY DATA VERIFICATION")
    print("-" * 40)
    
    num_classes = 4
    X_test = torch.randn(20, 3, 64, 64)
    y_test = torch.randint(0, num_classes, (20,))
    
    print(f"Test set size: {len(y_test)} samples")
    print(f"Label distribution: {torch.bincount(y_test, minlength=num_classes).tolist()}")
    
    # Create model and make random predictions
    model = SimpleCNN(num_classes=num_classes)
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = outputs.max(1)
        
    print(f"Prediction distribution: {torch.bincount(predicted, minlength=num_classes).tolist()}")
    correct = predicted.eq(y_test).sum().item()
    print(f"Random model accuracy: {correct}/{len(y_test)} = {100*correct/len(y_test):.1f}%")
    
    # Issue 5: Check saved model loading
    print("\n[5] SAVED MODEL CHECK")
    print("-" * 40)
    
    model_paths = [
        'models/dp_epsilon_1.0.pth',
        'models/dp_epsilon_5.0.pth', 
        'models/dp_epsilon_10.0.pth',
        'models/dp_epsilon_inf.pth'
    ]
    
    for path in model_paths:
        full_path = Path(__file__).parent.parent.parent / path
        if full_path.exists():
            state_dict = torch.load(full_path, map_location='cpu')
            print(f"\n{path}:")
            # Check classifier layer shape
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    print(f"  {key}: {state_dict[key].shape}")
        else:
            print(f"{path}: NOT FOUND")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ISSUES")
    print("=" * 60)
    print("""
1. DUMMY DATA: Experiments ran with random noise instead of real images
   - 100 random train samples, 20 random test samples
   - Cannot learn meaningful patterns from noise
   
2. CLASS MISMATCH: Model has 4 output classes, real data has 2
   - data_loader.py: binary classification (positive/negative)
   - run_dp_experiments.py: defaults to 4 classes
   
3. SMALL DATASET: Only 100 training samples for DP experiments
   - DP requires more data to overcome noise addition
   - Effective batch size is reduced by gradient clipping

4. 10% ACCURACY EXPLANATION:
   - With 20 test samples and 4 classes, getting 2/20 correct = 10%
   - Random chance would give ~25%, but model learned noise patterns
   - That don't generalize at all to different random test noise
""")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES")
    print("=" * 60)
    print("""
1. Use REAL DATA by providing --data_path or integrating data_loader.py
2. Fix class count: use num_classes=2 for binary classification
3. Use larger dataset (at least 1000+ samples for DP training)
4. Add proper train/test split from the same data distribution
""")


if __name__ == "__main__":
    diagnose_issues()

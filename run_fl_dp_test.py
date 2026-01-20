#!/usr/bin/env python
"""
Test Federated Learning with Differential Privacy.
Runs server and clients in subprocesses, then evaluates the final model.

Usage:
    python run_fl_dp_test.py --num_rounds 3 --epsilon 5.0
"""
import subprocess
import sys
import os
import time
import argparse
import signal
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_fl_dp_experiment(args):
    """Run federated learning experiment with DP."""
    
    print("="*70)
    print("FEDERATED LEARNING + DIFFERENTIAL PRIVACY EXPERIMENT")
    print("="*70)
    print(f"  Target Epsilon: {args.epsilon}")
    print(f"  Num Clients: {args.num_clients}")
    print(f"  Num Rounds: {args.num_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print("="*70)
    
    python_exe = sys.executable
    processes = []
    
    try:
        # Start server
        print("\n🚀 Starting FL Server...")
        server_cmd = [
            python_exe, "-m", "model_side.federated.server_dp",
            "--server_address", args.server_address,
            "--num_clients", str(args.num_clients),
            "--num_rounds", str(args.num_rounds),
            "--local_epochs", str(args.local_epochs),
            "--output_dir", args.output_dir
        ]
        
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(('Server', server_proc))
        
        # Wait for server to start
        time.sleep(5)
        
        # Start clients
        print(f"\n🔗 Starting {args.num_clients} DP Clients...")
        for client_id in range(args.num_clients):
            client_cmd = [
                python_exe, "-m", "model_side.federated.client_dp",
                "--server_address", args.server_address,
                "--client_id", str(client_id),
                "--batch_size", str(args.batch_size),
                "--num_classes", "2",
                "--target_epsilon", str(args.epsilon),
                "--target_delta", str(args.delta),
                "--max_grad_norm", str(args.max_grad_norm)
            ]
            
            client_proc = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            processes.append((f'Client {client_id}', client_proc))
            print(f"   Started Client {client_id} (ε={args.epsilon})")
            time.sleep(2)  # Stagger client starts
        
        print("\n⏳ Training in progress...\n")
        
        # Monitor server output
        while server_proc.poll() is None:
            line = server_proc.stdout.readline()
            if line:
                print(f"[Server] {line.strip()}")
        
        # Get remaining output
        stdout, _ = server_proc.communicate(timeout=10)
        if stdout:
            for line in stdout.strip().split('\n'):
                if line:
                    print(f"[Server] {line}")
        
        # Wait for all processes to finish
        for name, proc in processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f"⚠️ {name} did not finish in time, terminating...")
                proc.terminate()
        
        print("\n✅ FL+DP training complete!")
        
        # Evaluate final model
        if args.evaluate:
            evaluate_final_model(args)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
        
    finally:
        # Cleanup processes
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


def evaluate_final_model(args):
    """Evaluate the final federated model."""
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    from tqdm import tqdm
    
    from model_side.models.cnn_model import SimpleCNN
    from model_side.data.data_loader_enhanced import get_global_test_loader
    from opacus.validators import ModuleValidator
    
    print("\n" + "="*70)
    print("EVALUATING FINAL FL+DP MODEL")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find the latest model
    model_dir = Path('models/fl_dp')
    if not model_dir.exists():
        print("❌ No model directory found")
        return
    
    model_files = sorted(model_dir.glob('fl_dp_round_*.pth'))
    if not model_files:
        print("❌ No model files found")
        return
    
    latest_model = model_files[-1]
    print(f"📂 Loading: {latest_model}")
    
    # Load model
    model = SimpleCNN(num_classes=2)
    model = ModuleValidator.fix(model)
    model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("📊 Loading test data...")
    test_loader = get_global_test_loader(batch_size=args.batch_size)
    
    # Evaluate
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['Negative', 'Positive'],
                               zero_division=0))
    
    print(f"\n🎯 Final Accuracy: {accuracy:.4f}")
    
    # Save evaluation results
    results = {
        'model_path': str(latest_model),
        'epsilon': args.epsilon,
        'num_rounds': args.num_rounds,
        'num_clients': args.num_clients,
        'accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path(args.output_dir) / 'fl_dp_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Run FL+DP Experiment")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=5.0,
                       help="Target epsilon for DP (privacy budget)")
    parser.add_argument("--delta", type=float, default=1e-5,
                       help="Target delta for DP")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--output_dir", type=str, default="results/fl_dp")
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="Evaluate final model after training")
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = run_fl_dp_experiment(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

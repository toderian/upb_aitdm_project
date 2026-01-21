import matplotlib.pyplot as plt
import json
import os

# Apply a cleaner style if available
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('ggplot') 
    except:
        pass 

RESULTS_DIR = "results"

def load_history(folder_name):
    path = os.path.join(RESULTS_DIR, folder_name, "history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def save_single_plot(x_data_list, y_data_list, labels, title, ylabel, filename, colors, markers):
    """
    Helper to save a single metric plot to a file with the legend OUTSIDE/BELOW the plot.
    """
    plt.figure(figsize=(10, 7)) # Slightly taller to accommodate bottom legend
    
    for x, y, label, color, marker in zip(x_data_list, y_data_list, labels, colors, markers):
        plt.plot(x, y, label=label, color=color, marker=marker, linewidth=2, markersize=5)
        
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Rounds / Epochs", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Legend placed below the plot
    # bbox_to_anchor controls position relative to the plot anchor
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2, fontsize=10)
    
    # Use tight_layout + bbox_inches to ensure nothing is cropped
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    plt.close()

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory '{RESULTS_DIR}' not found.")
        return

    subfolders = [f.name for f in os.scandir(RESULTS_DIR) if f.is_dir()]
    subfolders.sort()
    
    if not subfolders:
        print("No experiment results found.")
        return

    print(f"Found experiments: {subfolders}")

    # Standard colors and markers for distinction
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    # We collect data into parallel lists to pass to the plotting helper
    labels = []
    exp_colors = []
    exp_markers = []
    
    # Structure: [ [list_of_x1, list_of_x2], [list_of_y1, list_of_y2] ]
    train_acc_xy = [[], []]
    test_acc_xy = [[], []]
    train_loss_xy = [[], []]
    test_loss_xy = [[], []]

    for i, exp_name in enumerate(subfolders):
        data = load_history(exp_name)
        if not data:
            continue
            
        x = data.get("rounds", data.get("epochs", []))
        if not x: continue
        
        # Assign style
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # 1. Test Accuracy
        if "accuracy" in data:
            test_acc_xy[0].append(x)
            test_acc_xy[1].append(data["accuracy"])
        
        # 2. Test Loss
        if "loss" in data:
            test_loss_xy[0].append(x)
            test_loss_xy[1].append(data["loss"])
            
        # 3. Train Accuracy (Optional)
        if "train_accuracy" in data and len(data["train_accuracy"]) > 0:
            train_acc_xy[0].append(x)
            train_acc_xy[1].append(data["train_accuracy"])
            
        # 4. Train Loss (Optional)
        if "train_loss" in data and len(data["train_loss"]) > 0:
            train_loss_xy[0].append(x)
            train_loss_xy[1].append(data["train_loss"])

        # Store metadata for legend
        labels.append(exp_name)
        exp_colors.append(color)
        exp_markers.append(marker)

    # --- Generate Files ---

    if test_acc_xy[0]:
        save_single_plot(test_acc_xy[0], test_acc_xy[1], labels, "Test Accuracy", "Accuracy", 
                         "comparison_test_acc.png", exp_colors, exp_markers)

    if test_loss_xy[0]:
        save_single_plot(test_loss_xy[0], test_loss_xy[1], labels, "Test Loss", "Loss", 
                         "comparison_test_loss.png", exp_colors, exp_markers)

    if train_acc_xy[0]:
        # Note: We assume if Test Acc exists, we use the same labels/colors. 
        # Ideally, we'd filter labels only for experiments that actually have train data, 
        # but usually, all experiments in a batch will have the same structure.
        save_single_plot(train_acc_xy[0], train_acc_xy[1], labels, "Training Accuracy", "Accuracy", 
                         "comparison_train_acc.png", exp_colors, exp_markers)

    if train_loss_xy[0]:
        save_single_plot(train_loss_xy[0], train_loss_xy[1], labels, "Training Loss", "Loss", 
                         "comparison_train_loss.png", exp_colors, exp_markers)

if __name__ == "__main__":
    main()
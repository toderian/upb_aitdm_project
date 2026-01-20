import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import COVIDxZipDataset, ZIP_FILE_PATH, TXT_FILE_NAME
from tqdm import tqdm  # For progress bar
import random
from collections import Counter
import pandas as pd

# --- CONFIGURATION ---
# Number of images to check for pixel stats (scanning all 67k is slow)
SAMPLE_SIZE = 1000
OUTPUT_IMG = "data_samples.png"


def analyze_metadata(dataset):
    print(f"\n[1] Metadata Analysis (Full Dataset: {len(dataset)} rows)")
    df = dataset.df

    # 1. Duplicates
    duplicates = df[df.duplicated(subset=['filename'])]
    if len(duplicates) > 0:
        print(f"   ! WARNING: Found {len(duplicates)} duplicate filenames.")
    else:
        print("   - No duplicate filenames found.")

    # 2. File Formats (inferred from extensions)
    formats = [f.split('.')[-1].lower() for f in df['filename']]
    print(f"   - File Formats: {dict(Counter(formats))}")

    # 3. Class Imbalance
    print(f"   - Class Distribution: {df['label'].value_counts().to_dict()}")

    # 4. Source Distribution
    print(f"   - Source Distribution: {df['source'].value_counts().to_dict()}")


def analyze_image_properties(dataset):
    print(f"\n[2] Image Property Analysis (Sampling {SAMPLE_SIZE} images)")

    widths = []
    heights = []
    pixel_means = []
    r_vs_g_diffs = []  # To check if truly grayscale
    corrupted_count = 0

    # Randomly sample indices
    indices = random.sample(range(len(dataset)),
                            min(len(dataset), SAMPLE_SIZE))

    for idx in tqdm(indices):
        try:
            # We access raw item via dataset (no transforms applied yet if we passed None)
            img, label = dataset[idx]

            # Dimensions
            w, h = img.size
            widths.append(w)
            heights.append(h)

            # Convert to numpy for stats
            img_np = np.array(img)
            pixel_means.append(img_np.mean())

            # Check Color Channels (RGB vs Grayscale content)
            # Even if converted to RGB, grayscale images have R=G=B
            # We calculate mean absolute difference between Red and Green channels
            diff = np.abs(img_np[:, :, 0] - img_np[:, :, 1]).mean()
            r_vs_g_diffs.append(diff)

        except Exception as e:
            corrupted_count += 1

    # Report
    print(f"   - Corrupted/Unreadable Images in sample: {corrupted_count}")

    # Sizes
    print(
        f"   - Image Dimensions (Width): Min={min(widths)}, Max={max(widths)}, Avg={int(sum(widths)/len(widths))}")
    print(
        f"   - Image Dimensions (Height): Min={min(heights)}, Max={max(heights)}, Avg={int(sum(heights)/len(heights))}")
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        print(
            "   ! NOTE: Image sizes are inconsistent. Preprocessing (Resize) is MANDATORY.")

    # Pixel Value Distributions
    print(
        f"   - Pixel Intensity Mean: {np.mean(pixel_means):.2f} (Range 0-255)")

    # Color Analysis
    avg_color_diff = np.mean(r_vs_g_diffs)
    if avg_color_diff < 1.0:
        print(f"   - Color Check: Images appear to be GRAYSCALE (R=G=B mostly).")
    else:
        print(f"   - Color Check: Images appear to be TRUE COLOR (Significant RGB differences).")


def visualize_grid(dataset):
    print(f"\n[3] Generating Visual Samples -> {OUTPUT_IMG}")

    # Get separate lists for positive and negative
    pos_indices = dataset.df[dataset.df['label'] == 'positive'].index.tolist()
    neg_indices = dataset.df[dataset.df['label'] == 'negative'].index.tolist()

    # Select 4 of each
    samples_pos = random.sample(pos_indices, 4)
    samples_neg = random.sample(neg_indices, 4)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "COVIDx Sample Data: Top Row (Positive), Bottom Row (Negative)", fontsize=16)

    # Plot Positive
    for i, idx in enumerate(samples_pos):
        img, _ = dataset[idx]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Positive\n{dataset.df.iloc[idx]['source']}")
        axes[0, i].axis('off')

    # Plot Negative
    for i, idx in enumerate(samples_neg):
        img, _ = dataset[idx]
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Negative\n{dataset.df.iloc[idx]['source']}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print("   - Saved visualization.")


if __name__ == "__main__":
    # Initialize dataset WITHOUT transforms to see raw properties
    # Note: We must pass None for source_filter to analyze the WHOLE dataset
    full_dataset = COVIDxZipDataset(
        ZIP_FILE_PATH, TXT_FILE_NAME, transform=None)

    analyze_metadata(full_dataset)
    analyze_image_properties(full_dataset)
    visualize_grid(full_dataset)

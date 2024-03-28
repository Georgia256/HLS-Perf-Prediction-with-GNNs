import os
import pandas as pd
import numpy as np

# Define the percentage of data to keep for the new splits
portion = 0.01  # 50% of the original data

# Define the base directory containing the dataset folders
base_dir = "/home/georgia/Documents/gnn/HLS-Perf-Prediction-with-GNNs/GNN/dataset"

# Iterate over each dataset folder
for dataset_folder in os.listdir(base_dir):
    dataset_path = os.path.join(base_dir, dataset_folder)
    if os.path.isdir(dataset_path):
        print(f"Processing dataset folder: {dataset_folder}")
        
        # Iterate over each split directory within the dataset folder
        split_dir = os.path.join(dataset_path, "split")
        if os.path.exists(split_dir):
            for split_folder in os.listdir(split_dir):
                split_path = os.path.join(split_dir, split_folder)
                if os.path.isdir(split_path):
                    print(f"Processing split folder: {split_folder}")
                    
                    # Read the existing split files
                    train_idx = pd.read_csv(os.path.join(split_path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
                    valid_idx = pd.read_csv(os.path.join(split_path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
                    test_idx = pd.read_csv(os.path.join(split_path, "test.csv.gz"), compression="gzip", header=None).values.T[0]
                    
                    # Take a portion of the indices
                    new_train_idx = np.random.choice(train_idx, size=int(len(train_idx) * portion), replace=False)
                    new_valid_idx = np.random.choice(valid_idx, size=int(len(valid_idx) * portion), replace=False)
                    new_test_idx = np.random.choice(test_idx, size=int(len(test_idx) * portion), replace=False)
                    
                    # Write the new split files
                    os.makedirs(os.path.join(split_path, "small"), exist_ok=True)
                    pd.DataFrame(new_train_idx).to_csv(os.path.join(split_path, "small", "train.csv.gz"), compression="gzip", index=False, header=False)
                    pd.DataFrame(new_valid_idx).to_csv(os.path.join(split_path, "small", "valid.csv.gz"), compression="gzip", index=False, header=False)
                    pd.DataFrame(new_test_idx).to_csv(os.path.join(split_path, "small", "test.csv.gz"), compression="gzip", index=False, header=False)

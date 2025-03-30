import argparse
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import os

warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="Preprocess scRNAseq count data")
parser.add_argument("--train-set", required=True, help="Path to training set CSV")
parser.add_argument("--test-set", required=True, help="Path to test set CSV")
parser.add_argument("--train-label", required=True, help="Path to training labels CSV")
parser.add_argument("--test-label", required=True, help="Path to test labels CSV")
parser.add_argument("--output-dir", required=True, help="Directory to save preprocessed data")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")

args = parser.parse_args()

print("Importing data...")

# Load data
data_train = pd.read_csv(args.train_set, index_col=0).T
data_test = pd.read_csv(args.test_set, index_col=0).T
labels_train = pd.read_csv(args.train_label, header=0)
labels_test = pd.read_csv(args.test_label, header=0)

# Encode labels
labels_train_map = {label: idx for idx, label in enumerate(labels_train['celltype'].unique())}
labels_train_df = labels_train['celltype'].map(labels_train_map)
labels_test_df = labels_test['celltype'].map(labels_train_map)  # Ensure consistency

# Reset index to align with features
labels_train_df = labels_train_df.reset_index(drop=True)
labels_test_df = labels_test_df.reset_index(drop=True)
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

print("Processing data...")

# Convert to AnnData
adata = sc.AnnData(pd.concat([data_train, data_test], ignore_index=True),
                   obs=pd.concat([labels_train_df, labels_test_df], ignore_index=True).to_frame())

# Preprocessing
sc.pp.filter_cells(adata, min_genes=200)  # Filter cells with less than 200 genes
sc.pp.filter_genes(adata, min_cells=3)  # Filter genes that appear in less than 3 cells
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize the count data
sc.pp.log1p(adata)  # Log transform the data

# Select highly variable genes
n_top_genes = 2000
sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
highly_variable_genes = adata.var.highly_variable
adata = adata[:, highly_variable_genes]

# Split into train and test
n_train = data_train.shape[0]
adata_train = adata[:n_train]
adata_test = adata[n_train:]

# Save preprocessed data in AnnData format
os.makedirs(args.output_dir, exist_ok=True)
adata.write_h5ad(f"{args.output_dir}/preprocessed_data.h5ad")

print("Preprocessing complete! Data saved in ", args.output_dir)


# Define dataset class for DataLoader
class AnnDataset(Dataset):
    def __init__(self, adata):
        self.data = torch.tensor(adata.X, dtype=torch.float32)
        self.labels = torch.tensor(adata.obs.values.flatten(), dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create datasets
train_dataset = AnnDataset(adata_train)
test_dataset = AnnDataset(adata_test)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Save PyTorch datasets
torch.save(train_dataset, f"{args.output_dir}/train_dataset.pt")
torch.save(test_dataset, f"{args.output_dir}/test_dataset.pt")

# Save DataLoader
torch.save(train_loader, f"{args.output_dir}/train_loader.pt")
torch.save(test_loader, f"{args.output_dir}/test_loader.pt")

print("DataLoader and dataset saved in ", args.output_dir)
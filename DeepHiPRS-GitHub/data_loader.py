"""Prepare SNP features, disease status and predefined hierarchical mappings."""
import numpy as np
import pandas as pd
import torch


def load_data(genotype_file, topology_file, split_file):
    # Read 0/1/2-recoded genotypes and encoded disease status.
    data = pd.read_csv(genotype_file, sep="\t", dtype={"IID": str})
    if data.columns[0] != "IID" or data.columns[-1] != "AD_label":
        raise ValueError("Genotype columns must be IID, SNP columns, AD_label")
    if data.IID.isna().any() or data.IID.duplicated().any():
        raise ValueError("Missing or duplicate IID")
    snp = data.iloc[:, 1:-1].apply(pd.to_numeric, errors="raise")
    labels = data[["AD_label"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(snp.to_numpy()).all() or not np.isfinite(labels.to_numpy()).all():
        raise ValueError("Missing or non-finite SNP/label values")
    if not np.isin(snp.to_numpy(), [0, 1, 2]).all() or not np.isin(labels.to_numpy(), [0, 1, 2]).all():
        raise ValueError("SNP dosages and AD_label must be coded 0/1/2")

    # Read the supplied SNP-gene and gene-pathway mappings.
    topology = pd.read_csv(topology_file)
    for name in ["layer0_node", "layer1_node", "layer2_node"]:
        values = pd.to_numeric(topology[name], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all() or (values < -1).any():
            raise ValueError("Topology indices must be integers >= -1")
        topology[name] = values.astype(int)
    if topology.layer0_node.max() >= snp.shape[1]:
        raise ValueError("Topology SNP index exceeds SNP column count")
    for level in [1, 2]:
        values = set(topology[f"layer{level}_node"]) - {-1}
        if not values or values != set(range(max(values) + 1)):
            raise ValueError("Gene/pathway indices must start at zero without gaps")

    # Read fixed train/test assignments.
    splits = pd.read_csv(split_file, dtype={"IID": str})
    if splits.IID.isna().any() or splits.IID.duplicated().any() or set(splits.IID) != set(data.IID):
        raise ValueError("Split table must assign each IID exactly once")
    if not set(splits.split).issubset({"train", "test"}):
        raise ValueError("Only train and test are allowed")
    positions = pd.Series(np.arange(len(data)), index=data.IID)
    train_idx = positions.loc[splits.loc[splits.split == "train", "IID"]].to_numpy()
    test_idx = positions.loc[splits.loc[splits.split == "test", "IID"]].to_numpy()
    if len(train_idx) < 3 or len(test_idx) < 3:
        raise ValueError("Both train and test need at least three samples")
    return snp, labels, data[["IID"]], topology, train_idx, test_idx


def standardize(train_df, test_df):
    """Standardize the data."""
    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()
    train_mean = np.mean(train_np, axis=0)
    train_std = np.std(train_np, axis=0, ddof=1)
    train_std_safe = np.where(train_std == 0, 1, train_std)
    train_np = (train_np - train_mean) / train_std_safe
    train_np[:, train_std == 0] = 0
    test_mean = np.mean(test_np, axis=0)
    test_std = np.std(test_np, axis=0, ddof=1)
    test_std_safe = np.where(test_std == 0, 1, test_std)
    test_np = (test_np - test_mean) / test_std_safe
    test_np[:, test_std == 0] = 0
    return torch.from_numpy(train_np).float(), torch.from_numpy(test_np).float()

"""Model training, multilevel individual PRS calculation and PRS evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_loader import load_data, standardize
from model import DeepHiPRS
from trainer import Trainer
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepHiPRS once using a fixed train/test split")
    parser.add_argument("--genotype-file", default=str(ROOT / "data" / "simulated_genotypes.tsv"))
    parser.add_argument("--topology-file", default=str(ROOT / "data" / "simulated_topology.csv"))
    parser.add_argument("--split-file", default=str(ROOT / "data" / "simulated_split.csv"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--lambda-snp", type=float, default=0.12)
    parser.add_argument("--lambda-gene", type=float, default=0.15)
    parser.add_argument("--lambda-pathway", type=float, default=2.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--snp-lr-mult", type=float, default=1.0)
    parser.add_argument("--gene-lr-mult", type=float, default=0.1)
    parser.add_argument("--pathway-lr-mult", type=float, default=0.1)
    parser.add_argument("--checkpoint-metric", choices=["test_loss", "snp_prs", "gene_prs", "pathway_prs", "pred_mean"], default="snp_prs")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def save_head_weights(model, topology, snp_names, output):
    # Export output-head weights for level-specific genetic feature interpretation.
    for level, index in [("snp", 0), ("gene", 1), ("pathway", 2)]:
        weights = getattr(model, f"{level}_head").weight.detach().cpu().numpy().reshape(-1)
        if level == "snp":
            names = list(snp_names)
        else:
            name_column = f"layer{index}_name"
            if name_column in topology:
                mapping = topology.drop_duplicates(f"layer{index}_node").set_index(f"layer{index}_node")[name_column]
                names = [str(mapping.loc[i]) for i in range(len(weights))]
            else:
                names = [f"{level}_{i}" for i in range(len(weights))]
        pd.DataFrame({"feature": names, "head_weight": weights,
                      "abs_head_weight": np.abs(weights)}).to_csv(output / f"importance_{level}.csv", index=False)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing results preserved; choose a new --output-dir: {output}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print("Using:", device)
    # Prepare SNP features, disease status and the supplied train/test partition.
    snp, labels, iid, topology, train_idx, test_idx = load_data(args.genotype_file, args.topology_file, args.split_file)
    train_x, test_x = standardize(snp.iloc[train_idx], snp.iloc[test_idx])
    train_y = torch.tensor(labels.iloc[train_idx].values, dtype=torch.float32)
    test_y = torch.tensor(labels.iloc[test_idx].values, dtype=torch.float32)
    print(f"Train: {len(train_idx)} samples; test: {len(test_idx)} samples; SNPs: {snp.shape[1]}")
    print(f"Batch size: {train_x.size(0)} (all training samples)")
    output.mkdir(parents=True, exist_ok=True)
    configuration = dict(vars(args), batch_size=train_x.size(0), standardization="original")
    (output / "run_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    # Train the hierarchical network with multilevel joint supervision.
    model = DeepHiPRS(snp.shape[1], topology)
    solver = Trainer(
        model=model, lambda_snp=args.lambda_snp, lambda_gene=args.lambda_gene,
        lambda_pathway=args.lambda_pathway, epochs=args.epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        save_path=str(output), device=device,
        checkpoint_metric=args.checkpoint_metric, snp_lr_mult=args.snp_lr_mult,
        gene_lr_mult=args.gene_lr_mult, pathway_lr_mult=args.pathway_lr_mult,
    )
    best_score, best_epoch = solver.fit(train_x, train_y, test_x, test_y)

    # Calculate individual PRS using learned level-specific weights.
    model.eval()
    with torch.no_grad():
        predictions = model.prs(test_x.to(device))
    result = pd.DataFrame({"IID": iid.iloc[test_idx].IID.tolist(),
                           "AD_Label": labels.iloc[test_idx].AD_label.tolist()})
    metrics = []
    for level, values in predictions.items():
        result[level] = values.detach().cpu().numpy().reshape(-1)
        r = float(pearsonr(result[level], result.AD_Label)[0])
        result[f"r({level}, AD_Label)"] = np.nan
        result.loc[0, f"r({level}, AD_Label)"] = r
        metrics.append({"PRS_layer": level, "r": r, "abs_r": abs(r), "n_samples": len(result)})
    # Save individual PRS and their correlations with disease status.
    result.to_csv(output / "DeepHiPRS_PRS_r.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "correlation_summary.csv", index=False)
    save_head_weights(model, topology, snp.columns, output)
    print(f"Best {args.checkpoint_metric}: {best_score:.6f}, epoch: {best_epoch}")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()

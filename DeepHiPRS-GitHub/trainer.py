"""Joint supervision across biological levels with level-specific regularization."""
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import pearsonr


class Trainer:
    # Optimize the shared network and risk heads with separate learning rates.
    def __init__(
        self,
        model,
        lambda_snp,
        lambda_gene,
        lambda_pathway,
        epochs,
        learning_rate,
        weight_decay,
        save_path,
        device,
        checkpoint_metric,
        snp_lr_mult,
        gene_lr_mult,
        pathway_lr_mult,
    ):
        self.model = model.to(device)
        self.lambda_snp = lambda_snp
        self.lambda_gene = lambda_gene
        self.lambda_pathway = lambda_pathway
        self.epochs = epochs
        self.save_path = save_path
        self.device = device
        self.checkpoint_metric = checkpoint_metric
        self.snp_lr_mult = snp_lr_mult
        self.gene_lr_mult = gene_lr_mult
        self.pathway_lr_mult = pathway_lr_mult

        snp_param_ids = {id(param) for param in model.snp_head.parameters()}
        gene_param_ids = {id(param) for param in model.gene_head.parameters()}
        pathway_param_ids = {id(param) for param in model.pathway_head.parameters()}
        head_param_ids = snp_param_ids | gene_param_ids | pathway_param_ids
        other_params = [
            param for param in model.parameters() if id(param) not in head_param_ids
        ]
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": list(model.snp_head.parameters()),
                    "lr": learning_rate * snp_lr_mult,
                    "weight_decay": weight_decay,
                },
                {
                    "params": list(model.gene_head.parameters()),
                    "lr": learning_rate * gene_lr_mult,
                    "weight_decay": weight_decay,
                },
                {
                    "params": list(model.pathway_head.parameters()),
                    "lr": learning_rate * pathway_lr_mult,
                    "weight_decay": weight_decay,
                },
                {
                    "params": other_params,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
            ]
        )

    # Joint objective: three MSE losses and level-specific smooth L1 penalties.
    def _loss(self, y_pathway, y_snp, y_gene, target):
        criterion = torch.nn.MSELoss()
        target = target.float()
        loss_snp = criterion(y_snp, target)
        loss_gene = criterion(y_gene, target)
        loss_pathway = criterion(y_pathway, target)
        eps = 1e-8
        reg_snp = torch.sum(torch.sqrt(self.model.snp_head.weight.pow(2) + eps))
        reg_gene = torch.sum(torch.sqrt(self.model.gene_head.weight.pow(2) + eps))
        reg_pathway = torch.sum(torch.sqrt(self.model.pathway_head.weight.pow(2) + eps))
        total = (
            loss_snp
            + loss_gene
            + loss_pathway
            + self.lambda_snp * reg_snp
            + self.lambda_gene * reg_gene
            + self.lambda_pathway * reg_pathway
        )
        return total, loss_snp, loss_gene, loss_pathway

    # Select the checkpoint using the configured monitored test-set metric.
    def fit(self, train_x, train_y, test_x, test_y):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        test_x = test_x.to(self.device)
        test_y = test_y.to(self.device)
        best_score = float("inf") if self.checkpoint_metric == "test_loss" else -float("inf")
        best_epoch = 0
        checkpoint_path = os.path.join(self.save_path, "DeepHiPRS_best_model.pt")
        history = []

        for epoch in tqdm(range(self.epochs), desc="Training", ncols=100):
            self.model.train()
            # Jointly optimize all three risk outputs in full-batch mode.
            self.optimizer.zero_grad()
            y_pathway, y_snp, y_gene, _, _ = self.model(train_x)
            loss, _, _, _ = self._loss(y_pathway, y_snp, y_gene, train_y)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                y_pathway, y_snp, y_gene, _, _ = self.model(test_x)
                test_loss, loss_snp, loss_gene, loss_pathway = self._loss(
                    y_pathway, y_snp, y_gene, test_y
                )

                labels = test_y.detach().cpu().numpy().reshape(-1)
                r_snp = float(pearsonr(y_snp.detach().cpu().numpy().reshape(-1), labels)[0])
                r_gene = float(pearsonr(y_gene.detach().cpu().numpy().reshape(-1), labels)[0])
                r_pathway = float(pearsonr(y_pathway.detach().cpu().numpy().reshape(-1), labels)[0])
                prs_values = np.array([r_snp, r_gene, r_pathway], dtype=float)
                pred_mean = np.nanmean(prs_values)

            metric_values = {
                "test_loss": test_loss.item(),
                "snp_prs": r_snp,
                "gene_prs": r_gene,
                "pathway_prs": r_pathway,
                "pred_mean": pred_mean,
            }
            current_score = metric_values[self.checkpoint_metric]
            if np.isnan(current_score):
                current_score = float("inf") if self.checkpoint_metric == "test_loss" else -float("inf")

            if self.checkpoint_metric == "test_loss":
                improved = current_score < best_score
            else:
                improved = current_score > best_score

            if improved:
                best_score = current_score
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), checkpoint_path)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "test_loss": test_loss.item(),
                    "snp_prs": r_snp,
                    "gene_prs": r_gene,
                    "pathway_prs": r_pathway,
                    "pred_mean": pred_mean,
                    "checkpoint_metric": self.checkpoint_metric,
                    "checkpoint_score": current_score,
                    "best_score_so_far": best_score,
                    "best_epoch_so_far": best_epoch,
                    "loss_snp": loss_snp.item(),
                    "loss_gene": loss_gene.item(),
                    "loss_pathway": loss_pathway.item(),
                }
            )

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True), strict=False)
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(self.save_path, "training_history.csv"), index=False)
        return best_score, best_epoch

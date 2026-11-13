# DeepHiPRS: A Deep Learning Method for Calculating Hierarchical Individual Polygenic Risk Scores

DeepHiPRS constructs a hierarchical SNP-gene-pathway network with a separate
risk output head at each biological level. Multilevel joint supervision and
level-specific regularization guide the calculation of individual polygenic
risk scores at the SNP, gene and pathway levels.

## Run

Use Python 3.8 or later. From this directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

## Files

- `main.py`: command-line parameters and execution.
- `model.py`: sparse network and three output heads.
- `trainer.py`: joint objective, optimizer and training.
- `data_loader.py`: input loading and standardization.
- `data/`: simulated genotypes, topology and train/test assignments.

## Data

The example contains 180 participants, 60 SNPs, 12 genes and 8 pathways, with
144 training and 36 test participants. No real participant data are included.

- `simulated_genotypes.tsv`: `IID`, ordered SNP columns, then `AD_label`.
  Genotypes and labels are encoded as 0/1/2.
- `simulated_topology.csv`: zero-based `layer0_node`, `layer1_node` and
  `layer2_node`, with corresponding names. SNP indices follow genotype column
  order. Connections may be many-to-one or one-to-many; `-1` means no mapping.
- `simulated_split.csv`: `IID,split`, with each participant assigned once to
  `train` or `test`.

## Training

Defaults: 500 epochs, full-batch Adam, learning rate 0.001, weight decay 0.00001,
head learning-rate multipliers 1/0.1/0.1 and regularization coefficients
0.12/0.15/2.5 for SNP/gene/pathway.

The joint objective combines prediction losses and level-specific regularization:

$$
\mathcal{L}_{\mathrm{total}} =
\sum_{m\in\{\mathrm{SNP},\mathrm{Gene},\mathrm{Pathway}\}}
\left[\operatorname{MSE}(\hat{\mathbf{y}}_m,\mathbf{y})
+ \lambda_m R_{\varepsilon}(\mathbf{w}_m)\right].
$$

Here, $\hat{\mathbf{y}}_m$ is the prediction at level $m$, $\mathbf{y}$ contains
the disease-status labels, $\mathbf{w}_m$ contains the output-head weights,
$\lambda_m$ controls regularization, and $R_{\varepsilon}$ is the smooth L1 penalty.

## Outputs

- `DeepHiPRS_best_model.pt`: selected model parameters.
- `DeepHiPRS_PRS_r.csv`: test participants' three PRS.
- `correlation_summary.csv`: signed and absolute Pearson correlations.
- `training_history.csv`: per-epoch losses and correlations.
- `importance_snp.csv`, `importance_gene.csv`, `importance_pathway.csv`:
  head coefficients, not full nonlinear attributions.
- `run_config.json`: execution parameters.


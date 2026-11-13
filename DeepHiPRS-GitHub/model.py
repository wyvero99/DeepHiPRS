"""Hierarchical risk learning with SNP-, gene- and pathway-level output heads."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SparseLayer(nn.Module):
    """Learn aggregation weights only for connections in the supplied topology."""
    def __init__(self, pairs, input_size, output_size):
        super().__init__()
        indices = torch.tensor(np.asarray(pairs).T[[1, 0]], dtype=torch.long)
        self.register_buffer("indices", indices)
        self.shape = (output_size, input_size)
        self.kernel = nn.Parameter(torch.empty(len(pairs), 1))
        nn.init.xavier_uniform_(self.kernel)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, inputs):
        matrix = torch.sparse_coo_tensor(self.indices, self.kernel[:, 0], self.shape,
                                         device=inputs.device)
        return torch.sparse.mm(matrix, inputs.T).T + self.bias


class DeepHiPRS(nn.Module):
    def __init__(self, input_size, topology):
        super().__init__()
        gene_size = int(topology.layer1_node.max()) + 1
        pathway_size = int(topology.layer2_node.max()) + 1
        # Generate a level-specific risk output from each feature representation.
        self.snp_head = nn.Linear(input_size, 1)
        self.gene_head = nn.Linear(gene_size, 1)
        self.pathway_head = nn.Linear(pathway_size, 1)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # Restrict inter-level aggregation to the supplied biological mappings.
        for columns, shape in [
            (["layer0_node", "layer1_node"], (input_size, gene_size)),
            (["layer1_node", "layer2_node"], (gene_size, pathway_size)),
        ]:
            # Missing mappings prevent aggregation, but preserve the direct heads.
            pairs = topology[columns].drop_duplicates()
            pairs = pairs[(pairs >= 0).all(axis=1)].to_numpy(dtype=np.int64)
            self.layers.append(SparseLayer(pairs, *shape))
            self.batch_norms.append(nn.BatchNorm1d(shape[1], affine=False))

    def representations(self, x):
        # Progressively aggregate SNP features into gene and pathway features.
        gene = self.batch_norms[0](torch.tanh(self.layers[0](x)))
        pathway = self.batch_norms[1](torch.tanh(self.layers[1](gene)))
        return gene, pathway

    def forward(self, x):
        """Provide three risk outputs for joint supervision by disease status."""
        gene, pathway = self.representations(x)
        return self.pathway_head(pathway), self.snp_head(x), self.gene_head(gene), gene, pathway

    def prs(self, x):
        """Calculate individual PRS from learned head weights."""
        gene, pathway = self.representations(x)
        return {
            "SNP_PRS": F.linear(x, self.snp_head.weight, bias=None),
            "Gene_PRS": F.linear(gene, self.gene_head.weight, bias=None),
            "Pathway_PRS": F.linear(pathway, self.pathway_head.weight, bias=None),
        }

# hgt_gnn.py
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv, LayerNorm
from torch_geometric.typing import EdgeType, NodeType


class HeteroHGT(torch.nn.Module):
    """
    A relbench-style wrapper around PyG's HGTConv:
    - stacks num_layers HGTConv layers
    - applies per-node-type LayerNorm after each layer (like relbench)
    - optional dropout
    - robust to node types that receive no messages in a sampled batch:
      falls back to the previous x_dict entry for that node type.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        act: str = "relu",  # "relu", "gelu", or "none"
    ):
        super().__init__()

        # HGTConv needs metadata = (node_types, edge_types)
        self.metadata = (list(node_types), list(edge_types))

        # HGTConv requires out_channels divisible by heads (PyG enforces this).
        # Choose heads accordingly (e.g., channels=256 -> heads=4/8/16).
        if channels % heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by heads={heads} "
                f"(HGTConv constraint)."
            )

        self.convs = torch.nn.ModuleList(
            [
                HGTConv(
                    in_channels=channels,
                    out_channels=channels,
                    metadata=self.metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for nt in node_types:
                norm_dict[nt] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.dropout = float(dropout)
        self.act = act.lower().strip()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def _apply_act(self, x: Tensor) -> Tensor:
        if self.act == "none":
            return x
        if self.act == "gelu":
            return F.gelu(x)
        # default
        return F.relu(x)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        # num_sampled_* are unused by HGTConv; kept for API compatibility
        for conv, norm_dict in zip(self.convs, self.norms):
            edge_index_sub = {
                et: edge_index_dict[et]
                for et in conv.edge_types
                if et in edge_index_dict
            }
            
            out_dict = conv(x_dict, edge_index_sub)

            new_x_dict = dict(x_dict)  # keep everything by default

            for nt in conv.dst_node_types:  # dst types only (HGTConv defines these)
                out = out_dict.get(nt, None)
                h = x_dict[nt] if out is None else out   # fallback if None
                h = norm_dict[nt](h)
                h = self._apply_act(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
                new_x_dict[nt] = h

            x_dict = new_x_dict


        return x_dict

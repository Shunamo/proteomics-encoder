"""
GIN (Graph Isomorphism Network) 모델 구현

논문: Xu et al. (2018) "How powerful are graph neural networks?"
GNN-SubNet 논문에서 사용한 아키텍처
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree


class GINConv(MessagePassing):
    """
    GIN Convolution Layer
    
    논문 Equation (3):
    h_v^(k) = MLP^(k)((1 + ε^(k)) * h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super(GINConv, self).__init__(aggr='add')
        self.eps = eps
        
        # MLP: 2 fully connected layers + batch normalization
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
        """
        # Self-loops 추가
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Message passing
        out = self.propagate(edge_index, x=x)
        
        # GIN formula: (1 + ε) * x + aggregated neighbors
        out = (1 + self.eps) * x + out
        
        # MLP
        out = self.mlp(out)
        
        return out
    
    def message(self, x_j):
        """Message function: neighbor features"""
        return x_j


class GIN(nn.Module):
    """
    GIN (Graph Isomorphism Network)
    
    논문 아키텍처:
    - 3 GIN layers
    - Global pooling (mean/max/sum)
    - Final MLP
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 256,
        num_layers: int = 3,
        pooling: str = 'mean'  # 'mean', 'max', 'sum', 'concat'
    ):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        for i in range(num_layers):
            eps = 0.0 if i == 0 else 0.0  # 논문에서는 보통 0.0 사용
            self.gin_layers.append(GINConv(hidden_dim, hidden_dim, eps=eps))
        
        # Pooling output dimension
        if pooling == 'concat':
            pool_dim = hidden_dim * 3  # mean + max + sum
        else:
            pool_dim = hidden_dim
        
        # Final MLP
        self.fc = nn.Sequential(
            nn.Linear(pool_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes] (optional)
        
        Returns:
            Graph embedding [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        
        # GIN layers
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
        
        # Global pooling
        if batch is None:
            # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling == 'mean':
            graph_emb = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_emb = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            graph_emb = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            sum_pool = global_add_pool(x, batch)
            graph_emb = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Final MLP
        graph_emb = self.fc(graph_emb)
        
        return graph_emb

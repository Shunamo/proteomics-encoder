"""
Graph Classifier for Dementia Prediction

GNN-SubNet 논문 방식:
- Graph classification: Dementia vs Control
- GIN backbone
- Binary classification head
"""

import torch
import torch.nn as nn
from .gin import GIN


class GraphClassifier(nn.Module):
    """
    Graph Classification Model
    
    논문 아키텍처:
    - GIN encoder
    - Classification head
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        embedding_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        pooling: str = 'mean',
        dropout: float = 0.1
    ):
        super(GraphClassifier, self).__init__()
        
        # GIN encoder
        self.gin = GIN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            pooling=pooling
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes] (optional)
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Graph embedding
        graph_emb = self.gin(x, edge_index, batch)
        
        # Classification
        logits = self.classifier(graph_emb)
        
        return logits
    
    def get_embedding(self, x, edge_index, batch=None):
        """
        Graph embedding만 반환 (View B용)
        
        Returns:
            Graph embedding [batch_size, embedding_dim]
        """
        return self.gin(x, edge_index, batch)

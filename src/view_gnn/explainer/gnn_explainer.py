"""
GNNExplainer for Model-wide Explanations

GNN-SubNet 논문 방식 (공식 구현의 GNNExplainer 사용):
- Node importance 계산
- Edge importance 계산
- Model-wide explanations (전체 모델에 대한 설명)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torch_geometric.data import Data
import sys
import os

# 공식 구현의 GNNExplainer 사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GNN-SubNet', 'GNNSubNet'))
from gnn_explainer import GNNExplainer


class ModifiedGNNExplainer:
    """
    Modified GNNExplainer
    
    1. Node importance: Gradient-based importance
    2. Edge importance: Edge weight * node importance
    3. Model-wide: 모든 샘플에 대한 평균 importance
    """
    
    def __init__(self, model: nn.Module, epochs: int = 100, lr: float = 0.01):
        """
        Args:
            model: 학습된 GraphClassifier 모델
            epochs: GNNExplainer 학습 epoch 수
            lr: Learning rate
        """
        self.model = model
        self.model.eval()
        self.epochs = epochs
        self.lr = lr
        
        # 공식 구현의 GNNExplainer 사용
        self.explainer = GNNExplainer(
            model,
            epochs=epochs,
            lr=lr,
            return_type='log_prob',
            log=False  # Progress bar 비활성화
        )
    
    def explain_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Single node importance 계산 (PyG GNNExplainer 사용)
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            node_idx: Target node index
            target_class: Target class (None이면 예측 클래스 사용)
        
        Returns:
            Node importance scores [num_nodes]
        """
        data = Data(x=x, edge_index=edge_index)
        node_feat_mask, edge_mask = self.explainer.explain_node(node_idx, data)
        
        # Node importance = node feature mask의 평균 (feature dimension에 대해)
        if node_feat_mask.dim() > 1:
            node_importance = node_feat_mask.mean(dim=-1)
        else:
            node_importance = node_feat_mask
        
        return node_importance
    
    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph-level importance 계산 (공식 구현의 GNNExplainer 사용)
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            target_class: Target class (None이면 예측 클래스 사용)
        
        Returns:
            node_importance: [num_nodes]
            edge_importance: [num_edges]
        """
        data = Data(x=x, edge_index=edge_index)
        
        # 모델을 data 인자를 받도록 래핑
        class ModelWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
            
            def forward(self, data=None, batch=None, x=None, edge_index=None, **kwargs):
                # 공식 구현은 data 인자를 사용
                if data is not None:
                    x = data.x
                    edge_index = data.edge_index
                # 우리 모델은 x, edge_index를 직접 받음
                return self.original_model(x, edge_index, batch)
        
        # Wrapper로 모델 감싸기
        wrapped_model = ModelWrapper(self.model)
        wrapped_explainer = GNNExplainer(
            wrapped_model,
            epochs=self.epochs,
            lr=self.lr,
            return_type='log_prob',
            log=False
        )
        
        # 공식 구현의 GNNExplainer 사용 (explain_graph 메서드)
        node_feat_mask, edge_mask = wrapped_explainer.explain_graph(data)
        
        # Node importance = node feature mask의 평균 (feature dimension에 대해)
        if node_feat_mask.dim() > 1:
            node_importance = node_feat_mask.mean(dim=-1)
        else:
            node_importance = node_feat_mask
        
        # Edge importance = edge mask
        edge_importance = edge_mask
        
        # Edge weights가 있으면 곱하기
        if edge_attr is not None:
            edge_weights = edge_attr.squeeze() if edge_attr.dim() > 1 else edge_attr
            edge_importance = edge_importance * edge_weights
        
        return node_importance, edge_importance
    
    def calc_edge_importance(self, node_mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Edge importance 계산 (공식 구현 방식)
        
        Args:
            node_mask: Node importance scores [num_nodes]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Edge importance [num_edges]
        """
        rows, cols = edge_index[0], edge_index[1]
        edge_importance = (node_mask[rows] + node_mask[cols]) / 2.0
        return edge_importance
    
    def explain_model_wide(
        self,
        graphs: List[Data],
        target_class: int = 1,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Model-wide importance 계산 (모든 샘플에 대한 평균)
        
        Args:
            graphs: Graph 리스트
            target_class: Target class (1 = Dementia)
            batch_size: Batch size
        
        Returns:
            avg_node_importance: [num_nodes]
            avg_edge_importance: [num_edges]
        """
        device = next(self.model.parameters()).device
        
        # 첫 번째 그래프로 크기 확인
        first_graph = graphs[0]
        num_nodes = first_graph.x.size(0)
        num_edges = first_graph.edge_index.size(1)
        
        # 누적 importance
        node_importance_sum = torch.zeros(num_nodes, device=device)
        edge_importance_sum = torch.zeros(num_edges, device=device)
        count = 0
        
        # Batch 처리
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i+batch_size]
            
            for graph in batch_graphs:
                graph = graph.to(device)
                
                try:
                    node_imp, edge_imp = self.explain_graph(
                        graph.x,
                        graph.edge_index,
                        graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                        target_class=target_class
                    )
                    
                    node_importance_sum += node_imp
                    edge_importance_sum += edge_imp
                    count += 1
                except Exception as e:
                    print(f"Warning: Graph {i} explanation failed: {e}")
                    continue
        
        # 평균
        avg_node_importance = node_importance_sum / max(count, 1)
        avg_edge_importance = edge_importance_sum / max(count, 1)
        
        return avg_node_importance, avg_edge_importance

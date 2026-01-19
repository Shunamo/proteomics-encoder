"""
Community Detection for Disease Subnetwork Discovery

GNN-SubNet 논문 방식 (igraph 사용):
- Node importance 기반 edge weights 조정
- Louvain method로 community detection (igraph)
- Top communities = Disease subnetworks
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    print("Warning: igraph not found. Install with: pip install python-igraph")


def detect_disease_subnetworks(
    edge_index: torch.Tensor,
    node_importance: torch.Tensor,
    edge_importance: torch.Tensor,
    original_edge_weights: Optional[torch.Tensor] = None,
    top_k: int = 5,
    resolution: float = 1.0
) -> List[Dict]:
    """
    Disease subnetwork detection using community detection (igraph)
    
    GNN-SubNet 공식 구현 방식:
    - igraph 사용
    - community_multilevel (Louvain method)
    
    Args:
        edge_index: Edge indices [2, num_edges]
        node_importance: Node importance scores [num_nodes]
        edge_importance: Edge importance scores [num_edges]
        original_edge_weights: Original edge weights [num_edges] (optional)
        top_k: Number of top communities to return
        resolution: Not used for igraph (kept for compatibility)
    
    Returns:
        List of community dicts with:
        - nodes: List of node indices
        - importance: Average importance score
        - size: Number of nodes
    """
    if not HAS_IGRAPH:
        raise ImportError("igraph is required. Install with: pip install python-igraph")
    
    # Edge index를 numpy로 변환
    edge_index_np = edge_index.cpu().numpy()
    edge_importance_np = edge_importance.cpu().numpy()
    node_importance_np = node_importance.cpu().numpy()
    
    # igraph 그래프 생성
    num_nodes = node_importance.size(0)
    g = ig.Graph()
    g.add_vertices(num_nodes)
    
    # Edge 추가 및 weight 설정
    edges = []
    edge_weights = []
    
    for i in range(edge_index.size(1)):
        u = int(edge_index_np[0, i])
        v = int(edge_index_np[1, i])
        
        # Edge weight = edge importance
        weight = float(edge_importance_np[i])
        
        # Original edge weights가 있으면 곱하기
        if original_edge_weights is not None:
            weight = weight * float(original_edge_weights[i].item())
        
        # 최소값 보장 (0이면 division by zero 발생 가능)
        weight = max(weight, 1e-6)
        
        edges.append((u, v))
        edge_weights.append(weight)
    
    # Edge 추가
    g.add_edges(edges)
    g.es['weight'] = edge_weights
    
    # Node importance를 vertex attribute로 추가
    g.vs['importance'] = node_importance_np.tolist()
    
    # Louvain community detection (igraph)
    try:
        partition = g.community_multilevel(weights='weight')
    except Exception as e:
        print(f"Warning: igraph community detection failed: {e}")
        print("Falling back to unweighted community detection")
        partition = g.community_multilevel()
    
    # Community별 통계 계산
    community_stats = {}
    for comm_id in range(len(partition)):
        nodes = partition[comm_id]
        if len(nodes) == 0:
            continue
        
        importance_sum = sum(g.vs[node]['importance'] for node in nodes)
        avg_importance = importance_sum / len(nodes)
        
        community_stats[comm_id] = {
            'nodes': nodes,
            'importance': avg_importance,
            'size': len(nodes)
        }
    
    # Community 리스트 생성
    community_list = list(community_stats.values())
    
    # Importance 기준으로 정렬
    community_list.sort(key=lambda x: x['importance'], reverse=True)
    
    # Top-K 반환
    return community_list[:top_k]


def get_subnetwork_proteins(
    communities: List[Dict],
    node_to_protein: Dict[int, str]
) -> List[Dict]:
    """
    Community를 단백질 이름으로 변환
    
    Args:
        communities: Community 리스트
        node_to_protein: Node index → protein name 매핑
    
    Returns:
        Community 리스트 (protein names 포함)
    """
    result = []
    for comm in communities:
        proteins = [node_to_protein.get(node, f"Node_{node}") for node in comm['nodes']]
        result.append({
            **comm,
            'proteins': proteins
        })
    return result

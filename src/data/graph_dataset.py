"""
Graph Data Loader for GNN-SubNet

각 환자를 PPI 네트워크로 표현:
- Topology: PPI 네트워크 (모든 환자 동일)
- Node features: 환자별 단백질 발현량
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GraphDataLoader:
    """
    Graph 데이터 로더
    
    논문 방식:
    - 각 환자 = 하나의 그래프
    - 동일한 PPI topology
    - 환자별 다른 node features (proteomics)
    """
    
    def __init__(
        self,
        edges_file: str = None,
        mapping_file: str = None,
        master_file: str = None,
        score_threshold: int = 950
    ):
        """
        Args:
            edges_file: PPI edges CSV 파일
            mapping_file: UKB protein → STRING ID 매핑
            master_file: UKB master 데이터
        """
        if edges_file is None:
            edges_file = os.path.join(BASE_DIR, "data/stringdb/ukb_string_edges_topk20.csv")
        if mapping_file is None:
            mapping_file = os.path.join(BASE_DIR, "data/stringdb/ukb_to_string_map.csv")
        if master_file is None:
            master_file = os.path.join(BASE_DIR, "data/ukb/ukb_usable_master.parquet")
        
        self.edges_file = edges_file
        self.mapping_file = mapping_file
        self.master_file = master_file
        self.score_threshold = score_threshold
        
        # PPI 네트워크 로드
        self._load_ppi_network(threshold=score_threshold)
        
        # 매핑 로드
        self._load_mapping()
    
    def _load_ppi_network(self, threshold=950):
        """
        PPI 네트워크 로드 및 edge_index 생성
        
        Args:
            threshold: combined_score 최소값 (0-1000). 높을수록 더 확실한 연결만 유지
        """
        print(f"[GraphDataLoader] PPI 네트워크 로드: {self.edges_file}")
        
        df_edges = pd.read_csv(self.edges_file)
        print(f"   - 원본 Edges: {len(df_edges):,}개")
        
        # [긴급 수정 1] 그래프 다이어트: threshold 이상만 유지
        if 'combined_score' in df_edges.columns:
            df_edges = df_edges[df_edges['combined_score'] >= threshold]
            print(f"   - 필터링 후 Edges (score>={threshold}): {len(df_edges):,}개")
        else:
            print(f"   - ⚠️  combined_score 컬럼 없음, 필터링 스킵")
        
        # STRING ID → node index 매핑
        all_nodes = set(df_edges['p1'].unique()) | set(df_edges['p2'].unique())
        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.num_nodes = len(self.node_to_idx)
        
        print(f"   - Nodes: {self.num_nodes:,}개")
        
        # Edge index 생성
        edge_list = []
        for _, row in df_edges.iterrows():
            u = self.node_to_idx[row['p1']]
            v = self.node_to_idx[row['p2']]
            edge_list.append([u, v])
            edge_list.append([v, u])  # Undirected
        
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        print(f"   - Edge index shape: {self.edge_index.shape}")
        
        # Edge weights (combined_score)
        edge_weights = []
        for _, row in df_edges.iterrows():
            score = row.get('combined_score', 700) / 1000.0  # 0-1 정규화
            edge_weights.append(score)
            edge_weights.append(score)  # Undirected
        
        self.edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    def _load_mapping(self):
        """UKB protein → STRING ID 매핑 로드"""
        print(f"[GraphDataLoader] 매핑 로드: {self.mapping_file}")
        
        df_map = pd.read_csv(self.mapping_file)
        
        # UKB protein (소문자) → STRING ID
        self.ukb_to_string = {}
        for _, row in df_map.iterrows():
            ukb_protein = row['gene_symbol'].lower()
            string_id = row['string_protein_id']
            if string_id in self.node_to_idx:
                self.ukb_to_string[ukb_protein] = string_id
        
        print(f"   - 매핑된 단백질: {len(self.ukb_to_string):,}개")
    
    def create_graph(
        self,
        protein_features: pd.Series,
        label: Optional[int] = None
    ) -> Data:
        """
        환자별 그래프 생성
        
        Args:
            protein_features: 단백질 발현량 (Series, index=protein_name)
            label: 타겟 레이블 (0/1, optional)
        
        Returns:
            PyTorch Geometric Data 객체
        """
        # Node features 초기화 (모든 노드는 0으로 시작)
        node_features = torch.zeros(self.num_nodes, dtype=torch.float)
        
        # UKB 단백질 → STRING ID → node index로 매핑하여 features 할당
        for ukb_protein, value in protein_features.items():
            if pd.notna(value):
                ukb_protein_lower = ukb_protein.lower()
                if ukb_protein_lower in self.ukb_to_string:
                    string_id = self.ukb_to_string[ukb_protein_lower]
                    if string_id in self.node_to_idx:
                        node_idx = self.node_to_idx[string_id]
                        node_features[node_idx] = float(value)
        
        # Graph data 생성
        data = Data(
            x=node_features.unsqueeze(1),  # [num_nodes, 1]
            edge_index=self.edge_index,
            edge_attr=self.edge_weights.unsqueeze(1),  # [num_edges, 1]
            y=torch.tensor([label], dtype=torch.long) if label is not None else None
        )
        
        return data
    
    def create_graph_dataset(
        self,
        df: pd.DataFrame,
        protein_cols: List[str],
        label_col: str = 'target_dementia',
        eid_col: str = 'eid'
    ) -> List[Data]:
        """
        전체 데이터셋을 그래프 리스트로 변환
        
        Args:
            df: 데이터프레임
            protein_cols: 단백질 컬럼 리스트
            label_col: 레이블 컬럼명
            eid_col: eid 컬럼명
        
        Returns:
            그래프 리스트
        """
        print(f"[GraphDataLoader] 그래프 데이터셋 생성...")
        print(f"   - 샘플 수: {len(df):,}")
        print(f"   - 단백질 수: {len(protein_cols):,}")
        
        graphs = []
        for idx, row in df.iterrows():
            protein_features = row[protein_cols]
            label = int(row[label_col]) if label_col in df.columns else None
            
            graph = self.create_graph(protein_features, label)
            graphs.append(graph)
            
            if (idx + 1) % 1000 == 0:
                print(f"   - 진행: {idx + 1:,}/{len(df):,}")
        
        print(f"   - 완료: {len(graphs):,}개 그래프 생성")
        return graphs
    
    def get_node_mapping(self) -> dict:
        """Node index → UKB protein name 매핑 반환"""
        # STRING ID → UKB protein 역매핑
        string_to_ukb = {v: k for k, v in self.ukb_to_string.items()}
        
        node_to_protein = {}
        for idx, string_id in self.idx_to_node.items():
            if string_id in string_to_ukb:
                node_to_protein[idx] = string_to_ukb[string_id]
            else:
                node_to_protein[idx] = string_id  # 매핑 안 된 경우 STRING ID 사용
        
        return node_to_protein

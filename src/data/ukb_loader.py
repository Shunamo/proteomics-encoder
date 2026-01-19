"""
On-the-fly UKB Graph Dataset

메모리 효율적인 그래프 데이터셋:
- 공통 edge_index는 한 번만 로드
- 각 샘플의 node features만 온더플라이로 생성
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Optional


class UKBGraphDataset(Dataset):
    """
    UKB Graph Dataset (On-the-fly)
    
    논문 방식:
    - 공통 PPI topology (edge_index)
    - 환자별 node features (x_i)
    - Graph classification (y_i)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        protein_cols: List[str],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        label_col: str = 'target_dementia',
        eid_col: str = 'eid',
        protein_to_node_idx: Optional[dict] = None,
        num_nodes: Optional[int] = None
    ):
        """
        Args:
            df: 데이터프레임 (환자별 단백질 발현량)
            protein_cols: 단백질 컬럼 리스트
            edge_index: 공통 PPI edge_index [2, num_edges]
            edge_attr: 공통 PPI edge weights [num_edges] (optional)
            label_col: 레이블 컬럼명
            eid_col: eid 컬럼명
            protein_to_node_idx: 단백질명 → node index 매핑
            num_nodes: 전체 노드 수
        """
        self.df = df.reset_index(drop=True)
        self.protein_cols = protein_cols
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.label_col = label_col
        self.eid_col = eid_col
        self.protein_to_node_idx = protein_to_node_idx or {}
        self.num_nodes = num_nodes or edge_index.max().item() + 1
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        환자별 그래프 생성 (온더플라이)
        
        Returns:
            Data(x=[num_nodes, 1], edge_index=[2, num_edges], y=[1])
        """
        row = self.df.iloc[idx]
        
        # Node features 초기화
        x = torch.zeros(self.num_nodes, 1, dtype=torch.float32)
        
        # 단백질 발현량을 node features로 매핑
        for protein in self.protein_cols:
            if protein in self.protein_to_node_idx:
                node_idx = self.protein_to_node_idx[protein]
                value = row[protein]
                if pd.notna(value):
                    x[node_idx, 0] = float(value)
        
        # Label
        y = torch.tensor([int(row[self.label_col])], dtype=torch.long) if self.label_col in row else None
        
        # Graph data 생성
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=y
        )
        
        return data

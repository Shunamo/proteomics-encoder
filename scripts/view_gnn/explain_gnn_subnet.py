"""
GNN-SubNet: Model Explanation & Disease Subnetwork Detection

학습된 모델을 사용하여:
1. Model-wide explanations
2. Disease subnetwork detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from src.data.graph_dataset import GraphDataLoader
from src.view_gnn.models.graph_classifier import GraphClassifier
from src.view_gnn.explainer.gnn_explainer import ModifiedGNNExplainer
from src.view_gnn.community.community_detection import detect_disease_subnetworks, get_subnetwork_proteins


def main():
    print("=" * 70)
    print("GNN-SubNet: Model Explanation & Subnetwork Detection")
    print("=" * 70)
    
    # 1. 데이터 로드 (학습과 동일한 방식)
    print("\n[1/4] 데이터 로드...")
    master_file = "../../data/ukb/ukb_usable_master.parquet"
    df = pd.read_parquet(master_file)
    
    # 단백질 컬럼 추출
    exclude_cols = ["eid", "sex", "target_age", "target_dementia", "participant.p42018"]
    protein_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and not c.startswith("pc__")
                   and not c.startswith("assess__")
                   and not c.startswith("online__")]
    
    # 데이터 정규화 (학습과 동일)
    from sklearn.preprocessing import StandardScaler
    print("   - 데이터 정규화 적용...")
    scaler = StandardScaler()
    df[protein_cols] = scaler.fit_transform(df[protein_cols])
    
    # 2. Graph 데이터 생성
    print("\n[2/4] Graph 데이터 생성...")
    score_threshold = 700
    loader = GraphDataLoader(score_threshold=score_threshold)
    
    # 학습과 동일한 샘플링
    target_total = 10000
    df_pos = df[df['target_dementia'] == 1]
    df_neg_all = df[df['target_dementia'] == 0]
    n_neg = min(target_total - len(df_pos), len(df_neg_all))
    df_neg = df_neg_all.sample(n=n_neg, random_state=42)
    df_sample = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    graphs = loader.create_graph_dataset(
        df_sample,
        protein_cols,
        label_col='target_dementia',
        eid_col='eid'
    )
    
    # Train/Test split (학습과 동일한 방식)
    labels = [g.y.item() for g in graphs]
    train_graphs, test_graphs = train_test_split(
        graphs,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"   - Test graphs: {len(test_graphs):,}개")
    
    # 3. 모델 로드
    print("\n[3/4] 모델 로드...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GraphClassifier(
        input_dim=1,
        hidden_dim=64,
        embedding_dim=256,
        num_layers=2,
        num_classes=2,
        pooling='mean',
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load('../../data/gnn_subnet_best_model.pt'))
    model.eval()
    print("   ✅ 모델 로드 완료")
    
    # 4. Model-wide Explanations & Subnetwork Detection
    print("\n[4/4] Model-wide Explanations & Subnetwork Detection...")
    
    explainer = ModifiedGNNExplainer(model, epochs=100, lr=0.01)
    
    # Dementia 샘플만 사용
    dementia_graphs = [g for g in test_graphs if g.y.item() == 1]
    print(f"   - Dementia 샘플: {len(dementia_graphs):,}개")
    
    # 여러 번 실행하여 평균 계산
    n_runs = 5
    print(f"   - Explainer 실행 횟수: {n_runs}회")
    
    all_node_imps = []
    all_edge_imps = []
    
    for run_idx in range(n_runs):
        print(f"   - Run {run_idx + 1}/{n_runs}...", end=' ')
        
        avg_node_imp, avg_edge_imp = explainer.explain_model_wide(
            dementia_graphs,
            target_class=1,
            batch_size=32
        )
        
        all_node_imps.append(avg_node_imp.cpu())
        all_edge_imps.append(avg_edge_imp.cpu())
        print("완료")
    
    # 평균 계산
    avg_node_imp = torch.stack(all_node_imps).mean(dim=0).to(device)
    avg_edge_imp = torch.stack(all_edge_imps).mean(dim=0).to(device)
    
    print(f"   - Node importance shape: {avg_node_imp.shape}")
    print(f"   - Edge importance shape: {avg_edge_imp.shape}")
    
    # Disease subnetwork detection
    first_graph = graphs[0]
    communities = detect_disease_subnetworks(
        first_graph.edge_index,
        avg_node_imp,
        avg_edge_imp,
        original_edge_weights=first_graph.edge_attr if hasattr(first_graph, 'edge_attr') else None,
        top_k=10,
        resolution=1.0
    )
    
    print(f"\n   ✅ 발견된 Disease Subnetworks: {len(communities)}개")
    
    # Node mapping
    node_to_protein = loader.get_node_mapping()
    
    # 단백질 이름으로 변환
    subnetworks = get_subnetwork_proteins(communities, node_to_protein)
    
    # 결과 저장
    results = []
    for i, subnet in enumerate(subnetworks):
        results.append({
            'subnetwork_id': i + 1,
            'num_proteins': subnet['size'],
            'importance': subnet['importance'],
            'proteins': ', '.join(subnet['proteins'][:20])
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('../../data/gnn_subnet_disease_subnetworks.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ Explanation & Subnetwork Detection 완료!")
    print("=" * 70)
    print("\n📁 출력 파일:")
    print("   - Disease Subnetworks: ../../data/gnn_subnet_disease_subnetworks.csv")
    print("\n📊 Top 5 Disease Subnetworks:")
    print(df_results.head().to_string(index=False))


if __name__ == "__main__":
    main()

"""
GNN-SubNet 전체 파이프라인

논문 방식:
1. Graph 데이터 생성 (PPI + Proteomics)
2. GIN 모델 학습
3. Model-wide explanations
4. Disease subnetwork detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import pickle
from pathlib import Path
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.graph_dataset import GraphDataLoader
from src.view_gnn.models.graph_classifier import GraphClassifier
# Explain 관련 import는 explain_gnn_subnet.py로 이동


def train_epoch(model, loader, optimizer, criterion, device):
    """한 epoch 학습 (BCEWithLogitsLoss 적용)"""
    model.train()
    total_loss = 0
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward
        logits = model(batch.x, batch.edge_index, batch.batch)
        
        # [수정] 차원 맞추기 (BCELoss는 [N] 형태의 float 입력을 원함)
        if logits.shape[1] == 2:
            # Class 1(Dementia)에 대한 Logit만 가져옴
            out = logits[:, 1]
        else:
            out = logits.squeeze()
            
        # Label을 float으로 변환 필수
        loss = criterion(out, batch.y.float())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (완화: 1.0 -> 5.0, 학습 촉진)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Stats accumulation
        total_loss += loss.item()
        
        # [수정] 확률 및 예측 계산 (Sigmoid 사용)
        probs = torch.sigmoid(out)
        threshold = 0.1  # 0.3 -> 0.1 (극심한 불균형 대응, 더 공격적)
        preds = (probs > threshold).long()  # threshold 넘으면 치매
        
        all_preds.extend(preds.cpu().detach().numpy())
        all_probs.extend(probs.cpu().detach().numpy())
        all_labels.extend(batch.y.cpu().detach().numpy())
    
    # Calculate Metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    try:
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except:
        acc, auc, f1 = 0.0, 0.5, 0.0
    
    return total_loss / len(loader), acc, auc, f1

def evaluate(model, loader, criterion, device):
    """평가 (BCEWithLogitsLoss 적용)"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            logits = model(batch.x, batch.edge_index, batch.batch)
            
            # [수정] 차원 맞추기
            if logits.shape[1] == 2:
                out = logits[:, 1]
            else:
                out = logits.squeeze()
                
            loss = criterion(out, batch.y.float())
            
            total_loss += loss.item()
            
            # [수정] 확률 계산
            probs = torch.sigmoid(out)
            
            # Threshold를 클래스 비율에 맞춰 조정 (30:1 비율이면 threshold를 매우 낮춤)
            # 극심한 불균형에서는 threshold를 낮춰서 더 많은 양성 예측 허용
            threshold = 0.1  # 0.3 -> 0.1 (더 공격적으로 양성 예측)
            preds = (probs > threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 예측 분포 확인 (디버깅용)
    n_pred_0 = (all_preds == 0).sum()
    n_pred_1 = (all_preds == 1).sum()
    n_label_0 = (all_labels == 0).sum()
    n_label_1 = (all_labels == 1).sum()
    
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except:
        auc = 0.5
        f1 = 0.0
    
    # 클래스별 확률 분포 분석 (AUC가 높은 이유 확인)
    probs_class_0 = all_probs[all_labels == 0]  # Control 그룹의 확률
    probs_class_1 = all_probs[all_labels == 1]  # Dementia 그룹의 확률
    
    # 예측 분포 출력 (F1이 낮거나 예측이 편향될 때만 상세 출력)
    if f1 < 0.2 or n_pred_1 < len(all_preds) * 0.05:  # F1이 낮거나 양성 예측이 5% 미만일 때
        print(f"      ⚠️  예측 분포: Pred 0={n_pred_0:,} ({n_pred_0/len(all_preds)*100:.1f}%), Pred 1={n_pred_1:,} ({n_pred_1/len(all_preds)*100:.1f}%)")
        print(f"      ⚠️  실제 분포: Label 0={n_label_0:,} ({n_label_0/len(all_labels)*100:.1f}%), Label 1={n_label_1:,} ({n_label_1/len(all_labels)*100:.1f}%)")
        print(f"      ⚠️  전체 확률 통계: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, median={np.median(all_probs):.3f}")
        print(f"      📊 Control 그룹 확률: mean={probs_class_0.mean():.3f}, median={np.median(probs_class_0):.3f}, std={probs_class_0.std():.3f}")
        print(f"      📊 Dementia 그룹 확률: mean={probs_class_1.mean():.3f}, median={np.median(probs_class_1):.3f}, std={probs_class_1.std():.3f}")
        
        # AUC가 높은 이유 분석
        if len(probs_class_0) > 0 and len(probs_class_1) > 0:
            prob_diff = probs_class_1.mean() - probs_class_0.mean()
            print(f"      🔍 AUC 분석: Dementia 평균 확률이 Control보다 {prob_diff:+.3f} 높음")
            if prob_diff > 0.05:
                print(f"      ✅ 모델이 확률 순위는 잘 맞추고 있습니다 (AUC={auc:.3f})")
                print(f"      ❌ 하지만 threshold={threshold:.2f}가 너무 높아서 F1이 낮습니다")
                print(f"      💡 해결책: threshold를 {probs_class_1.mean():.2f} 근처로 조정하면 F1이 개선될 수 있습니다")
            else:
                print(f"      ⚠️  모델이 두 그룹을 거의 구분하지 못하고 있습니다!")
                print(f"      ⚠️  AUC가 높은 것은 클래스 불균형 때문일 수 있습니다")
        
        print(f"      💡 해결책: pos_weight 증가 또는 threshold=0.1 이하로 조정 필요")
    
    return total_loss / len(loader), acc, auc, f1

def main():
    print("=" * 70)
    print("GNN-SubNet: Disease Subnetwork Detection")
    print("=" * 70)
    
    # 모델 저장 경로에 날짜 추가
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'../../data/gnn_subnet_best_model_{date_str}.pt'
    
    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드...")
    master_file = "../../data/ukb/ukb_usable_master.parquet"
    df = pd.read_parquet(master_file)
    
    # 단백질 컬럼 추출
    exclude_cols = ["eid", "sex", "target_age", "target_dementia", "participant.p42018"]
    protein_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and not c.startswith("pc__")
                   and not c.startswith("assess__")
                   and not c.startswith("online__")]
    
    print(f"   - 샘플 수: {len(df):,}")
    print(f"   - 단백질 수: {len(protein_cols):,}")
    print(f"   - Dementia: {(df['target_dementia'] == 1).sum():,}")
    print(f"   - Control: {(df['target_dementia'] == 0).sum():,}")
    
    # [핵심 수정 1] 데이터 정규화 (Standard Scaling) - 필수!
    from sklearn.preprocessing import StandardScaler
    print("\n   ⚠️  데이터 정규화 (Standard Scaling) 적용 중... (필수!)")
    scaler = StandardScaler()
    # 단백질 컬럼만 골라서 스케일링 (평균 0, 분산 1로 변환)
    df[protein_cols] = scaler.fit_transform(df[protein_cols])
    print("   ✅ 정규화 완료 (평균=0, 분산=1)")
    
    # 2. Graph 데이터 생성
    print("\n[2/5] Graph 데이터 생성...")
    
    # [최종 튜닝] 그래프 연결성 개선: Threshold 완화
    score_threshold = 700  # 800 -> 700 (High Confidence, 연결성 최대화)
    loader = GraphDataLoader(score_threshold=score_threshold)
    print(f"   - PPI Score Threshold: {score_threshold} (그래프 경량화)")
    
    # ---------------------------------------------------------
    # [수정] 전체 데이터(53,000명) 모두 사용!
    # ---------------------------------------------------------
    # 샘플링 없이 전체 사용
    df_sample = df.copy().sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 클래스 개수 확인
    n_pos = (df_sample['target_dementia'] == 1).sum()
    n_neg = (df_sample['target_dementia'] == 0).sum()
    class_ratio = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"   🔥 Full Dataset Mode On!")
    print(f"     > Dementia: {n_pos:,}명")
    print(f"     > Control:  {n_neg:,}명")
    print(f"     > Total:    {len(df_sample):,}명")
    print(f"     > 실제 비율: 1 : {class_ratio:.1f}")
    
    # 그래프 데이터셋 저장 경로
    dataset_dir = Path("../../data/gnn")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_dir / "graph_dataset_full.pkl"
    
    # 저장된 데이터셋이 있으면 로드, 없으면 생성 후 저장
    if dataset_file.exists():
        print(f"\n   📂 저장된 그래프 데이터셋 발견: {dataset_file}")
        print(f"   ⚡ 데이터셋 로드 중...")
        with open(dataset_file, 'rb') as f:
            graphs = pickle.load(f)
        print(f"   ✅ 그래프 데이터셋 로드 완료: {len(graphs):,}개")
    else:
        print(f"\n   🔨 그래프 데이터셋 생성 중... (처음 실행 시 시간이 걸립니다)")
        graphs = loader.create_graph_dataset(
            df_sample,
            protein_cols,
            label_col='target_dementia',
            eid_col='eid'
        )
        print(f"   - 생성된 그래프: {len(graphs):,}개")
        
        # 데이터셋 저장
        print(f"   💾 그래프 데이터셋 저장 중: {dataset_file}")
        with open(dataset_file, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"   ✅ 저장 완료! 다음 실행부터는 자동으로 로드됩니다.")
    
    # 3. Train/Test split (Downsampling 제거 - Weighted Loss 사용)
    print("\n[3/5] Train/Test split (전체 데이터 사용, Weighted Loss 적용)...")
    
    # 클래스 분포 확인
    labels = [g.y.item() for g in graphs]
    n_class_0 = labels.count(0)
    n_class_1 = labels.count(1)
    
    print(f"   - Class 0 (Control): {n_class_0:,}개")
    print(f"   - Class 1 (Dementia): {n_class_1:,}개")
    print(f"   - 클래스 비율: {n_class_0/n_class_1:.2f}:1")
    print(f"   ⚠️  Downsampling 없이 전체 {len(graphs):,}개 데이터 사용")
    print(f"   ✅ Weighted Loss로 클래스 불균형 처리 예정")
    
    # Train/Val/Test split (전체 데이터 사용, 클래스 비율 유지)
    # 1단계: Train + (Val + Test)로 분리
    train_graphs, temp_graphs = train_test_split(
        graphs,
        test_size=0.2,  # Val + Test = 20%
        random_state=42,
        stratify=labels
    )
    
    # 2단계: Val과 Test로 분리 (temp의 50%씩)
    temp_labels = [g.y.item() for g in temp_graphs]
    val_graphs_raw, test_graphs_raw = train_test_split(
        temp_graphs,
        test_size=0.5,  # Val 10%, Test 10%
        random_state=42,
        stratify=temp_labels
    )
    
    # 3단계: Val과 Test를 균형 데이터로 만들기 (1:1 비율)
    # Val 균형화
    val_labels = [g.y.item() for g in val_graphs_raw]
    val_class_0_raw = [g for g in val_graphs_raw if g.y.item() == 0]
    val_class_1_raw = [g for g in val_graphs_raw if g.y.item() == 1]
    n_val_class_1 = len(val_class_1_raw)
    # Control을 Dementia 개수만큼만 샘플링
    val_class_0_balanced = random.sample(val_class_0_raw, min(n_val_class_1, len(val_class_0_raw)))
    val_graphs = val_class_0_balanced + val_class_1_raw
    random.shuffle(val_graphs)
    
    # Test 균형화
    test_labels = [g.y.item() for g in test_graphs_raw]
    test_class_0_raw = [g for g in test_graphs_raw if g.y.item() == 0]
    test_class_1_raw = [g for g in test_graphs_raw if g.y.item() == 1]
    n_test_class_1 = len(test_class_1_raw)
    # Control을 Dementia 개수만큼만 샘플링
    test_class_0_balanced = random.sample(test_class_0_raw, min(n_test_class_1, len(test_class_0_raw)))
    test_graphs = test_class_0_balanced + test_class_1_raw
    random.shuffle(test_graphs)
    
    print(f"   - Train: {len(train_graphs):,}개 (80%, 불균형 유지)")
    print(f"   - Val:   {len(val_graphs):,}개 (균형 데이터, 1:1)")
    print(f"   - Test:  {len(test_graphs):,}개 (균형 데이터, 1:1)")
    
    # Train set 클래스 분포 확인
    train_class_0 = sum(1 for g in train_graphs if g.y.item() == 0)
    train_class_1 = sum(1 for g in train_graphs if g.y.item() == 1)
    train_ratio = train_class_0 / train_class_1 if train_class_1 > 0 else 0.0
    print(f"   - Train Class 0: {train_class_0:,}, Train Class 1: {train_class_1:,}")
    print(f"   - Train 비율: 1:{train_ratio:.2f} (불균형 유지 - 학습용)")
    
    # Val set 클래스 분포 확인
    val_class_0 = sum(1 for g in val_graphs if g.y.item() == 0)
    val_class_1 = sum(1 for g in val_graphs if g.y.item() == 1)
    val_ratio = val_class_0 / val_class_1 if val_class_1 > 0 else 0.0
    print(f"   - Val Class 0: {val_class_0:,}, Val Class 1: {val_class_1:,}")
    print(f"   - Val 비율: 1:{val_ratio:.2f} (균형 - 평가용)")
    
    # Test set 클래스 분포 확인
    test_class_0 = sum(1 for g in test_graphs if g.y.item() == 0)
    test_class_1 = sum(1 for g in test_graphs if g.y.item() == 1)
    test_ratio = test_class_0 / test_class_1 if test_class_1 > 0 else 0.0
    print(f"   - Test Class 0: {test_class_0:,}, Test Class 1: {test_class_1:,}")
    print(f"   - Test 비율: 1:{test_ratio:.2f} (균형 - 평가용)")
    
    print(f"   ✅ Train은 불균형 데이터로 학습, Val/Test는 균형 데이터로 평가")
    
    # 4. 모델 학습
    print("\n[4/5] 모델 학습...")
    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   - Device: {device}")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print(f"   - Device: {device} (⚠️  GPU not available, using CPU)")
    
    # [GPU 활용도 개선] 모델 크기 조정 (Over-smoothing 방지 + GPU 활용 균형)
    # Layers는 2 유지 (Over-smoothing 방지), Hidden/Embedding은 증가 (GPU 활용)
    model = GraphClassifier(
        input_dim=1,
        hidden_dim=64,      # 32 -> 64 (GPU 활용도 증가)
        embedding_dim=256,  # 128 -> 256 (GPU 활용도 증가)
        num_layers=2,       # 2 유지 (Over-smoothing 방지)
        num_classes=2,
        pooling='mean',
        dropout=0.2         # 0.2 유지 (규제)
    ).to(device)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # FP32 기준
    
    print(f"   - 모델 설정: Layers=2, Hidden=64, Embedding=256, Dropout=0.2")
    print(f"   - 모델 파라미터: {total_params:,}개 ({model_size_mb:.2f} MB)")
    
    # [GPU 활용도 개선] 배치 사이즈 증가 (전체 데이터 사용 시)
    # 전체 데이터(53,000개) 사용 시 메모리 여유 있으면 128~256 추천
    batch_size = 128 if torch.cuda.is_available() else 64  # GPU 있으면 128, 없으면 64
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"   - Batch Size: {batch_size} (전체 데이터 사용, GPU 활용도 최대화)")
    print(f"   - Num Workers: 4 (데이터 로딩 속도 향상)")
    
    # [핵심 수정 2] 학습률 및 Weight Decay 조정 (학습 촉진)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)  # 0.002 -> 0.005 (Loss 감소 촉진)
    
    # Learning rate scheduler 추가 (학습 안정화, patience 증가)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15  # 10 -> 15 (너무 빨리 감소하지 않도록)
    )
    print(f"   - Learning Rate: 0.005 (0.002 -> 0.005, Loss 감소 촉진)")
    print(f"   - Weight Decay: 1e-5 (더 자유로운 학습)")
    print(f"   - LR Scheduler: ReduceLROnPlateau (patience=15)")
    
    # ---------------------------------------------------------
    # [수정] 전체 데이터 비율에 맞춘 가중치 자동 계산
    # ---------------------------------------------------------
    
    # Train set에서 클래스 비율 계산
    train_labels = [g.y.item() for g in train_graphs]
    n_train_pos = train_labels.count(1)
    n_train_neg = train_labels.count(0)
    train_ratio = n_train_neg / n_train_pos if n_train_pos > 0 else 1.0
    
    # 비율에 맞춰 가중치 계산 (극심한 불균형 대응)
    # 전체 데이터 사용 시 비율이 30:1이므로 매우 강력한 가중치 필요
    # 비율 그대로 사용 (제한 없음) - 30:1이면 30.0 사용
    calculated_weight = train_ratio
    
    pos_weight = torch.tensor([calculated_weight]).to(device)
    
    print(f"   🔥 [Full Data 설정] Train Class Imbalance Ratio: 1:{train_ratio:.1f}")
    print(f"   🔥 [Full Data 설정] 적용된 pos_weight: {pos_weight.item():.1f} (비율 그대로, 제한 없음)")
    print(f"   ⚠️  극심한 불균형 대응: 치매 환자 틀리면 {pos_weight.item():.1f}배 벌점!")
    
    # BCEWithLogitsLoss가 학습 안정성이 훨씬 좋음
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 학습 (성능 향상을 위해 epoch 증가)
    num_epochs = 100  # 50 -> 100 (AUC가 계속 상승 중이므로 더 학습)
    best_auc = 0.0  # 명시적 초기화
    
    # Early stopping 설정 (완화)
    patience = 30  # 20 -> 30 (모델이 정신 차리는 데 시간 필요)
    patience_counter = 0
    best_loss = float('inf')
    
    print(f"\n   🚀 학습 시작 (최대 {num_epochs} epochs, patience={patience})")
    print(f"   📊 Validation set으로 모델 선택, Test set은 최종 평가에만 사용")
    
    for epoch in range(num_epochs):
        train_loss, train_acc, train_auc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduler 업데이트 (Validation loss 사용)
        scheduler.step(val_loss)
        
        # Best model 저장 (Validation AUC 기준)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            improved = "⭐"
        else:
            patience_counter += 1
            improved = ""
        
        # Loss 개선 체크
        if val_loss < best_loss:
            best_loss = val_loss
        
        # 매 epoch마다 출력 (학습 모니터링 개선)
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 1 == 0:  # 매 epoch마다 출력
            print(f"   Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Loss={train_loss:.4f}→{val_loss:.4f} | "
                  f"Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f} (Best: {best_auc:.4f}) {improved} | "
                  f"Train F1={train_f1:.4f} | Val F1={val_f1:.4f} | LR={current_lr:.6f} | Patience: {patience_counter}/{patience}")
        
        # Early stopping (Validation AUC 기준)
        if patience_counter >= patience:
            print(f"\n   ⚠️  Early stopping at epoch {epoch+1} (Val AUC 개선 없음 {patience} epochs)")
            break
    
    # 최종 평가: Test set으로 평가
    print(f"\n   📊 최종 평가 (Test set)...")
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_auc, test_f1 = evaluate(model, test_loader, criterion, device)
    
    print(f"\n   ✅ Best Val AUC: {best_auc:.4f}")
    print(f"   ✅ Final Test AUC: {test_auc:.4f}")
    print(f"   ✅ Final Test F1: {test_f1:.4f}")
    print(f"   ✅ 모델 저장: {model_path}")
    
    print("\n" + "=" * 70)
    print("✅ GNN 모델 학습 완료!")
    print("=" * 70)
    print("\n📁 출력 파일:")
    print(f"   - 모델: {model_path}")
    print("\n💡 다음 단계:")
    print("   - Explain & Subnetwork Detection은 별도 스크립트로 실행:")
    print("     python scripts/explain_gnn_subnet.py")


if __name__ == "__main__":
    main()

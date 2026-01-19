#!/usr/bin/env python3
"""
ProtRS (Proteomic Risk Score) 생성 스크립트

Age 기반 ElasticNet 모델로 ProtRS를 학습하고 저장합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 경로 설정
MASTER = "../../data/ukb/ukb_usable_master.parquet"
OUT_PROTRS = "../../data/protrs_age.parquet"
OUT_PLOT = "../../data/protrs_age_correlation.png"

def build_protrs_age():
    """
    Age 기반 ProtRS를 생성합니다.
    """
    print("=" * 60)
    print("ProtRS (Age-based) 생성")
    print("=" * 60)
    
    # -------------------------
    # 1. 데이터 로드
    # -------------------------
    print("\n[1/6] 데이터 로드...")
    df = pd.read_parquet(MASTER)
    print(f"   - 전체 샘플 수: {len(df):,}")
    print(f"   - 전체 컬럼 수: {len(df.columns):,}")
    
    # 단백질 컬럼 추출
    protein_cols = [
        c for c in df.columns
        if c not in ["eid", "sex", "target_age", "target_dementia", "participant.p42018"]
        and not c.startswith("pc__")
    ]
    
    print(f"   - 단백질 컬럼 수: {len(protein_cols):,}")
    
    X = df[protein_cols].copy()
    y = df["target_age"].copy()
    eids = df["eid"].copy()
    
    # Age 결측치 제거
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    eids = eids[valid_mask]
    
    print(f"   - Age 결측치 제거 후: {len(X):,} 샘플")
    
    # -------------------------
    # 2. Train / Validation split
    # -------------------------
    print("\n[2/6] Train/Validation split...")
    X_tr, X_va, y_tr, y_va, eid_tr, eid_va = train_test_split(
        X, y, eids, test_size=0.2, random_state=42
    )
    
    print(f"   - Train: {len(X_tr):,} 샘플")
    print(f"   - Valid: {len(X_va):,} 샘플")
    
    # -------------------------
    # 3. ElasticNet Pipeline 구성
    # -------------------------
    print("\n[3/6] ElasticNet 모델 학습...")
    
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Train set 기준 median
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-3, 1, 30),
            cv=5,
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    pipe.fit(X_tr, y_tr)
    
    # 최적 하이퍼파라미터 출력
    best_alpha = pipe.named_steps["enet"].alpha_
    best_l1_ratio = pipe.named_steps["enet"].l1_ratio_
    
    print(f"   - Best alpha: {best_alpha:.6f}")
    print(f"   - Best l1_ratio: {best_l1_ratio:.2f}")
    
    # -------------------------
    # 4. Validation 성능 평가
    # -------------------------
    print("\n[4/6] Validation 성능 평가...")
    
    y_va_pred = pipe.predict(X_va)
    r2_va = pipe.score(X_va, y_va)
    corr_va = np.corrcoef(y_va, y_va_pred)[0, 1]
    
    print(f"   - R² (valid): {r2_va:.4f}")
    print(f"   - Pearson r (valid): {corr_va:.4f}")
    
    # -------------------------
    # 5. 전체 데이터로 ProtRS 계산
    # -------------------------
    print("\n[5/6] 전체 데이터로 ProtRS 계산...")
    
    # 전체 데이터에 대해 ProtRS 계산
    protrs_all = pipe.predict(X)
    
    # 결과 저장
    df_protrs = pd.DataFrame({
        "eid": eids.values,
        "ProtRS_age": protrs_all
    })
    
    # 원본 데이터와 병합 (선택사항)
    df_protrs = df_protrs.merge(
        df[["eid", "target_age", "target_dementia", "sex"]],
        on="eid",
        how="left"
    )
    
    # -------------------------
    # 6. 품질 체크 및 시각화
    # -------------------------
    print("\n[6/6] 품질 체크 및 시각화...")
    
    # 전체 correlation
    corr_all = np.corrcoef(y, protrs_all)[0, 1]
    print(f"   - Pearson r (전체): {corr_all:.4f}")
    
    # 시각화
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y, protrs_all, alpha=0.1, s=1)
    plt.xlabel("Chronological Age")
    plt.ylabel("ProtRS (Age-based)")
    plt.title(f"ProtRS vs Age\nPearson r = {corr_all:.3f}")
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 2, 2)
    residuals = y - protrs_all
    plt.scatter(protrs_all, residuals, alpha=0.1, s=1)
    plt.xlabel("ProtRS")
    plt.ylabel("Residuals (Age - ProtRS)")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    # Distribution
    plt.subplot(2, 2, 3)
    plt.hist(protrs_all, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("ProtRS")
    plt.ylabel("Frequency")
    plt.title("ProtRS Distribution")
    plt.grid(True, alpha=0.3)
    
    # Age distribution
    plt.subplot(2, 2, 4)
    plt.hist(y, bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel("Chronological Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"   - 시각화 저장: {OUT_PLOT}")
    
    # -------------------------
    # 7. 결과 저장
    # -------------------------
    print("\n[7/7] 결과 저장...")
    df_protrs.to_parquet(OUT_PROTRS, index=False)
    print(f"   - ProtRS 저장: {OUT_PROTRS}")
    print(f"   - 샘플 수: {len(df_protrs):,}")
    
    # 통계 요약
    print("\n" + "=" * 60)
    print("ProtRS 통계 요약")
    print("=" * 60)
    print(f"평균: {df_protrs['ProtRS_age'].mean():.2f}")
    print(f"표준편차: {df_protrs['ProtRS_age'].std():.2f}")
    print(f"최소값: {df_protrs['ProtRS_age'].min():.2f}")
    print(f"최대값: {df_protrs['ProtRS_age'].max():.2f}")
    print(f"\nAge와의 상관관계: {corr_all:.4f}")
    print(f"R² (valid): {r2_va:.4f}")
    
    # 선택된 단백질 수 확인
    enet_model = pipe.named_steps["enet"]
    n_selected = np.sum(np.abs(enet_model.coef_) > 1e-6)
    print(f"\n선택된 단백질 수 (coef != 0): {n_selected:,} / {len(protein_cols):,}")
    
    print("\n✅ ProtRS 생성 완료!")
    
    # 모델 저장 (선택사항)
    import pickle
    model_path = "../../data/protrs_age_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"   - 모델 저장: {model_path}")
    
    return df_protrs, pipe


if __name__ == "__main__":
    df_protrs, pipe = build_protrs_age()

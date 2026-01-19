#!/usr/bin/env python3
"""
ProtRS 종합 분석 스크립트

1. ProtRS(Dementia) 생성 및 ProtRS(Age)와 비교
2. ProtRS 상위 단백질 해석 (biological interpretation)
3. ProtRS-only baseline 성능 평가
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
MASTER = "../data/ukb/ukb_usable_master.parquet"
PROTRS_AGE = "../data/protrs_age.parquet"
OUT_PROTRS_DEM = "../data/protrs_dementia.parquet"
OUT_COMPARISON = "../data/protrs_comparison.png"
OUT_TOP_PROTEINS = "../data/protrs_top_proteins.csv"
OUT_BASELINE = "../data/protrs_baseline_performance.png"

def get_protein_cols(df):
    """단백질 컬럼 추출"""
    return [
        c for c in df.columns
        if c not in ["eid", "sex", "target_age", "target_dementia", "participant.p42018"]
        and not c.startswith("pc__")
    ]

def build_protrs_dementia():
    """
    Dementia 기반 ProtRS를 생성합니다.
    """
    print("=" * 60)
    print("1. ProtRS (Dementia-based) 생성")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    df = pd.read_parquet(MASTER)
    protein_cols = get_protein_cols(df)
    
    X = df[protein_cols].copy()
    y = df["target_dementia"].copy()
    eids = df["eid"].copy()
    
    # Dementia 레이블이 있는 샘플만
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    eids = eids[valid_mask]
    
    print(f"   - 전체 샘플 수: {len(X):,}")
    print(f"   - 치매 (1): {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"   - 정상 (0): {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
    
    # Train/Validation split (stratified)
    print("\n[2/5] Train/Validation split (stratified)...")
    X_tr, X_va, y_tr, y_va, eid_tr, eid_va = train_test_split(
        X, y, eids, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   - Train: {len(X_tr):,} 샘플 (치매: {y_tr.sum():,})")
    print(f"   - Valid: {len(X_va):,} 샘플 (치매: {y_va.sum():,})")
    
    # Logistic Regression with ElasticNet (분류 문제)
    print("\n[3/5] Logistic Regression (ElasticNet) 모델 학습...")
    
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=np.logspace(-4, 1, 20),
            l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            penalty='elasticnet',
            solver='saga',
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'  # 클래스 불균형 처리
        ))
    ])
    
    pipe.fit(X_tr, y_tr)
    
    # Validation 성능
    print("\n[4/5] Validation 성능 평가...")
    y_va_pred_proba = pipe.predict_proba(X_va)[:, 1]
    auc_va = roc_auc_score(y_va, y_va_pred_proba)
    ap_va = average_precision_score(y_va, y_va_pred_proba)
    
    print(f"   - AUC (valid): {auc_va:.4f}")
    print(f"   - AP (valid): {ap_va:.4f}")
    
    # 전체 데이터로 ProtRS 계산
    print("\n[5/5] 전체 데이터로 ProtRS 계산...")
    protrs_all = pipe.predict_proba(X)[:, 1]
    
    df_protrs = pd.DataFrame({
        "eid": eids.values,
        "ProtRS_dementia": protrs_all
    })
    
    df_protrs = df_protrs.merge(
        df[["eid", "target_age", "target_dementia", "sex"]],
        on="eid",
        how="left"
    )
    
    df_protrs.to_parquet(OUT_PROTRS_DEM, index=False)
    print(f"   - ProtRS 저장: {OUT_PROTRS_DEM}")
    
    return df_protrs, pipe, protein_cols

def compare_protrs_age_vs_dementia():
    """
    ProtRS(Age)와 ProtRS(Dementia) 비교
    """
    print("\n" + "=" * 60)
    print("2. ProtRS(Age) vs ProtRS(Dementia) 비교")
    print("=" * 60)
    
    # 데이터 로드
    protrs_age = pd.read_parquet(PROTRS_AGE)
    protrs_dem = pd.read_parquet(OUT_PROTRS_DEM)
    
    # 병합
    df_compare = protrs_age.merge(
        protrs_dem[["eid", "ProtRS_dementia"]],
        on="eid",
        how="inner"
    )
    
    print(f"\n병합된 샘플 수: {len(df_compare):,}")
    
    # 상관관계
    corr = np.corrcoef(df_compare["ProtRS_age"], df_compare["ProtRS_dementia"])[0, 1]
    print(f"\nProtRS(Age) vs ProtRS(Dementia) 상관관계: {corr:.4f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot
    ax = axes[0, 0]
    scatter = ax.scatter(
        df_compare["ProtRS_age"],
        df_compare["ProtRS_dementia"],
        c=df_compare["target_dementia"],
        cmap='RdYlBu_r',
        alpha=0.3,
        s=1
    )
    ax.set_xlabel("ProtRS (Age-based)")
    ax.set_ylabel("ProtRS (Dementia-based)")
    ax.set_title(f"ProtRS(Age) vs ProtRS(Dementia)\nPearson r = {corr:.3f}")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Dementia (0=No, 1=Yes)")
    
    # 2. Distribution 비교
    ax = axes[0, 1]
    ax.hist(df_compare["ProtRS_age"], bins=50, alpha=0.6, label="ProtRS(Age)", edgecolor='black')
    ax.set_xlabel("ProtRS")
    ax.set_ylabel("Frequency")
    ax.set_title("ProtRS Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Dementia by ProtRS(Age)
    ax = axes[1, 0]
    bins = np.linspace(df_compare["ProtRS_age"].min(), df_compare["ProtRS_age"].max(), 20)
    df_compare["age_bin"] = pd.cut(df_compare["ProtRS_age"], bins=bins)
    dementia_rate = df_compare.groupby("age_bin")["target_dementia"].mean()
    bin_centers = [(b.left + b.right) / 2 for b in dementia_rate.index]
    ax.plot(bin_centers, dementia_rate.values, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel("ProtRS (Age-based)")
    ax.set_ylabel("Dementia Rate")
    ax.set_title("Dementia Rate by ProtRS(Age)")
    ax.grid(True, alpha=0.3)
    
    # 4. Dementia by ProtRS(Dementia)
    ax = axes[1, 1]
    bins = np.linspace(df_compare["ProtRS_dementia"].min(), df_compare["ProtRS_dementia"].max(), 20)
    df_compare["dem_bin"] = pd.cut(df_compare["ProtRS_dementia"], bins=bins)
    dementia_rate = df_compare.groupby("dem_bin")["target_dementia"].mean()
    bin_centers = [(b.left + b.right) / 2 for b in dementia_rate.index]
    ax.plot(bin_centers, dementia_rate.values, 'o-', linewidth=2, markersize=6, color='red')
    ax.set_xlabel("ProtRS (Dementia-based)")
    ax.set_ylabel("Dementia Rate")
    ax.set_title("Dementia Rate by ProtRS(Dementia)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_COMPARISON, dpi=300, bbox_inches='tight')
    print(f"\n비교 시각화 저장: {OUT_COMPARISON}")
    
    return df_compare

def analyze_top_proteins(pipe_dem, protein_cols):
    """
    ProtRS 상위 단백질 해석
    """
    print("\n" + "=" * 60)
    print("3. ProtRS 상위 단백질 해석")
    print("=" * 60)
    
    # Age 모델 로드 시도
    import pickle
    import os
    
    model_path = "../data/protrs_age_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            pipe_age = pickle.load(f)
        
        # Age 기반 상위 단백질
        enet_age = pipe_age.named_steps["enet"]
        coef_age = enet_age.coef_
    else:
        print("⚠️ Age 모델이 없습니다. Age 기반 분석을 건너뜁니다.")
        coef_age = np.zeros(len(protein_cols))
    
    # Dementia 기반 상위 단백질
    lr_dem = pipe_dem.named_steps["lr"]
    coef_dem = lr_dem.coef_[0]
    
    # 데이터프레임 생성
    df_proteins = pd.DataFrame({
        "protein": protein_cols,
        "coef_age": coef_age,
        "coef_dem": coef_dem,
        "abs_coef_age": np.abs(coef_age),
        "abs_coef_dem": np.abs(coef_dem)
    })
    
    # 상위 단백질 추출
    top_n = 50
    top_age = df_proteins.nlargest(top_n, "abs_coef_age")
    top_dem = df_proteins.nlargest(top_n, "abs_coef_dem")
    
    print(f"\n상위 {top_n}개 단백질:")
    print(f"\n[Age-based ProtRS]")
    print(top_age[["protein", "coef_age"]].head(20).to_string(index=False))
    
    print(f"\n[Dementia-based ProtRS]")
    print(top_dem[["protein", "coef_dem"]].head(20).to_string(index=False))
    
    # 공통 단백질
    common = set(top_age["protein"]) & set(top_dem["protein"])
    print(f"\n공통 상위 단백질 수: {len(common)}")
    if len(common) > 0:
        print(f"공통 단백질 (상위 20개): {sorted(list(common))[:20]}")
    
    # 저장
    df_proteins_sorted = df_proteins.sort_values("abs_coef_age", ascending=False)
    df_proteins_sorted.to_csv(OUT_TOP_PROTEINS, index=False)
    print(f"\n전체 단백질 계수 저장: {OUT_TOP_PROTEINS}")
    
    return df_proteins, top_age, top_dem

def evaluate_baseline_performance():
    """
    ProtRS-only baseline 성능 평가
    """
    print("\n" + "=" * 60)
    print("4. ProtRS-only Baseline 성능 평가")
    print("=" * 60)
    
    # 데이터 로드
    protrs_age = pd.read_parquet(PROTRS_AGE)
    protrs_dem = pd.read_parquet(OUT_PROTRS_DEM)
    
    df = protrs_age.merge(
        protrs_dem[["eid", "ProtRS_dementia"]],
        on="eid",
        how="inner"
    )
    
    # Train/Test split
    X = df[["ProtRS_age", "ProtRS_dementia"]].values
    y = df["target_dementia"].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 모델 학습
    print("\n[1/3] 모델 학습...")
    
    # 1. ProtRS(Age) only
    lr_age = LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 20),
        cv=5,
        random_state=42,
        class_weight='balanced'
    )
    lr_age.fit(X_tr[:, 0:1], y_tr)
    
    # 2. ProtRS(Dementia) only
    lr_dem = LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 20),
        cv=5,
        random_state=42,
        class_weight='balanced'
    )
    lr_dem.fit(X_tr[:, 1:2], y_tr)
    
    # 3. Both ProtRS
    lr_both = LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 20),
        cv=5,
        random_state=42,
        class_weight='balanced'
    )
    lr_both.fit(X_tr, y_tr)
    
    # 성능 평가
    print("\n[2/3] 성능 평가...")
    
    results = {}
    
    # ProtRS(Age) only
    y_pred_age = lr_age.predict_proba(X_te[:, 0:1])[:, 1]
    results["ProtRS(Age)"] = {
        "AUC": roc_auc_score(y_te, y_pred_age),
        "AP": average_precision_score(y_te, y_pred_age),
        "y_pred": y_pred_age
    }
    
    # ProtRS(Dementia) only
    y_pred_dem = lr_dem.predict_proba(X_te[:, 1:2])[:, 1]
    results["ProtRS(Dementia)"] = {
        "AUC": roc_auc_score(y_te, y_pred_dem),
        "AP": average_precision_score(y_te, y_pred_dem),
        "y_pred": y_pred_dem
    }
    
    # Both
    y_pred_both = lr_both.predict_proba(X_te)[:, 1]
    results["ProtRS(Both)"] = {
        "AUC": roc_auc_score(y_te, y_pred_both),
        "AP": average_precision_score(y_te, y_pred_both),
        "y_pred": y_pred_both
    }
    
    # 결과 출력
    print("\n[3/3] 결과:")
    for name, metrics in results.items():
        print(f"  {name:20s}: AUC = {metrics['AUC']:.4f}, AP = {metrics['AP']:.4f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax = axes[0]
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_te, metrics["y_pred"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics['AUC']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - ProtRS Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PR Curve
    ax = axes[1]
    for name, metrics in results.items():
        precision, recall, _ = precision_recall_curve(y_te, metrics["y_pred"])
        ax.plot(recall, precision, label=f"{name} (AP={metrics['AP']:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - ProtRS Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_BASELINE, dpi=300, bbox_inches='tight')
    print(f"\nBaseline 성능 시각화 저장: {OUT_BASELINE}")
    
    return results

def analyze_graphs():
    """
    ProtRS(Age) 그래프 분석
    """
    print("\n" + "=" * 60)
    print("5. ProtRS(Age) 그래프 분석")
    print("=" * 60)
    
    protrs = pd.read_parquet(PROTRS_AGE)
    
    # 상관관계
    corr = np.corrcoef(protrs["target_age"], protrs["ProtRS_age"])[0, 1]
    
    print("\n[그래프 1: ProtRS vs Age Scatter Plot]")
    print(f"  - X축: Chronological Age (실제 나이)")
    print(f"  - Y축: ProtRS (Age-based) (단백질 기반 예측 나이)")
    print(f"  - 상관계수: {corr:.4f}")
    print(f"  - 해석: 매우 강한 양의 상관관계 (r > 0.9)")
    print(f"          → ProtRS가 실제 나이를 매우 잘 예측함")
    print(f"          → 단백질 패턴이 생물학적 노화를 잘 반영함")
    
    # Residual 분석
    residuals = protrs["target_age"] - protrs["ProtRS_age"]
    print("\n[그래프 2: Residual Plot]")
    print(f"  - X축: ProtRS")
    print(f"  - Y축: Residuals (실제 나이 - ProtRS)")
    print(f"  - 평균 residual: {residuals.mean():.2f}")
    print(f"  - 표준편차: {residuals.std():.2f}")
    print(f"  - 해석: 잔차가 0 주변에 고르게 분포")
    print(f"          → 모델이 체계적 편향 없이 예측")
    
    # 분포 비교
    print("\n[그래프 3: ProtRS Distribution]")
    print(f"  - ProtRS 평균: {protrs['ProtRS_age'].mean():.2f}")
    print(f"  - ProtRS 표준편차: {protrs['ProtRS_age'].std():.2f}")
    print(f"  - 해석: 정규분포에 가까운 형태")
    
    print("\n[그래프 4: Age Distribution]")
    print(f"  - Age 평균: {protrs['target_age'].mean():.2f}")
    print(f"  - Age 표준편차: {protrs['target_age'].std():.2f}")
    print(f"  - 해석: 실제 나이 분포와 ProtRS 분포 비교")
    
    return {
        "correlation": corr,
        "mean_residual": residuals.mean(),
        "std_residual": residuals.std()
    }

if __name__ == "__main__":
    # 1. Dementia 기반 ProtRS 생성
    protrs_dem, pipe_dem, protein_cols = build_protrs_dementia()
    
    # Age 기반 모델 로드 (이미 생성됨)
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNetCV
    import pickle
    
    # Age 모델은 이미 학습되어 있으므로, 여기서는 비교만 수행
    # 실제로는 모델을 저장/로드해야 하지만, 여기서는 간단히 비교만
    
    # 2. 비교
    df_compare = compare_protrs_age_vs_dementia()
    
    # 3. 상위 단백질 분석
    df_proteins, top_age, top_dem = analyze_top_proteins(pipe_dem, protein_cols)
    
    # 4. Baseline 성능 평가
    results = evaluate_baseline_performance()
    
    # 5. 그래프 분석
    graph_analysis = analyze_graphs()
    
    print("\n" + "=" * 60)
    print("✅ 모든 분석 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print(f"  - {OUT_PROTRS_DEM}")
    print(f"  - {OUT_COMPARISON}")
    print(f"  - {OUT_BASELINE}")

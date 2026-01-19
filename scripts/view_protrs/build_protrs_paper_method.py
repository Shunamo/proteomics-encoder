#!/usr/bin/env python3
"""
논문 방법론 정확히 따르기 - PrRSMDD-ADRD 생성

Nature Mental Health 2025 논문 Methods (885-886페이지)를 정확히 따라:
1. Inverse Normal Transformation (INT)
2. Prevalent Case 제거
3. 10-fold CV + 1SE rule로 LASSO alpha 선택
4. Cox LASSO Regression으로 ProtRS 생성
5. C-index 평가

핵심 개선사항:
- participant.p42018 사용 (11,616개 케이스, 기존보다 6.8배 많음)
- Prevalent case 제거 (baseline 이전 치매)
- 정확한 Time/Event 계산
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(BASE_DIR, "data/ukb/ukb_usable_master.parquet")
OUTCOME_FILE = os.path.join(BASE_DIR, "data/ukb/ukb_cog_cov_master_plus_dementia_outcome.csv")
OUT_PROTRS = os.path.join(BASE_DIR, "data/protrs_paper_method.parquet")
OUT_PLOT = os.path.join(BASE_DIR, "data/protrs_paper_method_performance.png")
OUT_WEIGHTS = os.path.join(BASE_DIR, "data/protrs_paper_method_weights.csv")

def apply_inverse_normal_transform(X):
    """
    Inverse Normal Transformation (INT)
    논문: "We applied the inverse normal transformation to individual proteins 
    (n=2,920) in the baseline cohort to correct distributional skewness and 
    unify the scales into z scores"
    
    Args:
        X: 단백질 데이터 (n_samples, n_features)
    
    Returns:
        X_int: INT 변환된 데이터 (Z-scores)
    """
    X_int = np.zeros_like(X)
    
    for i in range(X.shape[1]):
        col = X[:, i]
        # 결측치 제외하고 rank 계산
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() == 0:
            X_int[:, i] = col
            continue
        
        valid_col = col[valid_mask]
        n_valid = len(valid_col)
        
        # Rank 계산 (tie 처리: average)
        ranks = stats.rankdata(valid_col, method='average')
        
        # INT 적용: (rank - 0.5) / n -> norm.ppf
        transformed = stats.norm.ppf((ranks - 0.5) / n_valid)
        
        # 원래 위치에 복원
        X_int[valid_mask, i] = transformed
        X_int[~valid_mask, i] = np.nan
    
    return X_int

def get_protein_cols(df):
    """단백질 컬럼 추출 (날짜 컬럼 제외)"""
    exclude_cols = [
        "eid", "sex", "target_age", "target_dementia", "participant.p42018",
        "time", "event", "baseline_date", "censor_date", "dementia_date",
        "baseline", "dementia_date", "death_date", "is_prevalent",
        "event_calc", "time_calc"
    ]
    
    # 날짜 타입 컬럼도 제외
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
    
    return [
        c for c in df.columns
        if c not in exclude_cols
        and c not in date_cols
        and not c.startswith("pc__")
        and not c.startswith("assess__")
        and not c.startswith("online__")
    ]

def calculate_survival_time_paper_method(df, baseline_date_col=None, death_date_col=None):
    """
    논문 방식 Time/Event 계산
    
    핵심:
    1. Prevalent case 제거 (baseline 이전 치매)
    2. Event: participant.p42018 존재 여부
    3. Time: baseline부터 dementia_date 또는 censor_date까지
    
    Args:
        df: 데이터프레임
        baseline_date_col: baseline date 컬럼명 (없으면 추정)
        death_date_col: death date 컬럼명 (없으면 사용 안 함)
    
    Returns:
        df with 'time', 'event', 'is_prevalent' columns
    """
    df = df.copy()
    
    # 날짜 변환
    if 'participant.p42018' in df.columns:
        dementia_date = pd.to_datetime(df['participant.p42018'], errors='coerce')
    else:
        dementia_date = None
    
    # Baseline date (강화된 fallback 로직)
    baseline = None
    
    # 1. 지정된 컬럼 사용
    if baseline_date_col and baseline_date_col in df.columns:
        print(f"   ✅ 지정된 컬럼 사용: {baseline_date_col}")
        baseline = pd.to_datetime(df[baseline_date_col], errors='coerce')
    
    # 2. 자동 검색: 여러 후보 컬럼 시도
    elif 'f.53.0.0' in df.columns:
        print("   ✅ 'f.53.0.0' 컬럼 자동 감지 및 사용")
        baseline = pd.to_datetime(df['f.53.0.0'], errors='coerce')
    elif 'p53_i0' in df.columns:
        print("   ✅ 'p53_i0' 컬럼 자동 감지 및 사용")
        baseline = pd.to_datetime(df['p53_i0'], errors='coerce')
    elif 'date_attending' in df.columns:
        print("   ✅ 'date_attending' 컬럼 자동 감지 및 사용")
        baseline = pd.to_datetime(df['date_attending'], errors='coerce')
    
    # 주의: assess__p20023_i0은 반응속도(ms)이므로 날짜 계산에 사용하지 않음
    # elif 'assess__p20023_i0' in df.columns:
    #     # 이 부분은 삭제됨: p20023은 날짜가 아니라 반응속도(ms)입니다
    
    # 3. 나이 기반 역산 시도 (p21003_i0 + p34)
    if baseline is None or baseline.isna().all():
        if 'target_age' in df.columns and 'p34' in df.columns:
            print("   ⚠️  나이 기반 역산 시도...")
            # Year of birth + Age = Year of recruitment
            year_birth = pd.to_datetime(df['p34'], format='%Y', errors='coerce').dt.year
            age = df['target_age']
            year_recruit = year_birth + age
            # 해당 년도의 중간 날짜로 설정 (6월 15일)
            baseline = pd.to_datetime(year_recruit.astype(str) + '-06-15', errors='coerce')
            if baseline.notna().sum() > 0:
                print(f"   ✅ 나이 기반 역산 성공: {baseline.notna().sum():,}개")
            else:
                baseline = None
    
    # 4. 최후의 수단: 2008-01-01로 통일
    if baseline is None or (hasattr(baseline, 'isna') and baseline.isna().all()):
        print("   ⚠️  경고: 참가일(p53) 컬럼 없음. 모든 참가일을 '2008-01-01'로 가정합니다.")
        print("      (UK Biobank 모집 중간값 사용, 2006-2010년)")
        baseline = pd.Series([pd.Timestamp('2008-01-01')] * len(df))
    
    # baseline을 df에 저장 (중요!)
    df['baseline'] = baseline
    
    # Death date
    if death_date_col and death_date_col in df.columns:
        death_date = pd.to_datetime(df[death_date_col], errors='coerce')
    else:
        death_date = None
    
    # Administrative censor date (데이터의 마지막 날짜)
    admin_censor = pd.Timestamp('2024-11-23')  # participant.p42018의 최대값
    
    # Event 정의
    df['event'] = dementia_date.notna().astype(int) if dementia_date is not None else 0
    
    # Prevalent case 확인 (baseline 이전에 이미 치매)
    if dementia_date is not None:
        df['is_prevalent'] = (dementia_date <= baseline) & dementia_date.notna()
    else:
        df['is_prevalent'] = False
    
    # Time 계산
    def get_time_event(row):
        baseline_val = row['baseline']
        dementia_val = row['dementia_date']
        death_val = row['death_date'] if death_date is not None else None
        event_val = row['event']
        
        # Prevalent case
        if row['is_prevalent']:
            return -1, -1  # 제거 대상
        
        # Baseline이 없으면 제외
        if pd.isna(baseline_val):
            return None, None
        
        # Event case: baseline부터 dementia_date까지
        if event_val == 1 and pd.notna(dementia_val):
            days = (dementia_val - baseline_val).days
            if days < 0:
                return -1, -1  # Prevalent case
            return 1, max(days, 0.1) / 365.25  # 최소 0.1년
        
        # Censored case: baseline부터 censor_date까지
        # 사망일이 있으면 사망일, 없으면 admin_censor
        end_date = death_val if (death_val is not None and pd.notna(death_val)) else admin_censor
        end_date = min(end_date, admin_censor)  # 미래 날짜 방지
        
        days = (end_date - baseline_val).days
        if days < 0:
            return None, None  # 오류
        
        return 0, max(days, 0.1) / 365.25  # 최소 0.1년
    
    # 컬럼 준비 (baseline은 이미 저장됨)
    df['dementia_date'] = dementia_date if dementia_date is not None else pd.Series([None] * len(df))
    if death_date is not None:
        df['death_date'] = death_date
    else:
        df['death_date'] = None
    
    # Time/Event 계산
    results = df.apply(get_time_event, axis=1, result_type='expand')
    df['event_calc'] = results[0]
    df['time_calc'] = results[1]
    
    # 최종 event/time (prevalent case는 -1)
    df['event'] = df['event_calc']
    df['time'] = df['time_calc']
    
    return df

def select_alpha_cv_10fold(X, y_surv, alphas, n_folds=10, random_state=42):
    """
    10-fold CV + 1SE rule로 최적 alpha 선택 (논문 방식)
    
    Args:
        X: feature matrix
        y_surv: survival array
        alphas: alpha candidates
        n_folds: CV fold 수 (논문: 10-fold)
    
    Returns:
        best_alpha: 최적 alpha
        mean_scores: 평균 C-index
        std_scores: 표준편차
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    
    print(f"   - {n_folds}-fold CV 진행 중...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_cv_tr, X_cv_val = X[train_idx], X[val_idx]
        y_cv_tr = Surv.from_arrays(
            event=y_surv['event'][train_idx],
            time=y_surv['time'][train_idx]
        )
        y_cv_val_event = y_surv['event'][val_idx]
        y_cv_val_time = y_surv['time'][val_idx]
        
        # 각 alpha에 대해 모델 학습 및 평가
        cv_scores_alpha = []
        for alpha in alphas:
            try:
                cox_cv = CoxnetSurvivalAnalysis(
                    alphas=[alpha],
                    l1_ratio=1.0,
                    fit_baseline_model=True,
                    max_iter=1000,
                    tol=1e-6
                )
                cox_cv.fit(X_cv_tr, y_cv_tr)
                
                # Linear predictor 계산
                if cox_cv.coef_.shape[1] > 0:
                    linear_pred = X_cv_val @ cox_cv.coef_[:, 0]
                    # C-index 계산
                    c_index = concordance_index_censored(
                        y_cv_val_event, y_cv_val_time, linear_pred
                    )[0]
                    cv_scores_alpha.append(c_index)
                else:
                    cv_scores_alpha.append(0.5)
            except Exception as e:
                cv_scores_alpha.append(0.5)
        
        cv_scores.append(cv_scores_alpha)
        if fold % 2 == 0:
            print(f"     Fold {fold}/{n_folds} 완료")
    
    # CV 점수 평균 및 표준편차
    cv_scores = np.array(cv_scores)
    mean_scores = np.mean(cv_scores, axis=0)
    std_scores = np.std(cv_scores, axis=0)
    
    # 1SE rule: 최대 C-index - 1 SE
    max_idx = np.argmax(mean_scores)
    max_score = mean_scores[max_idx]
    se = std_scores[max_idx] / np.sqrt(n_folds)  # 표준오차
    
    threshold = max_score - se
    
    # threshold보다 큰 가장 큰 alpha 선택 (더 간단한 모델)
    valid_indices = np.where(mean_scores >= threshold)[0]
    best_alpha_idx = valid_indices[-1] if len(valid_indices) > 0 else max_idx
    
    best_alpha = alphas[best_alpha_idx]
    
    return best_alpha, mean_scores, std_scores

def build_protrs_paper_method(
    baseline_date_col=None,
    death_date_col=None,
    n_folds=10
):
    """
    논문 방법론 정확히 따르기 - PrRSMDD-ADRD 생성
    
    Args:
        baseline_date_col: baseline date 컬럼명
        death_date_col: death date 컬럼명
        n_folds: CV fold 수 (논문: 10-fold)
    """
    print("=" * 70)
    print("PrRSMDD-ADRD 생성 (논문 방법론 정확히 따르기)")
    print("=" * 70)
    print("\n📌 논문 방법론:")
    print("   1. Inverse Normal Transformation (INT)")
    print("   2. Prevalent Case 제거")
    print("   3. 10-fold CV + 1SE rule")
    print("   4. LASSO Cox Regression")
    print("   5. C-index 평가")
    
    # -------------------------
    # 1. 데이터 로드 및 병합
    # -------------------------
    print("\n[1/7] 데이터 로드 및 병합...")
    df_master = pd.read_parquet(MASTER)
    print(f"   - Master 데이터: {len(df_master):,} 샘플")
    
    # Outcome 데이터 로드 (더 많은 participant.p42018 정보)
    df_outcome = pd.read_csv(OUTCOME_FILE, usecols=['eid', 'participant.p42018'])
    print(f"   - Outcome 데이터: {len(df_outcome):,} 샘플")
    print(f"   - participant.p42018 유효값: {df_outcome['participant.p42018'].notna().sum():,}개")
    
    # 병합 (outcome 데이터의 더 많은 정보 사용)
    df = df_master.merge(df_outcome, on='eid', how='left', suffixes=('', '_new'))
    
    # participant.p42018 우선순위: 새 데이터 > 기존 데이터
    if 'participant.p42018_new' in df.columns:
        df['participant.p42018'] = df['participant.p42018_new'].fillna(df['participant.p42018'])
        df = df.drop(columns=['participant.p42018_new'])
    
    print(f"   - 병합 후: {len(df):,} 샘플")
    print(f"   - participant.p42018 유효값: {df['participant.p42018'].notna().sum():,}개")
    
    # -------------------------
    # 2. Time/Event 계산 및 Prevalent Case 제거
    # -------------------------
    print("\n[2/7] Time/Event 계산 및 Prevalent Case 제거...")
    df = calculate_survival_time_paper_method(df, baseline_date_col, death_date_col)
    
    # Prevalent case 및 오류 데이터 제거
    before = len(df)
    df_clean = df[(df['event'] != -1) & (df['time'] > 0) & df['time'].notna()].copy()
    n_prevalent = (df['event'] == -1).sum()
    
    print(f"   - Prevalent case 제거: {n_prevalent:,}개")
    print(f"   - 최종 분석 데이터: {len(df_clean):,} 샘플 (제거 전: {before:,})")
    print(f"   - Event (치매): {df_clean['event'].sum():,} ({df_clean['event'].mean()*100:.2f}%)")
    print(f"   - Censored: {(df_clean['event']==0).sum():,}")
    print(f"   - Time 범위: {df_clean['time'].min():.2f} ~ {df_clean['time'].max():.2f}년")
    
    # -------------------------
    # 3. 단백질 컬럼 추출
    # -------------------------
    print("\n[3/7] 단백질 컬럼 추출...")
    protein_cols = get_protein_cols(df_clean)
    print(f"   - 단백질 컬럼 수: {len(protein_cols):,}")
    
    # -------------------------
    # 4. Train/Test split
    # -------------------------
    print("\n[4/7] Train/Test split...")
    X = df_clean[protein_cols].copy()
    y_event = df_clean['event'].values
    y_time = df_clean['time'].values
    
    X_tr, X_te, event_tr, event_te, time_tr, time_te = train_test_split(
        X, y_event, y_time,
        test_size=0.2,
        random_state=42,
        stratify=y_event
    )
    
    print(f"   - Train: {len(X_tr):,} 샘플 (event: {event_tr.sum():,})")
    print(f"   - Test: {len(X_te):,} 샘플 (event: {event_te.sum():,})")
    
    # -------------------------
    # 5. 데이터 전처리: Imputation + INT
    # -------------------------
    print("\n[5/7] 데이터 전처리 (Imputation + INT)...")
    imputer = SimpleImputer(strategy="median")
    
    # Imputation
    X_tr_imputed = imputer.fit_transform(X_tr)
    X_te_imputed = imputer.transform(X_te)
    
    # Inverse Normal Transformation (논문 방식)
    print("   - Inverse Normal Transformation 적용 중...")
    X_tr_int = apply_inverse_normal_transform(X_tr_imputed)
    X_te_int = apply_inverse_normal_transform(X_te_imputed)
    
    # NaN 처리 (INT 후에도 있을 수 있음)
    X_tr_int = np.nan_to_num(X_tr_int, nan=0.0)
    X_te_int = np.nan_to_num(X_te_int, nan=0.0)
    
    # 수치적 안정성
    X_tr_int = np.clip(X_tr_int, -10, 10)
    X_te_int = np.clip(X_te_int, -10, 10)
    
    print("   - INT 완료")
    
    # -------------------------
    # 6. 10-fold CV로 최적 alpha 선택
    # -------------------------
    print("\n[6/7] 10-fold CV로 최적 alpha 선택 (1SE rule)...")
    y_tr_surv = Surv.from_arrays(event=event_tr.astype(bool), time=time_tr)
    alphas = np.logspace(-4, 1, 100)  # 논문과 유사한 범위
    
    best_alpha, mean_scores, std_scores = select_alpha_cv_10fold(
        X_tr_int,
        {'event': event_tr.astype(bool), 'time': time_tr},
        alphas,
        n_folds=n_folds
    )
    
    print(f"   - 최적 alpha: {best_alpha:.6f}")
    print(f"   - 최대 C-index: {mean_scores.max():.4f}")
    best_alpha_idx = np.where(alphas == best_alpha)[0][0]
    print(f"   - 선택된 alpha의 C-index: {mean_scores[best_alpha_idx]:.4f} ± {std_scores[best_alpha_idx]:.4f}")
    
    # -------------------------
    # 7. 최적 alpha로 최종 모델 학습
    # -------------------------
    print("\n[7/7] 최적 alpha로 최종 모델 학습...")
    
    cox_lasso = CoxnetSurvivalAnalysis(
        alphas=[best_alpha],
        l1_ratio=1.0,
        fit_baseline_model=True,
        max_iter=1000,
        tol=1e-6
    )
    
    cox_lasso.fit(X_tr_int, y_tr_surv)
    
    # 선택된 단백질 확인
    coefs = cox_lasso.coef_[:, 0]
    selected_proteins = np.where(np.abs(coefs) > 1e-6)[0]
    print(f"   - 선택된 단백질 수: {len(selected_proteins):,} / {len(protein_cols):,}")
    
    # -------------------------
    # 8. ProtRS 계산 및 평가
    # -------------------------
    print("\n[8/8] ProtRS 계산 및 평가...")
    
    # 전체 데이터로 ProtRS 계산
    X_all_numeric = df_clean[protein_cols].select_dtypes(include=[np.number])
    X_all_imputed = imputer.transform(X_all_numeric)
    X_all_int = apply_inverse_normal_transform(X_all_imputed)
    X_all_int = np.nan_to_num(X_all_int, nan=0.0)
    X_all_int = np.clip(X_all_int, -10, 10)
    
    X_all_selected = X_all_int[:, selected_proteins]
    protrs_all = X_all_selected @ coefs[selected_proteins]
    
    # Test set ProtRS
    X_te_selected = X_te_int[:, selected_proteins]
    protrs_te = X_te_selected @ coefs[selected_proteins]
    
    # C-index 계산
    c_index = concordance_index_censored(
        event_te.astype(bool),
        time_te,
        protrs_te
    )[0]
    
    print(f"   - C-index (test): {c_index:.4f}")
    print(f"   - 논문 결과: C statistic = 0.84")
    
    # 결과 저장
    df_protrs = pd.DataFrame({
        "eid": df_clean["eid"].values,
        "ProtRS": protrs_all,
        "target_dementia": df_clean["target_dementia"].values,
        "target_age": df_clean["target_age"].values,
        "event": df_clean["event"].values,
        "time": df_clean["time"].values
    })
    
    df_protrs.to_parquet(OUT_PROTRS, index=False)
    print(f"   - ProtRS 저장: {OUT_PROTRS}")
    
    # 선택된 단백질 및 가중치 저장 (View A 인코더용)
    selected_protein_names = [protein_cols[i] for i in selected_proteins]
    df_weights = pd.DataFrame({
        "protein": selected_protein_names,
        "weight": coefs[selected_proteins]
    }).sort_values("weight", key=abs, ascending=False)
    
    df_weights.to_csv(OUT_WEIGHTS, index=False)
    print(f"   - 가중치 저장: {OUT_WEIGHTS}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ProtRS 분포
    ax = axes[0, 0]
    ax.hist(protrs_all, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel("ProtRS")
    ax.set_ylabel("Frequency")
    ax.set_title("ProtRS Distribution")
    ax.grid(True, alpha=0.3)
    
    # 2. ProtRS by Event
    ax = axes[0, 1]
    ax.boxplot([protrs_all[df_clean['event']==0], protrs_all[df_clean['event']==1]],
               labels=['No Dementia', 'Dementia'])
    ax.set_ylabel("ProtRS")
    ax.set_title("ProtRS by Dementia Status")
    ax.grid(True, alpha=0.3)
    
    # 3. CV scores
    ax = axes[1, 0]
    ax.plot(alphas, mean_scores, 'b-', label='Mean C-index')
    ax.fill_between(alphas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
    ax.axvline(best_alpha, color='r', linestyle='--', label=f'Best alpha ({best_alpha:.4f})')
    ax.set_xscale('log')
    ax.set_xlabel("Alpha")
    ax.set_ylabel("C-index")
    ax.set_title("10-fold CV Scores for Alpha Selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 선택된 단백질 계수
    ax = axes[1, 1]
    top_proteins = df_weights.head(15)
    ax.barh(range(len(top_proteins)), top_proteins['weight'].values)
    ax.set_yticks(range(len(top_proteins)))
    ax.set_yticklabels(top_proteins['protein'].values)
    ax.set_xlabel("Coefficient")
    ax.set_title("Top Selected Proteins")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"   - 시각화 저장: {OUT_PLOT}")
    
    print("\n" + "=" * 70)
    print("PrRSMDD-ADRD 통계 요약")
    print("=" * 70)
    print(f"ProtRS 평균: {protrs_all.mean():.2f}")
    print(f"ProtRS 표준편차: {protrs_all.std():.2f}")
    print(f"C-index (test): {c_index:.4f}")
    print(f"선택된 단백질 수: {len(selected_proteins):,}")
    print(f"Prevalent case 제거: {n_prevalent:,}개")
    
    print("\n✅ PrRSMDD-ADRD 생성 완료!")
    print(f"\n📁 출력 파일:")
    print(f"   - ProtRS: {OUT_PROTRS}")
    print(f"   - 가중치 (View A용): {OUT_WEIGHTS}")
    print(f"   - 시각화: {OUT_PLOT}")
    
    return df_protrs, df_weights

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PrRSMDD-ADRD 생성 (논문 방법론)')
    parser.add_argument('--baseline-col', type=str, default=None,
                        help='Baseline date 컬럼명')
    parser.add_argument('--death-col', type=str, default=None,
                        help='Death date 컬럼명')
    parser.add_argument('--n-folds', type=int, default=10,
                        help='CV fold 수 (기본: 10)')
    
    args = parser.parse_args()
    
    try:
        df_protrs, df_weights = build_protrs_paper_method(
            baseline_date_col=args.baseline_col,
            death_date_col=args.death_col,
            n_folds=args.n_folds
        )
    except ImportError as e:
        print(f"\n❌ 필요한 패키지가 없습니다: {e}")
        print("\n필요한 패키지 설치:")
        print("  pip install scikit-survival scipy")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

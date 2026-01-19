#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

def build_ukb_usable_datasets(
    file_prot="../../data/ukb/ukb_ppp_instance0.csv",
    file_meta="../../data/ukb/ukb_cog_cov_ppp.csv",
    file_outcome="../../data/ukb/ukb_cog_cov_master_plus_dementia_outcome.csv",
    out_master="../../data/ukb/ukb_usable_master.parquet",
    out_cases="../../data/ukb/ukb_dementia_cases.parquet",
):
    print("🚀 UKB usable dataset build start (Update for Survival Analysis)")
    
    # ---------------------------------------------------------
    # 0) 파일 존재 확인
    # ---------------------------------------------------------
    for fp in [file_prot, file_meta, file_outcome]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"파일이 없습니다: {fp}")

    # ---------------------------------------------------------
    # 1) Proteomics (Anchor)
    # ---------------------------------------------------------
    print("[1/6] Loading proteomics (anchor)...")
    df_prot = pd.read_csv(file_prot)
    if "eid" not in df_prot.columns:
        raise ValueError("PROT 파일에 'eid'가 없습니다.")
    print(f"   - PROT rows: {len(df_prot):,}, cols: {df_prot.shape[1]:,}")

    # ---------------------------------------------------------
    # 2) META: p53(참가일), p40000(사망일), p21003(나이), Sex, PCs
    # ---------------------------------------------------------
    print("[2/6] Loading metadata (Dates, Age, Sex, PCs)...")
    
    # 중요: p53(참가일)과 p40000(사망일)을 반드시 가져와야 함
    # 파일마다 컬럼명이 p53, p53_i0 등으로 다를 수 있어 체크 로직 추가
    possible_date_cols = ["p53", "p53_i0", "p40000", "p40000_i0", "p40000_i1"] 
    base_cols_wanted = ["eid", "p31", "p21003_i0"] + [f"pc__p22009_a{i}" for i in range(1, 11)]
    
    # 헤더 미리 읽기
    meta_header = pd.read_csv(file_meta, nrows=0).columns.tolist()
    
    # 실제 존재하는 컬럼만 선택
    real_cols = [c for c in base_cols_wanted if c in meta_header]
    date_cols = [c for c in possible_date_cols if c in meta_header]
    
    use_cols = list(set(real_cols + date_cols)) # 중복제거
    
    print(f"   - Loading cols: {len(use_cols)} columns including dates")
    df_meta = pd.read_csv(file_meta, usecols=use_cols)

    # 컬럼명 표준화 (분석하기 편하게)
    # p53(참가일) 찾기
    col_attend = next((c for c in ["p53_i0", "p53"] if c in df_meta.columns), None)
    # p40000(사망일) 찾기 (보통 인스턴스 0, 1 중 하나라도 있으면 됨. 여기선 우선순위 둠)
    col_death = next((c for c in ["p40000_i0", "p40000", "p40000_i1"] if c in df_meta.columns), None)

    if col_attend: df_meta = df_meta.rename(columns={col_attend: "date_attend"})
    if col_death: df_meta = df_meta.rename(columns={col_death: "date_death"})
    if "p21003_i0" in df_meta.columns: df_meta = df_meta.rename(columns={"p21003_i0": "age"})
    if "p31" in df_meta.columns: df_meta = df_meta.rename(columns={"p31": "sex"})

    if "date_attend" not in df_meta.columns:
        raise ValueError("⚠️ 치명적 오류: META 파일에 참가일(p53)이 없습니다. Cox 분석 불가!")

    # ---------------------------------------------------------
    # 3) OUTCOME: Dementia Date
    # ---------------------------------------------------------
    print("[3/6] Loading outcome (dementia date)...")
    out_cols = ["eid", "participant.p42018"] # 새로 찾은 파일 기준
    df_out = pd.read_csv(file_outcome, usecols=out_cols)
    df_out = df_out.rename(columns={"participant.p42018": "date_dementia"})

    # ---------------------------------------------------------
    # 4) MERGE & DATE PARSING
    # ---------------------------------------------------------
    print("[4/6] Merging & Parsing dates...")
    df_master = df_prot.merge(df_meta, on="eid", how="inner")
    df_master = df_master.merge(df_out, on="eid", how="left")
    
    # 날짜 변환 (에러 발생 시 NaT 처리)
    for col in ["date_attend", "date_death", "date_dementia"]:
        if col in df_master.columns:
            df_master[col] = pd.to_datetime(df_master[col], errors="coerce")

    # ---------------------------------------------------------
    # 5) ✨ CRITICAL: Survival Data Creation (Event & Time)
    # ---------------------------------------------------------
    print("[5/6] Calculating Event & Time for Cox Regression...")
    
    # 연구 종료일 (Censor Date): 데이터 추출 시점 (가장 최근 날짜)
    # 안전하게 2024-01-01 혹은 데이터 내 최대값 사용
    CENSOR_DATE = pd.Timestamp("2024-11-23") 
    
    def calculate_survival(row):
        start = row["date_attend"]
        event_date = row["date_dementia"]
        death_date = row.get("date_death", pd.NaT) # 없을 수도 있음
        
        if pd.isna(start): return -99, -99 # 참가일 모르면 삭제
        
        # A. Prevalent Case (참가 전에 이미 발병) -> 삭제 대상(-1)
        if pd.notna(event_date) and event_date <= start:
            return -1, -1
        
        # B. Incident Case (추적 중 발병) -> Event=1
        if pd.notna(event_date):
            days = (event_date - start).days
            return 1, days / 365.25
            
        # C. Censored Case (발병 안 함) -> Event=0
        # 종료 시점 = 사망일 vs 연구종료일 중 빠른 것
        end_date = CENSOR_DATE
        if pd.notna(death_date):
            end_date = min(death_date, CENSOR_DATE)
            
        days = (end_date - start).days
        # 날짜 오류로 음수 나오면 0.1년 처리
        return 0, max(days, 30) / 365.25 

    # 계산 적용
    surv_res = df_master.apply(calculate_survival, axis=1, result_type="expand")
    df_master["event"] = surv_res[0]
    df_master["time"] = surv_res[1]

    # Prevalent Case (-1) 및 오류 데이터 삭제
    n_total = len(df_master)
    df_master = df_master[df_master["event"] != -1]
    df_master = df_master[df_master["time"] > 0]
    df_master = df_master[df_master["event"] != -99]
    
    print(f"   - Removed Prevalent/Invalid cases: {n_total - len(df_master):,}")
    print(f"   - Final Cohort: {len(df_master):,}")
    print(f"   - Incident Dementia Cases (Event=1): {df_master['event'].sum():,}")

    # ---------------------------------------------------------
    # 6) Save
    # ---------------------------------------------------------
    print("[6/6] Saving outputs...")
    # 컬럼 정리 (필요한 것 위주로 정렬)
    cols_order = ["eid", "event", "time", "age", "sex", "date_attend", "date_dementia"] 
    # 나머지(단백질 등) 뒤에 붙이기
    cols_rest = [c for c in df_master.columns if c not in cols_order]
    df_master = df_master[cols_order + cols_rest]
    
    df_master.to_parquet(out_master, index=False)
    
    # Event=1 인 사람만 따로 저장 (분석용)
    df_cases = df_master[df_master["event"] == 1].copy()
    df_cases.to_parquet(out_cases, index=False)

    print("\n✅ DONE")
    print(f" - Master saved: {os.path.abspath(out_master)}")
    print(f" - Cases saved : {os.path.abspath(out_cases)}")

    return df_master

if __name__ == "__main__":
    build_ukb_usable_datasets()
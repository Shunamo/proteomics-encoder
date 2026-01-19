#!/usr/bin/env python3
import gzip
import pandas as pd
from collections import defaultdict

MASTER = "../../data/ukb/ukb_usable_master.parquet"
ALIASES = "../../data/stringdb/9606.protein.aliases.v12.0.txt"
LINKS = "../../data/stringdb/9606.protein.links.v12.0.txt"

OUT_MAP = "../../data/stringdb/ukb_to_string_map.csv"
OUT_EDGES = "../../data/stringdb/ukb_string_edges_topk20.csv"

TOPK = 20          # 노드당 상위 K개 이웃만 유지
MIN_SCORE = 700    # 0~1000 스케일 (STRING combined_score는 이미 *1000)

def get_ukb_proteins():
    df = pd.read_parquet(MASTER, columns=["eid"])
    # parquet에서 columns만 빠르게 읽기 어렵다면 전체 로드 후 columns 사용
    all_cols = pd.read_parquet(MASTER, engine="auto").columns
    prot = [
        c for c in all_cols
        if c not in ["eid","sex","target_age","target_dementia","participant.p42018"]
        and not c.startswith("pc__")
    ]
    return set(prot)

def load_alias_map(ukb_proteins):
    # gene_symbol(=alias) -> set(string_id)
    hit = defaultdict(set)

    with open(ALIASES, "rt") as f:
        header = f.readline().strip().split("\t")
        # expected columns: string_protein_id, alias, source
        for line in f:
            sp, alias, source = line.rstrip("\n").split("\t")
            a = alias.lower()
            if a in ukb_proteins:
                hit[a].add(sp)

    # gene_symbol -> single best id 선택 (여러개면 우선 하나 선택)
    # 실무적으로는 source 우선순위로 고르지만, 먼저 간단히 1개 선택
    mapping = {g: sorted(list(ids))[0] for g, ids in hit.items()}

    pd.DataFrame({"gene_symbol": list(mapping.keys()),
                  "string_protein_id": list(mapping.values())}).to_csv(OUT_MAP, index=False)
    return mapping

def build_subgraph_edges(mapped_ids):
    edges = []
    with open(LINKS, "rt") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue  # 컬럼이 3개 미만인 줄은 스킵
            p1, p2, s = parts[0], parts[1], parts[2]
            try:
                s = int(s)
            except ValueError:
                continue  # score를 정수로 변환할 수 없으면 스킵
            if s < MIN_SCORE:
                continue
            if (p1 in mapped_ids) and (p2 in mapped_ids):
                edges.append((p1, p2, s))
    return edges

def topk_sparsify(edges, topk=20):
    # 노드별로 score 높은 이웃 topk만 유지
    nbrs = defaultdict(list)
    for u, v, s in edges:
        nbrs[u].append((s, v))
        nbrs[v].append((s, u))

    kept = set()
    for u, lst in nbrs.items():
        lst.sort(reverse=True)
        for s, v in lst[:topk]:
            a, b = (u, v) if u < v else (v, u)
            kept.add((a, b, s))

    kept = list(kept)
    df = pd.DataFrame(kept, columns=["p1", "p2", "combined_score"])
    df.to_csv(OUT_EDGES, index=False)
    return df

if __name__ == "__main__":
    ukb_proteins = get_ukb_proteins()
    print("UKB proteins:", len(ukb_proteins))

    # UKB 컬럼은 소문자처럼 보이니까 소문자로 통일
    ukb_proteins = set([p.lower() for p in ukb_proteins])

    mapping = load_alias_map(ukb_proteins)
    print("Mapped proteins:", len(mapping))

    mapped_ids = set(mapping.values())
    edges = build_subgraph_edges(mapped_ids)
    print("Raw subgraph edges (score>=700):", len(edges))

    df_edges = topk_sparsify(edges, TOPK)
    print("TopK sparsified edges:", len(df_edges))

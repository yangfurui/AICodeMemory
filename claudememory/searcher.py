"""Hybrid retrieval: exact cosine over the whole library + BM25 rerank.

两阶段:
1. 向量召回 —— 归一化向量的内积 = cosine,一次矩阵点积算完全库(个人
   量级下这比任何 ANN 索引都简单且**精确**),取 top-CANDIDATES 候选
2. BM25 重排 —— jieba 分词后在候选集内算 Okapi BM25(IDF 也在候选集内,
   免建全库倒排),与向量分 6:4 融合

融合权重沿用 MemPalace 在 LongMemEval 上验证过的配比作起步值。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
CANDIDATES = 50  # 向量召回的候选池大小,重排在这个池子里进行


@dataclass
class Hit:
    score: float
    cos: float
    bm25: float
    session_id: str
    project: str
    date: str
    text: str


def _tokenize(text: str) -> list[str]:
    import jieba  # 惰性导入:首次调用会加载词典(~1s)

    return [t for t in jieba.cut_for_search(text.lower()) if t.strip()]


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]],
                 k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """标准 Okapi BM25,IDF 在候选集内计算。"""
    n = len(docs_tokens)
    if n == 0:
        return np.zeros(0)
    avgdl = sum(len(d) for d in docs_tokens) / n
    df: Counter = Counter()
    for d in docs_tokens:
        df.update(set(d))
    scores = np.zeros(n)
    for i, doc in enumerate(docs_tokens):
        tf = Counter(doc)
        dl = len(doc) or 1
        for q in query_tokens:
            if q not in tf:
                continue
            idf = math.log((n - df[q] + 0.5) / (df[q] + 0.5) + 1)
            scores[i] += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * dl / avgdl))
    return scores


def search(rows: list, matrix: np.ndarray, query_vec: np.ndarray,
           query: str, k: int = 5) -> list[Hit]:
    """rows/matrix 来自 Store.load_matrix(),query_vec 来自 Embedder.encode_query()。"""
    if not rows:
        return []

    # 阶段一:全库精确 cosine(归一化向量 → 内积即 cosine)
    cos = matrix @ query_vec
    pool = min(CANDIDATES, len(rows))
    cand_idx = np.argpartition(-cos, pool - 1)[:pool]

    # 阶段二:候选集内 BM25 + 融合
    query_tokens = _tokenize(query)
    docs_tokens = [_tokenize(rows[i][4]) for i in cand_idx]
    bm25 = _bm25_scores(query_tokens, docs_tokens)
    if bm25.max() > 0:  # 候选内 min-max 归一;全零(纯语义查询)则退化为纯向量排序
        bm25 = bm25 / bm25.max()

    fused = VECTOR_WEIGHT * cos[cand_idx] + BM25_WEIGHT * bm25
    order = np.argsort(-fused)[:k]

    hits = []
    for j in order:
        i = cand_idx[j]
        session_id, project, date, _, text = rows[i]
        hits.append(Hit(
            score=float(fused[j]), cos=float(cos[i]), bm25=float(bm25[j]),
            session_id=session_id, project=project, date=date, text=text,
        ))
    return hits

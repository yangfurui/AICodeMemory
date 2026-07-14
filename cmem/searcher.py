"""Hybrid retrieval: (vector top-N ∪ keyword top-N) → unified rerank.

v0.2 召回结构(修 v0.1 的架构缺陷):

    向量通道: 全库精确 cosine(numpy 矩阵点积)→ top-CANDIDATES
    关键词通道: SQLite FTS5(jieba 分词)   → top-CANDIDATES
                        │ 并集 │
                        ▼
    统一重排: 0.6 × cosine + 0.4 × BM25(并集内归一)→ top-k

v0.1 的教训:BM25 只在向量赢家里重排,导致"精确关键词命中但措辞与
查询不同构"的块(如错误码定案原话)连参赛资格都没有——小模型语义
对齐失灵的时刻,恰恰最需要关键词通道兜底,两通道必须各自独立召回。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .tokenizer import tokenize

VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
CANDIDATES = 50  # 每条通道各自的召回量,并集≤2×


@dataclass
class Hit:
    score: float
    cos: float
    bm25: float
    source: str
    session_id: str
    project: str
    date: str
    chunk_index: int
    text: str


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


def search(rows: list, matrix: np.ndarray, query_vec: np.ndarray, query: str,
           k: int = 5, fts_ids: set[str] | None = None) -> list[Hit]:
    """rows/matrix 来自 Store.load_matrix()(行首列是块 id),
    fts_ids 来自 Store.fts_candidates(),None 表示无关键词通道(退化为 v0.1 行为)。"""
    if not rows:
        return []

    # 通道一:全库精确 cosine(归一化向量 → 内积即 cosine)
    cos = matrix @ query_vec
    pool = min(CANDIDATES, len(rows))
    vec_idx = set(np.argpartition(-cos, pool - 1)[:pool].tolist())

    # 通道二:FTS5 关键词候选(id → 行号;rows 已被上游过滤时,滤掉的 id 自然失配丢弃)
    if fts_ids:
        id_to_idx = {r[0]: i for i, r in enumerate(rows)}
        vec_idx |= {id_to_idx[cid] for cid in fts_ids if cid in id_to_idx}

    cand_idx = np.array(sorted(vec_idx))

    # 统一重排:BM25 在并集内算并归一,与 cosine 凸组合
    query_tokens = tokenize(query)
    docs_tokens = [tokenize(rows[i][5]) for i in cand_idx]
    bm25 = _bm25_scores(query_tokens, docs_tokens)
    if bm25.max() > 0:  # 全零(纯语义查询)则退化为纯向量排序
        bm25 = bm25 / bm25.max()

    fused = VECTOR_WEIGHT * cos[cand_idx] + BM25_WEIGHT * bm25
    order = np.argsort(-fused)[:k]

    hits = []
    for j in order:
        i = int(cand_idx[j])
        _, session_id, project, date, chunk_index, text, source = rows[i]
        hits.append(Hit(
            score=float(fused[j]), cos=float(cos[i]), bm25=float(bm25[j]),
            source=source, session_id=session_id, project=project, date=date,
            chunk_index=chunk_index, text=text,
        ))
    return hits

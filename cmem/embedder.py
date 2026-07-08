"""Local embedding via bge-small-zh-v1.5 (512-dim, ~100MB, runs on CPU).

注意:中文版 small 是 512 维(英文版 bge-small-en 才是 384),别记混。

- 全程本地推理,首次使用时从 HuggingFace 下载模型(之后离线可用)
- normalize_embeddings=True:向量归一化后,cosine 相似度 = 纯内积,
  检索侧一次矩阵点积就能算完全库
- bge 官方约定:短查询→长文档的检索场景,**查询侧**要加指令前缀,
  文档侧不加——漏掉这个前缀召回质量会明显下降
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章:"
DIM = 512


class Embedder:
    def __init__(self) -> None:
        # 惰性导入:torch 链路加载秒级,别让 cmem status 这种命令也扛这个开销
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_NAME)

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """文档侧批量嵌入 → (n, 512) float32,已归一化。"""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """查询侧嵌入(带 bge 指令前缀)→ (512,) float32。"""
        return self.encode_texts([QUERY_INSTRUCTION + query])[0]

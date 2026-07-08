"""Shared tokenizer — 入库(FTS5)与查询(BM25)必须用同一套分词,否则对不上。"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    import jieba  # 惰性导入:首次调用加载词典(~1s),别拖累不需要分词的命令

    return [t for t in jieba.cut_for_search(text.lower()) if t.strip()]

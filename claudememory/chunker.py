"""Split dialogue into embedding-sized chunks.

切块原则:一问一答(exchange)是语义最完整的最小单元,优先一块装下。

硬约束:bge-small-zh 的上下文只有 512 token(中文约等于 500 字),超长文本
在嵌入时会被**静默截断**——截掉的部分对向量零贡献,等于检索不到。所以:

- 整个 exchange ≤ CHUNK_LIMIT:一块
- 超长:assistant 回答按段落贪心装箱切成多块,**每块都重复携带问题前缀**
  (问题截断到 QUESTION_PREFIX_LIMIT),保证任何一块单独拿出来语义都完整
- 单段落仍超限:按行、再按字符硬切兜底
"""

from __future__ import annotations

from dataclasses import dataclass

from .extractor import SessionDialogue

CHUNK_LIMIT = 450  # 字符;对齐 bge 512 token,留余量给角色标记与英文
QUESTION_PREFIX_LIMIT = 120  # 超长回答拆块时,问题前缀最多带这么多字


@dataclass
class Chunk:
    session_id: str
    project: str
    date: str
    index: int  # 会话内序号;切块是确定性的,同输入必得同序号
    text: str


def _split_long(text: str, limit: int) -> list[str]:
    """段落 → 行 → 硬切,三级兜底的贪心装箱。"""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        while len(para) > limit:  # 单段超限:按行再不行硬切
            cut = para.rfind("\n", 0, limit)
            cut = cut if cut > limit // 3 else limit
            head, para = para[:cut], para[cut:].lstrip("\n")
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(head)
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > limit:
            pieces.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p.strip()]


def chunk_session(sess: SessionDialogue) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(text: str) -> None:
        chunks.append(
            Chunk(
                session_id=sess.session_id,
                project=sess.project,
                date=sess.date,
                index=len(chunks),
                text=text,
            )
        )

    i, msgs = 0, sess.messages
    while i < len(msgs):
        # 组一个 exchange:一条 user + 紧随的 assistant(若有)
        if msgs[i].role == "user":
            user_text = msgs[i].text
            assistant_text = msgs[i + 1].text if i + 1 < len(msgs) and msgs[i + 1].role == "assistant" else ""
            i += 2 if assistant_text else 1
        else:  # 开头就是 assistant(如恢复的会话),单独成块
            user_text, assistant_text = "", msgs[i].text
            i += 1

        whole = (f"USER: {user_text}\nASSISTANT: {assistant_text}" if assistant_text
                 else f"USER: {user_text}") if user_text else f"ASSISTANT: {assistant_text}"
        if len(whole) <= CHUNK_LIMIT:
            add(whole)
            continue

        # 超长:问题作前缀,回答装箱拆块
        prefix = f"USER: {user_text[:QUESTION_PREFIX_LIMIT]}\nASSISTANT: " if user_text else "ASSISTANT: "
        body_limit = max(CHUNK_LIMIT - len(prefix), CHUNK_LIMIT // 2)
        if not assistant_text:  # 纯超长提问(粘贴大段材料)
            for piece in _split_long(user_text, CHUNK_LIMIT):
                add(f"USER: {piece}")
            continue
        for piece in _split_long(assistant_text, body_limit):
            add(prefix + piece)

    return chunks

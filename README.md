# AICodeMemory

**把 Claude Code / Codex 的本地会话变成永久、可检索的原话档案。**

Local semantic memory for your AI coding sessions — verbatim, offline, forever.

把 `~/.claude/projects/` 与 `~/.codex/sessions/` 下的会话历史归档成
**永久、可语义检索**的本地记忆库,让你以及 AI 自己随时找回"当时的原话",
哪怕上游记录已经被清理或格式发生变化。

```
$ cmem search "AGP 8 升级后 Kotlin 编译失败怎么解决"

[1] 0.880 · 2026-07-07 · jiage_mobile · 7ccce197
    ASSISTANT: AGP 8 升 Kotlin 2.2.20 后,老插件未设 jvmTarget 导致校验失败;
    在 android/build.gradle 的 subprojects 块里写死 kotlinOptions.jvmTarget = '1.8'
    才解决,不能用动态读 targetCompatibility……
```

## 为什么

AI 编码助手的本地会话首先是产品运行状态,不是稳定的长期档案;每个新会话也
不会可靠地带回全部旧上下文。AICodeMemory 在上游记录仍可见时把它们归档:

- **永久档案** — 原始会话先安全存成 gzip 底片,SQLite 只保存当前检索投影;源被清理后仍可从 raw 完整重建,算法升级永不触碰底片(有契约测试锁死)
- **极简** — 单文件 SQLite + numpy 精确检索,零服务、零守护进程、零配置。核心代码几百行,一顿饭的时间能读完
- **中文一等公民** — bge 中文 embedding + jieba 分词的混合检索(语义 0.6 + 关键词 0.4)
- **数据不出机** — 本地模型推理,无 API key,无云,无遥测

## 安装

```bash
git clone https://github.com/yangfurui/AICodeMemory.git
cd AICodeMemory
python3 -m venv .venv && .venv/bin/pip install -e .
# 把 .venv/bin/cmem 加进 PATH,或直接用完整路径调用
```

依赖 Python ≥ 3.10。首次索引时自动从 HuggingFace 下载 embedding 模型(`BAAI/bge-small-zh-v1.5`,约 100MB),之后完全离线。

## 使用

```bash
cmem index               # 同时索引 Claude Code + Codex(增量,首次即全量)
cmem index --provider codex  # 只索引 Codex;可换成 claude
cmem search "查询" -k 5  # 语义检索,返回原文块+出处(日期/项目/会话)
cmem search "查询" --source codex  # 只查指定来源
cmem status              # 库概况:块数、会话数、日期覆盖
```

实测量级参考:453 个会话(约一个月重度使用)→ 13k 块,首次索引 4.5 分钟(M 系列 CPU),增量重跑 12 秒,单次查询亚秒级。

## 让 AI 自己查

不需要 MCP、不需要插件——Claude Code 本来就会跑命令。在你的 `~/.claude/CLAUDE.md` 加一条:

```markdown
## 历史会话检索
被问到过去的讨论、决策、结论时,先跑 `cmem search "<问题>"` 查历史原文,
基于结果回答并注明出处;查不到就明说,不要凭印象编。
```

Codex 使用同样的软约定,写入全局 `~/.codex/AGENTS.md` 即可。之后问
"上次那个编译错误怎么解决的?",AI 会自行检索并引用原话回答。正式 MCP
集成仍在规划中,CLI 检索已经可用。

## 工作原理

```
~/.claude/projects/**/*.jsonl ─┐
~/.codex/sessions/**/*.jsonl  ─┴─(Claude/Codex 来源适配器)
    │  cmem index(一次扫描,双层归档)
    ├────────────────────────┐
    ▼                        ▼
【raw 永久档案】       【text 当前投影 + 索引层】
 源文件原样 gzip           去噪 → 一问一答切块 → 本地嵌入(bge, 512 维)
 ~/.cmem/raw/              → SQLite(原文 + 来源 + 向量 + FTS,~/.cmem/memory.sqlite3)
 永不自动删除;               ▲
 非追加改写拒绝覆盖             │
                              │  cmem search
              查询嵌入 → 全库精确 cosine ∪ FTS5 关键词召回
               → jieba BM25 + 向量 6:4 融合重排 → top-k 原文
```

三层数据的生命周期(这是本项目的核心契约,有测试锁死):

| 层 | 内容 | 能否再生 |
|---|---|---|
| raw 永久档案 | 源 jsonl 的 gzip 拷贝(底片) | 不可再生,**永不自动删除**;只接受字节级追加更新 |
| text 当前投影 | 去噪、切块后的对话文本 | 可从 raw 重提取;会话更新时原子替换 |
| 索引层 | 向量 + FTS | 随时可从 text 投影重算 |

数据安全底线:**原始记录永不丢,检索结果可随算法升级重新生成**。换 embedding 模型 → 只重算向量;改提取算法 → 从 raw 重建 text 投影。

设计取舍:**不用向量数据库、不用 ANN 索引**。个人量级(几万~几十万块)下,numpy 全库精确计算本身就是毫秒级,比任何近似索引都简单且召回=100%。

## ⚠️ 备份提醒

`~/.cmem/` 可能保存着**上游已清理、无法重新生成的唯一对话记录**——请把它纳入你的常规备份(Time Machine / 云盘同步任选)。这是整个工具里唯一需要你操心的一件事。

## License

MIT

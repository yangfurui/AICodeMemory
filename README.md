# AICodeMemory

**Local semantic memory for Claude Code — search your session history, verbatim, offline.**

给 Claude Code 的本地会话记忆:把 `~/.claude/projects/` 下的历史会话索引成可语义检索的本地库,让你——以及 Claude 自己——随时找回"当时的原话"。

```
$ cmem search "AGP 8 升级后 Kotlin 编译失败怎么解决"

[1] 0.880 · 2026-07-07 · jiage_mobile · 7ccce197
    ASSISTANT: AGP 8 升 Kotlin 2.2.20 后,老插件未设 jvmTarget 导致校验失败;
    在 android/build.gradle 的 subprojects 块里写死 kotlinOptions.jvmTarget = '1.8'
    才解决,不能用动态读 targetCompatibility……
```

## 为什么

Claude Code 的会话记录 **30 天后被自动清理**,而且每个新会话都是失忆重启——上周定的方案、踩过的坑、说过的原话,全部找不回。AICodeMemory 把它们变成一个可语义检索的本地库:

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
cmem index               # 索引会话历史(增量,首次即全量;之后随时重跑,只处理有变化的会话)
cmem search "查询" -k 5  # 语义检索,返回原文块+出处(日期/项目/会话)
cmem status              # 库概况:块数、会话数、日期覆盖
```

实测量级参考:453 个会话(约一个月重度使用)→ 13k 块,首次索引 4.5 分钟(M 系列 CPU),增量重跑 12 秒,单次查询亚秒级。

## 让 Claude 自己查

不需要 MCP、不需要插件——Claude Code 本来就会跑命令。在你的 `~/.claude/CLAUDE.md` 加一条:

```markdown
## 历史会话检索
被问到过去的讨论、决策、结论时,先跑 `cmem search "<问题>"` 查历史原文,
基于结果回答并注明出处;查不到就明说,不要凭印象编。
```

之后问 Claude"上次那个编译错误怎么解决的?",它会自己检索并引用原话回答。

## 工作原理

```
~/.claude/projects/**/*.jsonl
    │  cmem index
    ▼
去噪(只留人/AI 对话,滤工具输出、思维链、系统噪音)
 → 切块(一问一答一块,对齐 embedding 模型 512 token 上限)
 → 本地嵌入(bge-small-zh-v1.5,512 维)
 → SQLite(原文 + 向量,~/.cmem/memory.sqlite3)
    ▲
    │  cmem search
查询嵌入 → 全库精确 cosine(numpy 矩阵点积)→ top-50 候选
 → jieba BM25 重排 → 6:4 融合 → top-k 原文
```

设计取舍:**不用向量数据库、不用 ANN 索引**。个人量级(几万~几十万块)下,numpy 全库精确计算本身就是毫秒级,比任何近似索引都简单且召回=100%。索引库是纯派生缓存——删掉重跑 `cmem index` 即可完整重建,永远不需要备份或迁移。

## License

MIT

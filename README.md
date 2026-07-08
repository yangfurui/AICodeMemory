# AICodeMemory

**Claude Code 只给你 30 天,AICodeMemory 把它变成永久。**

Local semantic memory for your AI coding sessions — verbatim, offline, forever.

把 `~/.claude/projects/` 下的会话历史归档成**永久、可语义检索**的本地记忆库,让你——以及 Claude 自己——随时找回"当时的原话",哪怕原始记录早已被自动清理。

```
$ cmem search "AGP 8 升级后 Kotlin 编译失败怎么解决"

[1] 0.880 · 2026-07-07 · jiage_mobile · 7ccce197
    ASSISTANT: AGP 8 升 Kotlin 2.2.20 后,老插件未设 jvmTarget 导致校验失败;
    在 android/build.gradle 的 subprojects 块里写死 kotlinOptions.jvmTarget = '1.8'
    才解决,不能用动态读 targetCompatibility……
```

## 为什么

Claude Code 的会话记录 **30 天后被自动清理**,而且每个新会话都是失忆重启——上周定的方案、踩过的坑、说过的原话,全部找不回。AICodeMemory 在清理发生之前把它们归档:

- **永久档案** — 原始会话 gzip 存档 + 对话原文入库,双层保全;源被清理后记忆依然完整,任何升级都不丢数据(有契约测试锁死)
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
~/.claude/projects/**/*.jsonl(源:Claude Code 30 天滚动清理)
    │  cmem index(一次扫描,双层归档)
    ├────────────────────────┐
    ▼                        ▼
【raw 原始层】           【text 档案层 + 索引层】
 源文件原样 gzip           去噪 → 一问一答切块 → 本地嵌入(bge, 512 维)
 ~/.cmem/raw/              → SQLite(原文 + 向量 + FTS,~/.cmem/memory.sqlite3)
 永不自动删除                  ▲
                              │  cmem search
              查询嵌入 → 全库精确 cosine ∪ FTS5 关键词召回
               → jieba BM25 + 向量 6:4 融合重排 → top-k 原文
```

三层数据的生命周期(这是本项目的核心契约,有测试锁死):

| 层 | 内容 | 能否再生 |
|---|---|---|
| raw 原始层 | 源 jsonl 的 gzip 拷贝(底片) | 不可再生,**永不自动删除** |
| text 档案层 | 去噪后的对话原文(剪报) | 可从 raw 重提取,**永不自动删除** |
| 索引层 | 向量 + FTS | 随时可从 text 重算 |

升级永不丢数据:换 embedding 模型 → 只重算向量;改提取算法 → 从 raw 重提取。

设计取舍:**不用向量数据库、不用 ANN 索引**。个人量级(几万~几十万块)下,numpy 全库精确计算本身就是毫秒级,比任何近似索引都简单且召回=100%。

## ⚠️ 备份提醒

`~/.cmem/` 保存着**超过 30 天窗口的唯一对话记录**(源已被 Claude Code 清理)——请把它纳入你的常规备份(Time Machine / 云盘同步任选)。这是整个工具里唯一需要你操心的一件事。

## License

MIT

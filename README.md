# AICodeMemory

**把 AI 编码会话变成自己掌控、永久、跨客户端可检索的原话档案。**

Local semantic memory for your AI coding sessions — verbatim, offline, forever.

把 Claude Code、Codex 与 Cursor 的本地会话历史归档成**永久、可语义检索**
的本地记忆库,让你以及 AI 自己随时找回"当时的原话",哪怕上游记录已经
被清理或格式发生变化。

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

- **永久档案** — 会话先安全存成 gzip 底片,SQLite 只保存当前检索投影;源被清理后仍可从 raw 完整重建,算法升级永不触碰底片(有契约测试锁死)
- **极简** — 单文件 SQLite + numpy 精确检索,零服务、零守护进程、零配置。核心代码几百行,一顿饭的时间能读完
- **中文一等公民** — bge 中文 embedding + jieba 分词的混合检索(语义 0.6 + 关键词 0.4)
- **跨客户端** — Claude Code、Codex 与 Cursor 的原话进同一座档案库,三个客户端也都能通过同一个 MCP 查询
- **本地数据层** — 存档、embedding 与检索无 API key、无云、无遥测;AI 集成只把命中原文交给当前客户端(见下文隐私边界)

## 安装

依赖 Python ≥ 3.10 与 Git。下面命令以 macOS / Linux 为例。

推荐使用 [`uv tool`](https://docs.astral.sh/uv/guides/tools/):它会为
AICodeMemory 创建独立、持久的环境,同时把 `cmem` 与 `cmem-mcp` 放进 `PATH`。

```bash
uv tool install git+https://github.com/yangfurui/AICodeMemory.git
cmem --help
```

如果安装成功后找不到 `cmem`,运行 `uv tool update-shell`,然后重新打开
终端。不要用临时环境 `uvx` 代替安装:MCP 配置需要一个长期稳定的
`cmem-mcp` 可执行路径。

安装环境还包含 PyTorch 等运行依赖,体积因平台而异;本项目在 macOS /
Python 3.14 的隔离实测中约占 0.8–1.1 GB。首次索引所需的约 100 MB
embedding 模型不包含在这个数字里。

没有 `uv` 时,按下面顺序选一种。

<details>
<summary><strong>备选 1:pipx</strong></summary>

[`pipx`](https://pipx.pypa.io/stable/how-to/install-pipx/) 也会创建独立环境
并暴露全局命令:

```bash
pipx install git+https://github.com/yangfurui/AICodeMemory.git
cmem --help
```

如果找不到 `cmem`,运行 `pipx ensurepath` 后重新打开终端。

</details>

<details>
<summary><strong>备选 2:标准 venv + pip</strong></summary>

这个方案不需要额外的包管理工具:

```bash
python3 -m venv "$HOME/.local/share/ai-code-memory"
"$HOME/.local/share/ai-code-memory/bin/python" -m pip install \
  git+https://github.com/yangfurui/AICodeMemory.git
export PATH="$HOME/.local/share/ai-code-memory/bin:$PATH"
cmem --help
```

`export` 只对当前终端生效;要长期使用,请把同一行放进你的 shell 配置。
不要在执行 `cmem setup` 后移动这个 venv,MCP 配置会保存其绝对路径。

</details>

<details>
<summary><strong>开发者:从本地源码可编辑安装</strong></summary>

```bash
git clone https://github.com/yangfurui/AICodeMemory.git
cd AICodeMemory
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

</details>

## 快速开始

先把现有会话建成本地索引,再搜一个你肯定讨论过的问题:

```bash
cmem index
cmem status
cmem search "上次那个编译错误怎么解决的?"
```

`cmem index` 会同时扫描 Claude Code、Codex 和 Cursor,某个客户端未安装
不影响另外两个。首次索引会从 HuggingFace 下载 embedding 模型
(`BAAI/bge-small-zh-v1.5`,约 100MB),之后存档、embedding 和搜索都可以
离线进行。

如果希望 AI 自动查历史,再选一种集成方式:

```bash
cmem setup                 # 推荐:注册到 Claude/Codex/Cursor
# 或
cmem setup --instructions  # 不注册 MCP,只写入 CLI 搜索软约定
```

两种集成都是可选的;即使不执行 `cmem setup`,你也可以一直手动使用
`cmem index` / `cmem search`。

## 常用命令

```bash
cmem index               # 同时索引 Claude Code + Codex + Cursor
cmem index --provider cursor # 只索引 Cursor;也可换成 claude/codex
cmem search "查询" -k 5  # 语义检索,返回原文块+出处(日期/项目/会话)
cmem search "查询" --source codex  # 只查指定来源
cmem status              # 库概况:块数、会话数、日期覆盖
cmem verify              # 检查 raw gzip 底片和 SQLite 完整性
cmem show <会话ID前缀>  # 展开搜索结果对应的整场会话
```

实测量级参考:453 个会话(约一个月重度使用)→ 13k 块,首次索引 4.5 分钟(M 系列 CPU),增量重跑 12 秒,单次查询亚秒级。

## 让 AI 自己查

推荐把 AICodeMemory 注册为本地 stdio MCP server:

```bash
cmem setup
```

它会同时发现 Claude Code、Codex 与 Cursor:前两者通过 CLI 注册,
Cursor 则安全合并[官方全局配置](https://docs.cursor.com/context/model-context-protocol)
`~/.cursor/mcp.json`。重复执行不会重复写入;所有客户端都先只读检查,
任何一处存在不同的同名 `cmem` 配置或损坏的 Cursor JSON,就整体停止
而不覆盖。
server 只注册 3 个只读工具:

- `search_history`:搜索过去的讨论、决策与原话
- `get_session`:按搜索结果的 `session_key` 展开整场会话或命中块前后文
- `memory_status`:查看记忆覆盖范围、完整性和索引新鲜度

server 在客户端会话期间复用 embedding 模型与向量矩阵;另一进程运行
`cmem index` 后,下次查询会自动重载新索引。

Cursor 的普通 Agent 历史由[官方说明](https://docs.cursor.com/en/agent/chat/history)
保存在本地 SQLite。AICodeMemory 会把共享数据库规范化为逐会话、只追加的
JSONL 底片:保留所有非空 user/assistant 文字、项目与时间出处;排除工具调用、
代码上下文和执行状态等大体积内部载荷。当前检索投影每轮取最后一条 assistant
文字作为最终回答;消息被 Cursor 修改时会追加 revision,不会覆盖旧底片。
Background Agent 对话由 Cursor 存在远端,不在本地采集范围内。Cursor 官方
没有把 SQLite 内部表结构承诺为稳定接口;若未来版本不兼容,索引会明确失败且
不刷新成功心跳,避免静默漏采。

不想使用 MCP 时,Claude Code 和 Codex 也可以直接跑 CLI。下面命令会在回显
将写入的受管区块后,幂等更新 `~/.claude/CLAUDE.md` 与 Codex 当前
全局指令文件(`${CODEX_HOME:-~/.codex}/AGENTS.md`,存在非空
`AGENTS.override.md` 时改写它):

```bash
cmem setup --instructions
```

它写入的软约定等价于下面内容;Codex 需新开会话才会重新读取全局
`AGENTS.md`。`cmem setup --claude-md` 作为兼容命令,仍只修改 Claude 文件。

```markdown
## 历史会话检索
被问到过去的讨论、决策、结论时,先跑 `cmem search "<问题>"` 查历史原文,
基于结果回答并注明出处;查不到就明说,不要凭印象编。
```

Codex 和 Claude 之后被问到"上次那个编译错误怎么解决的?"时,会按这条软约定
自行检索并引用原话回答。
需要撤销时运行 `cmem setup --remove`,它只会移除 AICodeMemory 在
Claude/Codex/Cursor 中的 MCP 配置和
`CLAUDE.md` / `AGENTS.md` marker 区块。

隐私边界:embedding、存档和检索都在本地;但 AI 调用 MCP 或 CLI 后,返回的命中
原文会进入当前 Claude/Codex/Cursor 的模型上下文,并非"所有内容永不离机"。

## 工作原理

```
~/.claude/projects/**/*.jsonl ─┐
~/.codex/sessions/**/*.jsonl  ─┤
Cursor/User/globalStorage/state.vscdb ─(规范化逐字消息)─┘
    │  cmem index(一次扫描,双层归档)
    ├────────────────────────┐
    ▼                        ▼
【raw 永久档案】       【text 当前投影 + 索引层】
 JSONL gzip 底片           去噪 → 一问一答切块 → 本地嵌入(bge, 512 维)
 ~/.cmem/raw/              → SQLite(原文 + 来源 + 向量 + FTS,~/.cmem/memory.sqlite3)
 永不自动删除;               ▲
 非追加改写拒绝覆盖             │
                              │  cmem search / MCP search_history
              查询嵌入 → 全库精确 cosine ∪ FTS5 关键词召回
               → jieba BM25 + 向量 6:4 融合重排 → top-k 原文
```

三层数据的生命周期(这是本项目的核心契约,有测试锁死):

| 层 | 内容 | 能否再生 |
|---|---|---|
| raw 永久档案 | Claude/Codex 源 JSONL;Cursor 逐字消息事件 JSONL | 不可再生,**永不自动删除**;只接受字节级追加更新 |
| text 当前投影 | 去噪、切块后的对话文本 | 可从 raw 重提取;会话更新时原子替换 |
| 索引层 | 向量 + FTS | 随时可从 text 投影重算 |

数据安全底线:**原始记录永不丢,检索结果可随算法升级重新生成**。换 embedding 模型 → 只重算向量;改提取算法 → 从 raw 重建 text 投影。

设计取舍:**不用向量数据库、不用 ANN 索引**。个人量级(几万~几十万块)下,numpy 全库精确计算本身就是毫秒级,比任何近似索引都简单且召回=100%。

## ⚠️ 备份提醒

`~/.cmem/` 可能保存着**上游已清理、无法重新生成的唯一对话记录**——请把它纳入你的常规备份(Time Machine / 云盘同步任选)。这是整个工具里唯一需要你操心的一件事。

## License

MIT

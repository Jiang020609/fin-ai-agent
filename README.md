# FinAI — 基于大模型的全栈金融资产问答系统

一个结合实时行情 API、RAG 知识检索和大语言模型的金融问答系统。支持资产价格查询、涨跌原因分析、金融知识问答，并在架构层面系统性控制回答准确性。

## 目录

- [系统架构](#系统架构)
- [核心能力](#核心能力)
- [技术选型](#技术选型)
- [路由设计](#路由设计)
- [Prompt 设计思路](#prompt-设计思路)
- [准确性控制](#准确性控制--hallucination-mitigation)
- [数据来源](#数据来源)
- [项目结构](#项目结构)
- [本地运行](#本地运行)
- [Docker 部署](#docker-部署)
- [优化与扩展思考](#优化与扩展思考)

---

## 系统架构

```
    用户浏览器
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Frontend  (Next.js + TypeScript)                │
│                                                  │
│  InputBar → useChatHistory(状态机) → api.ts(SSE) │
│                                                  │
│  渲染组件:                                       │
│    MessageBubble   KPICards        FinancialChart │
│    ThoughtChain    TrendBadge     ComparisonTable │
│    SourceList      EvidenceAnalysis DataFreshness │
└──────────────────────┬───────────────────────────┘
                       │  SSE: thought/meta/chart/token/done
                       ▼
┌──────────────────────────────────────────────────┐
│  Backend  (FastAPI)                              │
│                                                  │
│  Phase 1 ─ Query Understanding     [agent.py]    │
│    Session消解 → 查询改写 → Ticker提取 → 意图分类│
│    (正则快速分类 → LLM Function Call fallback)   │
│                                                  │
│  Phase 2 ─ Plan Generation     [orchestrator.py] │
│    意图 → PLAN_TEMPLATE → 有序步骤列表           │
│                                                  │
│  Phase 3 ─ Step Execution          [steps.py]    │
│    22个@register_step函数, 共享StepContext        │
│    fetch → validate → prompt → LLM → assemble    │
│                                                  │
│  底层服务:                                       │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐│
│  │ market.py  │ │  rag.py     │ │ grounding.py ││
│  │ 4源级联    │ │ BM25+Vector │ │ 三层准确性   ││
│  │ +缓存降级  │ │ +LLM Rerank │ │ 控制         ││
│  └─────┬──────┘ └──────┬──────┘ └──────────────┘│
└────────┼───────────────┼────────────────────────-┘
         ▼               ▼
┌──────────────────────────────────────────────────┐
│  External Services                               │
│                                                  │
│  Yahoo Finance / Finnhub / Alpha Vantage / Stooq │
│  OpenAI / OpenRouter  (LLM + Embedding)          │
│  Tavily / SerpAPI     (Web Search)               │
└──────────────────────────────────────────────────┘
```

<details>
<summary>Mermaid 版本（支持渲染的环境可展开）</summary>

```mermaid
graph TB
    subgraph Frontend["Frontend (Next.js + TypeScript)"]
        UI["ChatPanel / InputBar"]
        Charts["FinancialChart + KPICards + TrendBadge + DataFreshness"]
        Sources["SourceList + DisclaimerBar"]
        Thought["ThoughtChain"]
        State["useChatHistory (状态机)"]
        UI --> State
        State -->|SSE 流式| API_Client["api.ts (SSE Parser)"]
    end

    subgraph Backend["Backend (FastAPI)"]
        Router["chat.py (API 路由)"]
        Agent["agent.py (Query Understanding)"]
        Orchestrator["orchestrator.py (Plan + Execute)"]
        Steps["steps.py (22 步骤函数)"]

        subgraph Intent["意图分类 (双层)"]
            Regex["正则快速分类"]
            LLM_Cls["LLM Fallback"]
        end

        subgraph Tools["工具服务"]
            Market["market.py\n4源级联 + 缓存"]
            RAG["rag.py\nChromaDB + BM25 混合检索"]
            WebSearch["web_search.py\nTavily / SerpAPI"]
            Trend["trend.py\n规则化趋势分类"]
        end

        subgraph Grounding["准确性控制层"]
            Validate["数据校验\n(NaN/Inf/负值拦截)"]
            FactCheck["事实核查\n(LLM 自检 + 重试)"]
            NumCheck["数字交叉验证\n(回答 vs 原始数据)"]
            MissingNotice["缺失字段标注"]
        end

        LLM["llm.py\nOpenAI API 封装"]
    end

    subgraph External["外部服务"]
        Yahoo["Yahoo Finance"]
        Finnhub["Finnhub API"]
        AlphaV["Alpha Vantage"]
        Stooq["Stooq CSV"]
        OpenAI["OpenAI / OpenRouter"]
        SearchAPI["Tavily / SerpAPI"]
    end

    API_Client -->|POST /api/chat/stream| Router
    Router --> Agent
    Agent --> Intent
    Agent --> Orchestrator
    Orchestrator --> Steps
    Steps --> Tools
    Steps --> Grounding
    Steps --> LLM

    Market --> Yahoo
    Market --> Finnhub
    Market --> AlphaV
    Market --> Stooq
    LLM --> OpenAI
    WebSearch --> SearchAPI

    Orchestrator -->|structured_response| Router
    Router -->|SSE events| API_Client
    API_Client --> Charts
    API_Client --> Sources
    API_Client --> Thought
```

</details>

## 核心能力

### 1. 资产价格与涨跌分析

- 获取实时/近期价格数据（4 层数据源级联，任一成功即返回）
- 计算 7 日、30 日涨跌幅和区间高低点
- **规则化趋势分类**：基于涨跌幅 + 振幅 + 线性斜率，不依赖 LLM 判断
- 返回结构化 `data_summary` + `trend_summary`，客观数据与分析严格分离

### 2. 涨跌原因分析

- 抽取资产和日期引用（支持"1月15日""上周""最近"等表达）
- 获取对应窗口的行情数据
- 搜索相关新闻/事件证据（Tavily / SerpAPI）
- LLM 基于行情 + 证据做归因分析，使用谨慎措辞
- 明确区分：观测数据 / 可能原因 / 证据来源 / 风险说明

### 3. 金融知识问答（RAG）

- **三级检索降级**：ChromaDB 向量 + BM25 混合检索 → 纯 BM25 关键词检索 → Web 搜索兜底
- Embedding 不可用时（如 OpenRouter 不支持 embedding API）自动降级为纯 BM25 模式，确保知识库始终可查
- 文档分块（500 字符/块，80 字符重叠，按 Markdown 层级分割）
- **混合检索**：BM25 关键词 + 向量语义 → RRF（Reciprocal Rank Fusion）融合 → LLM Rerank
- 10 篇知识文档（41KB+），覆盖估值指标、财报分析、技术分析、宏观经济、风险管理、衍生品、公司治理、行业分析、中国市场
- 事实核查：LLM 自检回答中的数值是否与参考资料一致，不通过则重试

### 4. 多资产对比

- 自动提取问题中的多个 Ticker（如"苹果和微软谁更好"）
- 计算量化对比指标：7 日/30 日收益率、波动率、**夏普比率**、**最大回撤**
- 自动判定各维度优胜者
- LLM 基于量化数据生成结构化对比分析

### 5. 多轮对话 & Session

- 保留最近 6 条消息作为上下文
- **Session 状态管理**（30 分钟 TTL）：记忆当前资产和意图
- Ticker 指代回溯（如先问"阿里巴巴股价"，再问"它为什么跌"能自动关联）
- 查询改写：模糊问题自动改写为明确查询（LLM 辅助）

### 5. 流式输出与思维链

- SSE 实时推送：thought → token → chart → meta → done
- 思维链可视化：展示完整的 Agent 决策过程
- 前端状态机管理：idle → sending → thinking → idle/error

---

## 技术选型

| 层级 | 技术 | 选型理由 |
|------|------|----------|
| **前端框架** | Next.js 14 + TypeScript | App Router、SSR/CSR 灵活、类型安全 |
| **UI 样式** | Tailwind CSS | 原子化 CSS、快速开发、深色主题 |
| **图表** | Recharts | React 原生、轻量、AreaChart 适合金融走势 |
| **状态管理** | useReducer 状态机 | 严格状态转换、防止并发不一致 |
| **后端框架** | FastAPI (Python) | 异步高性能、自动 OpenAPI 文档、SSE 原生支持 |
| **行情数据** | yfinance + Finnhub + Alpha Vantage + Stooq | 4 源级联、免费、覆盖面广 |
| **向量数据库** | ChromaDB | 本地嵌入、零基础设施依赖 |
| **Embedding** | text-embedding-3-small | 高质量、低成本、支持 OpenRouter 代理 |
| **LLM** | GPT-4o-mini / DeepSeek | 性价比高、支持 OpenRouter 统一接入 |
| **Web 搜索** | Tavily / SerpAPI | RAG 兜底 + 事件分析证据来源 |
| **部署** | Docker Compose | 前后端一键编排、含健康检查 |

---

## 路由设计

### Query Orchestrator — 三阶段架构

系统采用自研的 **Query Orchestrator** 框架，将每次请求拆为三个阶段：

| 阶段 | 模块 | 职责 |
|------|------|------|
| **Phase 1: Query Understanding** | `agent.py` | Session 消解 → 查询改写 → Ticker 提取 → 双层意图分类 |
| **Phase 2: Plan Generation** | `orchestrator.py` | 意图 → `PLAN_TEMPLATE` 步骤序列（`build_plan()`） |
| **Phase 3: Step Execution** | `steps.py` | 22 个 `@register_step` 步骤函数，共享 `StepContext`，支持 sync/streaming 双模式 |

> **设计动机**：将隐式的代码流程（函数调用顺序 = 执行逻辑）转为显式的步骤序列模板，使流程可配置、可测试、可复用。streaming 执行引擎自动将 `assemble_*` 步骤重排到 `generate_answer` 之前，确保 meta/chart 事件先于 token 流发送。

系统采用 **5 类意图路由**，通过 Query Orchestrator 统一编排为步骤序列：

```
用户问题
    │
    ├─ Session 消解 + 查询改写
    │
    ├─ Ticker 识别（映射表 + 历史回溯）
    │
    ├─ 意图分类（正则优先 → LLM Function Call fallback）
    │
    ├─ orchestrator.build_plan(intent, assets)
    │     意图 → PLAN_TEMPLATE → 步骤序列
    │
    ├─→ market_data（6 步）
    │     fetch_price → validate_data → build_market_prompt →
    │     generate_answer → validate_response_numbers → assemble_market_response
    │
    ├─→ market_reasoning（8 步）
    │     extract_date → fetch_price → validate_data → search_news →
    │     classify_evidence → build_reasoning_prompt → generate_answer →
    │     assemble_reasoning_response
    │
    ├─→ knowledge_rag（6 步）
    │     rag_search → web_search_fallback → build_knowledge_prompt →
    │     generate_answer → fact_check → assemble_knowledge_response
    │
    ├─→ compare（5 步）
    │     fetch_prices_multi → compute_comparison → build_compare_prompt →
    │     generate_answer → assemble_compare_response
    │
    └─→ general（3 步）
          build_general_prompt → generate_answer → assemble_general_response
```

### 意图分类策略

**第一层：正则快速分类（零成本，零延迟）**

| 优先级 | 类别 | 匹配关键词 |
|--------|------|-----------|
| 1 | `compare` | 对比、比较、vs、哪个更好（需 2+ ticker） |
| 2 | `market_reasoning` | 为什么、为何、原因、大涨.*原因、why、reason |
| 3 | `market_data` | 股价、走势、涨跌、多少钱、price、stock |
| 4 | `knowledge_rag` | 什么是、解释、定义、what is、explain |

> compare 优先级最高但需要 2+ ticker；reasoning 优先于 market_data。

**第二层：LLM Fallback 分类**

正则未命中时调用 LLM（temperature=0, max_tokens=10），返回结果由 `_normalize_intent()` 归一化到 4 类，兼容旧标签。

### 结构化回答

所有路由返回统一的 `structured_response`，由后端程序组装（非 LLM 生成），保证准确性：

```json
{
  "response_type": "market_data",
  "data_summary": {
    "price": 178.25,
    "change_7d": 3.52,
    "high_7d": 182.0,
    "low_7d": 170.1,
    "data_source": "yahoo",
    "timestamp": "2025-01-15T10:30:00"
  },
  "trend_summary": {
    "label": "uptrend",
    "label_cn": "上涨",
    "confidence": "high",
    "rationale": "7日涨幅 +3.52%，价格斜率向上"
  },
  "sources": [{"title": "行情数据", "source": "yahoo"}],
  "disclaimer": "以上数据来自第三方行情接口...不构成投资建议。"
}
```

---

## Prompt 设计思路

### 1. 行情数据 Prompt（market_data）

采用**强制三段结构**：

```
📊 客观数据 — 仅列出 API 返回的真实数字，标注来源和时间
📈 趋势总结 — 明确给出上涨/下跌/震荡判断及依据
💡 补充说明 — 缺失数据明确告知，不编造
```

核心约束：`所有数字必须来自提供的 market_data，不得自行生成价格数据`

### 2. 原因分析 Prompt（market_reasoning）

采用**强制四段结构 + 证据约束**：

```
📉 市场表现观察 — 实际涨跌数据（来自 API）
🔍 可能原因分析 — 基于新闻/事件归因，谨慎措辞
📑 证据来源     — 列出引用的新闻标题和出处
⚠️ 风险说明     — 不确定性声明 + 不构成投资建议
```

核心约束：`不臆造、区分事实与分析、证据不足时明确说明`

### 3. 知识问答 Prompt（knowledge_rag）

采用**四段渐进结构**：

```
📌 核心定义/结论 — 直接回答
📖 展开解释     — 分点阐述
💡 示例（如适用）— 具体例子辅助理解
📑 来源说明     — 标注信息来源
```

核心约束：`优先使用参考资料，无检索结果时标注"基于模型通用知识"`

### 4. 事实核查 Prompt

独立的核查 Agent，输出 JSON `{passed, issues}`，核查 LLM 回答中的数值是否与参考资料一致。不通过则注入反馈重试一次。

---

## 准确性控制 / Hallucination Mitigation

系统实现了**三层准确性防线**，整合在独立的 `grounding.py` 模块中：

```
┌─────────────────────────────────────────────────────────────┐
│ 第一层：数据源隔离                                          │
│                                                             │
│  market_data     → 数据只来自行情 API，LLM 只解释不编数据   │
│  market_reasoning→ 行情 API + Web 搜索证据，LLM 做归因     │
│  knowledge_rag   → RAG 向量检索 + Web 搜索，LLM 做整合     │
│  general         → LLM 通用知识（不涉及实时数据）           │
├─────────────────────────────────────────────────────────────┤
│ 第二层：检索过滤                                            │
│                                                             │
│  • RAG 相似度阈值 ≥ 0.3（低于丢弃）                        │
│  • 行情数据 NaN/Inf/负值拦截（market.py 数据清洗）          │
│  • 价格缓存 TTL 60s / 历史缓存 1h                           │
│  • 降级缓存显式标注 stale + age                             │
│  • 空结果拦截：API 无数据时直接返回提示，不调用 LLM          │
├─────────────────────────────────────────────────────────────┤
│ 第三层：输出约束                                            │
│                                                             │
│  • Prompt 约束："所有数字必须来自 market_data"               │
│  • 事实核查：LLM 自检回答 vs RAG 来源（不通过则重试）       │
│  • 数字交叉验证：回答中的数字 vs market_data（无需额外LLM）  │
│  • 缺失字段显式标注：注入"无法获取"提示到 Prompt            │
│  • structured_response 的 data_summary 由程序组装            │
│  • 原因分析强制使用谨慎措辞："可能原因""或与…有关"         │
└─────────────────────────────────────────────────────────────┘
```

### 趋势分类规则（trend.py）

趋势判断使用**纯规则模块**，不依赖 LLM：

| 指标 | 来源 | 作用 |
|------|------|------|
| 涨跌幅 | API 计算 | 主判据：> +2% 上涨，< -2% 下跌 |
| 振幅 | (high-low)/low | 窄幅涨跌 + 高振幅 → 震荡 |
| 斜率 | 最小二乘法 | ±2% 内用斜率区分微趋势 |

输出带 `confidence` 分级（high / medium / low），前端据此展示不同样式。

---

## 数据来源

| 数据类型 | 来源 | 刷新策略 | 备注 |
|----------|------|----------|------|
| 实时股价 | yfinance → Finnhub → Alpha Vantage → Stooq | 缓存 60s | 4 源级联，任一成功即返回 |
| 历史行情 | yfinance → Stooq | 缓存 1h | 7 日 / 30 日窗口 |
| 金融知识 | 本地 .md/.txt/.pdf → ChromaDB | 启动时加载 | 10 篇文档，41KB+，涵盖估值/财报/技术分析/宏观/风险/衍生品/行业/中国市场 |
| 新闻证据 | Tavily / SerpAPI | 实时搜索 | 用于原因分析链路 |
| LLM | OpenAI / OpenRouter | 按需调用 | 支持 GPT-4o-mini / DeepSeek |

---

## 项目结构

```
fin-ai-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 入口 + 生命周期
│   │   ├── routers/
│   │   │   └── chat.py              # API 路由（chat + stream）
│   │   ├── services/
│   │   │   ├── agent.py             # Query Understanding：意图分类 + Session 消解
│   │   │   ├── orchestrator.py      # Plan Generation + Execution Engine
│   │   │   ├── steps.py             # 22 个步骤函数（@register_step 注册）
│   │   │   ├── market.py            # 行情服务：4源级联 + 缓存 + 重试 + 清洗
│   │   │   ├── rag.py               # RAG 检索：ChromaDB + BM25 混合 + LLM Rerank
│   │   │   ├── llm.py               # LLM 封装：流式/非流式 + 意图分类
│   │   │   ├── compare.py           # 多资产对比：Sharpe/MaxDrawdown 计算
│   │   │   ├── web_search.py        # Web 搜索：Tavily / SerpAPI
│   │   │   ├── news_classifier.py   # 证据分类：关键词驱动的归因分类
│   │   │   ├── grounding.py         # 准确性控制层：事实核查 + 数据校验
│   │   │   ├── trend.py             # 趋势分析：规则化分类（涨跌幅+振幅+斜率）
│   │   │   ├── session.py           # Session 状态管理（30分钟 TTL）
│   │   │   └── metrics.py           # 监控指标收集器
│   │   ├── prompts/
│   │   │   └── templates.py         # Prompt 模板（行情/原因分析/RAG/核查/分类）
│   │   ├── models/
│   │   │   └── schemas.py           # Pydantic 模型 + StructuredResponse
│   │   └── utils/
│   │       ├── ticker_map.py        # 公司名/别名 → Ticker 映射（40+）
│   │       ├── date_extract.py      # 日期提取 + 搜索 query 构造
│   │       └── sanitize.py          # 输入安全：长度限制 + 注入检测
│   ├── data/                        # 金融知识库文档（.md / .txt）
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/page.tsx             # 主页面：Header + Messages + InputBar
│   │   ├── components/
│   │   │   ├── MessageBubble.tsx     # 消息气泡：Markdown + 结构化卡片
│   │   │   ├── FinancialChart.tsx    # 走势图：Recharts AreaChart
│   │   │   ├── KPICards.tsx          # KPI 卡片：价格/涨跌/PE/市值
│   │   │   ├── DataFreshness.tsx    # 数据更新时间标注
│   │   │   ├── TrendBadge.tsx        # 趋势标签：方向 + 置信度 + 依据
│   │   │   ├── ThoughtChain.tsx      # 思维链：可折叠推理步骤
│   │   │   ├── SourceList.tsx        # 来源列表：页码徽章 + 相关度指示器
│   │   │   ├── DisclaimerBar.tsx     # 风险声明条
│   │   │   ├── ExampleQuestions.tsx   # 示例问题卡片
│   │   │   └── InputBar.tsx          # 输入框
│   │   ├── lib/
│   │   │   ├── api.ts               # API 客户端 + SSE 解析 + 超时保护
│   │   │   ├── useChatHistory.ts     # 状态机 Hook + localStorage 持久化
│   │   │   ├── useServerStatus.ts    # 服务健康检查（15s 轮询）
│   │   │   └── useColorScheme.ts     # 涨跌配色切换（中国/美国）
│   │   └── types/
│   │       └── chat.ts              # TypeScript 类型（含 StructuredResponse）
│   └── package.json
├── .env.example                     # 环境变量模板
├── docker-compose.yml               # Docker 编排
└── README.md
```

---

## 本地运行

### 前置条件

- Python 3.10+
- Node.js 18+
- OpenAI API Key（或 OpenRouter Key）

### 1. 克隆 & 配置

```bash
git clone <repo-url>
cd fin-ai-agent
cp .env.example .env
# 编辑 .env，填入 API Key
```

`.env` 关键配置：

```bash
# LLM（二选一）
# 方案 A：OpenAI 直连
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# 方案 B：OpenRouter（可用 DeepSeek 等模型）
OPENAI_API_KEY=sk-or-v1-xxx
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=deepseek/deepseek-chat-v3-0324
EMBEDDING_MODEL=openai/text-embedding-3-small

# 行情数据（可选，不填仅用 yfinance）
FINNHUB_API_KEY=
ALPHA_VANTAGE_API_KEY=

# Web 搜索（可选，用于原因分析和 RAG 兜底）
SEARCH_API_KEY=
SEARCH_PROVIDER=tavily
```

### 2. 启动后端

```bash
cd backend
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> 首次启动会自动构建 ChromaDB 向量索引（需要 `data/` 目录有文档）。

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
# 打开 http://localhost:3000
```

## Docker 部署

```bash
docker compose up --build
# 前端：http://localhost:3000
# 后端：http://localhost:8000
# 健康检查：http://localhost:8000/health
```

---

## 优化与扩展思考

### 已实现

- [x] **Query Orchestrator 架构**：意图 → 步骤序列模板 → 统一执行引擎（sync + streaming 双模式）
- [x] **22 个注册步骤函数**：通过 `@register_step` 装饰器，可独立测试、动态组合
- [x] SSE 流式输出（打字机效果 + 思维链实时更新）
- [x] 4 源级联行情（yfinance → Finnhub → Alpha Vantage → Stooq）
- [x] 多轮对话（6 条上下文 + Ticker 指代回溯 + Session 状态管理）
- [x] 查询改写：模糊/指代性问题自动改写为明确查询
- [x] 三层准确性控制（数据源隔离 + 检索过滤 + 输出约束）
- [x] 结构化回答（data_summary / trend_summary 程序组装，非 LLM 生成）
- [x] 多资产对比分析 + **夏普比率** + **最大回撤**
- [x] RAG 混合检索（BM25 + 向量 + RRF 融合 + LLM Rerank）
- [x] **知识库 10 篇文档**（41KB+）：估值指标、财报分析、技术分析、宏观经济、风险管理、衍生品、公司治理、行业分析、中国市场
- [x] 引用溯源：RAG 来源附带 **页码** 和 **相关度评分**
- [x] 新闻证据分类：关键词驱动的自动归因（财报/政策/宏观/行业/公司）
- [x] Docker 一键部署

### 可继续优化

1. **实时财报接入**：对接 SEC EDGAR / 交易所公告 API，自动解析最新财报
2. **缓存升级**：生产环境替换为 Redis，支持分布式和 TTL 精细管理
3. **监控增强**：对接 Prometheus + Grafana，基于审计日志的延迟/错误率看板
4. **日期窗口定位**：对特定日期请求，从历史行情中精确提取该日前后数据
5. **多语言**：前端 i18n + Prompt 自适应语言
6. **更多资产类型**：加密货币、外汇、商品期货

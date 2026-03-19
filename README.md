# FinAI — 智能金融资产问答系统

基于大模型的全栈金融问答系统，支持 **实时行情分析** 和 **金融知识问答（RAG）**。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
│  ┌───────────┐  ┌──────────┐  ┌───────────────────────┐ │
│  │ ChatWindow│  │ InputBar │  │  FinancialChart       │ │
│  │ (Markdown)│  │          │  │  (Recharts AreaChart) │ │
│  └─────┬─────┘  └────┬─────┘  └───────────────────────┘ │
│        │              │    useChatHistory (状态机)        │
│        └──────┬───────┘    localStorage 持久化            │
└───────────────┼─────────────────────────────────────────┘
                │ POST /api/chat
                ▼
┌─────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                      │
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │              Agent Router                    │        │
│  │  Ticker识别 → 意图分类 → Tool分发 → 核查    │        │
│  │         (ThoughtStep 思考链记录)             │        │
│  └────────┬──────────────────┬──────────────────┘       │
│           │                  │                          │
│    ┌──────▼──────┐    ┌──────▼──────┐                   │
│    │ Market Tool │    │  RAG Tool   │                   │
│    │             │    │             │                   │
│    │ yfinance    │    │ ChromaDB    │                   │
│    │ + Cache 60s │    │ + Embedding │                   │
│    │ + Retry     │    │ + FactCheck │                   │
│    │ + Validate  │    │             │                   │
│    └──────┬──────┘    └──────┬──────┘                   │
│           │                  │                          │
│    ┌──────▼──────────────────▼──────┐                   │
│    │         LLM (OpenAI API)       │                   │
│    │   结构化 Prompt → 分区回答      │                   │
│    └────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## 技术选型

| 层级 | 技术 | 选型理由 |
|------|------|----------|
| 前端框架 | Next.js 14 + TypeScript | App Router、SSR/CSR 灵活、类型安全 |
| UI 样式 | Tailwind CSS | 原子化 CSS、快速开发、一致性好 |
| 图表 | Recharts | React 原生、轻量、AreaChart 适合金融数据 |
| 状态管理 | useReducer 状态机 | 严格的状态转换、无中间态不一致 |
| 后端框架 | FastAPI (Python) | 异步高性能、自动 OpenAPI 文档 |
| 行情数据 | yfinance | 免费、覆盖面广、Python 原生 |
| 向量数据库 | ChromaDB | 本地嵌入、零基础设施依赖 |
| Embedding | OpenAI text-embedding-3-small | 高质量、低成本 |
| LLM | OpenAI GPT-4o-mini | 性价比高、响应快 |

## 快速开始

### 前置条件

- Python 3.10+
- Node.js 18+
- OpenAI API Key

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/fin-ai-agent.git
cd fin-ai-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key
```

### 3. 启动后端

```bash
cd backend
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt

# 构建知识库向量索引
python -m scripts.ingest

# 启动 API 服务
uvicorn app.main:app --reload --port 8000
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
# 浏览器打开 http://localhost:3000
```

## Prompt 设计思路

### 1. 行情分析 Prompt

采用**强制分区**策略，要求 LLM 将回答分为：
- **客观数据**：仅包含 API 返回的真实数字，禁止编造
- **分析性描述**：基于数据的趋势总结，使用"可能""或与…有关"等不确定措辞

关键约束：`所有数字必须来自提供的 market_data，不得自行生成价格数据`

### 2. RAG 知识问答 Prompt

- 明确指示 LLM 优先使用检索到的参考资料
- 无检索结果时，强制标注"本回答基于模型通用知识"
- 后置**事实核查 Agent**：LLM 自检回答中的数值是否与上下文一致

### 3. 意图分类 Prompt

极简 few-shot：只要求返回 `market` / `knowledge` / `general` 一个词，
配合 Ticker 映射表做双重路由，避免单点依赖 LLM 分类。

## 数据来源

| 数据类型 | 来源 | 刷新策略 |
|----------|------|----------|
| 实时股价 | Yahoo Finance (yfinance) | 缓存 60s |
| 历史行情 | Yahoo Finance (yfinance) | 缓存 1h |
| 金融知识 | 本地 Markdown 知识库 → ChromaDB | 启动时加载 |

## 系统可靠性设计

- **重试机制**：yfinance 调用失败时指数退避重试（3 次，1s→2s→4s）
- **数据清洗**：拦截 NaN/Inf/负价格，向 LLM 发送明确的"数据不可用"信号
- **缓存管理**：MarketDataCache 减少 API 调用，日志记录 Hit/Miss 用于调优
- **事实核查**：RAG 链路末尾 LLM 自检，不通过则注入反馈重试一次
- **审计日志**：每次请求记录 Input/Output/耗时/思考链，支持生产监控
- **前端状态机**：useReducer 管理 idle→sending→thinking→success/error 严格转换
- **持久化**：localStorage 保存对话历史，刷新不丢失

## 优化与扩展思考

1. **流式响应**：当前为同步请求，可改为 SSE 实现打字机效果，改善体感
2. **多数据源**：接入 Alpha Vantage / Finnhub 作为 yfinance 的 fallback
3. **知识库扩展**：支持 PDF 财报解析，自动从 SEC EDGAR 拉取季度财报
4. **多轮对话**：引入对话上下文记忆，支持"它最近的财报呢？"等指代消解
5. **缓存升级**：生产环境可替换为 Redis，支持分布式部署
6. **监控告警**：对接 Prometheus + Grafana，基于审计日志实现慢请求告警

## 项目结构

```
fin-ai-agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── routers/chat.py      # POST /api/chat
│   │   ├── services/
│   │   │   ├── agent.py         # Agent 路由 + 思考链 + 审计日志
│   │   │   ├── market.py        # 行情服务 + 缓存 + 重试 + 清洗
│   │   │   ├── rag.py           # RAG 检索服务
│   │   │   └── llm.py           # LLM 调用封装
│   │   ├── prompts/templates.py # Prompt 模板
│   │   ├── models/schemas.py    # Pydantic 数据模型
│   │   └── utils/ticker_map.py  # 公司名 → Ticker 映射
│   ├── data/                    # 金融知识库文档
│   ├── scripts/ingest.py        # 知识库构建脚本
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/page.tsx         # 主页面
│   │   ├── components/          # UI 组件
│   │   ├── lib/
│   │   │   ├── api.ts           # API 客户端 + 拦截器
│   │   │   └── useChatHistory.ts # 状态机 Hook + 持久化
│   │   └── types/chat.ts        # 类型定义
│   └── package.json
├── .env.example
├── docker-compose.yml
└── README.md
```

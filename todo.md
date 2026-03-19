# 金融资产问答系统 — 开发计划

## 阶段一：项目初始化

- [ ] 确定技术选型（前端 Next.js + 后端 FastAPI + LLM Claude/OpenAI）
- [ ] 初始化前端项目（Next.js + TypeScript + Tailwind CSS）
- [ ] 初始化后端项目（FastAPI + Python）
- [ ] 配置 .gitignore、环境变量模板 (.env.example)
- [ ] 搭建 monorepo 目录结构
- [ ] 验证前后端可独立启动

## 阶段二：后端行情模块

- [ ] 集成 Yahoo Finance API（yfinance）获取实时/近期股票数据
- [ ] 实现股票价格查询接口（当前价、历史价格）
- [ ] 实现涨跌幅计算逻辑（7 日、30 日）
- [ ] 实现趋势结构化总结（上涨 / 下跌 / 震荡判定）
- [ ] 支持股票名称（中文/英文）到 ticker 的映射
- [ ] 编写行情模块单元测试

## 阶段三：RAG 知识库模块

- [ ] 准备金融知识文档语料（市盈率、财报术语、基础概念等）
- [ ] 实现文档分块（chunking）逻辑
- [ ] 集成向量化模型（OpenAI Embeddings / 本地模型）
- [ ] 搭建向量数据库（ChromaDB / FAISS）
- [ ] 实现向量检索接口（相似度查询 top-k）
- [ ] 集成 Web Search 作为补充检索源
- [ ] 编写 RAG 模块测试

## 阶段四：Agent 路由与 LLM 集成

- [ ] 设计查询意图分类逻辑（行情类 vs 知识类）
- [ ] 实现 Agent 路由框架（根据意图分发到对应模块）
- [ ] 集成 LLM API（Claude / OpenAI）
- [ ] 设计 Prompt 模板（行情分析 prompt、RAG 问答 prompt）
- [ ] 实现结构化回答生成（区分客观数据与分析性描述）
- [ ] 实现流式响应（SSE / WebSocket）
- [ ] 添加 hallucination 控制策略（引用数据源、限制自由发挥）
- [ ] 端到端集成测试

## 阶段五：前端 UI

- [ ] 设计聊天界面布局（对话框 + 输入区）
- [ ] 实现消息列表组件（支持 Markdown 渲染）
- [ ] 实现用户输入与发送逻辑
- [ ] 对接后端 API（流式展示回答）
- [ ] 实现数据可视化组件（股价走势图表，使用 Recharts / ECharts）
- [ ] 添加示例问题快捷入口
- [ ] 响应式适配与交互优化
- [ ] 前端组件测试

## 阶段六：部署与文档

- [ ] 编写 Dockerfile（前端 + 后端）
- [ ] 编写 docker-compose.yml 一键启动
- [ ] 编写 README.md（架构图、技术选型、Prompt 设计、数据来源、优化思考）
- [ ] 绘制系统架构图
- [ ] 录制 3 分钟演示视频
- [ ] 最终代码审查与清理

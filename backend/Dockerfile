FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（jieba 分词需要）
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 知识库在 FastAPI lifespan 中自动加载，启动失败不阻断服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

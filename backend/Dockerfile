FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 知识库构建在启动时执行（需要 OPENAI_API_KEY）
CMD ["sh", "-c", "python -m scripts.ingest && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

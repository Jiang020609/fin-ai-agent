"""知识库构建脚本 — 读取 data/ 目录下的文档，向量化后写入 ChromaDB。

Usage:
    cd backend
    python -m scripts.ingest          # 增量构建（已有数据则跳过）
    python -m scripts.ingest --force  # 强制重建
"""

import logging
import sys

sys.path.insert(0, ".")

from app.services.rag import build_vectorstore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    force = "--force" in sys.argv
    if force:
        print("Force rebuilding vector store...")
    else:
        print("Building vector store (skip if exists)...")

    vs = build_vectorstore(force_rebuild=force)
    count = vs._collection.count()
    print(f"Done. Vector store contains {count} chunks.")


if __name__ == "__main__":
    main()

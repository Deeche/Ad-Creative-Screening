FROM python:3.9-slim

WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY backend/ backend/
COPY app.py .

# テスト実行用のコマンド
CMD ["python", "-m", "unittest", "backend/test_rag_system.py", "-v"] 
FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow + faiss
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/index/ ./data/index/

ENV INDEX_DIR=data/index
ENV MODEL_NAME=ViT-B-32

EXPOSE 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

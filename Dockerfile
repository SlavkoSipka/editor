FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    wget \
    ca-certificates \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG QDRANT_VERSION=1.11.3
RUN wget -q "https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" -O /tmp/qdrant.tar.gz \
    && tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin \
    && rm /tmp/qdrant.tar.gz \
    && chmod +x /usr/local/bin/qdrant

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY process.py .

# Optional local bake: uncomment if your build context has data/ (not on clean GitHub CI).
# COPY data/library/ /app/bootstrap/library/
# COPY data/embeddings/ /app/bootstrap/embeddings/
RUN mkdir -p /app/bootstrap/library /app/bootstrap/embeddings

COPY tests/fixtures/ ./tests/fixtures/

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

CMD ["/usr/local/bin/docker-entrypoint.sh"]

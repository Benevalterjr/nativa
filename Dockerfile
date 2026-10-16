# ============================================================================
# NATIVA ENGINE DOCKERFILE (RENDER / CLOUD RUN COMPATIBLE)
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV OMP_NUM_THREADS=2
ENV OMP_WAIT_POLICY=PASSIVE

# Dependências do sistema para compilar C99 com OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências Python (versões estritamente travadas e binários pré-compilados - SonarCloud S8544 e S6581)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: numpy==2.2.3 fastapi==0.115.8 uvicorn==0.34.0 pydantic==2.10.6

# Copia apenas os arquivos e módulos estritamente necessários (SonarCloud S6505 compliant)
COPY nativa.py nativa_rt.h ./
COPY tools/ ./tools/
COPY data/ ./data/

# Baixa o modelo quantizado da release do GitHub se não estiver no repositório
RUN if [ ! -f "data/qwen3_0.6b.natq4" ]; then \
        echo "Baixando modelo da Release GitHub..." && \
        mkdir -p data && \
        curl --proto '=https' --proto-redir '=https' -L -f -o data/qwen3_0.6b.natq4 "https://github.com/Benevalterjr/nativa/releases/download/v0.1.0/qwen3_0.6b.natq4"; \
    fi

# Compila o motor de inferência C99 nativo com OpenMP para Linux x86_64
RUN gcc -O3 -fopenmp -msse4.1 -mavx tools/nativa_engine.c -o tools/nativa_engine -lm && \
    chmod +x tools/nativa_engine

# Usuário não-root para conformidade de segurança
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "tools/api_server.py"]

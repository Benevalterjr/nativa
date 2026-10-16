# ============================================================================
# NATIVA ENGINE DOCKERFILE (RENDER / CLOUD RUN COMPATIBLE)
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV OMP_NUM_THREADS=2
ENV OMP_WAIT_POLICY=PASSIVE

# Dependências do sistema para compilar C99 com OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências Python mínimas para o servidor FastAPI
RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy

# Copia o código fonte do repositório
COPY . .

# Baixa o modelo quantizado da release do GitHub se não estiver no repositório
RUN if [ ! -f "data/qwen3_0.6b.natq4" ]; then \
        echo "Baixando modelo da Release GitHub..." && \
        mkdir -p data && \
        curl -L -f -o data/qwen3_0.6b.natq4 "https://github.com/Benevalterjr/nativa/releases/download/v0.1.0/qwen3_0.6b.natq4"; \
    fi

# Compila o motor de inferência C99 nativo com OpenMP para Linux x86_64
RUN gcc -O3 -fopenmp -msse4.1 -mavx tools/nativa_engine.c -o tools/nativa_engine -lm

# Permissão de execução no binário compilado
RUN chmod +x tools/nativa_engine

EXPOSE 8000

CMD ["python", "tools/api_server.py"]

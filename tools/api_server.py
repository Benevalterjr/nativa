#!/usr/bin/env python3
"""
Nativa API Server - Servidor HTTP REST compatível com OpenAI, RAG e Branch.dev
Executa o motor C nativo (nativa_engine) em background com latência mínima e zero GPU.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import uuid

# Constantes para evitar duplicação de literais (SonarQube compliance)
TAG_THINK = "<think>"
TAG_THINK_END = "</think>"
MARKER_TURN_DONE = "[TURN_DONE"
MSG_ENGINE_NOT_READY = "Motor não inicializado."
STATUS_503_RESPONSES = {503: {"description": MSG_ENGINE_NOT_READY}}

# Adiciona a raiz do projeto ao sys.path para importar nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import nativa

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("[ERRO] FastAPI e Uvicorn são necessários: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Caminhos e configurações padrão
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4"))
DEFAULT_TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json"))
IS_WINDOWS = sys.platform.startswith("win")
ENGINE_EXEC = "nativa_engine.exe" if IS_WINDOWS else "nativa_engine"
DEFAULT_ENGINE_PATH = os.environ.get("ENGINE_PATH", os.path.join(ROOT_DIR, "tools", ENGINE_EXEC))


def extract_thinking(text: str) -> Tuple[Optional[str], str]:
    """Extrai conteúdo delimitado por <think>...</think> de forma padronizada."""
    if TAG_THINK in text and TAG_THINK_END in text:
        parts = text.split(TAG_THINK_END, 1)
        reasoning = parts[0].replace(TAG_THINK, "").strip()
        answer = parts[1].strip()
        return reasoning, answer
    return None, text


def write_vocab_file(tokenizer, path: str) -> None:
    """Exporta o vocabulário para o formato binário lido pelo motor C."""
    vocab_size = max(tokenizer.inv_vocab, default=-1) + 1
    with open(path, "wb") as f:
        f.write(struct.pack("<I", vocab_size))
        for token_id in range(vocab_size):
            piece = tokenizer.inv_vocab.get(token_id, "")
            if piece.startswith("<0x") and len(piece) == 6 and piece.endswith(">"):
                try:
                    token = bytes([int(piece[3:5], 16)])
                except ValueError:
                    token = piece.encode("utf-8", errors="replace")
            elif piece and hasattr(tokenizer, "byte_decoder") and all(c in tokenizer.byte_decoder for c in piece):
                token = bytes(tokenizer.byte_decoder[c] for c in piece)
            else:
                token = piece.encode("utf-8", errors="replace")
            if len(token) > 65535:
                raise ValueError(f"Token {token_id} excede 65535 bytes.")
            f.write(struct.pack("<H", len(token)))
            f.write(token)


def _compute_safe_len(buf: str, printed_idx: int) -> int:
    """Calcula tamanho seguro para corte sem quebrar marcador de turn."""
    safe_len = len(buf)
    if "[" in buf[printed_idx:]:
        last_bracket = buf.rfind("[")
        if last_bracket >= printed_idx and len(buf) - last_bracket < 12:
            safe_len = last_bracket
    return safe_len


class EngineRunner:
    """Gerenciador do subprocesso do motor C com lock para requisições concorrentes."""

    def __init__(self, model_path: str, tokenizer_path: str, engine_path: str):
        self.model_path = os.path.abspath(model_path)
        self.tokenizer_path = os.path.abspath(tokenizer_path)
        self.engine_path = os.path.abspath(engine_path)
        self.proc: Optional[subprocess.Popen] = None
        self.vocab_path: Optional[str] = None
        self.tokenizer: Optional[nativa.Tokenizer] = None
        self.lock = asyncio.Lock()
        self.model_name = os.path.basename(self.model_path).replace(".natq4", "")

    def start(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")
        if not os.path.isfile(self.tokenizer_path):
            raise FileNotFoundError(f"Arquivo do tokenizer não encontrado: {self.tokenizer_path}")
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"Executável do motor C não encontrado: {self.engine_path}")

        print(f"[*] Inicializando Tokenizer: {self.tokenizer_path}")
        self.tokenizer = nativa.Tokenizer(self.tokenizer_path)

        vocab_tmp = tempfile.NamedTemporaryFile(prefix="nativa-vocab-", suffix=".bin", delete=False)
        self.vocab_path = vocab_tmp.name
        vocab_tmp.close()
        write_vocab_file(self.tokenizer, self.vocab_path)

        env = os.environ.copy()
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

        print(f"[*] Subindo Nativa C Engine: {self.engine_path} com {self.model_path}...")
        self.proc = subprocess.Popen(
            [self.engine_path, self.model_path, "--chat", "--vocab", self.vocab_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
            env=env,
        )

        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = self.proc.stderr.read()
                raise RuntimeError(f"Motor C encerrou prematuramente. Stderr: {stderr}")
            if "[NATIVA_CHAT_READY]" in line:
                print(f"[OK] Nativa Engine pronto para inferência! Modelo: {self.model_name}")
                break

    def stop(self):
        if self.proc:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        if self.vocab_path and os.path.isfile(self.vocab_path):
            try:
                os.unlink(self.vocab_path)
            except OSError:
                pass
        print("[*] Nativa Engine finalizado.")

    async def generate(self, prompt: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
        """Gera tokens em streaming a partir do prompt com baixa complexidade cognitiva."""
        async with self.lock:
            tokens = self.tokenizer.encode(prompt).tolist()
            if not tokens:
                return

            cmd = f"INPUT {len(tokens)} {max_tokens} " + " ".join(map(str, tokens)) + "\n"
            self.proc.stdin.write(cmd)
            self.proc.stdin.flush()

            buf = ""
            printed_idx = 0

            while True:
                ch = await asyncio.to_thread(self.proc.stdout.read, 1)
                if not ch:
                    break
                buf += ch

                if MARKER_TURN_DONE in buf and "]" in buf[buf.find(MARKER_TURN_DONE):]:
                    break

                td_pos = buf.find(MARKER_TURN_DONE, printed_idx)
                if td_pos != -1:
                    chunk = buf[printed_idx:td_pos]
                    if chunk:
                        yield chunk
                    printed_idx = td_pos
                    continue

                safe_len = _compute_safe_len(buf, printed_idx)
                if safe_len > printed_idx:
                    chunk = buf[printed_idx:safe_len]
                    yield chunk
                    printed_idx = safe_len


runner: Optional[EngineRunner] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = EngineRunner(
        model_path=DEFAULT_MODEL_PATH,
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        engine_path=DEFAULT_ENGINE_PATH,
    )
    runner.start()
    yield
    runner.stop()


app = FastAPI(
    title="Nativa Qwen3-0.6B INT4 API",
    description="API Serverless de altíssimo desempenho em C99 puro para RAG, Agentes e Branch.dev",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# DTOs (Modelos Pydantic)
# ----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen3-0.6b-natq4"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class RagRequest(BaseModel):
    context: str = Field(..., description="Documentos recuperados para embasamento")
    question: str = Field(..., description="Pergunta do usuário")
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512


class DecisionRequest(BaseModel):
    state: Any = Field(..., description="Objeto ou texto representando o estado atual")
    choices: Dict[str, str] = Field(..., description="Dicionário de escolhas (chave -> descrição)")
    task: str = Field(..., description="Descrição da tarefa ou pergunta de decisão")
    temperature: Optional[float] = 0.0


# ----------------------------------------------------------------------------
# ROTAS
# ----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Nativa Qwen3-0.6B INT4 Engine API",
        "status": "online",
        "architecture": "C99 OpenMP Q4x4",
        "model": runner.model_name if runner else "carregando",
        "docs": "/docs",
    }


@app.get("/health", responses=STATUS_503_RESPONSES)
def health():
    if runner and runner.proc and runner.proc.poll() is None:
        return {"status": "healthy", "engine": "running", "model": runner.model_name}
    raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)


@app.post("/v1/chat/completions", responses=STATUS_503_RESPONSES)
async def chat_completions(req: ChatCompletionRequest):
    """Endpoint compatível com OpenAI Chat Completions API."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    prompt_parts = [f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n" for msg in req.messages]
    prompt_parts.append("<|im_start|>assistant\n")
    full_prompt = "".join(prompt_parts)

    created_ts = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        async def event_generator():
            async for chunk in runner.generate(full_prompt, max_tokens=req.max_tokens or 512):
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            final_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    accumulated = []
    async for chunk in runner.generate(full_prompt, max_tokens=req.max_tokens or 512):
        accumulated.append(chunk)

    full_response = "".join(accumulated).strip()
    reasoning, cleaned_content = extract_thinking(full_response)

    prompt_tokens_len = len(runner.tokenizer.encode(full_prompt).tolist())
    completion_tokens_len = len(runner.tokenizer.encode(full_response).tolist())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": cleaned_content,
                    **({"reasoning_content": reasoning} if reasoning else {}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens_len,
            "completion_tokens": completion_tokens_len,
            "total_tokens": prompt_tokens_len + completion_tokens_len,
        },
    }


@app.post("/v1/rag/answer", responses=STATUS_503_RESPONSES)
async def rag_answer(req: RagRequest):
    """Endpoint otimizado para RAG com groundedness estrito."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    system_prompt = (
        "Você é um assistente estritamente baseado em fatos. "
        "Responda à pergunta do usuário usando EXCLUSIVAMENTE o contexto abaixo. "
        "Se a informação não estiver descrita no contexto, diga apenas: "
        "'Essa informação não consta nos documentos fornecidos.'"
    )

    user_prompt = f"CONTEXTO:\n\"\"\"\n{req.context}\n\"\"\"\n\nPERGUNTA:\n{req.question}"
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    accumulated = []
    async for chunk in runner.generate(prompt, max_tokens=req.max_tokens or 512):
        accumulated.append(chunk)

    raw_text = "".join(accumulated).strip()
    reasoning, answer = extract_thinking(raw_text)

    return {
        "answer": answer,
        "reasoning": reasoning,
        "model": runner.model_name,
    }


@app.post("/v1/decide", responses=STATUS_503_RESPONSES)
async def branch_decide(req: DecisionRequest):
    """Endpoint direto para tomada de decisão probabilística do Branch.dev."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    choices_text = "\n".join([f"- '{k}': {desc}" for k, desc in req.choices.items()])
    state_str = json.dumps(req.state, ensure_ascii=False) if isinstance(req.state, (dict, list)) else str(req.state)

    system_prompt = (
        "Você é o árbitro de decisões lógicas do Branch.dev (Sistema 2). "
        "Analise o estado fornecido e a tarefa solicitada. "
        "Pense no problema dentro das tags <think> e termine indicando APENAS o nome exato da escolha vencedora."
    )

    user_prompt = (
        f"ESTADO DO SISTEMA:\n{state_str}\n\n"
        f"ESCOLHAS POSSÍVEIS:\n{choices_text}\n\n"
        f"TAREFA:\n{req.task}\n\n"
        f"Decida qual das opções acima é a mais correta:"
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    accumulated = []
    async for chunk in runner.generate(prompt, max_tokens=256):
        accumulated.append(chunk)

    raw_text = "".join(accumulated).strip()
    reasoning, winner_candidate = extract_thinking(raw_text)

    selected_winner = None
    for k in req.choices:
        if k.lower() in winner_candidate.lower():
            selected_winner = k
            break

    if not selected_winner:
        selected_winner = next(iter(req.choices.keys()))

    return {
        "winner": selected_winner,
        "reasoning": reasoning,
        "raw_response": winner_candidate,
        "system": "system_two_nativa",
    }


if __name__ == "__main__":
    import uvicorn
    # Por padrão vincula a 127.0.0.1 em dev local; HOST=0.0.0.0 é injetado pelo Docker/Render
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("tools.api_server:app", host=host, port=port, reload=False)

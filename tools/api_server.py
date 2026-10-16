#!/usr/bin/env python3
"""
Nativa API Server - Servidor HTTP REST compatível com OpenAI, RAG e Branch.dev
Executa o motor C nativo (nativa_engine) em background com latência mínima e zero GPU.
"""

import asyncio
import codecs
from contextlib import asynccontextmanager
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple
import uuid

# Constantes para evitar duplicação de literais (SonarQube compliance)
TAG_THINK = "<think>"
TAG_THINK_END = "</think>"
MARKER_TURN_DONE = "[TURN_DONE"
MSG_ENGINE_NOT_READY = "Motor não inicializado."
STATUS_503_RESPONSES = {503: {"description": MSG_ENGINE_NOT_READY}}
PREFIX_ASSISTANT_THINK = "<|im_start|>assistant\n<think>\n"
PREFIX_ASSISTANT_NO_THINK = "<|im_start|>assistant\n<think>\n</think>\n"
DEFAULT_MODEL_NAME = "qwen3-0.6b-natq4"
DEFAULT_CHAT_EXAMPLE_MSG = "Olá! Explique brevemente o que é o motor Nativa."
DEFAULT_RAG_CONTEXT = "Nativa é uma linguagem e runtime C99 de alta eficiência para inferência de LLMs em CPU."
DEFAULT_RAG_QUESTION = "O que é o Nativa?"
EXAMPLE_API_URL = "https://example.com/api"
CHOICE_RETRY_DESC = "Fazer nova tentativa de requisição"
CHOICE_ABORT_DESC = "Encerrar e registrar falha definitiva"
EXAMPLE_TASK_DECISION = "Qual ação deve ser tomada após 3 tentativas com falha 500?"
OBJECT_CHAT_CHUNK = "chat.completion.chunk"

# Adiciona a raiz do projeto ao sys.path para importar nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import nativa
from tools.nativa_bundle import write_vocab_file

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    print("[ERRO] FastAPI e Uvicorn são necessários: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Caminhos e configurações padrão
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4"))
DEFAULT_TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH", os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json")
)
IS_WINDOWS = sys.platform.startswith("win")
ENGINE_EXEC = "nativa_engine.exe" if IS_WINDOWS else "nativa_engine"
DEFAULT_ENGINE_PATH = os.environ.get("ENGINE_PATH", os.path.join(ROOT_DIR, "tools", ENGINE_EXEC))


def sanitize_chatml(text: str) -> str:
    """Remove marcadores especiais ChatML de entradas externas para evitar prompt injection."""
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")


def extract_thinking(text: str) -> Tuple[Optional[str], str]:
    """Extrai conteúdo delimitado por <think>...</think> (mesmo se <think> vier no prefixo do prompt)."""
    if TAG_THINK_END in text:
        parts = text.split(TAG_THINK_END, 1)
        reasoning = parts[0].replace(TAG_THINK, "").strip()
        answer = parts[1].strip()
        return (reasoning or None), answer
    return None, text.replace(TAG_THINK, "").strip()


def _compute_safe_len(buf: str, printed_idx: int) -> int:
    """Calcula tamanho seguro para corte sem quebrar marcador de turn."""
    safe_len = len(buf)
    if "[" in buf[printed_idx:]:
        last_bracket = buf.rfind("[")
        if last_bracket >= printed_idx and len(buf) - last_bracket < 16:
            safe_len = last_bracket
    return safe_len


def _extract_stream_chunk(buf: str, printed_idx: int) -> Tuple[Optional[str], int, bool]:
    """Extrai o próximo fragmento seguro para streaming e indica se o turno finalizou."""
    td_pos = buf.find(MARKER_TURN_DONE)
    if td_pos != -1:
        chunk = buf[printed_idx:td_pos] if td_pos > printed_idx else None
        is_done = "]" in buf[td_pos:]
        return chunk, td_pos, is_done

    safe_len = _compute_safe_len(buf, printed_idx)
    if safe_len > printed_idx:
        return buf[printed_idx:safe_len], safe_len, False
    return None, printed_idx, False


def _format_sse_chunk(
    completion_id: str,
    created_ts: int,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> str:
    """Formata um chunk SSE compatível com a API OpenAI."""
    data = {
        "id": completion_id,
        "object": OBJECT_CHAT_CHUNK,
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"



def _select_winner(candidate: str, choices: Dict[str, str]) -> str:
    """Seleciona a opção vencedora priorizando casamento exato de palavra do fim para o início."""
    cleaned = candidate.strip()
    keys = list(choices.keys())
    if not keys:
        return ""

    # 1. Casamento exato com a resposta inteira (ignorando pontuação/aspas nas bordas)
    normalized_full = cleaned.strip(" \t\n\r'\"`.,:;!()[]{}").lower()
    for k in keys:
        if k.lower() == normalized_full:
            return k

    # 2. Busca por palavra exata (word boundary), priorizando a última ocorrência no texto
    best_key = None
    best_pos = -1
    for k in keys:
        pattern = rf"(?<![\w-]){re.escape(k.lower())}(?![\w-])"
        for match in re.finditer(pattern, cleaned.lower()):
            if match.start() >= best_pos:
                best_pos = match.start()
                best_key = k
    if best_key is not None:
        return best_key

    # 3. Fallback por substring (última ocorrência)
    for k in keys:
        pos = cleaned.lower().rfind(k.lower())
        if pos > best_pos:
            best_pos = pos
            best_key = k

    return best_key if best_key is not None else keys[0]


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

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")
        if not os.path.isfile(self.tokenizer_path):
            raise FileNotFoundError(f"Arquivo do tokenizer não encontrado: {self.tokenizer_path}")
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"Executável do motor C não encontrado: {self.engine_path}")

        if self.tokenizer is None:
            print(f"[*] Inicializando Tokenizer: {self.tokenizer_path}")
            self.tokenizer = nativa.Tokenizer(self.tokenizer_path)

        if not self.vocab_path or not os.path.isfile(self.vocab_path):
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
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"Motor C encerrou prematuramente. Stderr: {stderr}")
            if "[NATIVA_CHAT_READY]" in line:
                print(f"[OK] Nativa Engine pronto para inferência! Modelo: {self.model_name}")
                break

    def stop(self):
        if self.proc:
            try:
                if self.proc.poll() is None and self.proc.stdin:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
            finally:
                self.proc = None
        if self.vocab_path and os.path.isfile(self.vocab_path):
            try:
                os.unlink(self.vocab_path)
            except OSError:
                pass
            self.vocab_path = None
        print("[*] Nativa Engine finalizado.")

    def _write_stdin(self, cmd: str) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError(MSG_ENGINE_NOT_READY)
        self.proc.stdin.write(cmd)
        self.proc.stdin.flush()

    async def generate(self, prompt: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
        """Gera tokens em streaming a partir do prompt sem perda do chunk final."""
        async with self.lock:
            if not self.is_alive():
                await asyncio.to_thread(self.start)

            tokens = self.tokenizer.encode(prompt).tolist()
            if not tokens:
                return

            cmd = f"RESET\nINPUT {len(tokens)} {max_tokens} " + " ".join(map(str, tokens)) + "\n"
            await asyncio.to_thread(self._write_stdin, cmd)

            buf = ""
            printed_idx = 0
            fd = self.proc.stdout.fileno()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            while True:
                raw = await asyncio.to_thread(os.read, fd, 1024)
                if not raw:
                    tail = decoder.decode(b"", final=True)
                    final_chunk = (buf + tail)[printed_idx:]
                    if final_chunk:
                        yield final_chunk
                    break

                buf += decoder.decode(raw, final=False)
                chunk, printed_idx, is_done = _extract_stream_chunk(buf, printed_idx)
                if chunk:
                    yield chunk
                if is_done:
                    break


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
    role: Literal["system", "user", "assistant"] = Field(
        default="user",
        description="Papel do autor da mensagem na conversa.",
        examples=["user"],
    )
    content: str = Field(
        default=DEFAULT_CHAT_EXAMPLE_MSG,
        min_length=1,
        description="Conteúdo textual da mensagem.",
        examples=[DEFAULT_CHAT_EXAMPLE_MSG],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"role": "user", "content": DEFAULT_CHAT_EXAMPLE_MSG},
            "examples": [{"role": "user", "content": DEFAULT_CHAT_EXAMPLE_MSG}],
        }
    )


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=DEFAULT_MODEL_NAME, examples=[DEFAULT_MODEL_NAME])
    messages: List[ChatMessage] = Field(
        default_factory=lambda: [ChatMessage(role="user", content=DEFAULT_CHAT_EXAMPLE_MSG)],
        min_length=1,
        description="Lista de mensagens da conversa no formato ChatML.",
        examples=[[{"role": "user", "content": DEFAULT_CHAT_EXAMPLE_MSG}]],
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    stream: Optional[bool] = Field(default=False)
    enable_thinking: Optional[bool] = Field(default=False)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": DEFAULT_MODEL_NAME,
                "messages": [
                    {"role": "user", "content": DEFAULT_CHAT_EXAMPLE_MSG}
                ],
                "temperature": 0.7,
                "max_tokens": 256,
                "stream": False,
                "enable_thinking": False,
            },
            "examples": [
                {
                    "model": DEFAULT_MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": DEFAULT_CHAT_EXAMPLE_MSG}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "stream": False,
                    "enable_thinking": False,
                }
            ],
        }
    )


class RagRequest(BaseModel):
    context: str = Field(
        default=DEFAULT_RAG_CONTEXT,
        min_length=1,
        description="Documentos recuperados para embasamento",
        examples=[DEFAULT_RAG_CONTEXT],
    )
    question: str = Field(
        default=DEFAULT_RAG_QUESTION,
        min_length=1,
        description="Pergunta do usuário",
        examples=[DEFAULT_RAG_QUESTION],
    )
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    enable_thinking: Optional[bool] = Field(default=False)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context": DEFAULT_RAG_CONTEXT,
                "question": DEFAULT_RAG_QUESTION,
                "temperature": 0.1,
                "max_tokens": 128,
                "enable_thinking": False,
            },
            "examples": [
                {
                    "context": DEFAULT_RAG_CONTEXT,
                    "question": DEFAULT_RAG_QUESTION,
                    "temperature": 0.1,
                    "max_tokens": 128,
                    "enable_thinking": False,
                }
            ],
        }
    )


class DecisionRequest(BaseModel):
    state: Any = Field(
        default={"url": EXAMPLE_API_URL, "status_code": 500, "retries": 3},
        description="Objeto ou texto representando o estado atual",
        examples=[{"url": EXAMPLE_API_URL, "status_code": 500, "retries": 3}],
    )
    choices: Dict[str, str] = Field(
        default={
            "tentar_novamente": CHOICE_RETRY_DESC,
            "abortar": CHOICE_ABORT_DESC,
        },
        min_length=1,
        description="Dicionário de escolhas (chave -> descrição)",
        examples=[{
            "tentar_novamente": CHOICE_RETRY_DESC,
            "abortar": CHOICE_ABORT_DESC,
        }],
    )
    task: str = Field(
        default=EXAMPLE_TASK_DECISION,
        min_length=1,
        description="Descrição da tarefa ou pergunta de decisão",
        examples=[EXAMPLE_TASK_DECISION],
    )
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    enable_thinking: Optional[bool] = Field(default=True)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": {"url": EXAMPLE_API_URL, "status_code": 500, "retries": 3},
                "choices": {
                    "tentar_novamente": CHOICE_RETRY_DESC,
                    "abortar": CHOICE_ABORT_DESC,
                },
                "task": EXAMPLE_TASK_DECISION,
                "temperature": 0.0,
                "enable_thinking": True,
            },
            "examples": [
                {
                    "state": {"url": EXAMPLE_API_URL, "status_code": 500, "retries": 3},
                    "choices": {
                        "tentar_novamente": CHOICE_RETRY_DESC,
                        "abortar": CHOICE_ABORT_DESC,
                    },
                    "task": EXAMPLE_TASK_DECISION,
                    "temperature": 0.0,
                    "enable_thinking": True,
                }
            ],
        }
    )


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
    if runner and runner.is_alive():
        return {"status": "healthy", "engine": "running", "model": runner.model_name}
    raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)


async def _stream_chat_chunks(
    runner_inst: EngineRunner,
    prompt: str,
    max_tokens: int,
    completion_id: str,
    created_ts: int,
    model: str,
    enable_thinking: bool,
):
    is_thinking = True
    buf = ""
    async for chunk in runner_inst.generate(prompt, max_tokens=max_tokens):
        if not is_thinking:
            yield _format_sse_chunk(completion_id, created_ts, model, {"content": chunk})
            continue

        buf += chunk
        if TAG_THINK_END in buf:
            parts = buf.split(TAG_THINK_END, 1)
            is_thinking = False
            thought = parts[0].replace(TAG_THINK, "").strip()
            ans = parts[1].lstrip("\n")
            if enable_thinking and thought:
                yield _format_sse_chunk(completion_id, created_ts, model, {"reasoning_content": thought})
            if ans:
                yield _format_sse_chunk(completion_id, created_ts, model, {"content": ans})
        elif enable_thinking:
            cleaned = chunk.replace(TAG_THINK, "")
            if cleaned:
                yield _format_sse_chunk(completion_id, created_ts, model, {"reasoning_content": cleaned})

    yield _format_sse_chunk(completion_id, created_ts, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", responses=STATUS_503_RESPONSES)
async def chat_completions(req: ChatCompletionRequest):
    """Endpoint compatível com OpenAI Chat Completions API."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    prompt_parts = [
        f"<|im_start|>{msg.role}\n{sanitize_chatml(msg.content)}<|im_end|>\n"
        for msg in req.messages
    ]
    prompt_parts.append(PREFIX_ASSISTANT_THINK)
    full_prompt = "".join(prompt_parts)

    created_ts = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    user_model = req.model or DEFAULT_MODEL_NAME
    engine_tokens = (req.max_tokens or 256) + 150

    if req.stream:
        return StreamingResponse(
            _stream_chat_chunks(
                runner,
                full_prompt,
                engine_tokens,
                completion_id,
                created_ts,
                user_model,
                bool(req.enable_thinking),
            ),
            media_type="text/event-stream",
        )

    accumulated = []
    async for chunk in runner.generate(full_prompt, max_tokens=engine_tokens):
        accumulated.append(chunk)

    full_response = "".join(accumulated).strip()
    reasoning, cleaned_content = extract_thinking(full_response)

    prompt_tokens_len = len(runner.tokenizer.encode(full_prompt).tolist())
    completion_tokens_len = len(runner.tokenizer.encode(full_response).tolist())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": user_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": cleaned_content,
                    **({"reasoning_content": reasoning} if (req.enable_thinking and reasoning) else {}),
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
    """Endpoint otimizado para RAG com resposta direta e fundamentada."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    system_prompt = (
        "Com base no contexto fornecido, responda à pergunta do usuário de forma direta e precisa em português."
    )

    safe_context = sanitize_chatml(req.context)
    safe_question = sanitize_chatml(req.question)
    user_prompt = f"Contexto:\n{safe_context}\n\nPergunta:\n{safe_question}"
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{PREFIX_ASSISTANT_THINK}"
    )

    engine_tokens = (req.max_tokens or 256) + 150
    accumulated = []
    async for chunk in runner.generate(prompt, max_tokens=engine_tokens):
        accumulated.append(chunk)

    raw_text = "".join(accumulated).strip()
    reasoning, answer = extract_thinking(raw_text)

    return {
        "answer": answer,
        "reasoning": reasoning if req.enable_thinking else None,
        "model": runner.model_name,
    }


@app.post("/v1/decide", responses=STATUS_503_RESPONSES)
async def branch_decide(req: DecisionRequest):
    """Endpoint direto para tomada de decisão probabilística do Branch.dev."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    choices_text = "\n".join([f"- '{k}': {desc}" for k, desc in req.choices.items()])
    state_str = (
        json.dumps(req.state, ensure_ascii=False)
        if isinstance(req.state, (dict, list))
        else str(req.state)
    )

    system_prompt = (
        "Você é o árbitro de decisões lógicas do Branch.dev (Sistema 2). "
        "Analise o estado e escolha a opção correta. "
        "Termine indicando o nome exato da escolha vencedora."
    )

    user_prompt = (
        f"Estado:\n{sanitize_chatml(state_str)}\n\n"
        f"Escolhas:\n{sanitize_chatml(choices_text)}\n\n"
        f"Tarefa:\n{sanitize_chatml(req.task)}\n\n"
        f"Qual escolha executar?"
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{PREFIX_ASSISTANT_THINK}"
    )

    engine_tokens = (req.max_tokens or 256) + 150
    accumulated = []
    async for chunk in runner.generate(prompt, max_tokens=engine_tokens):
        accumulated.append(chunk)

    raw_text = "".join(accumulated).strip()
    reasoning, winner_candidate = extract_thinking(raw_text)
    selected_winner = _select_winner(winner_candidate, req.choices)

    return {
        "winner": selected_winner,
        "reasoning": reasoning if req.enable_thinking else None,
        "raw_response": winner_candidate,
        "system": "system_two_nativa",
    }


if __name__ == "__main__":
    import uvicorn

    # Por padrão vincula a 127.0.0.1 em dev local; HOST=0.0.0.0 é injetado pelo Docker/Render
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("tools.api_server:app", host=host, port=port, reload=False)

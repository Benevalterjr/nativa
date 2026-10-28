#!/usr/bin/env python3
"""
Nativa API Server - OpenAI-, RAG-, and Branch.dev-compatible HTTP REST server.
Runs the native C engine (nativa_engine) in the background with minimal latency and zero GPU.
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

# Constants to avoid literal duplication (SonarQube compliance)
TAG_THINK = "<think>"
TAG_THINK_END = "</think>"
MARKER_TURN_DONE = "[TURN_DONE"
MSG_ENGINE_NOT_READY = "Engine not initialized."
STATUS_503_RESPONSES = {503: {"description": MSG_ENGINE_NOT_READY}}
PREFIX_ASSISTANT_THINK = "<|im_start|>assistant\n<think>\n"
PREFIX_ASSISTANT_NO_THINK = "<|im_start|>assistant\n"
DEFAULT_MODEL_NAME = "qwen3-0.6b-natq4"
DEFAULT_CHAT_EXAMPLE_MSG = "Hello! Briefly explain what the Nativa engine is."
DEFAULT_RAG_CONTEXT = "Nativa is a high-efficiency C99 language and runtime for CPU LLM inference."
DEFAULT_RAG_QUESTION = "What is Nativa?"
EXAMPLE_API_URL = "https://example.com/api"
CHOICE_RETRY_DESC = "Retry the request"
CHOICE_ABORT_DESC = "Abort and log permanent failure"
EXAMPLE_TASK_DECISION = "What action should be taken after 3 failed attempts with status 500?"
OBJECT_CHAT_CHUNK = "chat.completion.chunk"

# Add project root to sys.path to import nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import nativa
from tools.nativa_bundle import write_vocab_file

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    print("[ERROR] FastAPI and Uvicorn are required: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Default paths and settings
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4"))
DEFAULT_TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH", os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json")
)
IS_WINDOWS = sys.platform.startswith("win")
ENGINE_EXEC = "nativa_engine.exe" if IS_WINDOWS else "nativa_engine"
DEFAULT_ENGINE_PATH = os.environ.get("ENGINE_PATH", os.path.join(ROOT_DIR, "tools", ENGINE_EXEC))


def sanitize_chatml(text: str) -> str:
    """Remove special ChatML markers from external inputs to prevent prompt injection and clean surrogates."""
    if not isinstance(text, str):
        return ""
    cleaned = text.encode("utf-8", "replace").decode("utf-8", "replace")
    return cleaned.replace("<|im_start|>", "").replace("<|im_end|>", "")


def extract_thinking(text: str) -> Tuple[Optional[str], str]:
    """Extract content delimited by <think>...</think> (even if <think> comes in the prompt prefix)."""
    if TAG_THINK_END in text:
        parts = text.split(TAG_THINK_END, 1)
        reasoning = parts[0].replace(TAG_THINK, "").strip()
        answer = parts[1].strip()
        return (reasoning or None), answer
    return None, text.replace(TAG_THINK, "").strip()


def _compute_safe_len(buf: str, printed_idx: int) -> int:
    """Compute a safe slice length without breaking the turn marker."""
    safe_len = len(buf)
    if "[" in buf[printed_idx:]:
        last_bracket = buf.rfind("[")
        if last_bracket >= printed_idx and len(buf) - last_bracket < 16:
            safe_len = last_bracket
    return safe_len


def _extract_stream_chunk(buf: str, printed_idx: int) -> Tuple[Optional[str], int, bool]:
    """Extract the next safe fragment for streaming and indicate whether the turn has finished."""
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
    """Format an SSE chunk compatible with the OpenAI API."""
    data = {
        "id": completion_id,
        "object": OBJECT_CHAT_CHUNK,
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"



def _select_winner(candidate: str, choices: Dict[str, str]) -> str:
    """Select the winning option prioritizing exact word match from end to start."""
    cleaned = candidate.strip()
    keys = list(choices.keys())
    if not keys:
        return ""

    # 1. Exact match with the entire response (ignoring surrounding punctuation/quotes)
    normalized_full = cleaned.strip(" \t\n\r'\"`.,:;!()[]{}").lower()
    for k in keys:
        if k.lower() == normalized_full:
            return k

    # 2. Exact word search (word boundary), prioritizing the last occurrence in the text
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

    # 3. Substring fallback (last occurrence)
    for k in keys:
        pos = cleaned.lower().rfind(k.lower())
        if pos > best_pos:
            best_pos = pos
            best_key = k

    return best_key if best_key is not None else keys[0]


class EngineRunner:
    """C engine subprocess manager with a lock for concurrent requests."""

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
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.isfile(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"C engine executable not found: {self.engine_path}")

        if self.tokenizer is None:
            print(f"[*] Initializing Tokenizer: {self.tokenizer_path}")
            self.tokenizer = nativa.Tokenizer(self.tokenizer_path)

        if not self.vocab_path or not os.path.isfile(self.vocab_path):
            vocab_tmp = tempfile.NamedTemporaryFile(prefix="nativa-vocab-", suffix=".bin", delete=False)
            self.vocab_path = vocab_tmp.name
            vocab_tmp.close()
            write_vocab_file(self.tokenizer, self.vocab_path)

        env = os.environ.copy()
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

        print(f"[*] Starting Nativa C Engine: {self.engine_path} with {self.model_path}...")
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
                raise RuntimeError(f"C engine terminated prematurely. Stderr: {stderr}")
            if "[NATIVA_CHAT_READY]" in line:
                print(f"[OK] Nativa Engine ready for inference! Model: {self.model_name}")
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
        print("[*] Nativa Engine stopped.")

    def _write_stdin(self, cmd: str) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError(MSG_ENGINE_NOT_READY)
        self.proc.stdin.write(cmd)
        self.proc.stdin.flush()

    async def generate(self, prompt: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
        """Stream generated tokens from the prompt without losing the final chunk."""
        async with self.lock:
            if not self.is_alive():
                await asyncio.to_thread(self.start)

            tokens = self.tokenizer.encode(prompt).tolist()
            if not tokens:
                return

            # Keep prompt strictly within the model's physical window and latency budget
            MAX_ENGINE_CONTEXT = 1100
            if len(tokens) > MAX_ENGINE_CONTEXT:
                print(f"[!] Trimming prompt from {len(tokens)} to {MAX_ENGINE_CONTEXT} tokens to fit C engine window (2048)...", flush=True)
                tokens = tokens[:300] + tokens[-(MAX_ENGINE_CONTEXT - 300):]

            print(f"[*] Engine: Sending {len(tokens)} tokens to C engine | max_gen: {max_tokens}", flush=True)

            cmd = f"RESET\nINPUT {len(tokens)} {max_tokens} " + " ".join(map(str, tokens)) + "\n"
            await asyncio.to_thread(self._write_stdin, cmd)

            buf = ""
            printed_idx = 0
            fd = self.proc.stdout.fileno()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            try:
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
            except (GeneratorExit, asyncio.CancelledError):
                print("[*] Client cancelled turn. Killing stuck engine to reset cleanly...", flush=True)
                if self.proc:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    self.proc = None
                raise


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
    description="High-performance pure C99 Serverless API for RAG, Agents, and Branch.dev",
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raw_error = str(exc)
    safe_error = raw_error.encode("utf-8", "replace").decode("utf-8")
    return JSONResponse(status_code=422, content={"detail": safe_error})


# ----------------------------------------------------------------------------
# DTOs (Pydantic Models)
# ----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(
        default="user",
        description="Role of the message author in the conversation.",
    )
    content: Optional[str] = Field(
        default="",
        description="Text content of the message.",
    )
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=DEFAULT_MODEL_NAME)
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="List of conversation messages.",
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    stream: Optional[bool] = Field(default=False)
    enable_thinking: Optional[bool] = Field(default=False)
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None

    model_config = ConfigDict(extra="ignore")


class RagRequest(BaseModel):
    context: str = Field(
        default=DEFAULT_RAG_CONTEXT,
        min_length=1,
        description="Retrieved documents for grounding",
        examples=[DEFAULT_RAG_CONTEXT],
    )
    question: str = Field(
        default=DEFAULT_RAG_QUESTION,
        min_length=1,
        description="User question",
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
        description="Object or text representing the current state",
        examples=[{"url": EXAMPLE_API_URL, "status_code": 500, "retries": 3}],
    )
    choices: Dict[str, str] = Field(
        default={
            "tentar_novamente": CHOICE_RETRY_DESC,
            "abortar": CHOICE_ABORT_DESC,
        },
        min_length=1,
        description="Dictionary of choices (key -> description)",
        examples=[{
            "tentar_novamente": CHOICE_RETRY_DESC,
            "abortar": CHOICE_ABORT_DESC,
        }],
    )
    task: str = Field(
        default=EXAMPLE_TASK_DECISION,
        min_length=1,
        description="Task description or decision question",
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
# ROUTES
# ----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Nativa Qwen3-0.6B INT4 Engine API",
        "status": "online",
        "architecture": "C99 OpenMP Q4x4",
        "model": runner.model_name if runner else "loading",
        "docs": "/docs",
    }


@app.get("/health", responses=STATUS_503_RESPONSES)
def health():
    if runner and runner.is_alive():
        return {"status": "healthy", "engine": "running", "model": runner.model_name}
    raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)


@app.get("/v1/models")
def list_models():
    model_id = runner.model_name if runner else DEFAULT_MODEL_NAME
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nativa",
            }
        ],
    }


async def _stream_chat_chunks(
    runner_inst: EngineRunner,
    prompt: str,
    max_tokens: int,
    completion_id: str,
    created_ts: int,
    model: str,
    enable_thinking: bool,
):
    is_thinking = bool(enable_thinking)
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
    """Endpoint compatible with the OpenAI Chat Completions API."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    prompt_parts = []
    for msg in req.messages:
        role = msg.role
        content = sanitize_chatml(msg.content or "")
        if role == "tool":
            prompt_parts.append(f"<|im_start|>user\n[Tool Output ({msg.name or 'tool'})]:\n{content}<|im_end|>\n")
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # Ensure no-think directive if thinking is disabled
    if not req.enable_thinking:
        no_think_directive = "You are a concise assistant. Never explain your thought process or internal monologue. Output only the final response directly.\n"
        if prompt_parts:
            if prompt_parts[0].startswith("<|im_start|>system\n"):
                prompt_parts[0] = prompt_parts[0].replace("<|im_start|>system\n", f"<|im_start|>system\n{no_think_directive}")
            else:
                prompt_parts.insert(0, f"<|im_start|>system\n{no_think_directive}<|im_end|>\n")
        # Also append right before assistant turn so it survives prompt trimming
        prompt_parts.append(f"<|im_start|>system\nRespond directly. No internal monologue or reasoning.<|im_end|>\n")

    # If tools provided and not already in prompt, append schema instructions
    if req.tools and not any("# Tools" in (m.content or "") for m in req.messages):
        tools_desc = "\n\n# Tools\nYou have access to the following functions:\n"
        for t in req.tools:
            if isinstance(t, dict):
                func = t.get("function", t)
                tools_desc += f"\n## `{func.get('name')}`\nDescription: {func.get('description', '')}\nParameters: {json.dumps(func.get('parameters', {}), ensure_ascii=False)}\n"
        tools_desc += '\nWhen you choose to call a function, respond ONLY in this format:\n<tool_call>\n{"name": "function_name", "arguments": {"arg": "val"}}\n</tool_call>\n'
        if prompt_parts:
            if prompt_parts[0].startswith("<|im_start|>system\n"):
                prompt_parts[0] = prompt_parts[0].replace("<|im_end|>\n", f"{tools_desc}<|im_end|>\n")
            else:
                prompt_parts.insert(0, f"<|im_start|>system\n{tools_desc}<|im_end|>\n")

    prompt_prefix = (
        PREFIX_ASSISTANT_THINK if req.enable_thinking else PREFIX_ASSISTANT_NO_THINK
    )
    prompt_parts.append(prompt_prefix)
    full_prompt = "".join(prompt_parts)

    created_ts = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    user_model = req.model or DEFAULT_MODEL_NAME
    engine_tokens = (req.max_tokens or 256) + (150 if req.enable_thinking else 0)

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

    # Check for tool_call tag in response
    tool_calls = []
    finish_reason = "stop"
    if "<tool_call>" in cleaned_content:
        tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", cleaned_content, re.DOTALL)
        for tc_str in tc_matches:
            try:
                tc_data = json.loads(tc_str.strip())
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc_data.get("name", ""),
                        "arguments": json.dumps(tc_data.get("arguments", {}), ensure_ascii=False)
                        if isinstance(tc_data.get("arguments"), dict)
                        else str(tc_data.get("arguments", "{}")),
                    },
                })
            except Exception:
                pass
        if tool_calls:
            finish_reason = "tool_calls"

    prompt_tokens_len = len(runner.tokenizer.encode(full_prompt).tolist())
    completion_tokens_len = len(runner.tokenizer.encode(full_response).tolist())

    msg_data = {
        "role": "assistant",
        "content": cleaned_content,
    }
    if req.enable_thinking and reasoning:
        msg_data["reasoning_content"] = reasoning
    if tool_calls:
        msg_data["tool_calls"] = tool_calls

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": user_model,
        "choices": [
            {
                "index": 0,
                "message": msg_data,
                "finish_reason": finish_reason,
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
    """RAG-optimized endpoint returning a direct, grounded answer."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    system_prompt = (
        "Based on the provided context, answer the user's question directly and accurately."
    )

    safe_context = sanitize_chatml(req.context)
    safe_question = sanitize_chatml(req.question)
    user_prompt = f"Context:\n{safe_context}\n\nQuestion:\n{safe_question}"
    assistant_prefix = (
        PREFIX_ASSISTANT_THINK if req.enable_thinking else PREFIX_ASSISTANT_NO_THINK
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{assistant_prefix}"
    )

    engine_tokens = (req.max_tokens or 256) + (150 if req.enable_thinking else 0)
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
    """Direct endpoint for Branch.dev probabilistic decision-making."""
    if not runner:
        raise HTTPException(status_code=503, detail=MSG_ENGINE_NOT_READY)

    choices_text = "\n".join([f"- '{k}': {desc}" for k, desc in req.choices.items()])
    state_str = (
        json.dumps(req.state, ensure_ascii=False)
        if isinstance(req.state, (dict, list))
        else str(req.state)
    )

    system_prompt = (
        "You are the logical decision arbiter for Branch.dev (System 2). "
        "Analyze the state and choose the correct option. "
        "Finish by stating the exact name of the winning choice."
    )

    user_prompt = (
        f"State:\n{sanitize_chatml(state_str)}\n\n"
        f"Choices:\n{sanitize_chatml(choices_text)}\n\n"
        f"Task:\n{sanitize_chatml(req.task)}\n\n"
        f"Which choice should be executed?"
    )

    assistant_prefix = (
        PREFIX_ASSISTANT_THINK if req.enable_thinking else PREFIX_ASSISTANT_NO_THINK
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{assistant_prefix}"
    )

    engine_tokens = (req.max_tokens or 256) + (150 if req.enable_thinking else 0)
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

    # Defaults to binding 127.0.0.1 in local dev; HOST=0.0.0.0 is injected by Docker/Render
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("tools.api_server:app", host=host, port=port, reload=False)

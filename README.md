# Nativa

**A programming language and runtime for efficient AI inference**  
Focused on dense tensors, BitNet b1.58 (1.58-bit ternary), and local LLM execution.

[![Version](https://img.shields.io/badge/version-0.4-blue)](https://github.com/Benevalterjr/nativa)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-yellow)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-success)](https://github.com/Benevalterjr/nativa)

> **Looking for contributors.**  
> Nativa is currently maintained by a single developer. Help is welcome on kernels, quantization, the language frontend, documentation, and tooling. See [Contributing](#contributing) below.

---

## What is Nativa?

Nativa is a domain-specific language and high-performance runtime designed for:

- Local LLM inference (already runs **Qwen2.5-0.5B** end-to-end)
- Ternary / BitNet b1.58 computation (add-only GEMM)
- Custom quantization format (`.natq4`)
- Zero-copy SafeTensors loading via `mmap`
- AOT compilation to native C99 + OpenMP

It combines a clean, Python-inspired syntax with a dual execution model:
- **Interpreter** (fast prototyping with NumPy + AOT kernels)
- **AOT C99 backend** (native binaries, OpenMP, CPU dispatch)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Real model support** | Full inference of Qwen2.5-0.5B (24 layers, 490M parameters) |
| **Production INT4 Format** | `.natq4` (Q4x4 Lean, 4.50 bits/weight) with fused MLP SwiGLU, 4-row tiling, and full tail support |
| **BitNet b1.58 (Experimental)** | Native ternary type (`t2`), 2-bit SIMD and 5-trits packing for models trained natively in 1.58 bits |
| **Zero-copy loading** | SafeTensors via `mmap` (loads hundreds of MB in milliseconds) |
| **KV Cache Arena** | Contiguous pre-allocated memory (large speedup on autoregressive generation) |
| **AOT C99** | Compiles to native executables with OpenMP and runtime CPU dispatch |
| **OpenCL path** | Experimental multi-device support (CPU/GPU) via dynamic loading |

---

## Quick Start

New to the language? Read the [English beginner guide](GUIA_INICIANTE_EN.md)
or the [Portuguese beginner guide](GUIA_INICIANTE.md) for installation, syntax,
interpreter examples, and native builds.

```bash
# Clone
git clone https://github.com/Benevalterjr/nativa.git
cd nativa

# Install core dependencies
pip install -r requirements.txt

# Run the test suite
python nativa.py --test

# Run a mini LLM example
python nativa.py models/mini_llm_mha_real.nat --run

# Run full Qwen2.5-0.5B (requires weights in data/)
python nativa.py models/qwen2.5_completo.nat --run
```

### Quantize and chat with a model

```bash
# Convert a Hugging Face Qwen2/Qwen3 model
python nativa.py quantize path/to/hf_model data/my_model.natq4

# Interactive chat
python nativa.py chat data/my_model.natq4
```

### Compile to native binary

```bash
python nativa.py build tests/riemann.nat
./tests/riemann          # or riemann.exe on Windows
```

### REST API Server (OpenAI-compatible)

Nativa provides a zero-GPU, C99-accelerated HTTP server for RAG, local chat, and reasoning agents:

```bash
pip install -e ".[server]"
python tools/api_server.py
```

### Official Benchmarks & Performance

Nativa includes a built-in reproducible benchmark suite measuring throughput (tokens/s), time-to-first-token (TTFT), WikiText-2 perplexity (PPL), and memory footprint:

```bash
# Run full automated benchmark suite
python nativa.py bench --all
```

- 📊 [English Benchmark Report](BENCHMARKS_EN.md) — Throughput, PPL, and zero-shot reasoning evaluation.
- 📊 [Relatório de Benchmarks em Português](BENCHMARKS.md) — Avaliação auditável completa.
- 📋 [Q4x4 Lean Specification & Gatekeeping Protocol](docs/SPEC_Q4X4_LEAN_AND_GATEKEEPING.md) — Especificação técnica do baseline congelado, diagnóstico de banda e protocolo de gatekeeping.

---

## Project Structure

```text
nativa/
├── nativa.py                 # Compiler, interpreter, SafeTensors, AOT engine, CLI
├── GUIA_INICIANTE.md         # Beginner guide (Portuguese)
├── GUIA_INICIANTE_EN.md      # Beginner guide (English)
├── BENCHMARKS.md             # Official benchmark report (Portuguese)
├── BENCHMARKS_EN.md          # Official benchmark report (English)
├── models/                   # Full model examples (.nat)
│   ├── qwen2.5_completo.nat
│   ├── mini_llm_mha_real.nat
│   ├── mini_llm_bitnet.nat
│   └── ...
├── tests/                    # Numerical and regression tests
├── tools/                    # Quantization, download, benchmarks, C kernels
├── data/                     # Weights and tokenizers (not all committed)
└── scripts/                  # Chat launchers
```

---

## Architecture Highlights

- **SafeTensors zero-copy**: `mmap` + direct NumPy views. No unnecessary copies.
- **Memory Arena for KV Cache**: Pre-allocated contiguous buffers → big speedup on generation.
- **CPU dispatch**: Runtime detection (SSSE3/AVX) with pure C99 scalar fallback. No illegal instruction crashes on old CPUs.
- **Ternary kernels**: Dense base-3 packing (5 trits/byte) + Vec-LUT GEMV.
- **AOT cache**: SHA-256 based invalidation of compiled C kernels.

---

## Contributing

Nativa is a solo project today. Contributions of any size are welcome.

**Good first issues / areas that need help:**

1. **Kernels & performance** – AVX2/AVX-512, better OpenCL kernels, profiling
2. **Quantization quality** – outlier handling, group-wise BitNet, fidelity metrics
3. **Language frontend** – type system robustness, better error messages, missing syntax
4. **Tooling** – basic LSP / syntax highlighting, formatter, better CLI UX
5. **Documentation & examples** – English beginner guide, more tutorials, benchmarks

How to start:
1. Open an issue describing what you want to work on (or pick an existing one)
2. Fork → branch → pull request
3. Keep changes focused and add tests when possible

Even small contributions (typo fixes, clearer error messages, extra examples) help a lot.

---

## License

MIT (see `LICENSE` if present, otherwise the project follows a permissive open-source license).

---

## Acknowledgements

- Qwen team (Alibaba) for the excellent open models
- BitNet / Microsoft Research for the 1.58-bit direction
- The broader local LLM community (`llama.cpp`, ggml, etc.)

---

<br>

# Nativa (Português)

**Linguagem de programação e runtime para inferência eficiente de IA**  
Focada em tensores densos, BitNet b1.58 (ternário 1.58-bit) e execução local de LLMs.

> **Procurando contribuidores.**  
> A Nativa é mantida atualmente por um único desenvolvedor. Ajuda é bem-vinda em kernels, quantização, frontend da linguagem, documentação e tooling. Veja a seção [Contribuindo](#contribuindo) abaixo.

---

## O que é a Nativa?

A Nativa é uma linguagem de domínio específico + runtime de alta performance projetada para:

- Inferência local de LLMs (já roda o **Qwen2.5-0.5B** completo)
- Computação ternária / BitNet b1.58 (GEMM apenas com adições)
- Formato de quantização próprio (`.natq4`)
- Carregamento zero-copy de SafeTensors via `mmap`
- Compilação AOT para C99 nativo + OpenMP

Ela combina uma sintaxe limpa inspirada em Python com um modelo de execução duplo:
- **Interpretador** (prototipagem rápida com NumPy + kernels AOT)
- **Backend AOT C99** (binários nativos, OpenMP, despacho de CPU em runtime)

---

## Principais Recursos

| Recurso | Descrição |
|---------|-----------|
| **Suporte a modelo real** | Inferência completa do Qwen2.5-0.5B (24 camadas, 490M parâmetros) |
| **Formato de Produção INT4** | `.natq4` (Q4x4 Lean, 4.50 bits/peso) com MLP SwiGLU fundido, tiling de 4 linhas e suporte a caudas |
| **BitNet b1.58 (Experimental)** | Tipo ternário nativo (`t2`), suporte SIMD 2-bit e 5-trits para modelos treinados nativamente em 1.58 bits |
| **Carregamento zero-copy** | SafeTensors via `mmap` (centenas de MB em milissegundos) |
| **KV Cache com Arena** | Memória contígua pré-alocada (grande aceleração na geração) |
| **AOT C99** | Compila para executáveis nativos com OpenMP e despacho de CPU |
| **Caminho OpenCL** | Suporte experimental multi-dispositivo (CPU/GPU) via carregamento dinâmico |

---

## Início Rápido

```bash
git clone https://github.com/Benevalterjr/nativa.git
cd nativa

pip install -r requirements.txt

python nativa.py --test
python nativa.py models/mini_llm_mha_real.nat --run
python nativa.py models/qwen2.5_completo.nat --run   # precisa dos pesos em data/
```

### Quantizar e conversar com um modelo

```bash
python nativa.py quantize caminho/do/modelo_hf data/meu_modelo.natq4
python nativa.py chat data/meu_modelo.natq4
```

### Compilar para binário nativo

```bash
python nativa.py build tests/riemann.nat
./tests/riemann
```

---

## Contribuindo

A Nativa é um projeto solo hoje. Contribuições de qualquer tamanho são bem-vindas.

**Áreas que mais precisam de ajuda:**

1. **Kernels e performance** – AVX2/AVX-512, kernels OpenCL, profiling
2. **Qualidade de quantização** – outliers, BitNet por grupos, métricas de fidelidade
3. **Frontend da linguagem** – robustez do sistema de tipos, mensagens de erro, sintaxe faltante
4. **Tooling** – LSP básico / highlight, formatador, melhor UX do CLI
5. **Documentação e exemplos** – guia iniciante em inglês, mais tutoriais, benchmarks públicos

Como começar:
1. Abra uma issue descrevendo o que quer fazer (ou pegue uma existente)
2. Fork → branch → pull request
3. Mantenha as mudanças focadas e adicione testes sempre que possível

Mesmo correções pequenas (typos, mensagens de erro mais claras, exemplos extras) ajudam muito.

---

## Licença

MIT (veja o arquivo `LICENSE` se existir).

---

**Feito com foco em eficiência e simplicidade.**  
Se você gosta de compiladores, kernels de baixa precisão ou inferência local de LLMs, este é um bom lugar para contribuir.

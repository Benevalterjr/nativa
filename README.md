# Linguagem Nativa (v0.3 — AI, Data, Ternary & Universal Compute Edition)

A **Linguagem Nativa** é uma linguagem de programação moderna voltada para Inteligência Artificial, Tensores Densos, Computação Ternária de 1.58 bits (BitNet b1.58) e Inferência Heterogênea Universal (CPU e GPU). Ela combina um compilador AOT C99/OpenMP de alta velocidade, um interpretador dinâmico amigável e um runtime aberto multi-dispositivo via OpenCL.

Está começando? Acesse o [Guia de início para aprender Nativa](GUIA_INICIANTE.md).

---

## 🌟 Princípios de Design

1. **Universalidade Sem Amarras:** O código compilado e os modelos do Nativa rodam em **qualquer computador** (de PCs antigos a estações de trabalho de ponta). O sistema possui auto-detecção em tempo de execução (`CPUID`) e *graceful fallback* escalar em C99 puro para que nenhum desenvolvedor seja deixado para trás.
2. **Zero Dependência Pesada:** O compilador e o motor de inferência não exigem a instalação de SDKs de dezenas de gigabytes (como CUDA ou OpenCL SDK). As bibliotecas do sistema são carregadas dinamicamente sob demanda.
3. **Eficiência Extrema em Memória:** Utiliza mapeamento de arquivo virtual (*Zero-Copy Virtual mmap*) e alocação contígua em Arenas, eliminando cópias redundantes de tensores e permitindo carregar modelos pesados em computadores com memória RAM modesta.

---

## 📁 Estrutura do Workspace

```text
nativa/
├── nativa.py                     # Compilador AOT, Interpretador, SafeTensors mmap e CLI Unificada
├── nativa.bat / nativa.cmd       # Atalhos de linha de comando para Windows (CMD)
├── nativa.ps1                    # Atalho nativo para Windows PowerShell com autodetecção de .venv
├── requirements.txt              # Requisitos mínimos do runtime core (numpy)
├── requirements-tools.txt        # Dependências opcionais para tools/ (torch, transformers, etc.)
├── .gitignore                    # Regras de exclusão de binários, modelos pesados e caches
├── README.md                     # Documentação completa e guia do projeto
├── GUIA_INICIANTE.md             # Guia de início rápido da linguagem
├── SECURITY.md                   # Política de segurança e reporte de vulnerabilidades
│
├── scripts/                      # Launchers e scripts interativos de chat (.bat / .py)
│   ├── Chat_Universal_NatQ4.bat  # Launcher para chat universal com modelos .natq4
│   ├── Chat_Com_Qwen*.bat        # Launchers interativos com Qwen 0.5B, 1.5B e 3B
│   └── chat_qwen*.py             # Interfaces interativas de chat
│
├── models/                       # Modelos e arquiteturas completas (.nat)
│   ├── qwen2.5_completo.nat      # Inferência oficial do Qwen2.5-0.5B (24 camadas, 490M parâmetros)
│   ├── qwen2.5_bitnet_quant.nat  # Pipeline BitNet aplicado ao modelo Qwen2.5
│   ├── mini_llm_mha_real.nat     # MHA real (4 cabeças), KV Cache Memory Arena, BitNet e Top-P
│   ├── mini_llm_bitnet.nat       # Transformer 100% BitNet b1.58 puro (Add-Only GEMM)
│   ├── mini_llm_completo.nat     # Transformer Decoder-Only com pesos densos FP32
│   ├── criptografia_nativa.nat   # Prova de conceito de algoritmos criptográficos em Nativa
│   └── benchmark_rigoroso.nat    # Benchmark comparativo de fidelidade e latência
│
├── tests/                        # Validações numéricas e testes de regressão (.nat e .c)
│   ├── teste_opencl_benchmark.c  # Benchmark tensorial executado em CPU e GPU via OpenCL
│   ├── riemann.nat / riemann.c   # Exemplo numérico clássico (zeros da função Zeta)
│   ├── teste_qwen_real.nat       # Inferência de pesos reais da Camada 0 do Qwen2.5
│   ├── teste_qwen_multicamada.nat# Loop sequencial encadeado de múltiplas camadas do Qwen
│   ├── teste_modelo_llm.nat      # Testes das primitivas LLM (BPE, SDPA, RoPE, RMSNorm, KVCache)
│   ├── teste_bitnet_5trits.nat   # Teste do empacotador denso base-3 (5 trits/byte)
│   ├── teste_grande_safetensors.nat # Validação e streaming de matrizes SafeTensors
│   └── teste_llama7b_bloco.nat   # Benchmark de 202M pesos do LLaMA-7B via mmap
│
├── data/                         # Pesos reais e sintéticos em SafeTensors (.safetensors)
│   ├── qwen2.5_0.5b.safetensors  # Modelo oficial completo Qwen2.5-0.5B (988 MB)
│   ├── tokenizer_qwen.json       # Vocabulário oficial e 151.387 merges BPE do Qwen
│   ├── llama7b_bloco.safetensors # Bloco sintético da camada 0 do LLaMA-7B (809 MB)
│   └── pesos_llama.safetensors   # Arquivo minimalista para testes rápidos
│
├── tools/                        # Motores C nativos e utilitários de IA (.c / .h / .py)
│   ├── nativa_engine.c           # Motor universal C99 para modelos quantizados .natq4
│   ├── nativa_opencl.h           # Runtime Bridge dinâmico para aceleradores OpenCL (CPU & GPU)
│   ├── nativa_sampling.h         # Samplers de amostragem determinística (Argmax, Top-K, Temp)
│   ├── cpu_dispatch.h            # Detecção de hardware em runtime (CPUID) e sintonia OpenMP
│   ├── natq4_format.h            # Especificação binária autodescritiva do formato .natq4
│   ├── nativa_quantize.py        # Conversor de modelos Hugging Face para NATQ4X4
│   ├── chat_natq4.py             # Interface de chat streaming multi-turn
│   ├── qwen_1.5b_native.c        # Motor C99 nativo dedicado para Qwen2.5-1.5B
│   └── baixar_qwen.py            # Download automatizado de checkpoints Hugging Face
│
└── .nativa_cache/                # Cache da engine AOT C99 (DLL/SO e hash SHA-256 de invalidação)
```

---

## 🚀 Como Executar (CLI Moderna do Nativa)

Você pode usar o comando direto `nativa` no PowerShell (`.\nativa`), CMD (`nativa`) ou chamando `python nativa.py`:

### 1. Testes Unitários de Regressão (73 Testes)
Executa a validação formal de todos os componentes da linguagem (lexer, parser, AST, tensores, SafeTensors, quantização e compilação C):
```bash
nativa test
# ou: python nativa.py --test
```

### 2. Detecção de Dispositivos e Hardware (CPU & OpenCL)
Inspeciona em tempo real todas as CPUs e GPUs (dedicadas ou integradas) disponíveis no computador:
```bash
nativa opencl
# ou: nativa --devices
```
*Exemplo de saída:*
```text
====================================================================
  NATIVA - DISPOSITIVOS DE COMPUTAÇÃO DETECTADOS
====================================================================
  [CPU Host] SO: win32 | Python 3.11.1
  [0] [CPU] Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz (Plataforma: Intel(R) OpenCL)
  [1] [GPU] Intel(R) HD Graphics 4000 (Plataforma: Intel(R) OpenCL)
====================================================================
```

### 3. Execução de Programas e Exemplos (.nat)
```bash
# Executa um programa de usuário
nativa run meu_programa.nat

# Executa exemplos prontos de alta fidelidade
nativa run @riemann      # Zeros da função Zeta de Riemann
nativa run @rede         # Treinamento de Rede Neural Perceptron
nativa run @ia_tensor    # Operações matriciais e tensores
nativa run @ternario     # Computação Ternária BitNet b1.58
```

### 4. Compilação AOT C99 para Executável Nativo (.exe)
Compila o código Nativa para C99 padrão e gera um binário nativo de máquina:
```bash
nativa build tests/riemann.nat
./tests/riemann.exe
```

### 5. Quantização e Chat Interativo com Modelos de IA (.natq4)
Converte modelos do Hugging Face (Qwen2 / Qwen3) para o formato eficiente `.natq4` e inicia o chat streaming:
```bash
# Converte a pasta do modelo para .natq4
nativa quantize caminho/para/modelo_hf data/meu_modelo.natq4

# Inicia o chat interativo com streaming token a token em C
nativa chat data/meu_modelo.natq4
```

---

## ⚡ Pilares Arquiteturais do Nativa v0.3

### 1. Universalidade e Despacho Dinâmico de CPU (Graceful Fallback)
* O compilador C do Nativa não engessa flags proprietárias como `/arch:AVX` nos binários gerados.
* Através de [`tools/cpu_dispatch.h`](tools/cpu_dispatch.h), o motor consulta o processador em tempo de execução via `CPUID`:
  - **Se o processador suportar SSSE3/AVX:** ativa os kernels vetorizados SIMD para máxima velocidade.
  - **Se for um processador antigo ou de outra arquitetura (ARM/RISC-V):** chaveia instantaneamente para o fallback escalar puro C99 ([`q4x4_dot4_scalar`](tools/nativa_engine.c)), garantindo 0% de risco de falhas `Illegal Instruction`.

### 2. Aceleração Aberta Multi-Dispositivo com OpenCL (CPU & GPU)
* Através de [`tools/nativa_opencl.h`](tools/nativa_opencl.h), o Nativa conecta-se à `OpenCL.dll` (Windows) ou `libOpenCL.so` (Linux) via carregamento dinâmico sem necessitar de SDKs externos.
* Permite executar kernels matriciais tanto em placas de vídeo dedicadas (NVIDIA, AMD, Intel Arc) quanto em GPUs integradas (Intel HD/Iris, Radeon Vega) e CPUs multi-core, mantendo a portabilidade total do código.

### 3. Computação Ternária de 1.58 bits (BitNet b1.58)
* Primitivas ternárias nativas (`t2`), quantização baseada em média absoluta e empacotamento denso de **5 trits por byte** (base 3: $3^5 = 243 \le 256$).
* Operador de multiplicação matricial livre de multiplicações (Add-Only GEMM), reduzindo drasticamente o consumo energético e acelerando o processamento.

### 4. Carregador SafeTensors com Zero-Copy `mmap`
* Mapeamento virtual de memória instantâneo via `mmap.mmap` e visualização direta por buffers contíguos. Modelos de centenas de megabytes carregam em milissegundos sem duplicar o uso de memória RAM.
* Proteção ativa contra travamento de descritores no Windows e aceleração com `madvise` assíncrono no Linux.

### 5. Memory Arena para KV-Cache
* Pré-alocação contígua total upfront da janela de contexto (`_arena_k` e `_arena_v`).
* Elimina 100% das re-alocações dinâmicas a cada token gerado, acelerando a geração de texto em até 20x.

---

## 📦 Dependências e Instalação

### Requisitos Básicos (Runtime Nativa Core)
```bash
pip install -r requirements.txt
```
*(Requer apenas `numpy>=1.24.0` e Python 3.9+)*

### Requisitos Opcionais para Conversão de Modelos (`tools/`)
Caso queira baixar ou converter modelos diretos do Hugging Face:
```bash
pip install -r requirements-tools.txt
```
*(Inclui `torch`, `transformers`, `safetensors` e `huggingface-hub`)*

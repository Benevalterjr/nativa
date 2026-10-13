# Linguagem Nativa (v0.3 — AI, Data & Ternary Edition)

A **Linguagem Nativa** é uma linguagem de programação voltada para Inteligência Artificial, Tensores Densos e Computação Ternária de 1.58 bits (BitNet b1.58), combinando um compilador AOT C99/OpenMP com interpretador dinâmico de alta performance.

Está começando? Acesse o [Guia de início para aprender Nativa](GUIA_INICIANTE.md).

---

## 📁 Estrutura do Workspace

```text
nativa/
├── nativa.py                     # Compilador, Interpretador, SafeTensors mmap e Engine AOT C99
├── nativa.bat / nativa.cmd       # Atalhos de linha de comando para Windows
├── requirements.txt              # Requisitos mínimos do runtime core (numpy)
├── requirements-tools.txt        # Dependências opcionais para tools/ (torch, transformers, etc.)
├── .gitignore                    # Regras de exclusão de binários, modelos pesados e caches
├── README.md                     # Documentação completa e guia do projeto
├── GUIA_INICIANTE.md             # Guia de início rápido da linguagem
│
├── scripts/                      # Launchers e scripts interativos de chat (.bat / .py)
│   ├── Chat_Universal_NatQ4.bat  # Launcher para chat universal com modelos .natq4
│   ├── Chat_Com_Qwen*.bat        # Launchers interativos com Qwen 0.5B, 1.5B e 3B
│   └── chat_qwen*.py             # Interfaces interativas de chat
│
├── models/                       # Modelos e arquiteturas completas (.nat)
│   ├── qwen2.5_completo.nat      # Inserência oficial do Qwen2.5-0.5B (24 camadas, 490M parâmetros)
│   ├── qwen2.5_bitnet_quant.nat  # Pipeline BitNet aplicado ao modelo Qwen2.5
│   ├── mini_llm_mha_real.nat     # MHA real (4 cabeças), KV Cache Memory Arena, BitNet e Top-P
│   ├── mini_llm_bitnet.nat       # Transformer 100% BitNet b1.58 puro (Add-Only GEMM)
│   ├── mini_llm_completo.nat     # Transformer Decoder-Only com pesos densos FP32
│   ├── criptografia_nativa.nat   # Prova de conceito de algoritmos criptográficos em Nativa
│   └── benchmark_rigoroso.nat    # Benchmark comparativo de fidelidade e latência
│
├── tests/                        # Validações numéricas e testes de regressão (.nat)
│   ├── teste_llama7b_bloco.nat   # Benchmark de 202M pesos do LLaMA-7B via mmap
│   ├── teste_qwen_real.nat       # Inferência de pesos reais da Camada 0 do Qwen2.5
│   ├── teste_qwen_multicamada.nat# Loop sequencial encadeado de múltiplas camadas do Qwen
│   ├── teste_modelo_llm.nat      # Testes das primitivas LLM (BPE, SDPA, RoPE, RMSNorm, KVCache)
│   ├── teste_bitnet_5trits.nat   # Teste do empacotador denso base-3 (5 trits/byte)
│   ├── teste_grande_safetensors.nat # Validação e streaming de matrizes SafeTensors
│   ├── safetensors_demo.nat      # Demonstração básica de roundtrip SafeTensors
│   ├── riemann.nat               # Exemplo numérico clássico (zeros da função Zeta)
│   └── riemann.c / riemann.exe   # Binário C99 compilado nativamente via MSVC
│
├── data/                         # Pesos reais e sintéticos em SafeTensors (.safetensors)
│   ├── qwen2.5_0.5b.safetensors  # Modelo oficial completo Qwen2.5-0.5B da Alibaba Cloud (988 MB)
│   ├── qwen2.5_camada0.safetensors # Camada 0 oficial isolada do Qwen2.5
│   ├── qwen2.5_2camadas.safetensors# Camadas 0 e 1 encadeadas do Qwen2.5
│   ├── tokenizer_qwen.json       # Vocabulário oficial e 151.387 merges BPE do Qwen
│   ├── llama7b_bloco.safetensors # Bloco sintético da camada 0 do LLaMA-7B (809 MB)
│   ├── grande_modelo.safetensors # Modelo de teste com camadas de atenção e MLP
│   └── pesos_llama.safetensors   # Arquivo minimalista para testes rápidos
│
├── tools/                        # Utilitários de pesquisa, download e conversão (.py)
│   ├── baixar_qwen.py            # Download do modelo oficial Qwen2.5-0.5B do HuggingFace
│   ├── baixar_qwen_camada.py     # Extração e isolamento de camadas SafeTensors
│   ├── teste_qwen_int4.py        # Demonstração de inferência INT4 em CPU (435 MB RAM, 4.5 tok/s)
│   ├── teste_geracao_qwen.py     # Geração de texto autoregressiva experimental
│   ├── test_qwen_bitnet_generation.py # Estudo de quantização BitNet no Qwen2.5
│   ├── test_10pct_outliers.py    # Análise de preservação de outliers nos pesos
│   ├── test_groupwise_bitnet.py  # Quantização por grupos de canais
│   ├── test_hybrid_bitnet.py     # Quantização híbrida (Attn FP16 / MLP BitNet)
│   ├── test_mlp_hybrid.py        # Testes de fidelidade no bloco SwiGLU
│   ├── gerar_llama7b_block.py    # Gerador de pesos sintéticos LLaMA-7B
│   ├── gerar_safetensors.py      # Gerador de matrizes SafeTensors de teste
│   └── benchmark_rigoroso.py     # Suite de benchmarking analítico em Python
│
└── .nativa_cache/                # Cache da engine AOT C99 (DLL/SO e hash SHA-256 de invalidação)
```

---

## 🚀 Como Executar

### Quantização e chat local Qwen2/Qwen3
O conversor `tools/nativa_quantize.py` aceita pastas Hugging Face com `config.json`, `tokenizer.json` e SafeTensors (arquivo único ou shards):
```bash
python nativa.py quantize caminho/para/modelo caminho/para/modelo.natq4
python nativa.py chat caminho/para/modelo.natq4
```
O suporte atual cobre `Qwen2ForCausalLM` e `Qwen3ForCausalLM` com SwiGLU e RoPE padrão. Para Qwen3, o runtime também aplica as RMSNorm de Q e K por cabeça. Modelos Qwen3 com sliding window, RoPE escalado ou `attention_bias=true` são recusados explicitamente até essas operações estarem implementadas. O conversor grava os IDs de vocabulário ativos e copia o tokenizer para junto do `.natq4`, permitindo que o chat use os tokens do próprio modelo.

### 1. Testes Unitários Completos (65 Testes)
```bash
python nativa.py --test
```

### 2. Inferência Oficial do Qwen2.5-0.5B Completo (24 Camadas)
```bash
python nativa.py models/qwen2.5_completo.nat --run
```

### 3. Mini LLM com Multi-Head Attention Real e Memory Arena
```bash
python nativa.py models/mini_llm_mha_real.nat --run
```

### 4. Mini LLM BitNet b1.58 Puro (Add-Only GEMM)
```bash
python nativa.py models/mini_llm_bitnet.nat --run
```

### 5. Benchmark do Bloco LLaMA-7B (202M Parâmetros via mmap)
```bash
python nativa.py tests/teste_llama7b_bloco.nat --run
```

### 6. Compilação AOT C99 para Binário Nativo
```bash
python nativa.py tests/riemann.nat --build
./tests/riemann.exe
```

---

## ⚡ Pilares Arquiteturais do Nativa v0.3

### 1. Carregador SafeTensors com Zero-Copy `mmap` e Validador Estrito
* **Zero-Copy Instantâneo**: Mapeamento virtual de memória via `mmap.mmap` e visualização direta por `np.frombuffer`. Arquivos de centenas de megabytes carregam em frações de milissegundo (aceleração superior a 6.000x sobre `fread`).
* **Validação Rigorosa**:
  * Verificação estrita de integridade contra arquivos truncados ou corrompidos.
  * Validação de tamanho seguro do cabeçalho (limite de segurança de 100 MB).
  * Consistência estrita entre `shape`, `dtype` e quantidade real de bytes declarados nos offsets.
* **Segurança Multiplataforma**:
  * **Windows**: Gerenciador `close_safetensors_mmap()` eliminando o travamento `[WinError 32]`.
  * **Linux**: Prevenção de falhas de página `SIGBUS` e aceleração via syscall `madvise(MADV_WILLNEED)` para readahead assíncrono no kernel.

### 2. Tokenizador BPE Canônico com Merges e Byte-Fallback
* **Regras de Merge BPE Reais**: Aplicação das regras de mesclagem (`merges`) ordenadas por ranking canônico, compatível com tokenizers modernos como o do Qwen (`data/tokenizer_qwen.json`).
* **Mapeamento de Bytes GPT-2/Qwen**: Suporte completo a caracteres de fallback (`Ġ`, etc.) e reconstrução exata de sequências UTF-8 com suporte nativo a `<0xXX>`.

### 3. Memory Arena para KV-Cache (Estilo `llama2.c`)
* Pré-alocação contígua total upfront da janela de contexto (`_arena_k` e `_arena_v`).
* Elimina 100% das re-alocações dinâmicas e cópias redundantes de memória a cada token gerado.
* Geração de tokens no Mini LLM acelerada de **8.8 tok/s** para **180+ tok/s** (20x mais rápido).

### 4. Engine AOT C99 com Invalidação por Hash SHA-256
* Compilação automática de kernels em C com OpenMP usando MSVC (`cl.exe /O2 /openmp`) ou GCC/Clang (`-O3 -fopenmp`).
* **Cache Inteligente**: O arquivo `.nativa_cache/nativa_aot.hash` armazena o hash SHA-256 dos fontes C. Alterações nos kernels invalidam e recompilam a DLL automaticamente.

### 5. Semântica de Dispositivos (`device`)
* **Backend de Execução em v0.3**: O runtime executa nativamente no backend **CPU otimizado** (C99 AOT / OpenMP e NumPy vetorizado).
* **Tags `cuda` e `opencl`**: Operam como **metadados de grafo** experimentais para validação de código e compatibilidade futura, emitindo aviso explícito sobre a execução em CPU.

---

## 📦 Dependências e Instalação

### Requisitos Básicos (Runtime Nativa)
```bash
pip install -r requirements.txt
```
*(Requer apenas `numpy>=1.24.0` e Python 3.9+)*

### Requisitos para Scripts de Pesquisa e Modelos (`tools/`)
Caso queira executar os scripts de conversão, download do Hugging Face ou quantização experimental em `tools/`:
```bash
pip install -r requirements-tools.txt
```
*(Inclui `torch`, `transformers`, `safetensors` e `huggingface-hub`)*

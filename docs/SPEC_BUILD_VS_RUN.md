# Especificação Técnica: `nativa run` vs `nativa build` (v0.4)

Este documento define formalmente a matriz de suporte, o escopo arquitetural e os limites entre os dois modos de execução da linguagem Nativa:
1. **Modo Interpretador / Prototipagem (`nativa run` / `--run`)**
2. **Modo Compilador Nativo AOT (`nativa build` / `--build`)**

---

## 1. Filosofia e Arquitetura

Nativa adota uma separação intencional entre **velocidade de exploração científica** e **velocidade de execução de baixo nível**:

```
                  +----------------------------------+
                  |  Código-Fonte Nativa (.nat)     |
                  +----------------------------------+
                                   |
                  +----------------------------------+
                  | Frontend: Lexer + Parser + AST   |
                  |     Analisador Semântico Estático|
                  +----------------------------------+
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
+-----------------------------+             +-----------------------------+
| Modo Interpretador (--run)  |             | Modo Compilador (--build)   |
| Runtime Dinâmico:           |             | Backend AOT C99:            |
| - Python + NumPy Contíguo   |             | - Transpilação para C99     |
| - Kernels C AOT (ctypes)    |             | - Compilação GCC/Clang/MSVC |
| - OpenMP / Multi-threading  |             | - Otimização -O3 / -fopenmp |
| - Tokenizadores BPE Hugging |             | - Binário nativo standalone |
| - Safetensors mmap dinâmico |             | - Zero dependência externa  |
| - Dicionários e TurboQuant  |             | - Desempenho Bare-Metal     |
+-----------------------------+             +-----------------------------+
```

### O que significa "Nativo"?
- **`nativa run`**: É um interpretador de prototipagem em Python que executa sobre NumPy e invoca bibliotecas compartilhadas C (`.dll` / `.so`) para kernels críticos (BitNet, RMSNorm, Softmax, SiLU).
- **`nativa build`**: É um compilador AOT (Ahead-Of-Time) que transpila a AST da Nativa para **C99 puro** e invoca o compilador nativo do sistema (`gcc`, `clang` ou `cl.exe`), gerando um executável bare-metal sem Python, sem NumPy e sem interpretador.

---

## 2. Matriz de Compatibilidade (v0.4)

| Recurso / Primitiva | Interpretador (`--run`) | Compilador C99 (`--build`) | Observações |
| :--- | :---: | :---: | :--- |
| **Tipos Escalares e Estruturados** | | | |
| `i64`, `i32`, `i16`, `i8`, `u64`, `u32`, `u16`, `u8` | Sim | Sim | C99 `int64_t`, `int32_t`, etc. (`stdint.h`) |
| `f64`, `f32`, `f16`, `bf16` | Sim | Sim | C99 `double` / `float` |
| `c64`, `complex`, `Complex` | Sim | Sim | C99 `NatComplex` struct (`double real, imag;`) |
| `bool`, `str` | Sim | Sim | C99 `bool` (`stdbool.h`), `const char*` |
| `t2`, `ternary` ({-1, 0, +1}) | Sim | Sim | C99 `int8_t` |
| **Estruturas de Dados** | | | |
| Listas numéricas 1D (`[f64]`, `[i64]`, etc.) | Sim | Sim | Ponteiros C99 contíguos (`double*`, `int64_t*`) |
| Matrizes 2D (`[[f64]]`, etc.) | Sim | Sim | Achatamento contíguo row-major |
| Tensores (`tensor[T]`, `Tensor`) | Sim | Parcial | No C99 mapeado para buffers contíguos alinhados |
| Dicionários / Objetos dinâmicos | Sim | **Não** | Planejado via structs para v0.4 |
| **Controle de Fluxo** | | | |
| `if`, `elif`, `else` | Sim | Sim | Tradução direta para `if / else if / else` |
| `while`, `break`, `continue` | Sim | Sim | Tradução direta para loops C99 |
| `for i in start..end` (Intervalos numéricos) | Sim | Sim | Otimizado para loops `for (int64_t i = start; i < end; i++)` |
| `for item in colecao_dinamica` | Sim | **Não** | Apenas ranges numéricos são suportados em C99 |
| Funções e Recursão | Sim | Sim | Protótipos antecipados (Forward Declarations) |
| Escopo de bloco e Shadowing check | Sim | Sim | Validados estaticamente pelo analisador semântico |
| **Operadores** | | | |
| Aritméticos (`+`, `-`, `*`, `/`, `//`, `%`, `**`) | Sim | Sim | Mapeados para instruções C99 e `pow()` / `fmod()` |
| Comparações (`==`, `!=`, `<`, `<=`, `>`, `>=`) | Sim | Sim | Operadores relacionais nativos |
| Lógicos (`and`, `or`, `not`) | Sim | Sim | Operadores `&&`, `||`, `!` |
| `@` (Multiplicação Matricial Genérica) | Sim | Parcial | No C99 suportado para pesos ternários via GEMV |
| Fatiamento de Vetores `v[start..end]` | Sim | Parcial | Otimizado em loops no C99 |
| **Primitivas Matemáticas e Espectrais** | | | |
| `sqrt`, `exp`, `sin`, `cos`, `ln`, `floor`, `tanh` | Sim | Sim | Mapeadas para `<math.h>` (`sqrt`, `log`, etc.) |
| `rfft`, `irfft`, `power_spectrum` | Sim | Sim | Kernels ANSI C99 nativos embarcados (`nat_rfft`, `nat_irfft`, `nat_power_spectrum`) |
| `to_i64`, `to_f64` | Sim | Sim | Casts de tipo C99 `(int64_t)`, `(double)` |
| `print` (valores, literais, f-strings) | Sim | Sim | Mapeado para `printf` com especificadores de formato |
| **Primitivas BitNet b1.58 e Ternárias** | | | |
| `quantize_ternary`, `quantize_bitnet` | Sim | Sim | Kernels C99 dedicados (`nat_quantize_bitnet`) |
| `pack_ternary_5trits`, `unpack_ternary_5trits` | Sim | Sim | Codificação/Decodificação base-3 densa em C99 |
| `pack_ternary`, `unpack_ternary` (4 trits/byte)| Sim | Sim | Operações bitwise C99 (`nat_pack_ternary`) |
| `tensor_absmean` | Sim | Sim | Redução RMS/AbsMean em C99 |
| `gemv_vec_lut` | Sim | Sim | Kernel streaming Vec-LUT com look-up tables em C99 |
| `mae`, `relative_error`, `pearson_corr`, `cosine_sim` | Sim | Sim | Kernels bare-metal C99 com alta precisão |
| **Modelos, Tokenizadores e Cache** | | | |
| `Tokenizer`, `load_tokenizer`, `encode`, `decode`| Sim | **Não** | Requer tabelas dinâmicas de merges (v0.3 em Python) |
| `load_safetensors`, `save_safetensors` | Sim | **Não** | Parser SafeTensors/JSON dinâmico com mmap |
| `KVCache` dinâmico | Sim | **Não** | Arena dinâmica de memória; estática em v0.4 |
| `turbo_quant`, `turbo_dequant`, `turbo_dot` | Sim | **Não** | Depende de tabelas e engrams dicionários |
| `read_file`, `fill`, `seed` | Sim | Parcial | `fill()` e `seed()` suportados em v0.3.1 |
| **Tratamento de Erros (`Result`)** | | | |
| `Result[str]`, `read_file_safe`, `ok`, `err`, `is_ok`, `is_err`, `unwrap`, `unwrap_or`, `error_message` | Sim | Sim | C99 `NatResult` (carrega apenas `const char*`) |
| `Result[T]` com `T` ≠ `str` (ex.: `Result[i64]`) | Sim | **Não** | Compilador recusa com erro claro; só interpretador por enquanto |
| `?` em função que **não** retorna `Result` (ex.: `main`) | Sim (panic controlado) | Sim (`nat_unwrap_str`, `exit 1`) | Mesma semântica de unwrap com panic nos dois modos |
| `?` com propagação de `Err` em função `-> Result[T]` | Sim | **Não** | Exige early-return; C99 puro não tem statement-expressions. Compilador recusa com erro claro |

---

## 3. Comportamento e Detecção de Limites

Quando um programa Nativa contém código além do subconjunto suportado pelo compilador C99, o backend AOT **falha rápido** durante a transpilação, emitindo uma mensagem diagnóstica clara com número de linha:

```bash
$ nativa build modelo_complexo.nat
modelo_complexo.nat:15: erro: No backend C99, 'for' só é suportado sobre intervalos numéricos (start..end).
```

### Quando usar cada modo:

1. **Use `nativa run` durante:**
   - Prototipagem e experimentação rápida de novas arquiteturas neurais.
   - Carregamento de pesos reais de modelos HuggingFace (`.safetensors`).
   - Tokenização BPE interativa com arquivos `tokenizer.json`.
   - Algoritmos com compressão episódica de KV Cache (`turbo_quant`).

2. **Use `nativa build` durante:**
   - Deploy em servidores de borda (edge/embedded) sem interpretador.
   - Micro-benchmarks de kernels ternários Bare-Metal (Add-Only GEMM, Vec-LUT).
   - Ambientes que exigem binário estático único sem dependências de runtime.
   - Otimização extrema via OpenMP e flags de vetorização SIMD (`-O3 -mavx2`).

---

## 4. Roteiro de Convergência (v0.4+)

Para unificar gradualmente a paridade entre `--run` e `--build`:
- [ ] **Structs C99**: Transpilação de tipos compostos para `typedef struct` C99.
- [ ] **Tokenizador C99 Embutido**: Tokenizador BPE estático em C puro (sem dependência Python).
- [ ] **SafeTensors C99 Parser**: Leitor SafeTensors direto em C via `mmap()` e parser JSON minimalista.
- [ ] **Arena Estática de KV Cache**: Implementação no estilo `llama2.c` com memória contígua pré-alocada.
- [ ] **Backend OpenCL AOT**: Geração automática de kernels `.cl` para execução direta em GPUs heterogêneas.

# Roteiro de Issues Prontas para Abrir no GitHub (Good First Issues)

Copie e cole os títulos e corpos abaixo diretamente na aba **Issues -> New Issue** do seu repositório GitHub (`https://github.com/Benevalterjr/nativa/issues/new`).

---

## Issue 1: [Good First Issue] Implement AVX2 Vectorized Kernel for Ternary GEMV

**Title:**
`[Kernel]: Implement AVX2 vectorized fallback for Ternary GEMV (BitNet b1.58)`

**Labels:**
`good first issue`, `enhancement`, `performance`, `help wanted`

**Body:**
```markdown
### Motivation
Currently, Nativa supports ternary add-only GEMM / GEMV via pure scalar C99 loops and 5-trits-per-byte LUT (`nat_gemv_vec_lut_5trits`). Adding an explicit AVX2 SIMD path for machines with 256-bit vector registers will significantly boost throughput on modern x86_64 CPUs.

### Scope & Tasks
1. Implement an AVX2 kernel (in `tools/` or inline in `nativa.py` AOT emitter) using `_mm256_loadu_si256` / `_mm256_fmadd_ps` or integer sign-masking.
2. Integrate into runtime CPU dispatch (`tools/cpu_dispatch.h`) so it activates automatically when AVX2 is detected.
3. Keep pure C99 scalar fallback for non-AVX systems.
4. Add benchmark comparing scalar vs AVX2 throughput in `tests/`.

### References
- Current scalar ternary matmul: `nat_ternary_matmul_vec` in `nativa.py`
- CPU detection: `tools/cpu_dispatch.h`
```

---

## Issue 2: [Good First Issue] Enhance Parser & Type Error Diagnostics with Source Spans

**Title:**
`[Frontend]: Improve type error and syntax diagnostics with line/column source spans`

**Labels:**
`good first issue`, `enhancement`, `compiler`, `dx`

**Body:**
```markdown
### Motivation
When a syntax or type mismatch error occurs during parsing or semantic analysis, the current error message reports the line number, but displaying a caret `^` pointing to the exact column and printing the surrounding line of source code makes debugging much more user-friendly.

### Scope & Tasks
1. Update `NatError` to format errors in standard compiler style:
   ```text
   Erro semântico [linha 12, col 15]: tipo 'f64' incompatível com 'str'
     12 |     let x: f64 = "texto"
        |                  ^^^^^^^
   ```
2. Track start and end column offsets on AST nodes (`Expr`, `Stmt`).
3. Add unit tests in `TesteSemantico` verifying accurate error positioning.

### References
- Parser error handling: `NatError` class in `nativa.py`
- Semantic visitor: `SemanticAnalyzer` in `nativa.py`
```

---

## Issue 3: [Good First Issue] Create English Beginner's Guide (`GUIA_INICIANTE_EN.md`)

**Title:**
`[Docs]: Create English Beginner's Guide (GUIA_INICIANTE_EN.md)`

**Labels:**
`good first issue`, `documentation`, `help wanted`

**Body:**
```markdown
### Motivation
Nativa has a detailed Portuguese beginner guide (`GUIA_INICIANTE.md`), but international contributors and users need an English version to quickly learn the syntax, execution model, and CLI.

### Scope & Tasks
1. Translate and adapt `GUIA_INICIANTE.md` to `GUIA_INICIANTE_EN.md`.
2. Cover:
   - Installation & virtual environment setup
   - Language syntax (functions, variables, loops, slices, operators, `Result` and `?`)
   - Running via interpreter (`--run`) and building native binaries (`--build`)
   - Running full models (e.g. `models/mini_llm_mha_real.nat`)
3. Link the new guide in `README.md`.
```

---

## Issue 4: [Good First Issue] Basic VS Code Syntax Highlighting Extension for `.nat` Files

**Title:**
`[Tooling]: Create VS Code syntax highlighting extension for .nat files`

**Labels:**
`good first issue`, `tooling`, `dx`

**Body:**
```markdown
### Motivation
Writing `.nat` code currently requires setting VS Code file association to Python. A dedicated syntax highlighting extension provides a native developer experience for users of the language.

### Scope & Tasks
1. Create a minimal VS Code extension in `tools/vscode-nativa/` with:
   - `package.json` with language configuration (`.nat` extension)
   - `syntaxes/nativa.tmLanguage.json` with TextMate grammar for keywords (`fn`, `let`, `const`, `if`, `else`, `while`, `for`, `in`, `return`, `tensor`, `import`), operators (`@`, `?`), types (`f64`, `f32`, `i64`, `t2`, `bool`, `str`, `Result`), and f-strings.
2. Provide a quick installation guide in `tools/vscode-nativa/README.md`.
```

---

## Issue 5: [Good First Issue] Group-Wise BitNet b1.58 Quantization & Perplexity Benchmarking

**Title:**
`[Quantization]: Implement group-wise BitNet b1.58 scaling (group_size=128)`

**Labels:**
`good first issue`, `enhancement`, `quantization`, `ai`

**Body:**
```markdown
### Motivation
Currently, BitNet b1.58 quantization computes a single global scale factor $\gamma = \text{mean}(|W|)$ per weight matrix. In modern LLM quantization (such as BitNet b1.58 2.0 / QAT), computing $\gamma$ per block of 128 elements preserves local dynamic range and reduces quantization noise.

### Scope & Tasks
1. Add `group_size: Optional[int] = 128` parameter to `quantize_bitnet_fn` in `nativa.py`.
2. Store per-group scaling factors and support them in matrix-vector multiplication.
3. Validate perplexity / cosine similarity compared to FP32 baseline on sample layers.
```

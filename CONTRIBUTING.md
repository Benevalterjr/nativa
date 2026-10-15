# Contributing to Nativa / Contribuindo com a Nativa

Thank you for your interest in contributing to Nativa! / Obrigado pelo seu interesse em contribuir com a Nativa!

---

## English

Nativa is an open-source domain-specific language and lightweight runtime designed for efficient AI inference, BitNet b1.58 ternary computing, and local LLM execution. We welcome contributions from developers, researchers, and AI enthusiasts of all experience levels.

### Code of Conduct & Core Philosophy

When contributing to Nativa, keep in mind our core engineering principles:
1. **Lightweight & Portable**: Zero heavy external SDK requirements (no mandatory multi-gigabyte CUDA/OpenCL SDK installations). C runtime code must remain strictly ANSI C99 compliant and compile cleanly on GCC, Clang, and MSVC.
2. **Deterministic & Verifiable**: Any optimization or kernel addition must be grounded in empirical tests with baseline parity verification.
3. **Single Source of Truth**: When extending built-in functions, maintain the strict 1-to-1 correspondence between `BUILTINS` (semantic analyzer) and `BUILTIN_IMPLS` (runtime).

---

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/nativa.git
   cd nativa
   ```

2. **Create a Virtual Environment**
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Optional dependencies for benchmarks and HuggingFace tools:
   # pip install -r requirements-tools.txt
   ```

4. **Run the Test Suite**
   Ensure all existing tests pass before making any modifications:
   ```bash
   python nativa.py --test
   ```

---

### Pull Request Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feat/my-improvement
   ```
2. Make your changes with clear, focused commits.
3. Write automated unit and regression tests in `tests/`.
4. Ensure all unit tests pass:
   ```bash
   python nativa.py --test
   ```
5. If modifying C kernels or the AOT transpiler, verify compilation with both interpreter and native AOT compiler:
   ```bash
   python nativa.py run tests/my_test.nat
   python nativa.py build tests/my_test.nat --run
   ```
6. Push to your fork and submit a Pull Request describing:
   - What problem is being solved.
   - What benchmarks or test cases demonstrate correctness.

---

### Priority Areas for Contributions

- **Kernels & Performance**: AVX2 / AVX-512 SIMD vectorization, OpenCL GPU/CPU acceleration, fused linear operators.
- **Quantization**: Group-wise BitNet b1.58 quantization, outlier preservation, perplexity evaluation suites.
- **Language Frontend**: Rich type error diagnostic locations (line & column spans), enhanced AST checks, structured warning messages.
- **Tooling & DX**: VS Code syntax highlighting extension (`.nat`), basic language server (LSP), formatter.
- **Documentation**: Tutorials, English beginner guides, architectural diagrams.

---

<br>

## Português

A Nativa é uma linguagem de domínio específico de código aberto e runtime de alta eficiência para inferência de IA, computação ternária BitNet b1.58 e execução local de modelos de linguagem. Contribuições de desenvolvedores e pesquisadores são muito bem-vindas!

### Princípios de Engenharia

1. **Portabilidade e Leveza**: Sem dependências pesadas de SDKs proprietários. Todo código C gerado ou incluído em `tools/` deve ser estritamente compatível com ANSI C99 e compilar sem erros no GCC, Clang e MSVC (`cl.exe`).
2. **Fonte Única de Verdade**: Ao adicionar novas funções integradas (builtins), certifique-se de registrar a função tanto no analisador semântico (`BUILTINS`) quanto no interpretador (`BUILTIN_IMPLS`).
3. **Validação Numérica Rigorosa**: Otimizações não devem degradar a precisão ou fidelidade do modelo. Sempre compare com o baseline correspondente.

---

### Passo a Passo para Contribuir

1. Faça um Fork do repositório no GitHub.
2. Crie uma branch para a sua feature:
   ```bash
   git checkout -b feature/minha-melhoria
   ```
3. Instale o ambiente virtual e execute a bateria de testes:
   ```bash
   python nativa.py --test
   ```
4. Adicione testes para novas funcionalidades em `tests/`.
5. Abra o Pull Request descrevendo claramente as alterações e os resultados dos testes.

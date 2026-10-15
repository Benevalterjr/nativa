# Security Policy / Política de Segurança — Nativa

We take the security of Nativa's compiler, runtime, and ecosystem seriously. / Levamos a segurança do compilador, do runtime e do ecossistema da Nativa a sério.

---

## Supported Versions / Versões Suportadas

Only the latest active development versions receive security updates. / Apenas as versões ativas mais recentes recebem correções de segurança.

| Version / Versão | Supported / Suportada |
| :--- | :---: |
| 0.4.x | :white_check_mark: |
| 0.3.x | :white_check_mark: |
| < 0.3 | :x: |

---

## Reporting a Vulnerability / Como Reportar uma Vulnerabilidade

If you discover a security vulnerability in Nativa (compiler, C99 runtime, SafeTensors loader, or .natq4 engine):

1. **Do NOT open a public issue.** Keep details private until a fix is released.
2. **GitHub Private Advisory (Preferred):** Use the official [GitHub Private Vulnerability Reporting](https://github.com/Benevalterjr/nativa/security/advisories/new).
3. **Email:** Alternatively, report directly to `databenti@gmail.com` with the subject `[SECURITY - Nativa]`.

### What to include in your report / O que incluir no relato:
- A clear description of the vulnerability and its potential impact. / Descrição clara da vulnerabilidade e seu impacto.
- Minimal reproducible example (`.nat` script, malicious tensor header, or command sequence). / Exemplo mínimo reproduzível.
- Affected platforms (OS, Python version, compiler used: GCC/Clang/MSVC). / Plataformas afetadas.

### Response Timeline / Prazos de Resposta:
- **Initial acknowledgment:** Within 48 hours. / Confirmação de recebimento em até 48 horas.
- **Triage & Assessment:** Regular updates provided until a fix is published. / Atualizações regulares até a publicação da correção.

---

## Scope / Escopo de Segurança

### In Scope / Em Escopo:
- **Memory Safety in C99 Kernels:** Buffer overflows, out-of-bounds access, or memory leaks in generated C99 code or AOT runtime engines.
- **Untrusted File Parsing:** Malformed SafeTensors headers, corrupted `.natq4` files, or tokenizer deserialization exploits.
- **I/O Safety:** Path traversal vulnerabilities during model or file loading (`read_file_safe`).

### Out of Scope / Fora de Escopo:
- Running untrusted third-party `.nat` scripts without operating-system level sandboxing.
- Issues related to underlying operating system drivers or vendor OpenCL/CPU runtimes.

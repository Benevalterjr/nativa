# Getting Started: Learning Nativa

This guide introduces **Nativa v0.4** through small programs. It is adapted from
the [Portuguese beginner guide](GUIA_INICIANTE.md). Run the commands from the
repository root so relative paths to examples and data resolve correctly.

## 1. Installation and your first program

The core runtime requires Python 3.9+ and NumPy. Create an isolated environment:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell does not allow activation, use `.\.venv\Scripts\python.exe` in
place of `python` without changing the machine's execution policy.

Save this as `hello.nat`:

```nativa
fn main():
    print("Hello, world!")
```

Run it with either supported CLI spelling:

```bash
python nativa.py run hello.nat
python nativa.py hello.nat --run
```

Both execute the interpreter. To install the optional `nativa` command in your
active environment, run `python -m pip install -e .`, then use
`nativa run hello.nat`. Installing `requirements.txt` alone does not install
that command.

`main` is the program entry point. Indentation defines blocks, and `#` starts
a comment. Each complete program below can replace the contents of `hello.nat`.

## 2. Variables, constants, and types

Assignment creates a variable. Type annotations are optional where Nativa can
infer the type; `const` declares an immutable value:

```nativa
const LIMIT: i64 = 10

fn main():
    name: str = "Ada"
    age: i64 = 36
    height: f64 = 1.65
    student: bool = false
    print(f"{name} is {age} years old and {height}m tall.")
    print(LIMIT)
```

Supported type names include:

| Category | Types |
| --- | --- |
| Signed integers | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `u8`, `u16`, `u32`, `u64` |
| Floating point | `f16`, `bf16`, `f32`, `f64` |
| Complex numbers | `c64`, `Complex`, constructed with `Complex(re, im)` |
| Text and logic | `str`, `bool` with `true` / `false` |
| Ternary / BitNet | `t2`, restricted to `-1`, `0`, and `1` |

Strings accept single or double quotes. An f-string such as `f"Age: {age}"`
interpolates expressions; complex values expose `.real` and `.imag`.

## 3. Decisions, functions, and operators

Arithmetic operators are `+`, `-`, `*`, `/`, `%`, `//`, and `**`. Comparisons
use `==`, `!=`, `<`, `<=`, `>`, and `>=`; logic uses `and`, `or`, and `not`.
For tensors, `@` performs matrix multiplication.

```nativa
fn classify(score: f64) -> str:
    if score >= 7.0:
        return "Pass"
    elif score >= 5.0:
        return "Retake"
    else:
        return "Fail"

fn main():
    print(classify(8.5))
```

Function parameters can carry types, and `->` introduces the return type.

## 4. Loops

The end of `start..end` is excluded. This program prints 1 through 5, then
counts down from 3:

```nativa
fn main():
    for number in 1..6:
        print(number)

    counter = 3
    while counter > 0:
        print(counter)
        counter -= 1
```

Use `break` to leave a loop and `continue` to skip to its next iteration.

## 5. Lists and slices

Lists use brackets and zero-based indexing. `len` returns their length.
Slices use Nativa's `start..end` syntax, with the end excluded:

```nativa
fn total(values: [i64]) -> i64:
    result = 0
    for value in values:
        result += value
    return result

fn main():
    numbers = [10, 20, 30, 40]
    print(f"Sum: {total(numbers)}")
    print(f"Length: {len(numbers)}")
    print(numbers[0])
    middle = numbers[1..3]
    print(middle)
```

The sum is 100, the length is 4, the first item is 10, and `middle` contains
20 and 30. See [the buffer and slice example](tests/test_buffer_e_fatias.nat)
for slices of runtime-generated arrays.

## 6. Tensors and AI operations

```nativa
fn main():
    A = tensor([[1.0, 2.0], [3.0, 4.0]])
    B = tensor([[2.0], [1.0]])
    C = A @ B
    print(C)
    print(f"Shape: {shape(C)}")
```

The result has shape `[2, 1]` and values 4 and 10. Useful built-ins include:

- Creation: `zeros(shape)`, `ones(shape)`, `randn(shape)`.
- Transformation: `shape(t)`, `reshape(t, new_shape)`, `transpose(t)`.
- Reduction: `sum(t)`, `mean(t)`, `max(t)`, `min(t)`, `argmax(t)`.
- AI operations: `relu(t)`, `sigmoid(t)`, `tanh(t)`, `rmsnorm(t, w)`,
  `softmax(t)`, `sdpa(q, k, v)`.

## 7. Complex numbers and Fourier analysis

```nativa
fn main():
    first = Complex(3.0, 4.0)
    second = Complex(1.0, 2.0)
    combined = first + second
    print(f"Complex sum: {combined.real} + {combined.imag}i")

    signal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    spectrum = rfft(signal)
    print(f"Frequency bins: {len(spectrum)}")
    power = power_spectrum(spectrum)
    print(f"DC power: {power[0]}")
    reconstructed = irfft(spectrum)
    print(f"Reconstructed samples: {len(reconstructed)}")
```

`rfft` returns `N // 2 + 1` frequency bins for this real signal.
`power_spectrum` computes squared magnitudes, and `irfft` reconstructs the
signal. These functions and `Complex` are also accessible through `math`,
for example `math.rfft(signal)`; the C99 backend supplies native implementations.

## 8. Errors: `Result` and `?`

Fallible functions can return `Result[T]`, constructed with `ok(value)` or
`err(message)`. This example uses the repository's README as an existing file
and handles a missing configuration file without requiring extra setup:

```nativa
fn read_config(path: str) -> Result[str]:
    text = read_file_safe(path)?
    return ok(text)

fn main():
    result: Result[str] = read_file_safe("README.md")
    text = unwrap_or(result, "")
    print(f"Length: {len(text)}")
    if is_ok(result):
        print("Read succeeded")

    config = read_config("missing-beginner-config.txt")
    if is_err(config):
        print(f"Read failed: {error_message(config)}")
```

- `is_ok` / `is_err` inspect the result; `unwrap_or` supplies a fallback.
- `unwrap` extracts a value but panics on an error.
- `?` requires a `Result[T]` and extracts its `T` value. In an interpreted
  function returning `Result`, an error returns immediately to the caller.
  In a function that does not return `Result`, an error causes a controlled panic.

**C99 backend limitation:** native builds currently support only `Result[str]`.
There, `?` acts as unwrap with panic (`exit 1`). Error propagation with `?`
inside a `Result`-returning function, as in `read_config` above, is supported
only by the interpreter; the compiler rejects it, along with other Result
payload types. Use `--run` for this complete example.

## 9. Built-in examples and full models

The `@` prefix selects an example embedded in Nativa:

```bash
python nativa.py run @riemann
python nativa.py run @rede
python nativa.py run @ia_tensor
python nativa.py run @ternario
```

These demonstrate Riemann zeta zeros, perceptron training, tensor operations
and autodiff, and ternary computation respectively.

Run the full [mini LLM example](models/mini_llm_mha_real.nat) from the repository
root, using either spelling:

```bash
python nativa.py run models/mini_llm_mha_real.nat
python nativa.py models/mini_llm_mha_real.nat --run
```

It initializes random weights and demonstrates attention, a KV cache, and
sampling. Its generated text is not a trained model's quality benchmark.
The [full Qwen example](models/qwen2.5_completo.nat) instead needs model
weights in `data/`; large model files are not all committed. See the
[README](README.md) and [tools](tools/) before downloading or converting weights.
The optional `requirements-tools.txt` dependencies are for those tools, not
for the small language examples above.

## 10. Build a native executable

For a small first build, restore the hello-world program in `hello.nat`.
Install a C compiler with OpenMP support (GCC, Clang, or MSVC), then run:

```bash
python nativa.py build hello.nat
# Equivalent flag form:
python nativa.py hello.nat --build
```

The compiler writes `hello.c` and a native executable next to the input file.
Run `./hello` on Linux/macOS, or `.\hello.exe` in Windows PowerShell. To compile
and immediately run it, use `python nativa.py build hello.nat --run`.

The interpreter uses NumPy; the AOT path transpiles to C99 and invokes an
available compiler. Check compiler/OpenMP availability on your platform:
successful interpretation does not establish that native compilation works,
and not every interpreter feature is supported by the C99 backend.

## 11. Hardware discovery and validation

```bash
python nativa.py --devices
python nativa.py opencl
python nativa.py --test
```

The first two commands report detected compute devices and OpenCL information.
OpenCL acceleration depends on an available driver/runtime; it is not required
for basic interpreter examples. The last command runs the built-in validation
suite; review its failures and skips, since native checks depend on the local
C compiler. The test count can change as the project grows.

## 12. Self-contained single-executable packaging (`nativa bundle`)

Following the single-binary distribution concept (similar to *llamafile*),
Nativa can bundle the compiled C99 inference engine and quantized model
weights (`.natq4`) into a single standalone executable (`.exe` on Windows or
an ELF binary on Linux).

> **Prerequisites:**
> Large `.natq4` model weights and compiled C engine binaries are not committed to Git.
> Before creating a bundle or starting the local API server:
> ```bash
> # 1. Compile the C99 inference engine (if not already compiled in tools/)
> # Linux / macOS:
> gcc -O3 -fopenmp -mavx2 tools/nativa_engine.c -o tools/nativa_engine -lm
> # Windows PowerShell:
> gcc -O3 -fopenmp -mavx2 tools/nativa_engine.c -o tools/nativa_engine.exe
>
> # 2. Download the official Qwen3 quantized weights (release v0.1.0)
> curl -L -o data/qwen3_0.6b.natq4 https://github.com/Benevalterjr/nativa/releases/download/v0.1.0/qwen3_0.6b.natq4
> ```

```bash
# Package a model into a standalone binary
python nativa.py bundle data/qwen3_0.6b.natq4 -o qwen3-assistant.exe

# Inspect bundle metadata, trailer, and alignment
python nativa.py bundle --inspect qwen3-assistant.exe

# Run directly without Python, external DLLs, or separate weight files
./qwen3-assistant.exe "Explain the Nativa runtime in one paragraph."
```

### Key properties of Nativa bundles:
- **Zero runtime dependencies:** Runs offline without Python, PyTorch, or
  separate model files.
- **Instant memory-mapped loading:** The embedded weights are strictly aligned
  to 64 KB page boundaries, enabling zero-copy execution via `mmap` on POSIX
  or `MapViewOfFile` on Windows.
- **64-byte trailer:** The binary reads its own metadata and model offset
  from a fixed 64-byte trailer at the end of the file (`NATBNDL1` magic).

## 13. OpenAI-compatible API server and RAG (`tools/api_server.py`)

For integrating Nativa models with existing workflows, frontends, or tools
like LangChain, the repository provides an HTTP server based on FastAPI:

```bash
# Start the API server on port 8000
python tools/api_server.py --port 8000
```

> **Docker alternative:**
> To run the API without manually compiling the C engine or downloading weights:
> ```bash
> docker build -t nativa-api .
> docker run -p 8000:8000 nativa-api
> ```

### Supported endpoints:
- **Chat completions (`/v1/chat/completions`):** Compatible with the OpenAI
  API format, supporting real-time Server-Sent Events (SSE) token streaming
  and optional reasoning / thinking tokens (`enable_thinking`).
- **Grounding and RAG (`/v1/rag/answer`):** Accepts a context string and a
  question, returning answers grounded on the provided text.
- **Interactive documentation:** Visit `http://localhost:8000/docs` for the
  Swagger UI and pre-filled request examples.

## 14. Official Performance & Perplexity Benchmarks (`nativa bench`)

To evaluate and audit inference throughput (tokens/second), latency (TTFT), perplexity (PPL on WikiText-2), and RAM footprint (RSS):

```bash
# Run the complete automated benchmark suite
python nativa.py bench --all

# Or evaluate specific metrics
python nativa.py bench --threads 1,2,4,8
python nativa.py bench --ppl --ppl-tokens 512
```

See the full benchmark reports:
- [`BENCHMARKS_EN.md`](BENCHMARKS_EN.md) (Official Benchmark Report in English)
- [`BENCHMARKS.md`](BENCHMARKS.md) (Relatório Oficial em Português)

## Next steps

- [Benchmarks (EN)](BENCHMARKS_EN.md): Official benchmark suite with live empirical metrics.
- [README](README.md): architecture and supported model workflows.
- [Portuguese Guide](GUIA_INICIANTE.md): original Portuguese beginner guide.
- [Models](models/): Mini LLM, Transformer, and Qwen examples.
- [Tools](tools/): native inference, bundling, quantization, and research utilities.
- [Contributing](CONTRIBUTING.md): development setup and contribution checks.

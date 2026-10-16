# Guia de Início: Aprendendo a Linguagem Nativa

Este guia apresenta os recursos da linguagem **Nativa (v0.4)** através de exemplos práticos e diretos.

---

## 1. Instalação e Primeiro Programa

O Nativa requer apenas Python 3.9+ e NumPy instalados. Na pasta do projeto:

```powershell
python -m pip install -r requirements.txt
```

Crie um arquivo chamado `ola.nat`:

```nativa
fn main():
    print("Olá, mundo!")
```

### Como Executar

Você pode executar diretamente com a CLI do Nativa no PowerShell / Prompt de Comando ou chamando o Python:

```powershell
# Usando a CLI do Nativa (PowerShell / CMD)
nativa run ola.nat

# Ou usando o Python diretamente
python nativa.py run ola.nat
```

> **Nota:** `main` é o ponto de entrada obrigatório. Os blocos de código são definidos pela indentação (como em Python). Comentários começam com `#`.

---

## 2. Variáveis, Constantes e Tipos

Uma atribuição cria uma variável. Você também pode declarar seu tipo explicitamente. Use `const` para valores imutáveis:

```nativa
const LIMITE: i64 = 10

fn main():
    nome: str = "Ada"
    idade: i64 = 36
    altura: f64 = 1.65
    estudante: bool = false

    print(f"{nome} tem {idade} anos e mede {altura}m.")
```

### Tipos Nativos Suportados
* **Inteiros com sinal:** `i8`, `i16`, `i32`, `i64`
* **Inteiros sem sinal:** `u8`, `u16`, `u32`, `u64`
* **Ponto Flutuante:** `f16`, `bf16`, `f32`, `f64`
* **Números Complexos:** `c64`, `Complex` (campos `.real` e `.imag`, construtor `Complex(re, im)`)
* **Texto e Lógica:** `str`, `bool` (`true` / `false`)
* **Computação Ternária (BitNet b1.58):** `t2` (valores restritos a `-1`, `0` e `1`)

Strings aceitam aspas simples ou duplas. Para interpolar variáveis, use f-strings:

```nativa
valor = 42
print(f"O resultado é: {valor}")
```

---

## 3. Decisões e Funções

Nativa oferece operadores aritméticos (`+`, `-`, `*`, `/`, `%`, `//`, `**`), relacionais (`==`, `!=`, `<`, `<=`, `>`, `>=`) e lógicos (`and`, `or`, `not`). O operador `@` é nativo para multiplicação matricial de tensores.

```nativa
fn classificar(nota: f64) -> str:
    if nota >= 7.0:
        return "Aprovado"
    elif nota >= 5.0:
        return "Recuperação"
    else:
        return "Reprovado"

fn main():
    print(classificar(8.5))
```

Uma função declara parâmetros tipados e indica o tipo de retorno com `->`.

---

## 4. Laços de Repetição (`for` e `while`)

O laço `for` itera sobre intervalos ou listas. No intervalo `inicio..fim`, o limite final não é incluído:

```nativa
fn main():
    # Imprime de 1 a 5
    for numero in 1..6:
        print(numero)
```

Use `while` quando a repetição depender de uma condição:

```nativa
fn main():
    contador = 3
    while contador > 0:
        print(contador)
        contador -= 1
```

Use `break` para interromper o laço e `continue` para saltar para a próxima iteração.

---

## 5. Listas e Fatiamento

Listas são declaradas com colchetes e indexadas a partir de zero. A função `len` retorna a quantidade de itens:

```nativa
fn soma(valores: [i64]) -> i64:
    total = 0
    for valor in valores:
        total += valor
    return total

fn main():
    numeros = [10, 20, 30, 40]
    print(f"Soma: {soma(numeros)}")
    print(f"Tamanho: {len(numeros)}")
```

---

## 6. Tensores e Operações de IA

A Linguagem Nativa possui suporte de primeira classe a tensores densos e operações matriciais aceleradas:

```nativa
fn main():
    # Criação de tensores
    A = tensor([[1.0, 2.0], [3.0, 4.0]])
    B = tensor([[2.0], [1.0]])

    # Multiplicação matricial com o operador @
    C = A @ B
    print(C)
    print(f"Formato: {shape(C)}")
```

### Funções Built-in de Tensores:
* **Geração:** `zeros(shape)`, `ones(shape)`, `randn(shape)`
* **Transformação:** `shape(t)`, `reshape(t, novo_shape)`, `transpose(t)`
* **Redução:** `sum(t)`, `mean(t)`, `max(t)`, `min(t)`, `argmax(t)`
* **Operadores de IA:** `relu(t)`, `sigmoid(t)`, `tanh(t)`, `rmsnorm(t, w)`, `softmax(t)`, `sdpa(q, k, v)`

---

## 7. Análise Espectral e Números Complexos (Fourier)

O Nativa v0.4 inclui suporte de primeira classe a números complexos (`Complex`) e Transformadas de Fourier para análise espectral de sinais, filtros e inspeção de frequências em tensores de ativações:

```nativa
fn main():
    # 1. Numeros complexos e aritmetica
    c1 = Complex(3.0, 4.0)
    c2 = Complex(1.0, 2.0)
    soma = c1 + c2
    print(f"Soma complexa: {soma.real} + {soma.imag}i")

    # 2. Transformada Discreta de Fourier 1D (rfft)
    sinal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    spec = rfft(sinal)
    print(f"Qtd frequencias (N//2 + 1): {len(spec)}")

    # 3. Densidade Espectral de Potencia (|X|^2)
    ps = power_spectrum(spec)
    print(f"Potencia DC: {ps[0]}")

    # 4. Transformada Inversa (irfft) com reconstrucao exata
    rec = irfft(spec)
    print(f"Sinal reconstruido: {len(rec)} amostras")
```

As funções espectrais (`rfft`, `irfft`, `power_spectrum`, `Complex`) também estão disponíveis através do módulo `math` (`math.rfft(...)`) e são compiladas para código nativo C99 puro sem dependências externas.

---

## 8. Tratamento de Erros com `Result` e o Operador `?`

Funções que podem falhar devolvem um `Result[T]` (ou `Ok(valor)` ou `Err(mensagem)`) em vez de abortar o programa:

```python
fn ler_config(caminho: str) -> Result[str]:
    texto = read_file_safe(caminho)?   # se for Err, devolve o Err ao chamador
    return ok(texto)

fn main():
    r: Result[str] = read_file_safe("arquivo.txt")
    s = unwrap_or(r, "")               # valor padrão se der erro
    print(f"Tamanho: {len(s)}")
    if is_ok(r):
        print(unwrap(r))               # unwrap: aborta se for Err

    cfg = ler_config("config.txt")
    if is_err(cfg):
        print(f"Falhou: {error_message(cfg)}")
```

Regras do `?`:
* Só pode ser aplicado a um `Result[T]` — o resultado da expressão é `T` (verificado em tempo de compilação).
* Dentro de uma função que retorna `Result`, um `Err` é **propagado** (a função retorna o `Err` imediatamente).
* Em funções que **não** retornam `Result` (como `main`), um `Err` gera um **panic controlado** com a mensagem do erro.

> **Compilação nativa (`nativa build`):** por enquanto o backend C99 suporta apenas `Result[str]`, e `?` funciona como unwrap com panic (`exit 1`). A propagação de `Err` com `?` dentro de funções que retornam `Result` e `Result` de outros tipos rodam **só no interpretador** (`nativa run`); o compilador recusa esses casos com uma mensagem clara.

---

## 9. Exemplos Prontos Embutidos

Você pode rodar exemplos completos incluídos diretamente no Nativa usando o prefixo `@`:

```powershell
nativa run @riemann      # Zeros não-triviais da Função Zeta de Riemann
nativa run @rede         # Treinamento de Rede Neural Perceptron
nativa run @ia_tensor    # Operações matriciais avançadas e Autodiff
nativa run @ternario     # Computação Ternária BitNet 1.58b pura
```

---

## 10. Compilação AOT para Binário Nativo (.exe)

O Nativa permite compilar programas `.nat` diretamente para código de máquina C99 de alta velocidade:

```powershell
# Compila e gera o binário executável
nativa build ola.nat

# Executa o binário nativo gerado
./ola.exe
```

A compilação utiliza automaticamente o compilador C disponível no seu sistema (MSVC, GCC ou Clang). O código gerado é **universal** e não exige hardwares proprietários.

---

## 11. Descoberta de Hardware e Testes

### Inspecionar Dispositivos (CPU & OpenCL)
Para saber quais placas de vídeo e processadores estão disponíveis para aceleração:

```powershell
nativa opencl
# ou: nativa --devices
```

### Executar a Suíte de Validação
Para rodar a bateria com os **mais de 100 testes automatizados** do compilador e runtime:

```powershell
nativa test
```

---

## 12. Empacotamento em Binário Único Autocontido (`nativa bundle`)

Inspirado no conceito de executáveis portáteis de IA (como o *llamafile*), o Nativa permite empacotar a engine de inferência C99 e os pesos quantizados (`.natq4`) em um **único arquivo executável autossuficiente** (`.exe` no Windows ou binário ELF no Linux).

> **Pré-requisitos:**
> Como os arquivos de pesos são grandes e a engine C99 é compilada localmente, certifique-se de compilar a engine e obter o modelo antes do empacotamento:
> ```powershell
> # 1. Compilar o motor C99 (se ainda não existir em tools/)
> gcc -O3 -fopenmp -mavx2 tools/nativa_engine.c -o tools/nativa_engine.exe
>
> # 2. Baixar os pesos oficiais do Qwen3 (release v0.1.0)
> curl -L -o data/qwen3_0.6b.natq4 https://github.com/Benevalterjr/nativa/releases/download/v0.1.0/qwen3_0.6b.natq4
> ```

```powershell
# Cria um executável autocontido com o modelo embutido
nativa bundle data/qwen3_0.6b.natq4 -o qwen3-assistente.exe

# Inspeciona metadados, trailer e alinhamento do bundle
nativa bundle --inspect qwen3-assistente.exe

# Executa diretamente sem depender de Python, DLLs ou pesos externos
./qwen3-assistente.exe "Explique o que é a linguagem Nativa em um parágrafo."
```

### Características do Bundle Nativo:
- **Zero Dependências em Runtime:** Não requer Python, PyTorch ou arquivos adicionais; o executável carrega os tensores diretamente da própria imagem.
- **Carregamento Instantâneo via `mmap`:** O modelo é alinhado estritamente em limites de página de 64 KB, permitindo mapeamento de memória zero-copy (`MapViewOfFile` no Windows / `mmap` no Linux).
- **Trailer de 64 Bytes com Magic Check:** O executável localiza automaticamente o deslocamento dos tensores e metadados no final do arquivo (`NATBNDL1`).

---

## 13. Servidor de API OpenAI e RAG (`tools/api_server.py`)

Para integrar o Nativa com ecossistemas existentes (LangChain, LlamaIndex, interfaces web ou clientes OpenAI), o projeto oferece um servidor FastAPI de alta performance:

```powershell
# Inicia o servidor local na porta 8000 (utiliza a engine C e o modelo em data/)
python tools/api_server.py --port 8000
```

> **Alternativa com Docker:** Se preferir não compilar manualmente nem baixar o modelo à parte, o Dockerfile do projeto automatiza todo o setup e download:
> ```bash
> docker build -t nativa-api .
> docker run -p 8000:8000 nativa-api
> ```

### Principais Endpoints Disponíveis:
- **OpenAI Chat Completions (`/v1/chat/completions`):** Suporta streaming de tokens em tempo real via Server-Sent Events (SSE) e modo de raciocínio profundo (`enable_thinking`).
- **Endpoint Especializado em RAG (`/v1/rag/answer`):** Recebe contexto textual e uma pergunta, sintetizando respostas fundamentadas de baixa latência em CPU.
- **Documentação Interativa (Swagger):** Acesse `http://localhost:8000/docs` para testar as requisições interativamente via navegador.

---

## Próximos Passos

Após este guia, explore:
- [`README.md`](README.md): Documentação aprofundada da arquitetura e dos modelos suportados.
- [`GUIA_INICIANTE_EN.md`](GUIA_INICIANTE_EN.md): English version of this guide.
- [`models/`](models/): Implementações completas de modelos de linguagem (Mini LLM, Transformers e Qwen2.5).
- [`tools/`](tools/): Motores de inferência nativa em C99, empacotador de bundles e utilitários de quantização.

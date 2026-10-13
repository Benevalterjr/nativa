# Guia de início: aprendendo Nativa

Este guia apresenta os recursos básicos da linguagem Nativa com exemplos executados pelo interpretador do projeto (`nativa.py`). Salve cada exemplo em um arquivo com extensão `.nat`.

## 1. Instalação e primeiro programa

O runtime precisa de Python e NumPy. Na pasta do projeto, instale a dependência e crie `ola.nat`:

```powershell
python -m pip install -r requirements.txt
```

```nativa
fn main():
    print("Olá, mundo!")
```

Execute o programa com:

```powershell
python nativa.py ola.nat --run
```

`main` é o ponto de entrada. Os blocos são definidos pela indentação, como em Python. Comentários começam com `#`.

## 2. Variáveis, constantes e tipos

Uma atribuição cria uma variável. Você também pode declarar seu tipo explicitamente. Use `const` para valores que não devem ser reatribuídos:

```nativa
const LIMITE: i64 = 10

fn main():
    nome: str = "Ada"
    idade: i64 = 36
    altura: f64 = 1.65
    estudante: bool = false

    print(f"{nome} tem {idade} anos.")
```

Os tipos numéricos incluem `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `bf16`, `f32` e `f64`. Também há `str`, `bool` e o tipo ternário `t2` (valores `-1`, `0` e `1`). Anotações ajudam o analisador semântico a encontrar incompatibilidades antes da execução.

Strings aceitam aspas simples ou duplas. Para interpolar valores, use uma f-string:

```nativa
valor = 7
print(f"O valor é {valor}.")
```

## 3. Operações e decisões

Nativa oferece operações aritméticas (`+`, `-`, `*`, `/`, `%`, `//`, `**`), comparações (`==`, `!=`, `<`, `<=`, `>`, `>=`) e operadores booleanos (`and`, `or`, `not`). O operador `@` é usado para multiplicação matricial.

```nativa
fn classificar(nota: f64) -> str:
    if nota >= 7.0:
        return "aprovado"
    else:
        return "precisa estudar"

fn main():
    print(classificar(8.5))
```

Uma função declara parâmetros tipados e pode indicar seu tipo de retorno depois de `->`. Use `return` para devolver um valor.

## 4. Repetições

O laço `for` percorre uma lista ou um intervalo. Em `inicio..fim`, o limite final não é incluído:

```nativa
fn main():
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

`break` encerra o laço atual e `continue` pula para a próxima repetição.

## 5. Funções e listas

Separe uma tarefa em uma função para poder chamá-la de vários lugares. Listas usam colchetes, índices começam em zero e `len` retorna seu tamanho:

```nativa
fn soma(valores: [i64]) -> i64:
    total = 0
    for valor in valores:
        total += valor
    return total

fn main():
    numeros = [4, 8, 15]
    print(soma(numeros))
    print(f"Quantidade: {len(numeros)}")
```

Exemplo recursivo, calculando Fibonacci:

```nativa
fn fibonacci(n: i64) -> i64:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fn main():
    for i in 0..10:
        print(fibonacci(i))
```

## 6. Matemática e tensores

Funções matemáticas comuns estão disponíveis como built-ins. O módulo `math` também pode ser importado:

```nativa
import math

fn main():
    print(sqrt(81.0))
    print(math.cos(0.0))
```

Para trabalhar com matrizes, use `tensor` e `@`:

```nativa
fn main():
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[2.0], [1.0]])
    resultado = a @ b
    print(resultado)
```

Há também funções para criar tensores (`zeros`, `ones`, `randn`), inspecionar e transformar dados (`shape`, `reshape`, `transpose`, `sum`, `mean`) e para operações de IA, como `rmsnorm`, `silu`, `softmax` e `sdpa`.

## 7. SafeTensors e recursos de IA

O runtime inclui funções para listar e carregar tensores de arquivos SafeTensors:

```nativa
fn main():
    nomes = safetensors_keys("pesos.safetensors")
    print(nomes)
    peso = load_safetensors("pesos.safetensors", nomes[0])
    print(shape(peso))
```

O arquivo precisa existir no caminho indicado ou no diretório `data/`. O projeto também oferece recursos de tokenização, KV cache e quantização ternária/BitNet; consulte os exemplos em `models/` e `tests/` para ver esses recursos em contexto.

## 8. Verificar e compilar programas

O CLI pode analisar o programa sem executá-lo e mostrar tokens ou a AST:

```powershell
python nativa.py ola.nat
python nativa.py ola.nat --tokens
python nativa.py ola.nat --ast
```

Para executar os testes embutidos no projeto:

```powershell
python nativa.py --test
```

Para emitir código C ou compilar um executável nativo, use `--emit-c` ou `--build`:

```powershell
python nativa.py ola.nat --emit-c
python nativa.py ola.nat --build
```

A compilação nativa requer um compilador C compatível disponível no sistema. O interpretador é a forma mais direta de começar. Nem todos os recursos de tensores e IA necessariamente têm suporte equivalente no transpiler C.

## 9. Dicas para resolver erros

- Confira a indentação: o corpo de uma função, condição ou laço precisa estar recuado.
- Declare os tipos dos parâmetros das funções, por exemplo `fn dobro(x: i64) -> i64:`.
- Verifique se os tipos combinam; o analisador informa o arquivo e a linha do problema.
- Um aviso sobre variável não usada geralmente significa que um nome foi digitado ou usado de forma diferente do esperado.
- Comece executando exemplos pequenos com `--run` antes de explorar modelos grandes.

## Próximos passos

Depois deste guia, explore `tests/riemann.nat` para um exemplo numérico e os programas em `models/` para exemplos de tensores, tokenização e modelos de linguagem. O [README](README.md) descreve a arquitetura do projeto e suas dependências.

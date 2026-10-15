#!/usr/bin/env python3
"""
nativa.py — Linguagem Nativa (v0.3 - AI, Data & Ternary Edition)
Lexer, parser, analisador semântico e runtime para IA e computação em CPU/GPU.

Uso:
    python nativa.py programa.nat            # analisa (sintaxe + semântica)
    python nativa.py programa.nat --run      # executa o programa
    python nativa.py programa.nat --ast      # imprime a AST
    python nativa.py programa.nat --tokens   # imprime os tokens
    python nativa.py @ternario --run         # exemplos: @exemplo @riemann @rede @ia_tensor @ternario
    python nativa.py --test                  # roda a suíte de testes completa

Decisões de semântica e recursos para IA e Computação Ternária:
  - Suporte a tensores e precisões de IA: f32, f16, bf16, i32, i8, u8, tensor[T].
  - Computação Ternária de 1.58 bits nativa: tipos t2 e ternary (valores em {-1, 0, +1}).
  - Matmul Ternário Livre de Multiplicações: tensor[t2] @ tensor[f32] usa kernel Add-only GEMM.
  - Primitivas ternárias: quantize_ternary(x, threshold), pack_ternary(x), unpack_ternary(bytes, n).
  - Operador nativo de multiplicação matricial @ e atribuição composta @=.
  - Fatiamento contíguo nativo em listas, strings e tensores: v[start..end] e v[start..=end].
  - Funções de IA embutidas: matmul, relu, sigmoid, tanh, sum, mean, max, min,
    argmax, zeros, ones, shape, reshape, transpose, tensor, device.
  - len polimórfico suportando listas, tensores e strings.
  - Broadcasting elemento a elemento acelerado em CPU via vetores contíguos (NumPy).
  - Escopo de bloco, proibição de shadowing e sugestão de erros de digitação (fuzzy matching).
"""

from __future__ import annotations

import argparse
import ctypes
import difflib
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import unicodedata
import unittest
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Union

import numpy as np

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Erros
# ----------------------------------------------------------------------------

class NatError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def pretty(self, source: str, filename: str = "<entrada>") -> str:
        lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        text = lines[self.line - 1] if 1 <= self.line <= len(lines) else ""
        caret = " " * max(self.col - 1, 0) + "^"
        return (f"{filename}:{self.line}:{self.col}: erro: {self.message}\n"
                f"    {text}\n    {caret}")


# ----------------------------------------------------------------------------
# Tokens e lexer
# ----------------------------------------------------------------------------

@dataclass
class Token:
    kind: str    # NAME KW INT FLOAT STRING FSTRING OP NEWLINE INDENT DEDENT EOF
    value: Any
    line: int
    col: int

    def __repr__(self) -> str:
        return f"{self.kind}({self.value!r})@{self.line}:{self.col}"


KEYWORDS = {
    "fn", "const", "if", "elif", "else", "for", "in", "while", "return",
    "break", "continue", "pass", "import", "and", "or", "not", "true", "false",
}

OPS3 = {"..=", "**=", "//="}
OPS2 = {"->", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=",
        "**", "//", "..", "@="}
OPS1 = set("+-*/%<>=()[],:.@")
CLOSERS = {")": "(", "]": "["}

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0",
           "\\": "\\", '"': '"', "'": "'"}


def unescape(s: str, line: int, col: int) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            e = s[i + 1]
            if e not in ESCAPES:
                raise NatError(f"sequência de escape inválida '\\{e}'", line, col)
            out.append(ESCAPES[e])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(source: str, first_line: int = 1) -> list[Token]:
    src = source.replace("\r\n", "\n").replace("\r", "\n")
    n = len(src)
    toks: list[Token] = []
    indents = [0]
    brackets: list[tuple[str, int, int]] = []
    i = 0
    line = first_line
    line_start = 0
    at_line_start = True

    def err(msg: str, pos: Optional[int] = None):
        p = i if pos is None else pos
        raise NatError(msg, line, p - line_start + 1)

    def add(kind: str, value: Any, pos: int):
        toks.append(Token(kind, value, line, pos - line_start + 1))

    while i < n:
        if at_line_start and not brackets:
            j = i
            while j < n and src[j] == " ":
                j += 1
            if j < n and src[j] == "\t":
                err("tabs não são permitidos na indentação; use espaços", j)
            if j >= n:
                i = j
                break
            if src[j] == "\n":
                i = j + 1
                line += 1
                line_start = i
                continue
            if src[j] == "#":
                while j < n and src[j] != "\n":
                    j += 1
                i = j
                continue
            width = j - i
            if width > indents[-1]:
                indents.append(width)
                add("INDENT", width, j)
            else:
                while width < indents[-1]:
                    indents.pop()
                    add("DEDENT", width, j)
                if width != indents[-1]:
                    err("indentação inconsistente com nenhum bloco anterior", j)
            i = j
            at_line_start = False
            continue

        c = src[i]
        if c == " ":
            i += 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "\n":
            if not brackets:
                add("NEWLINE", None, i)
                at_line_start = True
            i += 1
            line += 1
            line_start = i
            continue
        if c == "\\" and i + 1 < n and src[i + 1] == "\n":
            i += 2
            line += 1
            line_start = i
            continue
        if c == "\t":
            err("tabs só são permitidos dentro de strings e comentários")

        if c.isdigit():
            j = i
            while j < n and (src[j].isdigit() or src[j] == "_"):
                j += 1
            is_float = False
            if j + 1 < n and src[j] == "." and src[j + 1].isdigit():
                is_float = True
                j += 1
                while j < n and (src[j].isdigit() or src[j] == "_"):
                    j += 1
            if j < n and src[j] in "eE":
                k = j + 1
                if k < n and src[k] in "+-":
                    k += 1
                if k < n and src[k].isdigit():
                    is_float = True
                    j = k
                    while j < n and src[j].isdigit():
                        j += 1
            text = src[i:j].replace("_", "")
            add("FLOAT" if is_float else "INT",
                float(text) if is_float else int(text), i)
            i = j
            continue

        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            if word == "f" and j < n and src[j] in "\"'":
                start, q = i, src[j]
                k = j + 1
                while k < n and src[k] != q:
                    if src[k] == "\n":
                        err("f-string não terminada", start)
                    if src[k] == "\\":
                        k += 1
                    k += 1
                if k >= n:
                    err("f-string não terminada", start)
                add("FSTRING", src[j + 1:k], start)
                i = k + 1
                continue
            add("KW" if word in KEYWORDS else "NAME", word, i)
            i = j
            continue

        if c in "\"'":
            k = i + 1
            while k < n and src[k] != c:
                if src[k] == "\n":
                    err("string não terminada")
                if src[k] == "\\":
                    k += 1
                k += 1
            if k >= n:
                err("string não terminada")
            value = unescape(src[i + 1:k], line, i - line_start + 1)
            add("STRING", value, i)
            i = k + 1
            continue

        for size, table in ((3, OPS3), (2, OPS2), (1, OPS1)):
            op = src[i:i + size]
            if len(op) == size and op in table:
                break
        else:
            err(f"caractere inesperado {c!r}")

        if op in ("(", "["):
            brackets.append((op, line, i - line_start + 1))
        elif op in CLOSERS:
            if not brackets or brackets[-1][0] != CLOSERS[op]:
                err(f"'{op}' sem abertura correspondente")
            brackets.pop()

        add("OP", op, i)
        i += size

    if brackets:
        ch, l, c_ = brackets[-1]
        raise NatError(f"'{ch}' não foi fechado", l, c_)

    if not at_line_start:
        add("NEWLINE", None, n)
    while len(indents) > 1:
        indents.pop()
        add("DEDENT", 0, n)
    add("EOF", None, n)
    return toks


# ----------------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------------

@dataclass(kw_only=True)
class Node:
    line: int = field(default=0, compare=False, repr=False)

# --- tipos ---
@dataclass
class TypeName(Node):
    name: str

@dataclass
class TypeList(Node):
    elem: "TypeExpr"

@dataclass
class TypeTensor(Node):
    elem: "TypeExpr"

TypeExpr = Union[TypeName, TypeList, TypeTensor]

# --- expressões ---
@dataclass
class Int(Node):
    value: int

@dataclass
class Float(Node):
    value: float

@dataclass
class Str(Node):
    value: str

@dataclass
class FormatPart(Node):
    expr: "Expr"
    spec: Optional[str]

@dataclass
class FString(Node):
    parts: list      # str | FormatPart

@dataclass
class Bool(Node):
    value: bool

@dataclass
class Name(Node):
    id: str

@dataclass
class ListLit(Node):
    items: list

@dataclass
class Unary(Node):
    op: str          # '-', '+', 'not'
    operand: "Expr"

@dataclass
class Binary(Node):
    op: str          # '+', '-', '*', '/', '%', '//', '**', '@', etc.
    left: "Expr"
    right: "Expr"

@dataclass
class Range(Node):
    start: Optional["Expr"]
    end: Optional["Expr"]
    inclusive: bool

@dataclass
class Call(Node):
    func: "Expr"
    args: list

@dataclass
class Index(Node):
    obj: "Expr"
    index: "Expr"

@dataclass
class Attr(Node):
    obj: "Expr"
    name: str

Expr = Union[Int, Float, Str, FString, Bool, Name, ListLit, Unary, Binary,
             Range, Call, Index, Attr]

# --- comandos ---
@dataclass
class Assign(Node):
    target: Expr     # Name | Index | Attr
    op: str          # '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=', '@='
    value: Expr

@dataclass
class VarDecl(Node):             # x: i64 = 0
    name: str
    type: TypeExpr
    value: Expr

@dataclass
class ConstDecl(Node):           # const PI = 3.14   |   const PI: f64 = 3.14
    name: str
    type: Optional[TypeExpr]
    value: Expr

@dataclass
class If(Node):
    cond: Expr
    body: list
    orelse: list                 # [] | [If] (elif) | [stmts] (else)

@dataclass
class While(Node):
    cond: Expr
    body: list

@dataclass
class For(Node):
    var: str
    iter: Expr
    body: list

@dataclass
class Return(Node):
    value: Optional[Expr]

@dataclass
class Break(Node):
    pass

@dataclass
class Continue(Node):
    pass

@dataclass
class Pass(Node):
    pass

@dataclass
class ExprStmt(Node):
    expr: Expr

# --- nível superior ---
@dataclass
class Param(Node):
    name: str
    type: TypeExpr

@dataclass
class Function(Node):
    name: str
    params: list
    ret: Optional[TypeExpr]
    body: list

@dataclass
class Import(Node):
    module: str                  # 'math' ou 'a.b'

@dataclass
class Program(Node):
    items: list                  # Import | ConstDecl | Function


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "**=", "//=", "@="}
CMP_OPS = {"==", "!=", "<", "<=", ">", ">="}

def describe(t: Token) -> str:
    special = {"NEWLINE": "fim de linha", "INDENT": "indentação inesperada",
               "DEDENT": "fim do bloco", "EOF": "fim do arquivo"}
    if t.kind in special:
        return special[t.kind]
    if t.kind == "FSTRING":
        return "uma f-string"
    return f"'{t.value}'"


class Parser:
    def __init__(self, toks: list[Token]):
        self.toks = toks
        self.pos = 0

    def peek(self, k: int = 0) -> Token:
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def advance(self) -> Token:
        t = self.toks[self.pos]
        if t.kind != "EOF":
            self.pos += 1
        return t

    def is_op(self, v: str) -> bool:
        t = self.peek()
        return t.kind == "OP" and t.value == v

    def is_kw(self, v: str) -> bool:
        t = self.peek()
        return t.kind == "KW" and t.value == v

    def error(self, msg: str, tok: Optional[Token] = None):
        tok = tok or self.peek()
        raise NatError(msg, tok.line, tok.col)

    def expect_op(self, v: str, ctx: str = "") -> Token:
        if not self.is_op(v):
            self.error(f"esperado '{v}'{ctx}, encontrado {describe(self.peek())}")
        return self.advance()

    def expect_kw(self, v: str) -> Token:
        if not self.is_kw(v):
            self.error(f"esperado '{v}', encontrado {describe(self.peek())}")
        return self.advance()

    def expect_name(self, what: str) -> Token:
        t = self.peek()
        if t.kind != "NAME":
            self.error(f"esperado {what}, encontrado {describe(t)}")
        return self.advance()

    def expect_newline(self):
        t = self.peek()
        if t.kind == "NEWLINE":
            self.advance()
        elif t.kind != "EOF":
            self.error(f"esperado fim de linha, encontrado {describe(t)}")

    def parse_program(self) -> Program:
        items = []
        while self.peek().kind != "EOF":
            t = self.peek()
            if t.kind == "NEWLINE":
                self.advance()
            elif t.kind == "INDENT":
                self.error("indentação inesperada")
            elif self.is_kw("import"):
                items.append(self.parse_import())
            elif self.is_kw("const"):
                items.append(self.parse_const())
            elif self.is_kw("fn"):
                items.append(self.parse_function())
            else:
                self.error("no nível superior só são aceitos 'import', 'const' "
                           f"e 'fn'; encontrado {describe(t)}")
        return Program(items, line=1)

    def parse_import(self) -> Import:
        kw = self.expect_kw("import")
        parts = [self.expect_name("nome do módulo").value]
        while self.is_op("."):
            self.advance()
            parts.append(self.expect_name("nome do módulo").value)
        self.expect_newline()
        return Import(".".join(parts), line=kw.line)

    def parse_const(self) -> ConstDecl:
        kw = self.expect_kw("const")
        name = self.expect_name("nome da constante").value
        ty = None
        if self.is_op(":"):
            self.advance()
            ty = self.parse_type()
        self.expect_op("=", " na declaração da constante")
        value = self.parse_expr()
        self.expect_newline()
        return ConstDecl(name, ty, value, line=kw.line)

    def parse_function(self) -> Function:
        kw = self.expect_kw("fn")
        name = self.expect_name("nome da função").value
        self.expect_op("(", " após o nome da função")
        params = []
        while not self.is_op(")"):
            nt = self.expect_name("nome do parâmetro")
            self.expect_op(":", " (parâmetros precisam de tipo)")
            params.append(Param(nt.value, self.parse_type(), line=nt.line))
            if self.is_op(","):
                self.advance()
            elif not self.is_op(")"):
                self.error(f"esperado ',' ou ')', encontrado {describe(self.peek())}")
        self.expect_op(")")
        ret = None
        if self.is_op("->"):
            self.advance()
            ret = self.parse_type()
        body = self.parse_block()
        return Function(name, params, ret, body, line=kw.line)

    def parse_type(self) -> TypeExpr:
        t = self.peek()
        if t.kind == "NAME":
            self.advance()
            if t.value in ("tensor", "Tensor") and self.is_op("["):
                self.advance()
                elem = self.parse_type()
                self.expect_op("]", " ao fechar o tipo tensor")
                return TypeTensor(elem, line=t.line)
            return TypeName(t.value, line=t.line)
        if self.is_op("["):
            self.advance()
            elem = self.parse_type()
            self.expect_op("]", " ao fechar o tipo lista")
            return TypeList(elem, line=t.line)
        self.error(f"esperado um tipo, encontrado {describe(t)}")

    def parse_block(self) -> list:
        self.expect_op(":", " no fim do cabeçalho")
        if self.peek().kind == "NEWLINE":
            self.advance()
            if self.peek().kind != "INDENT":
                self.error("esperado um bloco indentado")
            self.advance()
            body = []
            while self.peek().kind not in ("DEDENT", "EOF"):
                body.append(self.parse_statement())
            if self.peek().kind == "DEDENT":
                self.advance()
            return body
        stmt = self.parse_simple_statement()
        self.expect_newline()
        return [stmt]

    def parse_statement(self):
        t = self.peek()
        if t.kind == "INDENT":
            self.error("indentação inesperada")
        if t.kind == "KW":
            if t.value == "if":
                return self.parse_if()
            if t.value == "while":
                return self.parse_while()
            if t.value == "for":
                return self.parse_for()
            if t.value in ("elif", "else"):
                self.error(f"'{t.value}' sem 'if' correspondente")
            if t.value in ("fn", "import"):
                self.error(f"'{t.value}' só é permitido no nível superior")
        stmt = self.parse_simple_statement()
        self.expect_newline()
        return stmt

    def parse_if(self) -> If:
        kw = self.advance()
        cond = self.parse_expr()
        body = self.parse_block()
        orelse: list = []
        if self.is_kw("elif"):
            orelse = [self.parse_if()]
        elif self.is_kw("else"):
            self.advance()
            orelse = self.parse_block()
        return If(cond, body, orelse, line=kw.line)

    def parse_while(self) -> While:
        kw = self.expect_kw("while")
        cond = self.parse_expr()
        return While(cond, self.parse_block(), line=kw.line)

    def parse_for(self) -> For:
        kw = self.expect_kw("for")
        var = self.expect_name("nome da variável do laço").value
        self.expect_kw("in")
        it = self.parse_expr()
        return For(var, it, self.parse_block(), line=kw.line)

    def parse_simple_statement(self):
        t = self.peek()
        if t.kind == "KW":
            if t.value == "return":
                self.advance()
                value = None
                if self.peek().kind not in ("NEWLINE", "EOF"):
                    value = self.parse_expr()
                return Return(value, line=t.line)
            if t.value == "break":
                self.advance()
                return Break(line=t.line)
            if t.value == "continue":
                self.advance()
                return Continue(line=t.line)
            if t.value == "pass":
                self.advance()
                return Pass(line=t.line)
            if t.value == "const":
                return self.parse_const_inline()

        expr = self.parse_expr()
        if self.is_op(":"):
            if not isinstance(expr, Name):
                self.error("só é possível anotar o tipo de um nome simples")
            self.advance()
            ty = self.parse_type()
            self.expect_op("=", " na declaração com tipo")
            return VarDecl(expr.id, ty, self.parse_expr(), line=t.line)
        nt = self.peek()
        if nt.kind == "OP" and nt.value in ASSIGN_OPS:
            if not isinstance(expr, (Name, Index, Attr)):
                self.error("o lado esquerdo da atribuição precisa ser um nome, "
                           "um índice ou um atributo", t)
            self.advance()
            return Assign(expr, nt.value, self.parse_expr(), line=t.line)
        return ExprStmt(expr, line=t.line)

    def parse_const_inline(self) -> ConstDecl:
        kw = self.expect_kw("const")
        name = self.expect_name("nome da constante").value
        ty = None
        if self.is_op(":"):
            self.advance()
            ty = self.parse_type()
        self.expect_op("=", " na declaração da constante")
        return ConstDecl(name, ty, self.parse_expr(), line=kw.line)

    def parse_expr(self) -> Expr:
        left = self.parse_or()
        if self.is_op("..") or self.is_op("..="):
            op = self.advance()
            right = self.parse_or()
            if self.is_op("..") or self.is_op("..="):
                self.error("intervalos não podem ser encadeados")
            return Range(left, right, op.value == "..=", line=op.line)
        return left

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.is_kw("or"):
            op = self.advance()
            left = Binary("or", left, self.parse_and(), line=op.line)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_not()
        while self.is_kw("and"):
            op = self.advance()
            left = Binary("and", left, self.parse_not(), line=op.line)
        return left

    def parse_not(self) -> Expr:
        if self.is_kw("not"):
            op = self.advance()
            return Unary("not", self.parse_not(), line=op.line)
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        left = self.parse_arith()
        t = self.peek()
        if t.kind == "OP" and t.value in CMP_OPS:
            self.advance()
            left = Binary(t.value, left, self.parse_arith(), line=t.line)
            n = self.peek()
            if n.kind == "OP" and n.value in CMP_OPS:
                self.error("comparações não podem ser encadeadas; "
                           "use 'and' (ex.: a < b and b < c)")
        return left

    def parse_arith(self) -> Expr:
        left = self.parse_term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.advance()
            left = Binary(op.value, left, self.parse_term(), line=op.line)
        return left

    def parse_term(self) -> Expr:
        left = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%", "//", "@"):
            op = self.advance()
            left = Binary(op.value, left, self.parse_unary(), line=op.line)
        return left

    def parse_unary(self) -> Expr:
        if self.is_op("-") or self.is_op("+"):
            op = self.advance()
            return Unary(op.value, self.parse_unary(), line=op.line)
        return self.parse_power()

    def parse_power(self) -> Expr:
        base = self.parse_postfix()
        if self.is_op("**"):
            op = self.advance()
            return Binary("**", base, self.parse_unary(), line=op.line)
        return base

    def parse_postfix(self) -> Expr:
        e = self.parse_atom()
        while True:
            t = self.peek()
            if self.is_op("("):
                self.advance()
                args = []
                while not self.is_op(")"):
                    args.append(self.parse_expr())
                    if self.is_op(","):
                        self.advance()
                    elif not self.is_op(")"):
                        self.error("esperado ',' ou ')' na lista de argumentos, "
                                   f"encontrado {describe(self.peek())}")
                self.expect_op(")")
                e = Call(e, args, line=t.line)
            elif self.is_op("["):
                self.advance()
                idx = self.parse_expr()
                self.expect_op("]", " ao fechar o índice")
                e = Index(e, idx, line=t.line)
            elif self.is_op("."):
                self.advance()
                e = Attr(e, self.expect_name("nome do atributo").value, line=t.line)
            else:
                return e

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t.kind == "INT":
            self.advance()
            return Int(t.value, line=t.line)
        if t.kind == "FLOAT":
            self.advance()
            return Float(t.value, line=t.line)
        if t.kind == "STRING":
            self.advance()
            return Str(t.value, line=t.line)
        if t.kind == "FSTRING":
            self.advance()
            return self.parse_fstring(t)
        if t.kind == "KW" and t.value in ("true", "false"):
            self.advance()
            return Bool(t.value == "true", line=t.line)
        if t.kind == "NAME":
            self.advance()
            return Name(t.value, line=t.line)
        if self.is_op("("):
            self.advance()
            e = self.parse_expr()
            if self.is_op(","):
                self.error("tuplas ainda não existem na linguagem")
            self.expect_op(")", " ao fechar os parênteses")
            return e
        if self.is_op("["):
            self.advance()
            items = []
            while not self.is_op("]"):
                items.append(self.parse_expr())
                if self.is_op(","):
                    self.advance()
                elif not self.is_op("]"):
                    self.error("esperado ',' ou ']' na lista, "
                               f"encontrado {describe(self.peek())}")
            self.expect_op("]")
            return ListLit(items, line=t.line)
        self.error(f"esperada uma expressão, encontrado {describe(t)}")

    def parse_fstring(self, tok: Token) -> FString:
        raw, n = tok.value, len(tok.value)
        parts: list = []
        buf: list[str] = []
        i = 0

        def flush():
            if buf:
                parts.append(unescape("".join(buf), tok.line, tok.col))
                buf.clear()

        while i < n:
            c = raw[i]
            if c == "\\" and i + 1 < n:
                buf.append(raw[i:i + 2])
                i += 2
            elif c == "{":
                if i + 1 < n and raw[i + 1] == "{":
                    buf.append("{")
                    i += 2
                    continue
                flush()
                j, depth, quote, colon = i + 1, 0, None, None
                while j < n:
                    d = raw[j]
                    if quote:
                        if d == quote:
                            quote = None
                        elif d == "\\":
                            j += 1
                    elif d in "\"'":
                        quote = d
                    elif d in "([{":
                        depth += 1
                    elif d in ")]":
                        depth -= 1
                    elif d == "}":
                        if depth == 0:
                            break
                        depth -= 1
                    elif d == ":" and depth == 0 and colon is None:
                        colon = j
                    j += 1
                else:
                    self.error("'{' sem '}' correspondente na f-string "
                               "(use '{{' para um '{' literal)", tok)

                expr_text = raw[i + 1:colon if colon is not None else j]
                spec = raw[colon + 1:j] if colon is not None else None
                parts.append(FormatPart(self._sub_expr(expr_text, tok), spec,
                                        line=tok.line))
                i = j + 1
            elif c == "}":
                if i + 1 < n and raw[i + 1] == "}":
                    buf.append("}")
                    i += 2
                else:
                    self.error("'}' solto na f-string "
                               "(use '}}' para um '}' literal)", tok)
            else:
                buf.append(c)
                i += 1
        flush()
        return FString(parts, line=tok.line)

    def _sub_expr(self, text: str, tok: Token) -> Expr:
        text = text.strip()
        if not text:
            self.error("expressão vazia dentro de '{}' na f-string", tok)
        try:
            sub = Parser(tokenize(text, first_line=tok.line))
            e = sub.parse_expr()
            if sub.peek().kind not in ("NEWLINE", "EOF"):
                sub.error(f"trecho inesperado {describe(sub.peek())}")
            return e
        except NatError as ex:
            raise NatError(f"na f-string: {ex.message}", tok.line, tok.col) from None


# ----------------------------------------------------------------------------
# API pública
# ----------------------------------------------------------------------------

def parse(source: str) -> Program:
    return Parser(tokenize(source)).parse_program()


def parse_expression(source: str) -> Expr:
    p = Parser(tokenize(source.strip()))
    e = p.parse_expr()
    if p.peek().kind not in ("NEWLINE", "EOF"):
        p.error(f"trecho inesperado {describe(p.peek())}")
    return e


# ----------------------------------------------------------------------------
# Impressão da AST
# ----------------------------------------------------------------------------

def _compact(x: Any) -> str:
    if isinstance(x, Node):
        vals = [_compact(getattr(x, f.name)) for f in fields(x) if f.name != "line"]
        return f"{type(x).__name__}({', '.join(vals)})"
    if isinstance(x, list):
        return "[" + ", ".join(_compact(v) for v in x) + "]"
    return repr(x)


def dump(x: Any, ind: int = 0, width: int = 96) -> str:
    pad = "  " * ind
    c = _compact(x)
    if len(c) + len(pad) <= width or not isinstance(x, (Node, list)):
        return c
    if isinstance(x, list):
        inner = ",\n".join(f"{pad}  {dump(v, ind + 1, width)}" for v in x)
        return f"[\n{inner}\n{pad}]"
    inner = ",\n".join(
        f"{pad}  {f.name}={dump(getattr(x, f.name), ind + 1, width)}"
        for f in fields(x) if f.name != "line")
    return f"{type(x).__name__}(\n{inner}\n{pad})"


# ============================================================================
# ANALISADOR SEMÂNTICO
# ============================================================================

@dataclass(frozen=True)
class Prim:
    name: str
    def __repr__(self) -> str:
        return self.name

@dataclass(frozen=True)
class ListT:
    elem: Any
    def __repr__(self) -> str:
        return f"[{self.elem}]"

@dataclass(frozen=True)
class TensorT:
    elem: Any
    def __repr__(self) -> str:
        return f"tensor[{self.elem}]"

# Tipos numéricos expandidos (incluindo t2 / ternary para 1.58-bit AI)
I8, U8, I16, U16, I32, U32, I64, U64 = (Prim(n) for n in ("i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"))
F16, BF16, F32, F64 = (Prim(n) for n in ("f16", "bf16", "f32", "f64"))
T2 = Prim("t2")   # Ternary: -1, 0, +1
STR, BOOL, VOID, ANY = (Prim(n) for n in ("str", "bool", "void", "any"))

FLOAT_TYPES = {"f16", "bf16", "f32", "f64"}
INT_TYPES = {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "t2", "ternary"}

PRIM_TYPES = {
    "i8": I8, "u8": U8, "i16": I16, "u16": U16,
    "i32": I32, "u32": U32, "i64": I64, "u64": U64,
    "f16": F16, "bf16": BF16, "f32": F32, "f64": F64,
    "t2": T2, "ternary": T2,
    "str": STR, "bool": BOOL, "void": VOID, "any": ANY,
}

@dataclass
class FuncSig:
    params: list
    ret: Any
    variadic: bool = False
    builtin: bool = False


BUILTINS = {
    "print":            FuncSig([ANY], VOID, variadic=True, builtin=True),
    "len":              FuncSig([ANY], I64, builtin=True),
    "sqrt":             FuncSig([F64], F64, builtin=True),
    "exp":              FuncSig([F64], F64, builtin=True),
    "sin":              FuncSig([F64], F64, builtin=True),
    "cos":              FuncSig([F64], F64, builtin=True),
    "ln":               FuncSig([F64], F64, builtin=True),
    "floor":            FuncSig([F64], I64, builtin=True),
    "to_i64":           FuncSig([ANY], I64, builtin=True),
    "to_f64":           FuncSig([ANY], F64, builtin=True),
    "dot":              FuncSig([ListT(F64), ListT(F64)], F64, builtin=True),
    # Builtins adicionados para IA e manipulação de tensores
    "tanh":             FuncSig([F64], F64, builtin=True),
    "relu":             FuncSig([ANY], ANY, builtin=True),
    "sigmoid":          FuncSig([ANY], ANY, builtin=True),
    "matmul":           FuncSig([ANY, ANY], ANY, builtin=True),
    "sum":              FuncSig([ANY], ANY, builtin=True),
    "mean":             FuncSig([ANY], F64, builtin=True),
    "max":              FuncSig([ANY], ANY, builtin=True),
    "min":              FuncSig([ANY], ANY, builtin=True),
    "argmax":           FuncSig([ANY], I64, builtin=True),
    "zeros":            FuncSig([ANY], ANY, builtin=True),
    "ones":             FuncSig([ANY], ANY, builtin=True),
    "shape":            FuncSig([ANY], ListT(I64), builtin=True),
    "reshape":          FuncSig([ANY, ANY], ANY, builtin=True),
    "transpose":        FuncSig([ANY], ANY, builtin=True),
    "tensor":           FuncSig([ANY], TensorT(ANY), builtin=True),
    "device":           FuncSig([ANY], STR, builtin=True),
    # Primitivas para computação ternária
    "quantize_ternary": FuncSig([ANY], TensorT(T2), builtin=True),
    "pack_ternary":     FuncSig([ANY], ListT(I64), builtin=True),
    "unpack_ternary":   FuncSig([ANY, I64], ListT(I64), builtin=True),
    "clock":            FuncSig([], F64, builtin=True),
    "random_ternary":   FuncSig([ANY], ANY, builtin=True),
    "randn":            FuncSig([ANY], ANY, builtin=True),
    "rotate_rht":       FuncSig([ANY], ListT(F64), builtin=True),
    "turbo_quant":      FuncSig([ANY], ANY, builtin=True),
    "turbo_dequant":    FuncSig([ANY], ListT(F64), builtin=True),
    "turbo_dot":        FuncSig([ANY, ANY], F64, builtin=True),
    "rmsnorm":          FuncSig([ANY, ANY], ANY, builtin=True),
    "apply_rope":       FuncSig([ANY, I64], ListT(F64), builtin=True),
    "silu":             FuncSig([ANY], ANY, builtin=True),
    # SafeTensors I/O (Hugging Face standard)
    "load_safetensors": FuncSig([STR, STR], ANY, builtin=True),
    "safetensors_keys": FuncSig([STR], ListT(STR), builtin=True),
    "save_safetensors": FuncSig([STR, STR, ANY], VOID, builtin=True),
    # BitNet b1.58 Canônico (AbsMean Scaling & Dense Base-3 Packing 5 trits/byte)
    "quantize_bitnet":       FuncSig([ANY], TensorT(T2), builtin=True),
    "tensor_absmean":        FuncSig([ANY], F64, builtin=True),
    "pack_ternary_5trits":   FuncSig([ANY], ListT(I64), builtin=True),
    "unpack_ternary_5trits": FuncSig([ANY, I64], ListT(I64), builtin=True),
    # Validação de Fidelidade Numérica & Vec-LUT Kernel
    "mae":                   FuncSig([ANY, ANY], F64, builtin=True),
    "relative_error":        FuncSig([ANY, ANY], F64, builtin=True),
    "pearson_corr":          FuncSig([ANY, ANY], F64, builtin=True),
    "cosine_sim":            FuncSig([ANY, ANY], F64, builtin=True),
    "gemv_vec_lut":          FuncSig([ANY, ANY, I64, I64], ANY, builtin=True),
    # Primitivas de Modelo (Attention, Tokenizador, KV Cache)
    "softmax":               FuncSig([ANY], ANY, builtin=True),
    "causal_mask":           FuncSig([I64], ANY, builtin=True),
    "sdpa":                  FuncSig([ANY, ANY, ANY], ANY, builtin=True),
    "Tokenizer":             FuncSig([], ANY, builtin=True),
    "load_tokenizer":        FuncSig([], ANY, builtin=True),
    "encode":                FuncSig([ANY, STR], ANY, builtin=True),
    "decode":                FuncSig([ANY, ANY], STR, builtin=True),
    "KVCache":               FuncSig([I64, I64], ANY, builtin=True),
    "concat":                FuncSig([ANY], ANY, builtin=True),
    "sample":                FuncSig([ANY], I64, builtin=True),
}

KNOWN_MODULES = {"math", "time"}
MATH_FUNCS = {"sqrt", "exp", "sin", "cos", "ln", "floor", "tanh"}
MATH_CONSTS = {"PI": F64, "E": F64}


@dataclass
class Symbol:
    name: str
    type: Any
    is_const: bool = False
    implicit: bool = False
    line: int = 0
    used: bool = False


@dataclass
class Diagnostic:
    level: str                  # "erro" | "aviso"
    msg: str
    line: int


def is_num(t) -> bool:
    return isinstance(t, Prim) and (t.name in FLOAT_TYPES or t.name in INT_TYPES)

def is_float(t) -> bool:
    return isinstance(t, Prim) and t.name in FLOAT_TYPES

def is_int(t) -> bool:
    return isinstance(t, Prim) and t.name in INT_TYPES

def base_num(t):
    while isinstance(t, (ListT, TensorT)):
        t = t.elem
    return t if is_num(t) else None

def same(a, b) -> bool:
    if a == ANY or b == ANY:
        return True
    if isinstance(a, ListT) and isinstance(b, ListT):
        return same(a.elem, b.elem)
    if isinstance(a, TensorT) and isinstance(b, TensorT):
        return same(a.elem, b.elem)
    return a == b

def assignable(src, dst) -> bool:
    if dst == ANY or src == ANY:
        return True
    if is_num(src) and is_num(dst):
        if is_int(src) and is_int(dst):
            return True
        if is_float(src) and is_float(dst):
            return True
        if is_int(src) and is_float(dst):
            return True
        return False
    if isinstance(src, ListT) and isinstance(dst, ListT):
        if is_float(src.elem) and is_float(dst.elem):
            return True
        return same(src, dst)   # Listas inteiras não promovem para float
    if isinstance(dst, TensorT):
        bn = base_num(src)
        return assignable(bn, dst.elem) if bn is not None else False
    if isinstance(src, TensorT) and isinstance(dst, ListT):
        return assignable(src.elem, dst.elem)
    return same(src, dst)

def unify(a, b):
    if a == ANY:
        return b
    if b == ANY:
        return a
    if a == b:
        return a
    if is_num(a) and is_num(b):
        if is_float(a) or is_float(b):
            return F64 if (F64 in (a, b)) else F32
        return I64
    if isinstance(a, ListT) and isinstance(b, ListT):
        e = unify(a.elem, b.elem)
        return ListT(e) if e is not None else None
    if isinstance(a, TensorT) and isinstance(b, TensorT):
        e = unify(a.elem, b.elem)
        return TensorT(e) if e is not None else None
    return None


class SemanticAnalyzer:
    def __init__(self):
        self.functions: dict[str, FuncSig] = dict(BUILTINS)
        self.modules: set[str] = set()
        self.scopes: list[dict[str, Symbol]] = [{}]
        self.diagnostics: list[Diagnostic] = []
        self.cur_ret = None
        self.loop_depth = 0

    def report(self, msg: str, node, level: str = "erro"):
        self.diagnostics.append(Diagnostic(level, msg, getattr(node, "line", 0)))

    @property
    def errors(self):
        return [d for d in self.diagnostics if d.level == "erro"]

    @property
    def warnings(self):
        return [d for d in self.diagnostics if d.level == "aviso"]

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        scope = self.scopes.pop()
        visible = [n for s in self.scopes for n in s] + list(scope)
        for sym in scope.values():
            if sym.implicit and not sym.used:
                msg = f"a variável '{sym.name}' recebe valor mas nunca é usada"
                near = difflib.get_close_matches(
                    sym.name, [n for n in visible if n != sym.name], n=1, cutoff=0.75)
                if near:
                    msg += f" (erro de digitação de '{near[0]}'?)"
                self.diagnostics.append(Diagnostic("aviso", msg, sym.line))

    def define(self, name: str, node, type_, *, const=False, implicit=False) -> bool:
        if name in self.modules:
            self.report(f"'{name}' já é o nome de um módulo; escolha outro nome", node)
            return False
        if name in self.functions:
            self.report(f"'{name}' já é o nome de uma função; escolha outro nome", node)
            return False
        for scope in self.scopes:
            if name in scope:
                self.report(f"'{name}' já foi declarado na linha {scope[name].line} "
                            "(shadowing é proibido)", node)
                return False
        self.scopes[-1][name] = Symbol(name, type_, const, implicit, getattr(node, "line", 0))
        return True

    def lookup(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def undefined_msg(self, name: str) -> str:
        names = [n for s in self.scopes for n in s] + list(self.functions)
        near = difflib.get_close_matches(name, names, n=1, cutoff=0.7)
        return f"nome '{name}' não definido" + (f" (você quis dizer '{near[0]}'?)" if near else "")

    def resolve_type(self, t):
        if isinstance(t, TypeName):
            r = PRIM_TYPES.get(t.name)
            if r is None:
                self.report(f"tipo desconhecido '{t.name}'", t)
                return ANY
            return r
        if isinstance(t, TypeTensor):
            return TensorT(self.resolve_type(t.elem))
        return ListT(self.resolve_type(t.elem))

    def analyze(self, program: Program) -> list[Diagnostic]:
        for it in program.items:
            if isinstance(it, Import):
                if it.module in KNOWN_MODULES:
                    self.modules.add(it.module)
                else:
                    self.report(f"módulo desconhecido '{it.module}'", it)

        sigs: dict[int, FuncSig] = {}
        for it in program.items:
            if isinstance(it, Function):
                if it.name in self.functions:
                    why = ("é uma função embutida" if self.functions[it.name].builtin
                           else "já foi definida")
                    self.report(f"a função '{it.name}' {why}", it)
                    continue
                params = [self.resolve_type(p.type) for p in it.params]
                ret = self.resolve_type(it.ret) if it.ret else VOID
                sig = FuncSig(params, ret)
                self.functions[it.name] = sig
                sigs[id(it)] = sig

        for it in program.items:
            if isinstance(it, ConstDecl):
                self.visit_ConstDecl(it)

        for it in program.items:
            if isinstance(it, Function) and id(it) in sigs:
                self.visit_Function(it, sigs[id(it)])

        self.diagnostics.sort(key=lambda d: d.line)
        return self.diagnostics

    def visit(self, node):
        m = getattr(self, "visit_" + type(node).__name__, None)
        if m is None:
            self.report(f"construção não suportada pelo analisador: {type(node).__name__}", node)
            return ANY
        return m(node)

    def expr(self, node):
        t = self.visit(node)
        if t == VOID:
            self.report("esta expressão não produz valor", node)
            return ANY
        return t

    def cond(self, node):
        t = self.expr(node)
        if t != BOOL and t != ANY:
            self.report(f"a condição deve ser bool, mas é {t}", node)

    def always_returns(self, stmts) -> bool:
        for s in stmts:
            if isinstance(s, Return):
                return True
            if (isinstance(s, If) and s.orelse
                    and self.always_returns(s.body) and self.always_returns(s.orelse)):
                return True
        return False

    def visit_Function(self, node: Function, sig: FuncSig):
        self.cur_ret, self.loop_depth = sig.ret, 0
        self.enter_scope()
        for p, t in zip(node.params, sig.params):
            self.define(p.name, p, t)
        for s in node.body:
            self.visit(s)
        if sig.ret != VOID and not self.always_returns(node.body):
            self.report(f"a função '{node.name}' declara retorno {sig.ret}, mas nem todos "
                        "os caminhos retornam um valor", node)
        self.exit_scope()
        self.cur_ret = None

    def visit_ConstDecl(self, node: ConstDecl):
        vt = self.expr(node.value)
        dt = self.resolve_type(node.type) if node.type else vt
        if node.type and not assignable(vt, dt):
            self.report(f"constante '{node.name}' declarada como {dt}, mas o valor é {vt}", node)
        self.define(node.name, node, dt, const=True)

    def visit_VarDecl(self, node: VarDecl):
        vt = self.expr(node.value)
        dt = self.resolve_type(node.type)
        if not assignable(vt, dt):
            self.report(f"'{node.name}' declarado como {dt}, mas o valor é {vt}", node)
        self.define(node.name, node, dt)

    def visit_Assign(self, node: Assign):
        tgt = node.target
        if isinstance(tgt, Name):
            return self.assign_name(node)
        if isinstance(tgt, Index):
            elem = self.expr(tgt)
            vt = self.expr(node.value)
            rt = vt if node.op == "=" else self.binop(node.op[:-1], elem, vt, node)
            if not assignable(rt, elem):
                self.report(f"um elemento {elem} não pode receber {rt}", node)
            return
        self.expr(node.value)
        self.report("atribuição a atributos ainda não é suportada", node)

    def assign_name(self, node: Assign):
        name, op = node.target.id, node.op
        vt = self.expr(node.value)
        sym = self.lookup(name)
        if sym is None:
            if op != "=":
                self.report(f"'{name}' usado com '{op}' antes de ser declarado", node)
            else:
                self.define(name, node, vt, implicit=True)
            return
        if sym.is_const:
            self.report(f"'{name}' é uma constante e não pode ser reatribuída", node)
        elif op == "=":
            if not assignable(vt, sym.type):
                self.report(f"'{name}' é {sym.type} e não pode receber {vt}", node)
        else:
            rt = self.binop(op[:-1], sym.type, vt, node)
            if not assignable(rt, sym.type):
                self.report(f"'{name}' é {sym.type}, mas '{op}' produz {rt}", node)

    def block(self, stmts):
        self.enter_scope()
        for s in stmts:
            self.visit(s)
        self.exit_scope()

    def visit_If(self, node: If):
        self.cond(node.cond)
        self.block(node.body)
        if node.orelse:
            self.block(node.orelse)

    def visit_While(self, node: While):
        self.cond(node.cond)
        self.loop_depth += 1
        self.block(node.body)
        self.loop_depth -= 1

    def visit_For(self, node: For):
        it = self.expr(node.iter)
        if isinstance(it, (ListT, TensorT)):
            et = it.elem
        else:
            et = ANY
        if it != ANY and not isinstance(it, (ListT, TensorT)):
            self.report(f"não é possível iterar sobre {it}", node.iter)
        self.loop_depth += 1
        self.enter_scope()
        self.define(node.var, node, et)
        for s in node.body:
            self.visit(s)
        self.exit_scope()
        self.loop_depth -= 1

    def visit_Return(self, node: Return):
        want = self.cur_ret
        if node.value is None:
            if want != VOID:
                self.report(f"esta função deve retornar {want}", node)
            return
        vt = self.expr(node.value)
        if want == VOID:
            self.report("uma função sem '-> tipo' não pode retornar valor", node)
        elif not assignable(vt, want):
            self.report(f"retorno esperado {want}, recebeu {vt}", node)

    def visit_Break(self, node):
        if self.loop_depth == 0:
            self.report("'break' fora de um laço", node)

    def visit_Continue(self, node):
        if self.loop_depth == 0:
            self.report("'continue' fora de um laço", node)

    def visit_Pass(self, node):
        pass

    def visit_ExprStmt(self, node: ExprStmt):
        self.visit(node.expr)

    def visit_Int(self, n): return I64
    def visit_Float(self, n): return F64
    def visit_Str(self, n): return STR
    def visit_Bool(self, n): return BOOL

    def visit_Name(self, node: Name):
        sym = self.lookup(node.id)
        if sym:
            sym.used = True
            return sym.type
        if node.id in self.modules:
            self.report(f"'{node.id}' é um módulo; acesse um membro (ex.: {node.id}.sqrt)", node)
        elif node.id in self.functions:
            self.report(f"'{node.id}' é uma função; chame-a com parênteses", node)
        else:
            self.report(self.undefined_msg(node.id), node)
        return ANY

    def visit_Attr(self, node: Attr):
        o = node.obj
        if isinstance(o, Name) and not self.lookup(o.id) and o.id in KNOWN_MODULES:
            if o.id not in self.modules:
                self.report(f"módulo '{o.id}' não importado (use 'import {o.id}')", node)
            elif node.name in MATH_CONSTS:
                return MATH_CONSTS[node.name]
            elif node.name in MATH_FUNCS:
                self.report(f"'{o.id}.{node.name}' é uma função; chame-a com parênteses", node)
            else:
                self.report(f"o módulo '{o.id}' não tem '{node.name}'", node)
            return ANY
        self.expr(o)
        return ANY

    def visit_FString(self, node: FString):
        for p in node.parts:
            if isinstance(p, FormatPart):
                self.expr(p.expr)
        return STR

    def visit_ListLit(self, node: ListLit):
        if not node.items:
            return ListT(ANY)
        t = self.expr(node.items[0])
        for it in node.items[1:]:
            u = self.expr(it)
            n = unify(t, u)
            if n is None:
                self.report(f"elementos da lista com tipos incompatíveis: {t} e {u}", it)
            else:
                t = n
        return ListT(t)

    def visit_Range(self, node: Range):
        for b in (node.start, node.end):
            if b is not None:
                t = self.expr(b)
                if t != I64 and t != ANY and not (is_num(t) and is_int(t)):
                    self.report(f"os limites de um intervalo devem ser i64, mas é {t}", b)
        return ListT(I64)

    def visit_Index(self, node: Index):
        ot = self.expr(node.obj)
        if isinstance(node.index, Range):
            self.visit_Range(node.index)
            if isinstance(ot, (ListT, TensorT)):
                return ot
            if ot == STR:
                return STR
            if ot != ANY:
                self.report(f"um valor {ot} não pode ser fatiado", node)
            return ANY
        it = self.expr(node.index)
        if it != I64 and it != ANY and not (is_num(it) and is_int(it)):
            self.report(f"o índice deve ser i64, mas é {it}", node.index)
        if isinstance(ot, (ListT, TensorT)):
            return ot.elem
        if ot == STR:
            return STR
        if ot != ANY:
            self.report(f"um valor {ot} não pode ser indexado", node)
        return ANY

    def visit_Unary(self, node: Unary):
        t = self.expr(node.operand)
        if node.op == "not":
            if t != BOOL and t != ANY:
                self.report(f"'not' exige bool, mas recebeu {t}", node)
            return BOOL
        ok = t == ANY or is_num(t) or (isinstance(t, (ListT, TensorT)) and base_num(t) is not None)
        if not ok:
            self.report(f"o operador '{node.op}' não é definido para {t}", node)
            return ANY
        return t

    def visit_Binary(self, node: Binary):
        l = self.expr(node.left)
        r = self.expr(node.right)
        return self.binop(node.op, l, r, node)

    def binop(self, op, l, r, node):
        if op in ("and", "or"):
            for t in (l, r):
                if t != BOOL and t != ANY:
                    self.report(f"'{op}' exige bool, mas recebeu {t}", node)
            return BOOL
        if op in ("==", "!="):
            if not (same(l, r) or (is_num(l) and is_num(r))):
                self.report(f"não é possível comparar {l} com {r}", node)
            return BOOL
        if op in ("<", "<=", ">", ">="):
            ok = (l == ANY or r == ANY or (is_num(l) and is_num(r)) or (l == STR and r == STR))
            if not ok:
                self.report(f"não é possível comparar {l} com {r}", node)
            return BOOL
        return self.arith(op, l, r, node)

    def arith(self, op, l, r, node):
        if l == ANY or r == ANY:
            return ANY
        if is_num(l) and is_num(r):
            if op == "/":
                return F64
            if op in ("//", "%", "+", "-", "*", "**"):
                if is_float(l) or is_float(r):
                    return F64 if (F64 in (l, r)) else F32
                return I64
        if op == "+" and l == STR and r == STR:
            return STR
        if op == "+" and isinstance(l, ListT) and isinstance(r, ListT):
            elem = r.elem if l.elem == ANY else (l.elem if r.elem == ANY else (l.elem if l.elem == r.elem else ANY))
            return ListT(elem)

        # Multiplicação matricial com operador @ (suporta ternários e contínuos)
        if op == "@":
            nl = base_num(l)
            nr = base_num(r)
            if nl and nr and isinstance(l, (ListT, TensorT)) and isinstance(r, (ListT, TensorT)):
                dl, c = 0, l
                while isinstance(c, (ListT, TensorT)):
                    dl += 1
                    c = c.elem
                dr, c = 0, r
                while isinstance(c, (ListT, TensorT)):
                    dr += 1
                    c = c.elem

                # Tipo resultante do elemento
                if F64 in (nl, nr):
                    res_elem = F64
                elif F32 in (nl, nr):
                    res_elem = F32
                elif is_float(nl) or is_float(nr):
                    res_elem = F32
                else:
                    res_elem = I64

                wrap = TensorT if (isinstance(l, TensorT) or isinstance(r, TensorT)) else ListT
                if dl == 1 and dr == 1:
                    return res_elem
                elif dl == 2 and dr == 1:
                    return wrap(res_elem)
                elif dl == 1 and dr == 2:
                    return wrap(res_elem)
                elif dl == 2 and dr == 2:
                    return wrap(wrap(res_elem))
                return wrap(wrap(res_elem))
            self.report(f"o operador '@' não é definido para {l} e {r}", node)
            return ANY

        ll, rl = isinstance(l, (ListT, TensorT)), isinstance(r, (ListT, TensorT))
        numlike = lambda t: is_num(t) or (isinstance(t, (ListT, TensorT)) and base_num(t) is not None)
        if (ll or rl) and numlike(l) and numlike(r):          # broadcasting
            wrap = TensorT if (isinstance(l, TensorT) or isinstance(r, TensorT)) else ListT
            if ll and rl:
                return wrap(self.arith(op, l.elem, r.elem, node))
            if ll:
                return wrap(self.arith(op, l.elem, r, node))
            return wrap(self.arith(op, l, r.elem, node))

        self.report(f"o operador '{op}' não é definido para {l} e {r}", node)
        return ANY

    def visit_Call(self, node: Call):
        f, sig, label = node.func, None, ""
        if isinstance(f, Name):
            if self.lookup(f.id):
                self.report(f"'{f.id}' é uma variável, não uma função", f)
            elif f.id in self.functions:
                sig, label = self.functions[f.id], f.id
            else:
                self.report(self.undefined_msg(f.id), f)
        elif isinstance(f, Attr) and isinstance(f.obj, Name) and not self.lookup(f.obj.id):
            mod = f.obj.id
            if mod not in KNOWN_MODULES:
                self.report(f"'{mod}' não é um módulo", f)
            elif mod not in self.modules:
                self.report(f"módulo '{mod}' não importado (use 'import {mod}')", f)
            elif f.name in MATH_FUNCS:
                sig, label = BUILTINS[f.name], f"{mod}.{f.name}"
            else:
                self.report(f"o módulo '{mod}' não tem '{f.name}'", f)
        elif isinstance(f, Attr):
            self.expr(f.obj)
            for a in node.args:
                self.expr(a)
            return ANY
        else:
            self.report("só é possível chamar funções pelo nome", f)

        ats = [self.expr(a) for a in node.args]
        if sig is None:
            return ANY

        if label in ("Tokenizer", "load_tokenizer"):
            if len(ats) not in (0, 1):
                self.report(f"'{label}' espera 0 ou 1 argumento(s), mas recebeu {len(ats)}", node)
            return ANY

        if label == "sdpa":
            if len(ats) not in (3, 4, 5):
                self.report(f"'sdpa' espera entre 3 e 5 argumentos, mas recebeu {len(ats)}", node)
            return TensorT(F32)

        if label == "KVCache":
            if len(ats) not in (0, 1, 2):
                self.report(f"'KVCache' espera até 2 argumentos, mas recebeu {len(ats)}", node)
            return ANY

        if label == "sample":
            if len(ats) not in (1, 2, 3):
                self.report(f"'sample' espera entre 1 e 3 argumentos (logits, temp, top_p), mas recebeu {len(ats)}", node)
            return I64

        if label == "concat":
            if len(ats) not in (1, 2):
                self.report(f"'concat' espera 1 ou 2 argumento(s), mas recebeu {len(ats)}", node)
            return TensorT(F32)

        # Tratamento polimórfico de len para listas, tensores e strings
        if label == "len":
            if len(ats) != 1:
                self.report(f"'len' espera 1 argumento(s), mas recebeu {len(ats)}", node)
            else:
                t = ats[0]
                if not (isinstance(t, (ListT, TensorT)) or t in (STR, ANY)):
                    self.report(f"argumento 1 de 'len': esperado lista, tensor ou str, recebeu {t}", node.args[0])
            return I64

        # Tratamento de quantize_ternary e quantize_bitnet
        if label in ("quantize_ternary", "quantize_bitnet"):
            if label == "quantize_ternary" and len(ats) not in (1, 2):
                self.report(f"'quantize_ternary' espera 1 ou 2 argumento(s), mas recebeu {len(ats)}", node)
            elif label == "quantize_bitnet" and len(ats) != 1:
                self.report(f"'quantize_bitnet' espera 1 argumento, mas recebeu {len(ats)}", node)
            if ats and ats[0] == ANY:
                return ANY
            if ats and isinstance(ats[0], (ListT, TensorT)):
                c = ats[0]
                depth = 0
                while isinstance(c, (ListT, TensorT)):
                    depth += 1
                    c = c.elem
                res = T2
                for _ in range(depth):
                    res = TensorT(res)
                return res
            return T2

        if label == "turbo_quant":
            if len(ats) not in (1, 2):
                self.report(f"'turbo_quant' espera 1 ou 2 argumento(s), mas recebeu {len(ats)}", node)
            return ANY

        if label == "rmsnorm":
            if len(ats) not in (2, 3):
                self.report(f"'rmsnorm' espera 2 ou 3 argumento(s), mas recebeu {len(ats)}", node)
            return ats[0] if ats else ANY

        if label == "apply_rope":
            if len(ats) not in (2, 3):
                self.report(f"'apply_rope' espera 2 ou 3 argumento(s), mas recebeu {len(ats)}", node)
            return ats[0] if ats else ANY

        if sig.variadic:
            for i, (a, t) in enumerate(zip(node.args, ats)):
                if not assignable(t, sig.params[0]):
                    self.report(f"argumento {i + 1} de '{label}': esperado {sig.params[0]}, "
                                f"recebeu {t}", a)
        else:
            if len(ats) != len(sig.params):
                self.report(f"'{label}' espera {len(sig.params)} argumento(s), mas recebeu "
                            f"{len(ats)}", node)
            for i, (a, t, p) in enumerate(zip(node.args, ats, sig.params)):
                if not assignable(t, p):
                    self.report(f"argumento {i + 1} de '{label}': esperado {p}, recebeu {t}", a)
        return sig.ret


def analyze(program: Program) -> list[Diagnostic]:
    return SemanticAnalyzer().analyze(program)


# ============================================================================
# MOTOR DE EXECUÇÃO / RUNTIME (NumPy SIMD + Ternary Add-Only Backend)
# ============================================================================

class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

class BreakSignal(Exception):
    pass

class ContinueSignal(Exception):
    pass


class MathModule:
    PI = 3.14159265358979323846
    E = 2.71828182845904523536
    sqrt = staticmethod(math.sqrt)
    exp = staticmethod(math.exp)
    sin = staticmethod(math.sin)
    cos = staticmethod(math.cos)
    ln = staticmethod(math.log)
    floor = staticmethod(lambda x: int(math.floor(x)))
    tanh = staticmethod(math.tanh)


CURRENT_DEVICE = "cpu"

def device_fn(dev: Any = None) -> str:
    """Consulta ou define o dispositivo de execução do runtime Nativa.
    Nota arquitetural (v0.3): O pipeline de execução opera nativamente no backend CPU
    acelerado por OpenMP/AOT C99. As marcações 'cuda' e 'opencl' operam como metadados
    de posicionamento de grafo para compatibilidade futura com backends GPU.
    """
    global CURRENT_DEVICE
    if dev is not None:
        valid = ("cpu", "cuda", "aot", "opencl")
        dev_str = str(dev).lower().strip("\"'")
        if dev_str in valid:
            if dev_str in ("cuda", "opencl") and CURRENT_DEVICE != dev_str:
                sys.stderr.write(
                    f"[Nativa v0.3 Aviso]: Dispositivo '{dev_str}' registrado como metadado de grafo. "
                    f"O runtime v0.3 executa tensores no backend CPU otimizado (C99 AOT / OpenMP). "
                    f"Aceleração nativa via hardware '{dev_str}' está reservada para as próximas versões.\n"
                )
            CURRENT_DEVICE = dev_str
        return dev_str
    return CURRENT_DEVICE


AOT_C_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void nat_c_gemv_ternary(const int8_t* W, const float* x, float* y, int64_t M, int64_t K) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc = 0.0f;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc;
    }
}

static const uint8_t DEC_01[256] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3
};

static const uint8_t DEC_23[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1
};

static const uint8_t DEC_4[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

EXPORT void nat_c_gemv_veclut_5trits(const uint8_t* W_packed, const float* x, float* y, int64_t M, int64_t K, float gamma) {
    int64_t num_groups = (K + 4) / 5;
    float* lut_full = (float*)malloc(num_groups * 256 * sizeof(float));
    if (!lut_full) return;

    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 5;
        float x0 = (base < K) ? x[base] : 0.0f;
        float x1 = (base + 1 < K) ? x[base + 1] : 0.0f;
        float x2 = (base + 2 < K) ? x[base + 2] : 0.0f;
        float x3 = (base + 3 < K) ? x[base + 3] : 0.0f;
        float x4 = (base + 4 < K) ? x[base + 4] : 0.0f;

        float sub01[9];
        float sub23[9];
        float sub4[3];
        for (int t1 = 0; t1 < 3; t1++) {
            for (int t0 = 0; t0 < 3; t0++) {
                sub01[t0 + 3 * t1] = (float)(t0 - 1) * x0 + (float)(t1 - 1) * x1;
                sub23[t0 + 3 * t1] = (float)(t0 - 1) * x2 + (float)(t1 - 1) * x3;
            }
        }
        for (int t4 = 0; t4 < 3; t4++) {
            sub4[t4] = (float)(t4 - 1) * x4;
        }

        float* g_lut = lut_full + g * 256;
        for (int b = 0; b < 243; b++) {
            g_lut[b] = sub01[DEC_01[b]] + sub23[DEC_23[b]] + sub4[DEC_4[b]];
        }
        for (int b = 243; b < 256; b++) {
            g_lut[b] = 0.0f;
        }
    }

    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const uint8_t* row = W_packed + i * num_groups;
        const float* lut_ptr = lut_full;
        int64_t g = 0;

        for (; g + 3 < num_groups; g += 4) {
            acc0 += lut_ptr[row[g]];
            acc1 += (lut_ptr + 256)[row[g + 1]];
            acc2 += (lut_ptr + 512)[row[g + 2]];
            acc3 += (lut_ptr + 768)[row[g + 3]];
            lut_ptr += 1024;
        }
        for (; g < num_groups; g++) {
            acc0 += lut_ptr[row[g]];
            lut_ptr += 256;
        }
        y[i] = (acc0 + acc1 + acc2 + acc3) * gamma;
    }

    free(lut_full);
}

EXPORT void nat_c_pack_5trits_2d(const int8_t* in, uint8_t* out_bytes, int64_t M, int64_t K) {
    int64_t bpr = (K + 4) / 5;
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < M; i++) {
        const int8_t* row_in = in + i * K;
        uint8_t* row_out = out_bytes + i * bpr;
        for (int64_t g = 0; g < bpr; g++) {
            int64_t base = g * 5;
            uint16_t b_val = 0;
            uint16_t mul = 1;
            for (int j = 0; j < 5; j++) {
                int8_t t = (base + j < K) ? (row_in[base + j] + 1) : 1;
                b_val += (uint16_t)(t * mul);
                mul *= 3;
            }
            row_out[g] = (uint8_t)b_val;
        }
    }
}

EXPORT void nat_c_rmsnorm(const float* x, const float* gamma, float* out, int64_t N, float eps) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)N + eps);
    float inv_rms = 1.0f / rms;
    for (int64_t i = 0; i < N; i++) out[i] = (x[i] * inv_rms) * gamma[i];
}

EXPORT void nat_c_silu(const float* x, float* out, int64_t N) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < N; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

EXPORT void nat_c_softmax(const float* x, float* out, int64_t N) {
    float max_val = x[0];
    for (int64_t i = 1; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }
    float inv = 1.0f / sum;
    for (int64_t i = 0; i < N; i++) out[i] *= inv;
}
"""

class AOTKernelEngine:
    _dll = None
    _init_done = False

    @classmethod
    def get_dll(cls):
        if cls._init_done:
            return cls._dll
        cls._init_done = True
        cls._dll = cls._load_or_compile()
        return cls._dll

    @classmethod
    def _load_or_compile(cls):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nativa_cache")
        os.makedirs(cache_dir, exist_ok=True)
        dll_name = "nativa_aot.dll" if sys.platform == "win32" else "nativa_aot.so"
        dll_path = os.path.join(cache_dir, dll_name)
        c_src_path = os.path.join(cache_dir, "nativa_aot.c")
        hash_path = os.path.join(cache_dir, "nativa_aot.hash")

        src_hash = hashlib.sha256(AOT_C_SOURCE.encode("utf-8")).hexdigest()
        need_recompile = (not os.path.exists(dll_path)) or (not os.path.exists(hash_path))
        if not need_recompile:
            try:
                with open(hash_path, "r", encoding="utf-8") as hf:
                    saved_hash = hf.read().strip()
                if saved_hash != src_hash:
                    need_recompile = True
            except Exception:
                need_recompile = True

        if need_recompile:
            if os.path.exists(hash_path):
                try:
                    os.remove(hash_path)
                except Exception:
                    pass
            if os.path.exists(dll_path):
                try:
                    os.remove(dll_path)
                except Exception:
                    pass

            with open(c_src_path, "w", encoding="utf-8") as f:
                f.write(AOT_C_SOURCE)

            compiled_ok = False
            if sys.platform == "win32":
                vcvars_candidates = [
                    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
                    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
                    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
                ]
                for vc in vcvars_candidates:
                    if os.path.exists(vc):
                        bat = os.path.join(cache_dir, "_build_aot.bat")
                        with open(bat, "w") as bf:
                            bf.write(f'@echo off\ncall "{vc}" >nul 2>&1\ncl.exe /LD /O2 /openmp /nologo "{c_src_path}" /Fe:"{dll_path}"\n')
                        proc = subprocess.run([bat], capture_output=True, text=True)
                        if proc.returncode == 0 and os.path.exists(dll_path):
                            compiled_ok = True
                            break

            if not compiled_ok:
                gcc = shutil.which("gcc") or shutil.which("clang")
                if gcc:
                    cmd = [gcc, "-O3", "-shared", "-fPIC", "-fopenmp", c_src_path, "-o", dll_path, "-lm"]
                    proc = subprocess.run(cmd, capture_output=True)
                    if proc.returncode == 0 and os.path.exists(dll_path):
                        compiled_ok = True

            if compiled_ok and os.path.exists(dll_path):
                try:
                    with open(hash_path, "w", encoding="utf-8") as hf:
                        hf.write(src_hash)
                except Exception:
                    pass
            elif not compiled_ok:
                sys.stderr.write("[Nativa v0.3 Aviso]: Recompilação AOT C99 falhou. O cache não foi atualizado.\n")

        if os.path.exists(dll_path):
            try:
                dll = ctypes.CDLL(dll_path)
                cls._setup_signatures(dll)
                return dll
            except Exception:
                return None
        return None

    @classmethod
    def _setup_signatures(cls, dll):
        dll.nat_c_gemv_ternary.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64
        ]
        dll.nat_c_gemv_ternary.restype = None

        dll.nat_c_gemv_veclut_5trits.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float
        ]
        dll.nat_c_gemv_veclut_5trits.restype = None

        dll.nat_c_pack_5trits_2d.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_int64
        ]
        dll.nat_c_pack_5trits_2d.restype = None

        dll.nat_c_rmsnorm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_float
        ]
        dll.nat_c_rmsnorm.restype = None

        dll.nat_c_silu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64
        ]
        dll.nat_c_silu.restype = None

        dll.nat_c_softmax.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64
        ]
        dll.nat_c_softmax.restype = None


class Tensor:
    """Estrutura densa de dados de primeira classe no runtime do Nativa.
    Mantém memória contígua com metadados de shape, dtype, escala e device.
    """
    def __init__(self, data: Any, dtype: str = "f32", device: Optional[str] = None):
        self.device = device if device is not None else CURRENT_DEVICE
        self.scale = 1.0

        if dtype in ("f32", "float"):
            np_dt = np.float32
            self.dtype = "f32"
        elif dtype in ("f64", "double"):
            np_dt = np.float64
            self.dtype = "f64"
        elif dtype in ("t2", "ternary"):
            np_dt = np.int8
            self.dtype = "t2"
        elif dtype in ("t2_packed_5t", "5trits"):
            np_dt = np.uint8
            self.dtype = "t2_packed_5t"
        elif dtype in ("i64", "int"):
            np_dt = np.int64
            self.dtype = "i64"
        elif dtype in ("i32",):
            np_dt = np.int32
            self.dtype = "i32"
        elif dtype in ("i8",):
            np_dt = np.int8
            self.dtype = "i8"
        else:
            np_dt = np.float32
            self.dtype = dtype

        if isinstance(data, Tensor):
            self.data = data.data.copy()
            self.scale = data.scale
            self.dtype = dtype if dtype != "f32" else data.dtype
            self.device = device if device is not None else data.device
        elif isinstance(data, np.ndarray):
            self.data = data if data.dtype == np_dt else data.astype(np_dt)
        else:
            self.data = np.asarray(data, dtype=np_dt)

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            res = self.data
        else:
            res = self.data.astype(dtype)
        if copy:
            return res.copy()
        return res

    @property
    def shape(self) -> list[int]:
        return list(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def numel(self) -> int:
        return int(self.data.size)

    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def to(self, device: str) -> "Tensor":
        t = Tensor(self.data, dtype=self.dtype, device=device)
        t.scale = self.scale
        return t

    def quantize_bitnet(self) -> "Tensor":
        gamma = float(np.mean(np.abs(self.data)))
        scaled = self.data / (gamma + 1e-7)
        rounded = np.clip(np.round(scaled), -1, 1).astype(np.int8)
        t = Tensor(rounded, dtype="t2", device=self.device)
        t.scale = 1.0
        t.gamma = gamma
        return t

    def pack_5trits(self) -> "Tensor":
        arr = self.data
        if arr.ndim == 2:
            M, K = arr.shape
            bytes_per_row = (K + 4) // 5
            dll = AOTKernelEngine.get_dll()
            if dll:
                arr_c = np.ascontiguousarray(arr, dtype=np.int8)
                out_bytes = np.empty((M, bytes_per_row), dtype=np.uint8)
                dll.nat_c_pack_5trits_2d(
                    arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    out_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    M, K
                )
                t = Tensor(out_bytes.ravel(), dtype="t2_packed_5t", device=self.device)
                t.scale = self.scale
                t.gamma = getattr(self, "gamma", 1.0)
                t._orig_shape = (M, K)
                return t
            pad_k = bytes_per_row * 5 - K
            padded = np.pad(arr, ((0, 0), (0, pad_k)), constant_values=0) if pad_k > 0 else arr
            chunks = (padded + 1).astype(np.uint16).reshape(M, bytes_per_row, 5)
            weights = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
            packed = (chunks * weights).sum(axis=2).astype(np.uint8).ravel()
            t = Tensor(packed, dtype="t2_packed_5t", device=self.device)
            t.scale = self.scale
            t.gamma = getattr(self, "gamma", 1.0)
            t._orig_shape = (M, K)
            return t
        else:
            packed = pack_ternary_5trits_fn(self.data)
            t = Tensor(packed, dtype="t2_packed_5t", device=self.device)
            t.scale = self.scale
            t.gamma = getattr(self, "gamma", 1.0)
            return t

    def unpack_5trits(self, length: int) -> "Tensor":
        unpacked = unpack_ternary_5trits_fn(self.data, length)
        t = Tensor(unpacked, dtype="t2", device=self.device)
        t.scale = self.scale
        return t

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Any) -> Any:
        res = self.data[idx]
        if isinstance(res, np.ndarray):
            t = Tensor(res, dtype=self.dtype, device=self.device)
            t.scale = self.scale
            return t
        return res.item() if hasattr(res, "item") else res

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(val, Tensor):
            self.data[idx] = val.data
        else:
            self.data[idx] = val

    def __iter__(self):
        for item in self.data:
            if isinstance(item, np.ndarray):
                t = Tensor(item, dtype=self.dtype, device=self.device)
                t.scale = self.scale
                yield t
            else:
                yield item.item() if hasattr(item, "item") else item

    def __matmul__(self, other: Any) -> Any:
        if self.dtype == "t2_packed_5t":
            x_arr = other.data if isinstance(other, Tensor) else np.asarray(other)
            orig_shape = getattr(self, "_orig_shape", None)
            if orig_shape:
                m, k = orig_shape[0], orig_shape[1]
            else:
                k = x_arr.shape[0] if x_arr.ndim > 1 else len(x_arr)
                m = self.shape[0] if len(self.shape) > 1 else (len(self.data) // ((k + 4) // 5))

            gamma_val = getattr(self, "gamma", 1.0)
            if x_arr.ndim <= 1:
                res = gemv_vec_lut_fn(self.data, x_arr, m, k, gamma=gamma_val)
                out = Tensor(res, dtype="f32", device=self.device)
            else:
                cols = []
                for j in range(x_arr.shape[1]):
                    col_j = x_arr[:, j]
                    res_j = gemv_vec_lut_fn(self.data, col_j, m, k, gamma=gamma_val)
                    col_data = res_j.data if isinstance(res_j, Tensor) else np.asarray(res_j)
                    cols.append(col_data)
                res_mat = np.column_stack(cols)
                out = Tensor(res_mat, dtype="f32", device=self.device)

            if self.scale != 1.0:
                out.data *= self.scale
            return out

        if isinstance(other, Tensor) and other.dtype == "t2_packed_5t":
            x_arr = self.data if isinstance(self, Tensor) else np.asarray(self)
            orig_shape = getattr(other, "_orig_shape", None)
            if orig_shape:
                m, k = orig_shape[0], orig_shape[1]
            else:
                k = x_arr.shape[0] if x_arr.ndim > 1 else len(x_arr)
                m = other.shape[0] if len(other.shape) > 1 else (len(other.data) // ((k + 4) // 5))

            gamma_val = getattr(other, "gamma", 1.0)
            if x_arr.ndim <= 1:
                res = gemv_vec_lut_fn(other.data, x_arr, m, k, gamma=gamma_val)
                out = Tensor(res, dtype="f32", device=other.device)
            else:
                cols = []
                for j in range(x_arr.shape[1]):
                    col_j = x_arr[:, j]
                    res_j = gemv_vec_lut_fn(other.data, col_j, m, k, gamma=gamma_val)
                    col_data = res_j.data if isinstance(res_j, Tensor) else np.asarray(res_j)
                    cols.append(col_data)
                res_mat = np.column_stack(cols)
                out = Tensor(res_mat, dtype="f32", device=other.device)

            if other.scale != 1.0:
                out.data *= other.scale
            return out

        res = ternary_add_only_matmul(self, other)
        out = res if isinstance(res, Tensor) else Tensor(res, dtype="f32", device=self.device)
        return out

    def __mul__(self, other: Any) -> "Tensor":
        val = other.data if isinstance(other, Tensor) else other
        res = self.data * val
        t = Tensor(res, dtype=self.dtype, device=self.device)
        t.scale = self.scale
        return t

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __add__(self, other: Any) -> "Tensor":
        val = other.data if isinstance(other, Tensor) else other
        res = self.data + val
        return Tensor(res, dtype=self.dtype, device=self.device)

    def __sub__(self, other: Any) -> "Tensor":
        val = other.data if isinstance(other, Tensor) else other
        res = self.data - val
        return Tensor(res, dtype=self.dtype, device=self.device)

    def __truediv__(self, other: Any) -> "Tensor":
        val = other.data if isinstance(other, Tensor) else other
        res = self.data / val
        return Tensor(res, dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        s = f"Tensor(shape={self.shape}, dtype='{self.dtype}'"
        if self.device != "cpu":
            s += f", device='{self.device}'"
        if self.scale != 1.0:
            s += f", scale={self.scale:.6f}"
        s += f", data={self.data.tolist() if self.numel() <= 8 else f'[{self.data.ravel()[0]} ... {self.data.ravel()[-1]}]'})"
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def tolist(self) -> list:
        return self.data.tolist()


def quantize_ternary_fn(x: Any, threshold: Optional[float] = None) -> Any:
    if isinstance(x, Tensor):
        arr = x.data
        t = threshold if threshold is not None else float(np.mean(np.abs(arr)) * 0.7)
        res = np.zeros_like(arr, dtype=np.int8)
        res[arr > t] = 1
        res[arr < -t] = -1
        return Tensor(res, dtype="t2")
    arr = np.asarray(x, dtype=float)
    t = threshold if threshold is not None else float(np.mean(np.abs(arr)) * 0.7)
    res = np.zeros_like(arr, dtype=np.int8)
    res[arr > t] = 1
    res[arr < -t] = -1
    if isinstance(x, list):
        return res.tolist()
    if isinstance(x, np.ndarray):
        return res
    if np.isscalar(x) or res.size == 1:
        return int(res.item())
    return Tensor(res, dtype="t2")


def pack_ternary_fn(x: Any) -> list[int]:
    arr = np.array(x, dtype=np.int8).ravel()
    enc = np.zeros(len(arr), dtype=np.uint8)
    enc[arr == 1] = 1
    enc[arr == -1] = 2
    pad_len = (4 - (len(enc) % 4)) % 4
    if pad_len > 0:
        enc = np.pad(enc, (0, pad_len))
    chunks = enc.reshape(-1, 4)
    packed = chunks[:, 0] | (chunks[:, 1] << 2) | (chunks[:, 2] << 4) | (chunks[:, 3] << 6)
    return packed.tolist()


def unpack_ternary_fn(packed_bytes: list[int], length: int) -> list[int]:
    if not packed_bytes or length == 0:
        return []
    b = np.array(packed_bytes, dtype=np.uint8)
    inv = np.array([0, 1, -1, 0], dtype=np.int8)
    c0 = b & 0b11
    c1 = (b >> 2) & 0b11
    c2 = (b >> 4) & 0b11
    c3 = (b >> 6) & 0b11
    stacked = np.stack([c0, c1, c2, c3], axis=1).ravel()
    res = inv[stacked[:length]]
    return res.tolist()


def tensor_absmean_fn(x: Any) -> float:
    arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=float)
    return float(np.mean(np.abs(arr)))


def quantize_bitnet_fn(x: Any) -> Any:
    """Quantizador canônico do BitNet b1.58: Clip(Round(W / (gamma + eps)), -1, 1)."""
    if isinstance(x, Tensor):
        return x.quantize_bitnet()
    arr = np.asarray(x, dtype=float)
    gamma = float(np.mean(np.abs(arr)))
    eps = 1e-7
    scaled = arr / (gamma + eps)
    rounded = np.clip(np.round(scaled), -1, 1).astype(np.int8)
    if isinstance(x, np.ndarray):
        return rounded
    return rounded.tolist() if isinstance(x, list) else int(rounded.item())


def pack_ternary_5trits_fn(x: Any) -> Any:
    """Empacotamento denso de base-3: 5 trits por byte (3^5 = 243 <= 256), atingindo 1.60 bits/trit."""
    if isinstance(x, Tensor):
        return x.pack_5trits()
    arr = np.asarray(x, dtype=np.int8)
    if arr.ndim == 2:
        M, K = arr.shape
        bytes_per_row = (K + 4) // 5
        dll = AOTKernelEngine.get_dll()
        if dll:
            arr_c = np.ascontiguousarray(arr, dtype=np.int8)
            out_bytes = np.empty((M, bytes_per_row), dtype=np.uint8)
            dll.nat_c_pack_5trits_2d(
                arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                out_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                M, K
            )
            res = out_bytes.ravel()
            return res if isinstance(x, np.ndarray) else res.tolist()
        pad_k = bytes_per_row * 5 - K
        padded = np.pad(arr, ((0, 0), (0, pad_k)), constant_values=0) if pad_k > 0 else arr
        chunks = (padded + 1).astype(np.uint16).reshape(M, bytes_per_row, 5)
        weights = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
        packed = (chunks * weights).sum(axis=2).astype(np.uint8).ravel()
        return packed if isinstance(x, np.ndarray) else packed.tolist()
    else:
        arr = arr.ravel()
        trits = (arr + 1).astype(np.uint16)
        pad_len = (5 - (len(trits) % 5)) % 5
        if pad_len > 0:
            trits = np.pad(trits, (0, pad_len), constant_values=1)
        chunks = trits.reshape(-1, 5)
        weights = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
        packed = (chunks * weights).sum(axis=1).astype(np.uint8)
        return packed if isinstance(x, np.ndarray) else packed.tolist()


def unpack_ternary_5trits_fn(packed_bytes: Any, length: int) -> Any:
    """Desempacotamento denso de base-3: extrai trits {0, 1, 2} e mapeia para {-1, 0, +1}."""
    if isinstance(packed_bytes, Tensor):
        return packed_bytes.unpack_5trits(length)
    if (len(packed_bytes) if hasattr(packed_bytes, "__len__") else 0) == 0 or length == 0:
        return []
    b = np.asarray(packed_bytes, dtype=np.uint16)
    t0 = b % 3
    b = b // 3
    t1 = b % 3
    b = b // 3
    t2 = b % 3
    b = b // 3
    t3 = b % 3
    t4 = b // 3
    unpacked_trits = np.stack([t0, t1, t2, t3, t4], axis=1).ravel()
    res = unpacked_trits.astype(np.int8) - 1
    return res[:length] if isinstance(packed_bytes, np.ndarray) else res[:length].tolist()


def mae_fn(a: Any, b: Any) -> float:
    """Erro Absoluto Médio (Mean Absolute Error): 1/N * sum(|a - b|)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    return float(np.mean(np.abs(a_arr - b_arr)))


def relative_error_fn(a: Any, b: Any) -> float:
    """Erro Relativo de Norma L2: ||a - b|| / ||a||."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    norm_a = float(np.linalg.norm(a_arr))
    if norm_a == 0.0:
        return 0.0
    return float(np.linalg.norm(a_arr - b_arr) / norm_a)


def pearson_corr_fn(a: Any, b: Any) -> float:
    """Coeficiente de Correlação Linear de Pearson entre dois tensores."""
    a_arr = np.array(a, dtype=float).ravel()
    b_arr = np.array(b, dtype=float).ravel()
    if len(a_arr) < 2:
        return 1.0
    a_diff = a_arr - np.mean(a_arr)
    b_diff = b_arr - np.mean(b_arr)
    denom = np.sqrt(np.sum(a_diff ** 2) * np.sum(b_diff ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a_diff * b_diff) / denom)


def cosine_sim_fn(a: Any, b: Any) -> float:
    """Similaridade de Cosseno entre dois vetores: (a . b) / (||a|| * ||b||)."""
    a_arr = np.array(a, dtype=float).ravel()
    b_arr = np.array(b, dtype=float).ravel()
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def gemv_vec_lut_fn(W_packed: Any, X: Any, M: int, K: int, gamma: float = 1.0) -> list[float]:
    """Kernel Vec-LUT (Vectorized Look-Up Table) diretamente sobre matriz 5 trits/byte."""
    W_p = W_packed.data if isinstance(W_packed, Tensor) else np.asarray(W_packed, dtype=np.uint8)
    bytes_per_row = (K + 4) // 5
    expected_row_aligned = M * bytes_per_row
    gamma_f = float(gamma)

    if W_p.size == expected_row_aligned:
        dll = AOTKernelEngine.get_dll()
        if dll:
            W_c = np.ascontiguousarray(W_p, dtype=np.uint8)
            x_arr = np.ascontiguousarray(X.data if isinstance(X, Tensor) else X, dtype=np.float32).ravel()
            y = np.empty(M, dtype=np.float32)
            dll.nat_c_gemv_veclut_5trits(
                W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, K, gamma_f
            )
            return Tensor(y, dtype="f32") if isinstance(W_packed, Tensor) else (y if isinstance(W_packed, np.ndarray) else y.tolist())

        W_2d = W_p.reshape(M, bytes_per_row)
        num_groups = bytes_per_row
        x = np.asarray(X.data if isinstance(X, Tensor) else X, dtype=np.float64).ravel()

        lut01 = np.zeros((num_groups, 9), dtype=np.float64)
        lut23 = np.zeros((num_groups, 9), dtype=np.float64)
        lut4  = np.zeros((num_groups, 3), dtype=np.float64)

        for g in range(num_groups):
            base = g * 5
            x0 = x[base] if base < K else 0.0
            x1 = x[base + 1] if base + 1 < K else 0.0
            x2 = x[base + 2] if base + 2 < K else 0.0
            x3 = x[base + 3] if base + 3 < K else 0.0
            x4 = x[base + 4] if base + 4 < K else 0.0
            for t1 in range(3):
                for t0 in range(3):
                    lut01[g, t0 + 3 * t1] = (t0 - 1) * x0 + (t1 - 1) * x1
                    lut23[g, t0 + 3 * t1] = (t0 - 1) * x2 + (t1 - 1) * x3
            for t4 in range(3):
                lut4[g, t4] = (t4 - 1) * x4

        idx01 = W_2d % 9
        idx23 = (W_2d // 9) % 9
        idx4  = W_2d // 81

        group_indices = np.arange(num_groups)
        acc = (lut01[group_indices, idx01] + lut23[group_indices, idx23] + lut4[group_indices, idx4]).sum(axis=1) * gamma_f
        return acc if isinstance(W_packed, (Tensor, np.ndarray)) else acc.tolist()
    else:
        unpacked = unpack_ternary_5trits_fn(W_p, M * K)
        w_mat = np.array(unpacked, dtype=np.float32).reshape(M, K)
        x_vec = np.array(X.data if isinstance(X, Tensor) else X, dtype=np.float32).ravel()
        res = w_mat.dot(x_vec) * gamma_f
        return res if isinstance(W_packed, (Tensor, np.ndarray)) else res.tolist()


def ternary_add_only_matmul(W: Any, X: Any) -> Any:
    """Multiplicação matricial livre de multiplicações para pesos ternários {-1, 0, 1}."""
    W_arr = W.data if isinstance(W, Tensor) else np.asarray(W)
    X_arr = X.data if isinstance(X, Tensor) else np.asarray(X)
    is_ternary_W = (isinstance(W, Tensor) and W.dtype in ("t2", "ternary")) or getattr(W, "dtype", None) in ("t2", "ternary")
    is_ternary_X = (isinstance(X, Tensor) and X.dtype in ("t2", "ternary")) or getattr(X, "dtype", None) in ("t2", "ternary")
    dll = AOTKernelEngine.get_dll()

    if is_ternary_W and W_arr.ndim == 2 and X_arr.ndim == 1:
        M, K = W_arr.shape
        if dll and len(X_arr) == K:
            W_c = np.ascontiguousarray(W_arr, dtype=np.int8)
            X_c = np.ascontiguousarray(X_arr, dtype=np.float32)
            y = np.empty(M, dtype=np.float32)
            dll.nat_c_gemv_ternary(
                W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, K
            )
            out = Tensor(y, dtype="f32") if isinstance(W, Tensor) or isinstance(X, Tensor) else (y if isinstance(W, np.ndarray) else y.tolist())
            if isinstance(W, Tensor) and hasattr(W, "gamma") and W.gamma != 1.0:
                out = out * W.gamma
            return out
        res = W_arr.astype(np.float32).dot(X_arr.astype(np.float32))
        if isinstance(W, Tensor) and hasattr(W, "gamma") and W.gamma != 1.0:
            res = res * W.gamma
    elif is_ternary_X and W_arr.ndim == 1 and X_arr.ndim == 2:
        K, M = X_arr.shape
        if dll and len(W_arr) == K:
            X_trans = np.ascontiguousarray(X_arr.T, dtype=np.int8)
            W_c = np.ascontiguousarray(W_arr, dtype=np.float32)
            y = np.empty(M, dtype=np.float32)
            dll.nat_c_gemv_ternary(
                X_trans.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, K
            )
            out = Tensor(y, dtype="f32") if isinstance(W, Tensor) or isinstance(X, Tensor) else (y if isinstance(X, np.ndarray) else y.tolist())
            if isinstance(X, Tensor) and hasattr(X, "gamma") and X.gamma != 1.0:
                out = out * X.gamma
            return out
        res = W_arr.astype(np.float32).dot(X_arr.astype(np.float32))
        if isinstance(X, Tensor) and hasattr(X, "gamma") and X.gamma != 1.0:
            res = res * X.gamma
    elif is_ternary_W and W_arr.ndim == 1 and X_arr.ndim == 1:
        pos = (W_arr == 1)
        neg = (W_arr == -1)
        res = np.sum(X_arr[pos]) - np.sum(X_arr[neg])
    else:
        res = np.matmul(W_arr, X_arr)

    if isinstance(W, Tensor) or isinstance(X, Tensor):
        return Tensor(res, dtype="f32")
    if isinstance(W, np.ndarray) or isinstance(X, np.ndarray):
        return res
    if isinstance(res, np.ndarray):
        return res.tolist()
    return res.item() if hasattr(res, "item") else res


GAUSSIAN_CODEBOOKS = {
    1: np.array([-0.7979, 0.7979], dtype=np.float32),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32),
    3: np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152], dtype=np.float32),
    4: np.array([-2.73, -2.07, -1.62, -1.26, -0.94, -0.66, -0.39, -0.13,
                 0.13, 0.39, 0.66, 0.94, 1.26, 1.62, 2.07, 2.73], dtype=np.float32)
}


def fwht(a):
    a = np.array(a, dtype=np.float32)
    n = len(a)
    p2 = 1 << (n - 1).bit_length()
    if p2 != n:
        a = np.pad(a, (0, p2 - n))
    n = p2
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        x = a[:, 0, :].copy()
        y = a[:, 1, :].copy()
        a[:, 0, :] = x + y
        a[:, 1, :] = x - y
        h *= 2
    a = a.reshape(-1)
    return a / np.sqrt(n)


def rht_fn(x, seed=42):
    rng = np.random.RandomState(seed)
    x_arr = np.array(x, dtype=np.float32)
    n = len(x_arr)
    p2 = 1 << (n - 1).bit_length()
    signs = rng.choice([-1.0, 1.0], size=p2).astype(np.float32)
    x_pad = np.pad(x_arr, (0, p2 - n)) if n < p2 else x_arr
    res = fwht(x_pad * signs)
    return res[:n].tolist() if n != p2 else res.tolist()


def turbo_quant_fn(x, bits=2):
    x_arr = np.array(x, dtype=np.float32)
    norm = float(np.linalg.norm(x_arr))
    if norm == 0:
        return {'indices': [], 'qjl': [], 'norm': 0.0, 'bits': bits, 'dim': len(x_arr)}
    unit_x = x_arr / norm
    rot_x = np.array(rht_fn(unit_x))
    d = len(rot_x)
    std_rot = rot_x * np.sqrt(d)
    cb = GAUSSIAN_CODEBOOKS.get(bits, GAUSSIAN_CODEBOOKS[2])
    diffs = np.abs(std_rot[:, None] - cb[None, :])
    indices = np.argmin(diffs, axis=1).astype(np.int8)
    recon_std = cb[indices]
    recon_rot = recon_std / np.sqrt(d)
    res = rot_x - recon_rot
    qjl = np.sign(res).astype(np.int8)
    return {
        'indices': indices.tolist(),
        'qjl': qjl.tolist(),
        'norm': norm,
        'bits': bits,
        'dim': len(x_arr)
    }


def turbo_dequant_fn(q_dict):
    cb = GAUSSIAN_CODEBOOKS.get(q_dict['bits'], GAUSSIAN_CODEBOOKS[2])
    indices = np.array(q_dict['indices'])
    d = len(indices)
    recon_rot = cb[indices] / np.sqrt(d)
    p2 = 1 << (d - 1).bit_length()
    rng = np.random.RandomState(42)
    signs = rng.choice([-1.0, 1.0], size=p2).astype(np.float32)
    recon_rot_pad = np.pad(recon_rot, (0, p2 - d)) if len(recon_rot) < p2 else recon_rot
    inv = fwht(recon_rot_pad) * signs
    orig_len = q_dict['dim']
    return (inv[:orig_len] * q_dict['norm']).tolist()


def turbo_dot_fn(query, q_dict):
    q_arr = np.array(query, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_arr))
    if q_norm == 0 or q_dict['norm'] == 0:
        return 0.0
    unit_q = q_arr / q_norm
    rot_q = np.array(rht_fn(unit_q))
    d = len(rot_q)
    cb = GAUSSIAN_CODEBOOKS.get(q_dict['bits'], GAUSSIAN_CODEBOOKS[2])
    indices = np.array(q_dict['indices'])
    recon_k_rot = cb[indices] / np.sqrt(d)
    dot_main = np.dot(rot_q, recon_k_rot)
    return float(dot_main * q_norm * q_dict['norm'])


def rmsnorm_fn(x: Any, gamma: Any, eps: float = 1e-6) -> Any:
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    g_arr = gamma.data if isinstance(gamma, Tensor) else np.asarray(gamma, dtype=np.float32)
    dll = AOTKernelEngine.get_dll()
    if dll and x_arr.ndim == 1 and len(x_arr) == len(g_arr):
        x_c = np.ascontiguousarray(x_arr, dtype=np.float32)
        g_c = np.ascontiguousarray(g_arr, dtype=np.float32)
        out = np.empty_like(x_c)
        dll.nat_c_rmsnorm(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x_c), eps
        )
        return Tensor(out, dtype="f32") if isinstance(x, Tensor) else (out if isinstance(x, np.ndarray) else out.tolist())
    rms = np.sqrt(np.mean(x_arr ** 2) + eps)
    res = (x_arr / rms) * g_arr
    return Tensor(res, dtype="f32") if isinstance(x, Tensor) else (res if isinstance(x, np.ndarray) else res.tolist())


def apply_rope_fn(x: Any, pos: Any, base: float = 10000.0) -> Any:
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    dim = len(x_arr)
    half = dim // 2
    x1 = x_arr[:half]
    x2 = x_arr[half:]
    freqs = 1.0 / (base ** (np.arange(0, half, dtype=float) / half))
    angles = float(pos) * freqs
    out1 = x1 * np.cos(angles) - x2 * np.sin(angles)
    out2 = x1 * np.sin(angles) + x2 * np.cos(angles)
    res = np.concatenate([out1, out2])
    return Tensor(res, dtype="f32") if isinstance(x, Tensor) else (res if isinstance(x, np.ndarray) else res.tolist())


def silu_fn(x: Any) -> Any:
    if isinstance(x, (int, float)):
        return float(x / (1.0 + math.exp(-x)))
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    dll = AOTKernelEngine.get_dll()
    if dll and x_arr.ndim == 1:
        x_c = np.ascontiguousarray(x_arr, dtype=np.float32)
        out = np.empty_like(x_c)
        dll.nat_c_silu(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x_c)
        )
        return Tensor(out, dtype="f32") if isinstance(x, Tensor) else (out if isinstance(x, np.ndarray) else out.tolist())
    res = x_arr / (1.0 + np.exp(-x_arr))
    return Tensor(res, dtype="f32") if isinstance(x, Tensor) else (res if isinstance(x, np.ndarray) else res.tolist())


def softmax_fn(x: Any, axis: int = -1) -> Any:
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    dll = AOTKernelEngine.get_dll()
    if dll and x_arr.ndim == 1:
        x_c = np.ascontiguousarray(x_arr, dtype=np.float32)
        out = np.empty_like(x_c)
        dll.nat_c_softmax(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x_c)
        )
        return Tensor(out, dtype="f32") if isinstance(x, Tensor) else (out if isinstance(x, np.ndarray) else out.tolist())
    max_v = np.max(x_arr, axis=axis, keepdims=True)
    exp_v = np.exp(x_arr - max_v)
    sum_v = np.sum(exp_v, axis=axis, keepdims=True)
    res = exp_v / sum_v
    return Tensor(res, dtype="f32") if isinstance(x, Tensor) else (res if isinstance(x, np.ndarray) else res.tolist())


def causal_mask_fn(seq_len: int) -> Tensor:
    mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
    mask = np.triu(mask, k=1)
    return Tensor(mask, dtype="f32")


def sdpa_fn(q: Any, k: Any, v: Any, mask: Any = None, scale: Optional[float] = None) -> Tensor:
    Q = q.data if isinstance(q, Tensor) else np.asarray(q, dtype=np.float32)
    K = k.data if isinstance(k, Tensor) else np.asarray(k, dtype=np.float32)
    V = v.data if isinstance(v, Tensor) else np.asarray(v, dtype=np.float32)

    d_k = Q.shape[-1]
    inv_sqrt = (1.0 / math.sqrt(d_k)) if scale is None else scale
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * inv_sqrt

    if mask is not None:
        M = mask.data if isinstance(mask, Tensor) else np.asarray(mask, dtype=np.float32)
        scores = scores + M

    max_s = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    out = np.matmul(probs, V)
    return Tensor(out, dtype="f32")


def concat_fn(tensors: list[Any], axis: int = 0) -> Tensor:
    """Concatena uma lista de tensores ao longo de um eixo."""
    arrs = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    res = np.concatenate(arrs, axis=axis)
    return Tensor(res, dtype="f32")


def sample_fn(logits: Any, temperature: float = 1.0, top_p: float = 0.9) -> int:
    """Amostragem estocástica com Temperatura e Nucleus Sampling (Top-P)."""
    arr = logits.data if isinstance(logits, Tensor) else np.asarray(logits, dtype=np.float32)
    arr = arr.ravel().astype(np.float64)
    if temperature <= 1e-4:
        return int(np.argmax(arr))
    scaled = arr / max(temperature, 1e-4)
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    probs = exp_vals / np.sum(exp_vals)
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_idx = int(np.searchsorted(cumulative_probs, top_p))
        valid_indices = sorted_indices[:cutoff_idx + 1]
        valid_probs = sorted_probs[:cutoff_idx + 1]
        sum_p = np.sum(valid_probs)
        if sum_p > 0:
            valid_probs = valid_probs / sum_p
            return int(np.random.choice(valid_indices, p=valid_probs))
    return int(np.random.choice(len(probs), p=probs))


class KVCache:
    """Cache de Chaves e Valores para inferência autoregressiva token-a-token com Memory Arena (estilo llama2.c).
    Pré-aloca a memória contígua total upfront no início da sessão, eliminando re-alocações dinâmicas,
    cópias redundantes e fragmentação de memória a cada token gerado.
    """
    def __init__(self, max_seq_len: int = 2048, dim: int = 4096):
        self.max_seq_len = int(max_seq_len) if max_seq_len > 0 else 2048
        self.dim = int(dim) if dim > 0 else 64
        self.seq_len: int = 0

        # Memory Arena contígua pré-alocada (estilo state->key_cache e state->value_cache em llama2.c)
        self._arena_k = np.zeros((self.max_seq_len, self.dim), dtype=np.float32)
        self._arena_v = np.zeros((self.max_seq_len, self.dim), dtype=np.float32)
        self.k: Optional[Tensor] = None
        self.v: Optional[Tensor] = None

    def update(self, new_k: Any, new_v: Any) -> tuple[Tensor, Tensor]:
        """Insere novos tokens na Memory Arena sem re-alocação de memória."""
        nk = new_k.data if isinstance(new_k, Tensor) else np.asarray(new_k, dtype=np.float32)
        nv = new_v.data if isinstance(new_v, Tensor) else np.asarray(new_v, dtype=np.float32)

        # Normaliza formato para 2D: [n_tokens, dim]
        if nk.ndim == 1:
            nk = nk.reshape(1, -1)
            nv = nv.reshape(1, -1)
        elif nk.ndim > 2:
            nk = nk.reshape(-1, nk.shape[-1])
            nv = nv.reshape(-1, nv.shape[-1])

        n_tokens = int(nk.shape[0])
        token_dim = int(nk.shape[-1])

        # Se a dimensão não corresponder e nenhum token foi inserido, redimensiona a arena
        if token_dim != self.dim and self.seq_len == 0:
            self.dim = token_dim
            self._arena_k = np.zeros((self.max_seq_len, self.dim), dtype=np.float32)
            self._arena_v = np.zeros((self.max_seq_len, self.dim), dtype=np.float32)

        # Se ultrapassar a capacidade máxima, expande a arena (dobra capacidade como salvaguarda)
        if self.seq_len + n_tokens > self.max_seq_len:
            new_max = max(self.max_seq_len * 2, self.seq_len + n_tokens + 1024)
            new_ak = np.zeros((new_max, self.dim), dtype=np.float32)
            new_av = np.zeros((new_max, self.dim), dtype=np.float32)
            if self.seq_len > 0:
                new_ak[:self.seq_len] = self._arena_k[:self.seq_len]
                new_av[:self.seq_len] = self._arena_v[:self.seq_len]
            self._arena_k = new_ak
            self._arena_v = new_av
            self.max_seq_len = new_max

        # Cópia direta para o slot contíguo pré-alocado (ZERO REALLOC / ZERO CONCAT)
        self._arena_k[self.seq_len : self.seq_len + n_tokens] = nk
        self._arena_v[self.seq_len : self.seq_len + n_tokens] = nv
        self.seq_len += n_tokens

        # Slices em visualização contígua (O(1), zero cópia de buffer)
        self.k = Tensor(self._arena_k[:self.seq_len], dtype="f32")
        self.v = Tensor(self._arena_v[:self.seq_len], dtype="f32")
        return self.k, self.v

    def clear(self):
        """Reinicia o ponteiro de sequência sem desalocar a arena de memória."""
        self.seq_len = 0
        self.k = None
        self.v = None

    def reset(self):
        self.clear()

    @property
    def capacity(self) -> int:
        return self.max_seq_len

    @property
    def memory_bytes(self) -> int:
        return int(self._arena_k.nbytes + self._arena_v.nbytes)

    @property
    def memory_mb(self) -> float:
        return float(self.memory_bytes / (1024.0 * 1024.0))

    def slice_k(self, start: int, end: int) -> Tensor:
        start_idx = max(0, min(start, self.seq_len))
        end_idx = max(0, min(end, self.seq_len))
        return Tensor(self._arena_k[start_idx:end_idx], dtype="f32")

    def slice_v(self, start: int, end: int) -> Tensor:
        start_idx = max(0, min(start, self.seq_len))
        end_idx = max(0, min(end, self.seq_len))
        return Tensor(self._arena_v[start_idx:end_idx], dtype="f32")

    def __repr__(self) -> str:
        return f"KVCache(seq_len={self.seq_len}/{self.max_seq_len}, dim={self.dim}, arena={self.memory_mb:.2f}MB)"


def _bytes_to_unicode() -> dict[int, str]:
    """Tabela canônica de mapeamento byte -> caractere unicode (estilo GPT-2 / HuggingFace / Qwen)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tokenizer:
    """Tokenizador BPE com regras completas de Merge e Byte-Fallback (estilo LLaMA/Mistral/Qwen/GPT).
    Suporta:
      1. Vocabulário carregado de arquivos tokenizer.json (HuggingFace/Qwen/LLaMA) ou padrão interno.
      2. Regras canônicas de BPE Merge com prioridades de ranking.
      3. Added tokens / special tokens (<|endoftext|>, etc.).
      4. Byte-level mapping GPT-2/Qwen e fallback robusto para <0xXX>.
    """
    def __init__(self, vocab_file: Optional[str] = None):
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        self.special_tokens: dict[str, int] = {}
        self._special_regex: Optional[Any] = None
        self.normalizer_type: Optional[str] = None
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._pre_regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÿ_]+| ?[0-9]+| ?[^\s\w]+|\s+""")

        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)
        else:
            self._init_default_vocab()
        self.vocab_size = len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def _init_default_vocab(self):
        specials = ["<unk>", "<s>", "</s>", "<pad>"]
        for idx, sp in enumerate(specials):
            self.vocab[sp] = idx
            self.inv_vocab[idx] = sp

        for b in range(256):
            b_tok = f"<0x{b:02X}>"
            idx = len(self.vocab)
            self.vocab[b_tok] = idx
            self.inv_vocab[idx] = b_tok

        common_words = [
            " ", "a", "e", "o", "s", "r", "i", "n", "d", "m", "u", "t", "c", "l", "p", "v", "g", "b", "f", "h", "q", "j", "x", "z", "k", "w", "y",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            ".", ",", ";", ":", "!", "?", "-", "_", "/", "\\", "(", ")", "[", "]", "{", "}", "=", "+", "*", "<", ">", "\"", "'", "\n", "\t",
            " de", " a", " o", " que", " e", " do", " da", " em", " um", " para", " com", " não", " uma", " os", " no", " se", " na", " por", " mais", " as", " dos", " como", " mas", " foi", " ao", " ele", " das", " tem", " à", " seu", " sua", " ou", " ser", " quando", " muito", " há", " nos", " já", " está", " eu", " também", " só", " pelo", " pela", " até", " isso", " ela", " entre", " era", " depois", " sem", " mesmo", " aos", " ter", " seus", " quem", " nas", " me", " esse", " eles", " estão", " você", " tinha", " foram", " essa", " num", " nem", " suas", " meu", " às", " minha", " têm", " numa", " qual", " nós",
            " the", " of", " and", " to", " in", " is", " you", " that", " it", " he", " was", " for", " on", " are", " as", " with", " his", " they", " I", " at", " be", " this", " have", " from", " or", " one", " had", " by", " word", " but", " not", " what", " all", " were", " we", " when", " your", " can", " said", " there", " use", " an", " each", " which", " she", " do", " how", " their", " if", " will", " up", " other", " about", " out", " many", " then", " them", " these", " so", " some", " her", " would", " make", " like", " him", " into", " time", " has", " look", " two", " more", " write", " go", " see", " number", " no", " way", " could", " people", " my", " than", " first", " water", " been", " call", " who", " oil", " its", " now", " find",
            "fn", "val", "var", "const", "return", "if", "else", "while", "for", "in", "import", "def", "class", "print", "tensor", "Nativa", "llama", "bitnet", "model", "self", "attention", "token", "layer"
        ]
        for w in common_words:
            if w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.inv_vocab[idx] = w

    def _bpe_merge(self, word: str) -> list[str]:
        """Aplica as regras BPE de merges sobre a sequência de caracteres."""
        if not self.bpe_ranks or len(word) <= 1:
            return [word] if word in self.vocab else list(word)
        parts = list(word)
        while len(parts) >= 2:
            min_rank = float("inf")
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> Tensor:
        tokens: list[int] = []
        if not text:
            return Tensor([], dtype="i64")

        if self.normalizer_type == "NFC":
            text = unicodedata.normalize("NFC", text)

        # Se houver regras de merges BPE carregadas (ex: Qwen, LLaMA):
        if self.bpe_ranks:
            if self._special_regex:
                segments = [s for s in self._special_regex.split(text) if s]
            else:
                segments = [text]

            for seg in segments:
                if seg in self.special_tokens:
                    tokens.append(self.special_tokens[seg])
                    continue

                chunks = self._pre_regex.findall(seg)
                for chunk in chunks:
                    if chunk in self.vocab:
                        tokens.append(self.vocab[chunk])
                        continue
                    byte_encoded = "".join(self.byte_encoder.get(b, chr(b)) for b in chunk.encode("utf-8"))
                    if byte_encoded in self.vocab:
                        tokens.append(self.vocab[byte_encoded])
                        continue
                    subwords = self._bpe_merge(byte_encoded)
                    for sub in subwords:
                        if sub in self.vocab:
                            tokens.append(self.vocab[sub])
                        else:
                            for char in sub:
                                if char in self.vocab:
                                    tokens.append(self.vocab[char])
                                else:
                                    b_val = self.byte_decoder.get(char, ord(char) % 256)
                                    b_tok = f"<0x{b_val:02X}>"
                                    tokens.append(self.vocab.get(b_tok, self.vocab.get("<unk>", 0)))
            return Tensor(tokens, dtype="i64")

        # Fallback: Greedy Substring Match com Byte-Fallback (Vocabulário padrão sem merges)
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for length in range(min(32, n - i), 0, -1):
                sub = text[i:i + length]
                if sub in self.vocab:
                    tokens.append(self.vocab[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                char_bytes = text[i].encode("utf-8")
                for b in char_bytes:
                    tok_str = f"<0x{b:02X}>"
                    tokens.append(self.vocab.get(tok_str, 0))
                i += 1
        return Tensor(tokens, dtype="i64")

    def decode(self, tokens: Any) -> str:
        tok_list = tokens.tolist() if isinstance(tokens, Tensor) else list(tokens)
        res = []
        byte_buf = bytearray()
        for t in tok_list:
            piece = self.inv_vocab.get(t, "<unk>")
            if piece in ("<s>", "</s>", "<pad>", "<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"):
                continue
            if piece.startswith("<0x") and piece.endswith(">"):
                try:
                    b_val = int(piece[3:5], 16)
                    byte_buf.append(b_val)
                    continue
                except ValueError:
                    pass
            is_mapped = all(c in self.byte_decoder for c in piece) if piece else False
            if is_mapped:
                for c in piece:
                    byte_buf.append(self.byte_decoder[c])
                continue
            if byte_buf:
                res.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()
            res.append(piece)
        if byte_buf:
            res.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(res)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "model" in data and "vocab" in data["model"]:
            self.vocab = dict(data["model"]["vocab"])
        elif isinstance(data, dict) and "vocab" in data:
            self.vocab = dict(data["vocab"])
        elif isinstance(data, dict):
            self.vocab = dict(data)

        if "added_tokens" in data and isinstance(data["added_tokens"], list):
            for item in data["added_tokens"]:
                if isinstance(item, dict) and "content" in item and "id" in item:
                    self.vocab[item["content"]] = item["id"]
                    self.special_tokens[item["content"]] = item["id"]

        if self.special_tokens:
            try:
                import regex as _reg
                self._special_regex = _reg.compile("(" + "|".join(_reg.escape(k) for k in sorted(self.special_tokens.keys(), key=len, reverse=True)) + ")")
            except Exception:
                self._special_regex = re.compile("(" + "|".join(re.escape(k) for k in sorted(self.special_tokens.keys(), key=len, reverse=True)) + ")")

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        merges_raw = []
        if "model" in data and "merges" in data["model"]:
            merges_raw = data["model"]["merges"]
        elif "merges" in data and isinstance(data["merges"], list):
            merges_raw = data["merges"]

        self.merges = []
        self.bpe_ranks = {}
        for rank, m in enumerate(merges_raw):
            if isinstance(m, str):
                parts = m.split(" ")
                if len(parts) == 2:
                    pair = (parts[0], parts[1])
                    self.merges.append(pair)
                    self.bpe_ranks[pair] = rank
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                pair = (str(m[0]), str(m[1]))
                self.merges.append(pair)
                self.bpe_ranks[pair] = rank

        # Extrai normalizador (ex: NFC)
        if "normalizer" in data and isinstance(data["normalizer"], dict):
            self.normalizer_type = data["normalizer"].get("type")

        # Extrai padrão do pré-tokenizador oficial se declarado no JSON
        if "pre_tokenizer" in data and isinstance(data["pre_tokenizer"], dict):
            pt = data["pre_tokenizer"]
            pat_str = None
            if pt.get("type") == "Sequence" and "pretokenizers" in pt:
                for sub_pt in pt["pretokenizers"]:
                    if isinstance(sub_pt, dict) and "pattern" in sub_pt:
                        p_info = sub_pt["pattern"]
                        if isinstance(p_info, dict) and "Regex" in p_info:
                            pat_str = p_info["Regex"]
                            break
            elif "pattern" in pt and isinstance(pt["pattern"], dict) and "Regex" in pt["pattern"]:
                pat_str = pt["pattern"]["Regex"]

            if pat_str:
                try:
                    import regex as _regex_mod
                    self._pre_regex = _regex_mod.compile(pat_str)
                except Exception:
                    translated = pat_str.replace(r"\p{L}", r"[^\W\d_]").replace(r"\p{N}", r"\d")
                    try:
                        self._pre_regex = re.compile(translated)
                    except Exception:
                        pass


DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def _resolve_safetensors_path(path: str) -> str:
    if os.path.exists(path):
        return path
    candidate = os.path.join("data", path)
    if os.path.exists(candidate):
        return candidate
    candidate_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", os.path.basename(path))
    if os.path.exists(candidate_parent):
        return candidate_parent
    return path


_MMAP_CACHE: dict[str, tuple[Any, mmap.mmap, int, dict]] = {}


def close_safetensors_mmap(path: Optional[str] = None) -> None:
    """Fecha mapeamento de memória mmap aberto para o caminho informado ou todos."""
    if path is None:
        for p in list(_MMAP_CACHE.keys()):
            close_safetensors_mmap(p)
        return
    resolved = _resolve_safetensors_path(path)
    if resolved in _MMAP_CACHE:
        fh, mm, _, _ = _MMAP_CACHE.pop(resolved)
        try:
            mm.close()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass


def _get_mmap(path: str) -> tuple[mmap.mmap, int, dict]:
    resolved = _resolve_safetensors_path(path)
    if resolved in _MMAP_CACHE:
        _, mm, data_start, header = _MMAP_CACHE[resolved]
        return mm, data_start, header
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Arquivo safetensors não encontrado: '{resolved}'")
    file_size = os.path.getsize(resolved)
    if file_size < 8:
        raise ValueError(f"Arquivo safetensors inválido ou truncado: '{resolved}' (tamanho {file_size} bytes < cabeçalho de 8 bytes)")

    fh = open(resolved, "rb")
    try:
        header_size_bytes = fh.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"Arquivo safetensors truncado ao ler tamanho do cabeçalho: '{resolved}'")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size <= 0:
            raise ValueError(f"Arquivo safetensors corrompido: header_size = {header_size} em '{resolved}'")
        if header_size > file_size - 8:
            raise ValueError(
                f"Arquivo safetensors corrompido em '{resolved}': header_size ({header_size} bytes) "
                f"excede o tamanho total do arquivo ({file_size} bytes)"
            )
        if header_size > 100 * 1024 * 1024:
            raise ValueError(f"Cabeçalho safetensors excede o limite de segurança de 100MB ({header_size} bytes) em '{resolved}'")

        header_json_bytes = fh.read(header_size)
        if len(header_json_bytes) != header_size:
            raise ValueError(f"Arquivo safetensors truncado: esperava {header_size} bytes de cabeçalho, leu {len(header_json_bytes)} em '{resolved}'")
        try:
            header = json.loads(header_json_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Cabeçalho JSON do safetensors malformado em '{resolved}': {e}")
        if not isinstance(header, dict):
            raise ValueError(f"Cabeçalho safetensors inválido (esperado objeto JSON) em '{resolved}'")

        data_start = 8 + header_size
        tensor_intervals: list[tuple[int, int, str]] = []
        for tname, tinfo in header.items():
            if tname == "__metadata__":
                continue
            if not isinstance(tinfo, dict):
                raise ValueError(f"Metadados inválidos para tensor '{tname}' em '{resolved}': esperado dicionário")
            for req in ("dtype", "shape", "data_offsets"):
                if req not in tinfo:
                    raise ValueError(f"Metadados incompletos para tensor '{tname}' em '{resolved}': campo obrigatório '{req}' ausente")
            offsets = tinfo["data_offsets"]
            if not (isinstance(offsets, list) and len(offsets) == 2 and isinstance(offsets[0], int) and isinstance(offsets[1], int)):
                raise ValueError(f"Offsets inválidos para o tensor '{tname}' em '{resolved}': {offsets}")
            if offsets[0] < 0 or offsets[1] < offsets[0]:
                raise ValueError(f"Intervalo de offsets inválido para o tensor '{tname}' em '{resolved}': {offsets}")
            if data_start + offsets[1] > file_size:
                raise ValueError(
                    f"Arquivo safetensors truncado em '{resolved}': tensor '{tname}' requer offset final "
                    f"{data_start + offsets[1]}, mas arquivo tem apenas {file_size} bytes"
                )
            tensor_intervals.append((offsets[0], offsets[1], tname))

        # Validação canônica SafeTensors: verificação de sobreposição e cobertura contígua
        if tensor_intervals:
            tensor_intervals.sort(key=lambda x: (x[0], x[1]))
            if tensor_intervals[0][0] != 0:
                raise ValueError(
                    f"Arquivo safetensors inválido em '{resolved}': primeiro tensor inicia no offset {tensor_intervals[0][0]} "
                    f"(esperado offset 0, bytes não referenciados antes do primeiro tensor)."
                )
            for idx in range(len(tensor_intervals) - 1):
                cur_start, cur_end, cur_name = tensor_intervals[idx]
                nxt_start, nxt_end, nxt_name = tensor_intervals[idx + 1]
                if nxt_start < cur_end:
                    raise ValueError(
                        f"Sobreposição ilegal de tensores em '{resolved}': tensor '{cur_name}' [{cur_start}:{cur_end}] "
                        f"sobrepõe tensor '{nxt_name}' [{nxt_start}:{nxt_end}]."
                    )
                if nxt_start > cur_end:
                    raise ValueError(
                        f"Lacuna ilegal de bytes em '{resolved}': intervalo [{cur_end}:{nxt_start}] "
                        f"entre '{cur_name}' e '{nxt_name}' não pertence a nenhum tensor."
                    )
            last_end = tensor_intervals[-1][1]
            total_payload = file_size - data_start
            if last_end < total_payload:
                raise ValueError(
                    f"Bytes excedentes não referenciados em '{resolved}': último tensor termina em {last_end}, "
                    f"mas o payload do arquivo possui {total_payload} bytes ({total_payload - last_end} bytes excedentes no final)."
                )

        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        if hasattr(mm, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
            try:
                mm.madvise(mmap.MADV_WILLNEED)
            except Exception:
                pass
        _MMAP_CACHE[resolved] = (fh, mm, data_start, header)
        return mm, data_start, header
    except Exception:
        fh.close()
        raise


def safetensors_read_header(path: str) -> tuple[int, dict]:
    _, data_start, header = _get_mmap(path)
    return data_start, header


def safetensors_keys_fn(path: str) -> list[str]:
    _, header = safetensors_read_header(path)
    return [k for k in header.keys() if k != "__metadata__"]


def load_safetensors_fn(path: str, tensor_name: str) -> Any:
    mm, data_start, header = _get_mmap(path)
    if tensor_name not in header:
        available = [k for k in header.keys() if k != "__metadata__"]
        raise KeyError(f"Tensor '{tensor_name}' não encontrado em {path}. Tensores disponíveis: {available}")
    info = header[tensor_name]
    dtype_str = info.get("dtype", "F32")
    shape = info.get("shape", [])
    start_offset, end_offset = info["data_offsets"]

    if dtype_str not in DTYPE_MAP and dtype_str != "BF16":
        supported = list(DTYPE_MAP.keys()) + ["BF16"]
        raise ValueError(f"Tipo de dado '{dtype_str}' não suportado para o tensor '{tensor_name}'. Tipos suportados: {supported}")

    # Validação rigorosa de dimensões e contagem de bytes
    declared_bytes = end_offset - start_offset
    item_size = 2 if dtype_str in ("F16", "BF16", "I16") else (8 if dtype_str in ("F64", "I64") else (1 if dtype_str in ("I8", "U8", "BOOL") else 4))

    expected_numel = 1
    for dim_len in shape:
        if not isinstance(dim_len, int) or dim_len < 0:
            raise ValueError(f"Dimensão inválida {dim_len} no shape {shape} do tensor '{tensor_name}' em '{path}'")
        expected_numel *= dim_len

    if shape:
        expected_bytes = expected_numel * item_size
        if expected_bytes != declared_bytes:
            raise ValueError(
                f"Inconsistência de bytes no SafeTensors para o tensor '{tensor_name}' em '{path}': "
                f"shape {shape} ({expected_numel} elementos) com dtype '{dtype_str}' requer {expected_bytes} bytes, "
                f"mas data_offsets declaram {declared_bytes} bytes."
            )

    np_dtype = DTYPE_MAP.get(dtype_str, np.float32)
    byte_start = data_start + start_offset
    byte_end = data_start + end_offset

    if byte_end > len(mm):
        raise ValueError(
            f"Leitura fora dos limites no SafeTensors para o tensor '{tensor_name}': "
            f"offset final {byte_end} excede tamanho mapeado {len(mm)}"
        )

    if dtype_str == "BF16":
        u16 = np.frombuffer(mm, dtype=np.uint16, count=expected_numel, offset=byte_start)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    else:
        num_elements = declared_bytes // item_size
        arr = np.frombuffer(mm, dtype=np_dtype, count=num_elements, offset=byte_start)

    if shape:
        arr = arr.reshape(shape)
    return Tensor(arr, dtype="f32")


def save_safetensors_fn(path: str, tensor_name: str, data: Any) -> None:
    close_safetensors_mmap(path)
    arr = np.array(data)
    if arr.dtype == np.float64:
        dtype_str = "F64"
    elif arr.dtype == np.float32 or arr.dtype == np.float16:
        dtype_str = "F32"
        arr = arr.astype(np.float32)
    elif arr.dtype == np.int64:
        dtype_str = "I64"
    elif arr.dtype in (np.int32, np.int16, np.int8):
        dtype_str = "I32"
        arr = arr.astype(np.int32)
    else:
        arr = arr.astype(np.float32)
        dtype_str = "F32"

    raw_bytes = arr.tobytes()
    header = {
        tensor_name: {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [0, len(raw_bytes)]
        }
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(raw_bytes)


class Interpreter:
    def __init__(self):
        self.globals: dict[str, Any] = {}
        self.const_globals: set[str] = set()
        self.scopes: list[dict[str, Any]] = [{}]
        self.const_scopes: list[set[str]] = [set()]
        self.last_scope: dict[str, Any] = {}
        self.init_builtins()

    def init_builtins(self):
        self.globals = {
            "print": print,
            "len": len,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "ln": math.log,
            "floor": lambda x: int(math.floor(x)),
            "to_i64": int,
            "to_f64": float,
            "dot": lambda a, b: float(np.dot(np.array(a), np.array(b))),
            "matmul": lambda a, b: (Tensor(np.matmul(np.asarray(a), np.asarray(b))) if isinstance(a, Tensor) or isinstance(b, Tensor) else (np.matmul(np.array(a), np.array(b))).tolist()),
            "tanh": math.tanh,
            "relu": lambda x: (Tensor(np.maximum(0, x.data)) if isinstance(x, Tensor) else ((np.maximum(0, np.array(x))).tolist() if isinstance(x, list) else max(0, x))),
            "sigmoid": lambda x: (Tensor(1.0 / (1.0 + np.exp(-x.data))) if isinstance(x, Tensor) else ((1.0 / (1.0 + np.exp(-np.array(x, dtype=float)))).tolist() if isinstance(x, list) else (1.0 / (1.0 + math.exp(-x))))),
            "sum": lambda x: (np.sum(x.data if isinstance(x, Tensor) else np.asarray(x))).item() if isinstance(x, (list, Tensor, np.ndarray)) else x,
            "mean": lambda x: float(np.mean(x.data if isinstance(x, Tensor) else np.asarray(x))),
            "max": lambda x: (np.max(x.data if isinstance(x, Tensor) else np.asarray(x))).item() if isinstance(x, (list, Tensor, np.ndarray)) else x,
            "min": lambda x: (np.min(x.data if isinstance(x, Tensor) else np.asarray(x))).item() if isinstance(x, (list, Tensor, np.ndarray)) else x,
            "argmax": lambda x: int(np.argmax(x.data if isinstance(x, Tensor) else np.asarray(x))),
            "zeros": lambda s: Tensor(np.zeros(s, dtype=np.float32), dtype="f32"),
            "ones": lambda s: Tensor(np.ones(s, dtype=np.float32), dtype="f32"),
            "shape": lambda x: list(x.shape) if hasattr(x, "shape") else list(np.shape(x)),
            "reshape": lambda x, s: (Tensor(np.reshape(np.array(x), s)) if isinstance(x, Tensor) else (np.reshape(np.array(x), s)).tolist()),
            "transpose": lambda x: (Tensor(np.transpose(np.array(x))) if isinstance(x, Tensor) else (np.transpose(np.array(x))).tolist()),
            "tensor": lambda x, dtype="f32", dev=None: Tensor(x, dtype=dtype, device=dev),
            "Tensor": Tensor,
            "device": device_fn,
            "quantize_ternary": quantize_ternary_fn,
            "pack_ternary": pack_ternary_fn,
            "unpack_ternary": unpack_ternary_fn,
            "clock": time.perf_counter,
            "random_ternary": lambda s: Tensor(np.random.choice([-1, 0, 1], size=s).astype(np.int8), dtype="t2"),
            "randn": lambda s: Tensor(np.random.randn(*s if isinstance(s, list) else (s,)).astype(np.float32), dtype="f32"),
            "rotate_rht": rht_fn,
            "turbo_quant": turbo_quant_fn,
            "turbo_dequant": turbo_dequant_fn,
            "turbo_dot": turbo_dot_fn,
            "rmsnorm": rmsnorm_fn,
            "apply_rope": apply_rope_fn,
            "silu": silu_fn,
            "softmax": softmax_fn,
            "causal_mask": causal_mask_fn,
            "sdpa": sdpa_fn,
            "KVCache": KVCache,
            "Tokenizer": Tokenizer,
            "load_tokenizer": lambda p=None: Tokenizer(p),
            "encode": lambda tok, text: tok.encode(text),
            "decode": lambda tok, tokens: tok.decode(tokens),
            "load_safetensors": load_safetensors_fn,
            "safetensors_keys": safetensors_keys_fn,
            "save_safetensors": save_safetensors_fn,
            "quantize_bitnet": quantize_bitnet_fn,
            "tensor_absmean": tensor_absmean_fn,
            "pack_ternary_5trits": pack_ternary_5trits_fn,
            "unpack_ternary_5trits": unpack_ternary_5trits_fn,
            "mae": mae_fn,
            "relative_error": relative_error_fn,
            "pearson_corr": pearson_corr_fn,
            "cosine_sim": cosine_sim_fn,
            "gemv_vec_lut": gemv_vec_lut_fn,
            "concat": concat_fn,
            "sample": sample_fn,
            "math": MathModule(),
        }

    def run(self, program: Program):
        self.const_globals.clear()
        for it in program.items:
            if isinstance(it, Function):
                self.globals[it.name] = it
            elif isinstance(it, ConstDecl):
                val = self.eval_expr(it.value)
                self.globals[it.name] = val
                self.const_globals.add(it.name)
        if "main" in self.globals and isinstance(self.globals["main"], Function):
            return self.call_function(self.globals["main"], [])
        return None

    def call_function(self, fn: Function, args: list[Any]) -> Any:
        local_scope = {}
        for p, arg in zip(fn.params, args):
            local_scope[p.name] = arg
        self.scopes.append(local_scope)
        self.const_scopes.append(set())
        try:
            for stmt in fn.body:
                self.eval_stmt(stmt)
        except ReturnSignal as ret:
            return ret.value
        finally:
            self.const_scopes.pop()
            self.last_scope = self.scopes.pop()
        return None

    def eval_block(self, stmts: list[Node]):
        self.scopes.append({})
        self.const_scopes.append(set())
        try:
            for s in stmts:
                self.eval_stmt(s)
        finally:
            self.const_scopes.pop()
            self.scopes.pop()

    def eval_stmt(self, s: Node):
        if isinstance(s, VarDecl) or isinstance(s, ConstDecl):
            val = self.eval_expr(s.value)
            if hasattr(s, "type") and isinstance(s.type, TypeTensor):
                dt = s.type.elem.name if hasattr(s.type.elem, "name") else "f32"
                val = Tensor(val, dtype=dt)
            self.scopes[-1][s.name] = val
            if isinstance(s, ConstDecl):
                self.const_scopes[-1].add(s.name)
        elif isinstance(s, Assign):
            self.eval_assign(s)
        elif isinstance(s, ExprStmt):
            self.eval_expr(s.expr)
        elif isinstance(s, If):
            if self.eval_expr(s.cond):
                self.eval_block(s.body)
            elif s.orelse:
                self.eval_block(s.orelse)
        elif isinstance(s, While):
            while self.eval_expr(s.cond):
                try:
                    self.eval_block(s.body)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
        elif isinstance(s, For):
            iterator = self.eval_expr(s.iter)
            for item in iterator:
                val = item.item() if hasattr(item, "item") else item
                self.scopes.append({s.var: val})
                self.const_scopes.append(set())
                try:
                    for st in s.body:
                        self.eval_stmt(st)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
                finally:
                    self.const_scopes.pop()
                    self.scopes.pop()
        elif isinstance(s, Return):
            val = self.eval_expr(s.value) if s.value is not None else None
            raise ReturnSignal(val)
        elif isinstance(s, Break):
            raise BreakSignal()
        elif isinstance(s, Continue):
            raise ContinueSignal()
        elif isinstance(s, Pass):
            pass

    def eval_assign(self, node: Assign):
        tgt = node.target
        val = self.eval_expr(node.value)
        if isinstance(tgt, Name):
            name = tgt.id
            target_scope = None
            for sc, c_sc in zip(reversed(self.scopes), reversed(self.const_scopes)):
                if name in sc:
                    if name in c_sc:
                        raise RuntimeError(f"Não é permitido reatribuir a constante '{name}'")
                    target_scope = sc
                    break
            else:
                if name in self.const_globals:
                    raise RuntimeError(f"Não é permitido reatribuir a constante '{name}'")

            if target_scope is None:
                target_scope = self.scopes[-1]

            if node.op == "=":
                target_scope[name] = val
            else:
                base_op = node.op[:-1]
                target_scope[name] = self.eval_binary_op(base_op, target_scope[name], val)
        elif isinstance(tgt, Index):
            obj = self.eval_expr(tgt.obj)
            idx = self.eval_index(tgt.index)
            if node.op == "=":
                obj[idx] = val
            else:
                base_op = node.op[:-1]
                obj[idx] = self.eval_binary_op(base_op, obj[idx], val)
        else:
            raise RuntimeError(f"Alvo de atribuição inválido: {tgt}")

    def eval_index(self, node: Expr):
        if isinstance(node, Range):
            start = self.eval_expr(node.start) if node.start is not None else None
            end = self.eval_expr(node.end) if node.end is not None else None
            if end is not None and node.inclusive:
                end += 1
            return slice(start, end)
        return self.eval_expr(node)

    def eval_expr(self, e: Expr) -> Any:
        if isinstance(e, Int) or isinstance(e, Float) or isinstance(e, Str) or isinstance(e, Bool):
            return e.value
        if isinstance(e, Name):
            for sc in reversed(self.scopes):
                if e.id in sc:
                    return sc[e.id]
            if e.id in self.globals:
                return self.globals[e.id]
            raise RuntimeError(f"Variável não definida: {e.id}")
        if isinstance(e, ListLit):
            return [self.eval_expr(it) for it in e.items]
        if isinstance(e, Range):
            start = self.eval_expr(e.start) if e.start is not None else 0
            end = self.eval_expr(e.end) if e.end is not None else 0
            return range(start, end + 1 if e.inclusive else end)
        if isinstance(e, Index):
            obj = self.eval_expr(e.obj)
            idx = self.eval_index(e.index)
            return obj[idx]
        if isinstance(e, Attr):
            obj = self.eval_expr(e.obj)
            return getattr(obj, e.name)
        if isinstance(e, Unary):
            operand = self.eval_expr(e.operand)
            return self.eval_unary_op(e.op, operand)
        if isinstance(e, Binary):
            left = self.eval_expr(e.left)
            if e.op == "and":
                return left and self.eval_expr(e.right)
            if e.op == "or":
                return left or self.eval_expr(e.right)
            right = self.eval_expr(e.right)
            return self.eval_binary_op(e.op, left, right)
        if isinstance(e, Call):
            fn = self.eval_expr(e.func) if not isinstance(e.func, Attr) else getattr(self.eval_expr(e.func.obj), e.func.name)
            args = [self.eval_expr(a) for a in e.args]
            if isinstance(fn, Function):
                return self.call_function(fn, args)
            return fn(*args)
        if isinstance(e, FString):
            out = []
            for p in e.parts:
                if isinstance(p, FormatPart):
                    val = self.eval_expr(p.expr)
                    if p.spec:
                        out.append(format(val, p.spec))
                    else:
                        out.append(str(val))
                else:
                    out.append(p)
            return "".join(out)
        raise RuntimeError(f"Expressão desconhecida: {e}")

    def eval_unary_op(self, op: str, operand: Any) -> Any:
        if op == "not":
            return not operand
        if op == "-":
            if isinstance(operand, list):
                return (-np.array(operand)).tolist()
            return -operand
        if op == "+":
            return operand
        raise RuntimeError(f"Operador unário desconhecido: {op}")

    def eval_binary_op(self, op: str, left: Any, right: Any) -> Any:
        if op == "==": return left == right
        if op == "!=": return left != right
        if op == "<":  return left < right
        if op == "<=": return left <= right
        if op == ">":  return left > right
        if op == ">=": return left >= right

        if isinstance(left, Tensor) or isinstance(right, Tensor):
            if op == "+": return left + right
            if op == "-": return left - right
            if op == "*": return left * right
            if op == "/": return left / right
            if op == "@": return left @ right

        is_l_arr = isinstance(left, (list, np.ndarray))
        is_r_arr = isinstance(right, (list, np.ndarray))

        if op == "@":
            return ternary_add_only_matmul(left, right)

        if is_l_arr or is_r_arr:
            l_np = np.asarray(left) if is_l_arr else left
            r_np = np.asarray(right) if is_r_arr else right
            if op == "+":   res = l_np + r_np
            elif op == "-": res = l_np - r_np
            elif op == "*": res = l_np * r_np
            elif op == "/": res = l_np / r_np
            elif op == "//": res = l_np // r_np
            elif op == "%": res = l_np % r_np
            elif op == "**": res = l_np ** r_np
            else: raise RuntimeError(f"Operador desconhecido: {op}")

            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                return res
            if isinstance(res, np.ndarray):
                return res.tolist()
            return res.item() if hasattr(res, "item") else res

        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/": return left / right
        if op == "//": return left // right
        if op == "%": return left % right
        if op == "**": return left ** right
        raise RuntimeError(f"Operador não suportado: {op}")


# ============================================================================
# EXEMPLOS EMBUTIDOS
# ============================================================================

EXEMPLOS = {
'exemplo': r'''# Arquivo: exemplo.nat
fn soma(valores: [i64]) -> i64:
    total = 0
    for valor in valores:
        total += valor
    return total

fn main():
    numeros = [10, 20, 30, 40, 50]
    resultado = soma(numeros)
    print(f"Resultado: {resultado}")
''',

'riemann': r'''# riemann.nat
# Protótipo experimental: soma principal da fórmula de Riemann-Siegel.
const PI = 3.14159265358979323846

fn theta(t: f64) -> f64:
    return (
        0.5 * t * ln(t / (2.0 * PI))
        - 0.5 * t
        - PI / 8.0
        + 1.0 / (48.0 * t)
        + 7.0 / (5760.0 * t * t * t)
    )

fn hardy_z_aprox(t: f64) -> f64:
    N: i64 = to_i64(floor(sqrt(t / (2.0 * PI))))
    fase = theta(t)
    soma = 0.0
    for n in 1..N + 1:
        nf = to_f64(n)
        soma += cos(fase - t * ln(nf)) / sqrt(nf)
    return 2.0 * soma

fn bissecao(a_in: f64, b_in: f64) -> f64:
    a = a_in
    b = b_in
    fa = hardy_z_aprox(a)
    for passo in 0..60:
        meio = (a + b) / 2.0
        fm = hardy_z_aprox(meio)
        if fm == 0.0:
            return meio
        if fa * fm < 0.0:
            b = meio
        else:
            a = meio
            fa = fm
    return (a + b) / 2.0

fn main():
    inicio = 10.0
    fim = 100.0
    passo = 0.25
    anterior_t = inicio
    anterior_z = hardy_z_aprox(anterior_t)
    encontrados = 0
    t = inicio + passo
    while t <= fim:
        z = hardy_z_aprox(t)
        if anterior_z * z < 0.0:
            raiz = bissecao(anterior_t, t)
            print(f"Zero candidato: t = {raiz}")
            encontrados += 1
        anterior_t = t
        anterior_z = z
        t += passo
    print(f"Candidatos encontrados: {encontrados}")
''',

'rede': r'''import math

fn main():
    entradas = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [0.0, 1.0, 1.0, 1.0]
    const lr = 0.1
    w = [0.5, -0.2]

    for epoca in 0..1000:
        for i in 0..4:
            soma = entradas[i] @ w
            pred = 1.0 / (1.0 + math.exp(-soma))
            erro = targets[i] - pred
            w += lr * erro * entradas[i]

    print(f"Pesos finais: {w}")
    for i in 0..4:
        soma = entradas[i] @ w
        pred = 1.0 / (1.0 + math.exp(-soma))
        print(f"Entrada {entradas[i]} -> Pred: {pred:.3f} (Alvo: {targets[i]})")
''',

'ia_tensor': r'''import math
import time

fn main():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[2.0, 0.0], [1.0, 2.0]]
    C = A @ B
    print(f"Multiplicacao Matricial C = A @ B: {C}")

    s = sum(C)
    m = mean(C)
    print(f"Soma dos elementos: {s}, Media: {m}")

    ativ = relu(C - 5.0)
    sig = sigmoid(C * 0.1)
    print(f"ReLU(C - 5.0): {ativ}")
    print(f"Sigmoid(C * 0.1): {sig}")

    v = [10, 20, 30, 40, 50]
    fatia = v[1..4]
    print(f"Fatia de vetor v[1..4]: {fatia}")
''',

'ternario': r'''# Exemplo de Computacao Ternaria (1.58 bits) e Matrizes Ternarias
fn main():
    # 1. Matriz de Pesos Ternarios {-1, 0, +1} inspirada em BitNet
    W: tensor[t2] = [
        [1,  0, -1,  1],
        [-1, 1,  0, -1],
        [0, -1,  1,  0]
    ]

    # Vetor de ativacoes continuas f32
    x: [f32] = [0.8, -1.2, 2.5, 0.4]

    # 2. Multiplicacao Matricial Ternaria Nativa (Add-only GEMM livre de multiplicacoes)
    y = W @ x
    print(f"Ativacoes de entrada: {x}")
    print(f"Resultado do Matmul Ternario (W @ x): {y}")

    # 3. Quantizacao Ternaria dinamica das ativacoes
    ativacoes = [1.8, -0.3, 0.1, -1.5, 0.9, -0.1]
    trits = quantize_ternary(ativacoes, 0.5)
    print(f"Ativacoes originais: {ativacoes}")
    print(f"Ativacoes quantizadas em {{-1, 0, 1}}: {trits}")

    # 4. Compactacao de memoria (Bit Packing: 2 bits por trit, 4 trits por byte)
    compactado = pack_ternary(trits)
    descompactado = unpack_ternary(compactado, len(trits))
    print(f"Bytes compactados (4 trits/byte): {compactado}")
    print(f"Valores descompactados: {descompactado}")
''',
}


# ============================================================================
# TESTES — parser e analisador semântico
# ============================================================================

def src(nome): return EXEMPLOS[nome.replace('.nat', '')]
def E(s): return parse_expression(s)
def body(codigo):
    linhas = "\n".join("    " + l for l in codigo.strip("\n").split("\n"))
    return parse("fn main():\n" + linhas + "\n").items[0].body


class Exemplos(unittest.TestCase):
    def test_exemplo(self):
        p = parse(src("exemplo.nat"))
        self.assertEqual([f.name for f in p.items], ["soma", "main"])
        soma = p.items[0]
        self.assertEqual(soma.params[0].type, TypeList(TypeName("i64")))
        self.assertEqual(soma.ret, TypeName("i64"))
        self.assertIsInstance(soma.body[1], For)
        self.assertEqual(soma.body[1].body[0].op, "+=")

    def test_riemann(self):
        p = parse(src("riemann.nat"))
        self.assertEqual([type(i).__name__ for i in p.items],
                         ["ConstDecl", "Function", "Function", "Function", "Function"])
        self.assertEqual([i.name for i in p.items[1:]],
                         ["theta", "hardy_z_aprox", "bissecao", "main"])
        z = p.items[2]
        self.assertEqual(z.body[0], VarDecl("N", TypeName("i64"),
            Call(Name("to_i64"), [Call(Name("floor"), [Call(Name("sqrt"), [
                Binary("/", Name("t"), Binary("*", Float(2.0), Name("PI")))])])])))
        laco = z.body[3]
        self.assertEqual(laco.iter, Range(Int(1), Binary("+", Name("N"), Int(1)), False))
        biss = p.items[3]
        self.assertIsInstance(biss.body[3].body[3], If)
        self.assertEqual(len(biss.body[3].body[3].orelse), 2)

    def test_return_multilinha(self):
        r = parse(src("riemann.nat")).items[1].body[0]
        self.assertIsInstance(r, Return)
        self.assertEqual(r.value.op, "+")


class Expressoes(unittest.TestCase):
    def test_precedencia(self):
        self.assertEqual(E("1 + 2 * 3"), Binary("+", Int(1), Binary("*", Int(2), Int(3))))
        self.assertEqual(E("(1 + 2) * 3"), Binary("*", Binary("+", Int(1), Int(2)), Int(3)))
        self.assertEqual(E("a - b - c"), Binary("-", Binary("-", Name("a"), Name("b")), Name("c")))

    def test_potencia(self):
        self.assertEqual(E("2 ** 3 ** 2"), Binary("**", Int(2), Binary("**", Int(3), Int(2))))
        self.assertEqual(E("-x ** 2"), Unary("-", Binary("**", Name("x"), Int(2))))
        self.assertEqual(E("2 ** -1"), Binary("**", Int(2), Unary("-", Int(1))))

    def test_logicos(self):
        self.assertEqual(E("a or b and not c"),
                         Binary("or", Name("a"), Binary("and", Name("b"), Unary("not", Name("c")))))
        self.assertEqual(E("a + 1 < b * 2"),
                         Binary("<", Binary("+", Name("a"), Int(1)), Binary("*", Name("b"), Int(2))))

    def test_ranges(self):
        self.assertEqual(E("0..N"), Range(Int(0), Name("N"), False))
        self.assertEqual(E("1..=N + 1"), Range(Int(1), Binary("+", Name("N"), Int(1)), True))
        self.assertEqual(E("1..5"), Range(Int(1), Int(5), False))

    def test_postfixo(self):
        self.assertEqual(E("math.sqrt(x)[0]"),
                         Index(Call(Attr(Name("math"), "sqrt"), [Name("x")]), Int(0)))
        self.assertEqual(E("f(a, b,)"), Call(Name("f"), [Name("a"), Name("b")]))

    def test_literais(self):
        self.assertEqual(E("1_000"), Int(1000))
        self.assertEqual(E("1e3"), Float(1000.0))
        self.assertEqual(E("2.5e-1"), Float(0.25))
        self.assertEqual(E("[1, 2.0, true]"), ListLit([Int(1), Float(2.0), Bool(True)]))
        self.assertEqual(E('"a\\tb"'), Str("a\tb"))

    def test_fstring(self):
        e = E('f"x={x:.2f} {{lit}} {a + 1}!"')
        self.assertEqual(e, FString(["x=", FormatPart(Name("x"), ".2f"), " {lit} ",
                                     FormatPart(Binary("+", Name("a"), Int(1)), None), "!"]))
        self.assertEqual(E("f\"{d['k']}\""), FString([FormatPart(Index(Name("d"), Str("k")), None)]))


class Comandos(unittest.TestCase):
    def test_atribuicoes(self):
        b = body("x = 1\nx += 2\nv[0] = 3\na.b = 2\nn: i64 = 0")
        self.assertEqual(b, [Assign(Name("x"), "=", Int(1)), Assign(Name("x"), "+=", Int(2)),
                             Assign(Index(Name("v"), Int(0)), "=", Int(3)), Assign(Attr(Name("a"), "b"), "=", Int(2)),
                             VarDecl("n", TypeName("i64"), Int(0))])

    def test_elif_else(self):
        b = body("if a:\n    x = 1\nelif b:\n    x = 2\nelse:\n    x = 3")
        self.assertEqual(len(b), 1)
        self.assertIsInstance(b[0].orelse[0], If)
        self.assertEqual(b[0].orelse[0].orelse, [Assign(Name("x"), "=", Int(3))])

    def test_bloco_em_uma_linha(self):
        b = body("if a: return 1\nelse: return 2")
        self.assertEqual(b[0].body, [Return(Int(1))])
        self.assertEqual(b[0].orelse, [Return(Int(2))])

    def test_while_break_continue_pass(self):
        b = body("while true:\n    if x:\n        break\n    continue\npass")
        self.assertIsInstance(b[0].body[1], Continue)
        self.assertIsInstance(b[1], Pass)

    def test_return_vazio(self):
        self.assertEqual(body("return"), [Return(None)])

    def test_comentarios_e_linhas_em_branco(self):
        b = body("# c1\nx = 1  # c2\n\n    # indentado demais, mas é só comentário\ny = 2")
        self.assertEqual(len(b), 2)

    def test_dedent_multiplo(self):
        b = body("if a:\n    if b:\n        x = 1\ny = 2")
        self.assertEqual(len(b), 2)

    def test_top_level(self):
        p = parse("import math\nimport a.b\nconst PI: f64 = 3.14\nfn f(a: i64, b: [f64],) -> [f64]:\n    pass\n")
        self.assertEqual(p.items[0], Import("math"))
        self.assertEqual(p.items[1], Import("a.b"))
        self.assertEqual(p.items[2], ConstDecl("PI", TypeName("f64"), Float(3.14)))
        self.assertEqual(len(p.items[3].params), 2)

    def test_sem_newline_final(self):
        parse("fn main():\n    x = 1")
        parse("fn main():\r\n    x = 1\r\n")


class Erros(unittest.TestCase):
    def falha(self, codigo, trecho, linha=None, expr=False):
        with self.assertRaises(NatError) as cm:
            parse_expression(codigo) if expr else parse(codigo)
        self.assertIn(trecho, cm.exception.message)
        if linha: self.assertEqual(cm.exception.line, linha)

    def test_erros(self):
        self.falha("fn f(x):\n    pass\n", "precisam de tipo", 1)
        self.falha("fn f():\nx = 1\n", "bloco indentado", 2)
        self.falha("fn f():\n    x = (1 + 2\n", "não foi fechado", 2)
        self.falha("fn f():\n    x = 1)\n", "sem abertura", 2)
        self.falha("fn f():\n    a = 1\n  b = 2\n", "indentação inconsistente", 3)
        self.falha("fn f():\n    a = 1\n        b = 2\n", "indentação inesperada", 3)
        self.falha("fn f():\n\tx = 1\n", "tabs", 2)
        self.falha("fn f():\n    x = 1 2\n", "fim de linha", 2)
        self.falha("fn f():\n    else:\n        pass\n", "sem 'if'", 2)
        self.falha("fn f():\n    f() = 1\n", "lado esquerdo", 2)
        self.falha("fn f():\n    x = \n", "esperada uma expressão", 2)
        self.falha("fn f():\n    x = $\n", "caractere inesperado", 2)
        self.falha("x = 1\n", "nível superior", 1)
        self.falha('fn f():\n    s = "abc\n', "string não terminada", 2)
        self.falha("fn f():\n    fn g():\n        pass\n", "nível superior", 2)

    def test_encadeamentos_proibidos(self):
        self.falha("a < b < c", "encadeadas", expr=True)
        self.falha("1..2..3", "encadeados", expr=True)

    def test_fstring_erros(self):
        self.falha('f"{}"', "vazia", expr=True)
        self.falha('f"{x"', "sem '}'", expr=True)
        self.falha('f"x}"', "solto", expr=True)
        self.falha('f"{1 +}"', "na f-string", expr=True)

    def test_mensagem_formatada(self):
        s = "fn f():\n    x = (1\n"
        with self.assertRaises(NatError) as cm: parse(s)
        out = cm.exception.pretty(s, "t.nat")
        self.assertIn("t.nat:2:9: erro:", out)
        self.assertIn("x = (1", out)


def analisar(fonte):
    d = SemanticAnalyzer().analyze(parse(fonte))
    return ([x.msg for x in d if x.level == "erro"], [x.msg for x in d if x.level == "aviso"])

def M(corpo, antes=""):
    linhas = "\n".join(("    " + l) if l.strip() else l for l in corpo.strip("\n").split("\n"))
    return antes + "fn main():\n" + linhas + "\n"


class SemBase(unittest.TestCase):
    def ok(self, fonte):
        e, _ = analisar(fonte)
        self.assertEqual(e, [], e)

    def erro(self, fonte, trecho):
        e, _ = analisar(fonte)
        self.assertTrue(any(trecho in x for x in e), f"esperava '{trecho}' em {e}")

    def aviso(self, fonte, trecho):
        e, w = analisar(fonte)
        self.assertEqual(e, [], e)
        self.assertTrue(any(trecho in x for x in w), f"esperava '{trecho}' em {w}")


class SemExemplos(SemBase):
    def test_exemplos_limpos(self):
        for nome in ("exemplo", "riemann", "rede", "ia_tensor", "ternario"):
            self.assertEqual(analisar(EXEMPLOS[nome]), ([], []), nome)


class SemTipos(SemBase):
    def test_tipo_estavel(self):
        self.erro(M('x = 1\nx = "a"\nprint(x)'), "não pode receber")
        self.erro(M("x = 1\nx = 2.5\nprint(x)"), "não pode receber")
        self.ok(M("x = 1.0\nx = 2\nprint(x)"))

    def test_divisao(self):
        self.erro(M("n: i64 = 4 / 2\nprint(n)"), "declarado como")
        self.ok(M("a = 4 / 2\na = 1.5\nprint(a)"))
        self.ok(M("n: i64 = 7 // 2\nprint(n)"))

    def test_operadores(self):
        self.erro(M('a = 1 + "x"\nprint(a)'), "não é definido para")
        self.ok(M('s = "a" + "b"\nprint(s)'))
        self.erro(M('print(1 < "a")'), "não é possível comparar")
        self.erro(M('print(1 == "a")'), "não é possível comparar")

    def test_logicos(self):
        self.erro(M("print(1 and true)"), "'and' exige bool")
        self.erro(M("print(not 1)"), "'not' exige bool")
        self.erro(M("if 1:\n    pass"), "a condição deve ser bool")
        self.erro(M("while 1.5:\n    pass"), "a condição deve ser bool")
        self.ok(M("a = 1 < 2 and not false\nprint(a)"))

    def test_listas(self):
        self.ok(M("a = [1, 2.5]\nb: [f64] = a\nprint(b)"))
        self.erro(M('a = [1, "x"]\nprint(a)'), "incompatíveis")
        self.erro(M("a = [1.0]\nprint(a[0.5])"), "o índice deve ser i64")
        self.erro(M("x = 5\nprint(x[0])"), "não pode ser indexado")
        self.erro(M("for i in 0..2.5:\n    pass"), "limites de um intervalo")
        self.erro(M("for i in 5:\n    pass"), "iterar")
        self.erro(M('v = [1.0, 2.0]\nv[1] = "a"\nprint(v)'), "não pode receber")
        self.erro(M("a: [f64] = [1, 2]\nprint(a)"), "declarado como")

    def test_broadcasting(self):
        self.ok(M("w = [1.0, 2.0]\nw += 2.0\nw = w * w + w\nw = 2.0 * w\nprint(w)"))
        self.erro(M('w = [1.0]\nw += "a"\nprint(w)'), "não é definido para")

    def test_composta(self):
        self.erro(M("x = 1\nx += 2.5\nprint(x)"), "produz")
        self.erro(M("y += 1"), "antes de ser declarado")

    def test_sem_valor(self):
        self.erro(M("x = print(1)"), "não produz valor")


class SemFuncoes(SemBase):
    F = "fn f(a: i64) -> i64:\n    return a\n\n"

    def test_aridade(self):
        self.erro(M("print(sqrt(1.0, 2.0))"), "espera 1")
        self.erro(M("print(sqrt())"), "espera 1")
        self.ok(M("print(1, 2, 3)"))
        self.ok(M("print()"))

    def test_argumentos(self):
        self.erro(M("print(f(2.5))", self.F), "argumento 1")
        self.ok(M("print(f(2))", self.F))
        self.ok(M("print(g(2))", "fn g(a: f64) -> f64:\n    return a\n\n"))
        self.erro(M("print(dot([1, 2], [1.0]))"), "argumento 1")

    def test_retorno(self):
        self.erro("fn f() -> i64:\n    pass\n", "nem todos os caminhos")
        self.erro("fn f() -> i64:\n    if true:\n        return 1\n", "nem todos os caminhos")
        self.ok("fn f(x: bool) -> i64:\n    if x:\n        return 1\n    else:\n        return 2\n")
        self.ok("fn f(x: bool) -> i64:\n    if x:\n        return 1\n    elif true:\n        return 2\n    else:\n        return 3\n")
        self.erro("fn f():\n    return 1\n", "não pode retornar valor")
        self.erro("fn f() -> i64:\n    return 1.5\n", "retorno esperado")
        self.erro("fn f() -> i64:\n    return\n", "deve retornar")
        self.ok("fn f() -> f64:\n    return 1\n")

    def test_hoisting(self):
        self.ok("fn main():\n    print(f())\n\nfn f() -> i64:\n    return 1\n")

    def test_nomes(self):
        self.erro("fn sqrt():\n    pass\n", "embutida")
        self.erro("fn f():\n    pass\nfn f():\n    pass\n", "já foi definida")
        self.erro(M("print = 1"), "já é o nome")
        self.erro(M("x = 1\nx()\nprint(x)"), "é uma variável")
        self.erro(M("print(f)", "fn f():\n    pass\n\n"), "é uma função")
        self.erro("fn f(x: foo):\n    pass\n", "tipo desconhecido")

    def test_math(self):
        self.ok(M("print(math.sqrt(4.0) + math.PI)", "import math\n\n"))
        self.erro(M("print(math.sqrt(4.0))"), "não importado")
        self.erro(M("print(math.foo(1.0))", "import math\n\n"), "não tem")
        self.erro("import xyz\n", "módulo desconhecido")

    def test_break_continue(self):
        self.erro(M("break"), "fora de um laço")
        self.erro(M("continue"), "fora de um laço")
        self.ok(M("while true:\n    if true:\n        break"))


class SemEscopo(SemBase):
    def test_shadowing(self):
        self.erro(M("x = 1\nfor x in 0..3:\n    pass\nprint(x)"), "shadowing")
        self.erro(M("x: i64 = 1\nx: i64 = 2\nprint(x)"), "shadowing")
        self.erro("const N = 5\nfn f(N: i64):\n    pass\n", "shadowing")

    def test_sem_falso_shadowing(self):
        self.ok(M("if true:\n    a = 1\n    print(a)\nif true:\n    a = 2\n    print(a)"))
        self.ok(M("for i in 0..3:\n    pass\nfor i in 0..3:\n    pass"))
        self.ok("fn f(x: i64):\n    pass\nfn g(x: i64):\n    pass\n")

    def test_escopo_de_bloco(self):
        self.erro(M("if true:\n    t = 5\nprint(t)"), "não definido")
        self.ok(M("t = 0\nif true:\n    t = 5\nprint(t)"))
        self.erro(M("if true:\n    pass\nelse:\n    z = 1\n    print(z)\nprint(z)"), "não definido")
        self.erro(M("for i in 0..3:\n    pass\nprint(i)"), "não definido")

    def test_constantes(self):
        self.erro(M("const k = 1\nk = 2"), "constante")
        self.erro("const PI = 3.14\nfn main():\n    PI = 3.0\n", "constante")
        self.ok(M("const x: f64 = 3\nprint(x)"))
        self.erro(M('const x: i64 = "a"\nprint(x)'), "declarada como")
        self.ok("fn main():\n    print(PI)\n\nconst PI = 3.14\n")

    def test_nao_definido(self):
        self.erro(M("while true:\n    y = naoexiste\n    print(y)\n    break"), "não definido")
        self.erro(M('print(f"{naoexiste}")'), "não definido")
        self.erro(M("print(-naoexiste)"), "não definido")
        self.erro(M("total = 1\nprint(totl)"), "você quis dizer 'total'")
        self.erro(M("x = x + 1\nprint(x)"), "não definido")


class SemAvisos(SemBase):
    def test_typo(self):
        self.aviso(M("total = 0\ntotl = total + 1\nprint(total)"), "erro de digitação de 'total'")

    def test_nunca_usada(self):
        self.aviso(M("x = 1"), "nunca é usada")
        self.aviso(M("x = 1\nx += 1"), "nunca é usada")

    def test_sem_falso_positivo(self):
        self.assertEqual(analisar(M("valores = [1, 2]\nvalor = 3\nprint(valores, valor)"))[1], [])
        self.assertEqual(analisar(M("for i in 0..3:\n    pass"))[1], [])
        self.assertEqual(analisar("fn f(a: i64):\n    pass\n")[1], [])


class SemAIETensores(SemBase):
    def test_novos_tipos_ia(self):
        self.ok(M("x: f32 = 1.0\ny: f16 = 2.0\nz: bf16 = 3.0\na: i32 = 10\nb: i8 = 1\nprint(x, y, z, a, b)"))

    def test_tipo_tensor(self):
        self.ok(M("t: tensor[f32] = [1.0, 2.0]\nprint(t)"))

    def test_operador_matmul(self):
        self.ok(M("A = [[1.0, 2.0], [3.0, 4.0]]\nB = [1.0, 2.0]\nC = A @ B\nprint(C)"))
        self.ok(M("v1 = [1.0, 2.0]\nv2 = [3.0, 4.0]\ns = v1 @ v2\nprint(s)"))
        self.erro(M("x = 1.0 @ 2.0"), "não é definido para")

    def test_fatiamento(self):
        self.ok(M("v = [10, 20, 30, 40]\nf = v[1..3]\nprint(f)"))
        self.ok(M('s = "nativa"\nf = s[0..4]\nprint(f)'))

    def test_builtins_ia(self):
        self.ok(M("v = [1.0, -2.0, 3.0]\nr = relu(v)\ns = sigmoid(v)\nm = mean(v)\nsoma = sum(v)\nprint(r, s, m, soma)"))
        self.ok(M("M = zeros([2, 3])\nsh = shape(M)\nprint(sh)"))
        self.ok(M('s = "teste"\nprint(len(s))'))

    def test_safetensors_semantica(self):
        self.ok(M('k = safetensors_keys("teste.safetensors")\nprint(k)'))
        self.ok(M('t = load_safetensors("teste.safetensors", "peso")\nprint(t)'))
        self.ok(M('v = [1.0, 2.0]\nsave_safetensors("teste.safetensors", "peso", v)'))


# --- Testes de Computação Ternária ---
class SemTernario(SemBase):
    def test_tipo_t2_e_ternary(self):
        self.ok(M("w: t2 = 1\nw2: ternary = -1\nprint(w, w2)"))

    def test_tensor_ternario_matmul(self):
        self.ok(M("W: tensor[t2] = [[1, 0, -1], [0, 1, 0]]\nx: [f32] = [0.5, 1.0, 2.0]\ny = W @ x\nprint(y)"))

    def test_primitivas_ternarias(self):
        self.ok(M("v = [0.9, -0.4, 0.1]\nt = quantize_ternary(v, 0.5)\np = pack_ternary(t)\nu = unpack_ternary(p, 3)\nprint(t, p, u)"))


class RuntimeExecucao(unittest.TestCase):
    def test_run_exemplo(self):
        prog = parse(src("exemplo.nat"))
        interp = Interpreter()
        interp.run(prog)

    def test_run_rede_neural(self):
        prog = parse(src("rede.nat"))
        interp = Interpreter()
        interp.run(prog)
        w = interp.last_scope.get("w")
        self.assertIsNotNone(w)
        self.assertGreater(w[0], 2.0)
        self.assertGreater(w[1], 2.0)

    def test_run_ia_tensor(self):
        prog = parse(src("ia_tensor.nat"))
        interp = Interpreter()
        interp.run(prog)

    def test_run_ternario(self):
        prog = parse(src("ternario.nat"))
        interp = Interpreter()
        interp.run(prog)

    def test_turbo_quant_roundtrip(self):
        v = [1.0, 2.0, -1.5, 0.5]
        q = turbo_quant_fn(v, 2)
        rec = turbo_dequant_fn(q)
        dot_est = turbo_dot_fn(v, q)
        self.assertEqual(len(rec), 4)
        self.assertGreater(dot_est, 0.0)

    def test_pack_unpack_roundtrip(self):
        trits = [1, -1, 0, 1, -1, 0, 0, 1, -1, 1]
        packed = pack_ternary_fn(trits)
        unpacked = unpack_ternary_fn(packed, len(trits))
        self.assertEqual(trits, unpacked)

    def test_pack_5trits_roundtrip(self):
        trits = [1, -1, 0, 1, -1, 0, 0, 1, -1, 1, 1, -1, 0]
        packed = pack_ternary_5trits_fn(trits)
        unpacked = unpack_ternary_5trits_fn(packed, len(trits))
        self.assertEqual(trits, unpacked)

    def test_quantize_bitnet_canonical(self):
        dados = [0.8, -0.4, 0.05, -0.9, 0.2]
        gamma = tensor_absmean_fn(dados)
        self.assertAlmostEqual(gamma, 0.47, places=2)
        q = quantize_bitnet_fn(dados)
        self.assertEqual(q, [1, -1, 0, -1, 0])

    def test_safetensors_roundtrip(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tf:
            tmp_path = tf.name
        try:
            dados = [[1.5, -2.0, 3.25], [0.0, 4.5, -1.0]]
            save_safetensors_fn(tmp_path, "camada_mlp", dados)
            chaves = safetensors_keys_fn(tmp_path)
            self.assertEqual(chaves, ["camada_mlp"])
            carregado = load_safetensors_fn(tmp_path, "camada_mlp")
            np.testing.assert_allclose(carregado, dados, rtol=1e-5)
        finally:
            close_safetensors_mmap(tmp_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ============================================================================
# TRANSPILADOR AOT PARA C99
# ============================================================================

class CTranspiler:
    def __init__(self, program: Program):
        self.program = program
        self.indent_level = 0
        self.types_map = {
            "i64": "int64_t", "i32": "int32_t", "i16": "int16_t", "i8": "int8_t",
            "u64": "uint64_t", "u32": "uint32_t", "u16": "uint16_t", "u8": "uint8_t",
            "f64": "double", "f32": "double", "f16": "double", "bf16": "double",
            "t2": "int8_t", "ternary": "int8_t",
            "bool": "bool", "str": "const char*", "void": "void"
        }
        self.var_types: dict[str, str] = {}
        self.func_signatures: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        self._collect_signatures()

    def _collect_signatures(self):
        for it in self.program.items:
            if isinstance(it, Function):
                ret_t = self.c_type(it.ret) if it.ret else "void"
                params = [(p.name, self.c_type(p.type)) for p in it.params]
                self.func_signatures[it.name] = (ret_t, params)

    def indent(self) -> str:
        return "    " * self.indent_level

    def c_type(self, t: Any) -> str:
        if isinstance(t, TypeName):
            return self.types_map.get(t.name, "double")
        if isinstance(t, Prim):
            return self.types_map.get(t.name, "double")
        if isinstance(t, (TypeList, TypeTensor)):
            elem_t = self.c_type(t.elem)
            return f"{elem_t}*"
        if isinstance(t, (ListT, TensorT)):
            elem_t = self.c_type(t.elem)
            return f"{elem_t}*"
        return "double"

    def infer_type(self, expr: Expr) -> str:
        if isinstance(expr, Int):
            return "int64_t"
        if isinstance(expr, Float):
            return "double"
        if isinstance(expr, Bool):
            return "bool"
        if isinstance(expr, Str):
            return "const char*"
        if isinstance(expr, Name):
            return self.var_types.get(expr.id, "double")
        if isinstance(expr, Binary):
            if expr.op in ("<", "<=", ">", ">=", "==", "!=", "and", "or"):
                return "bool"
            if expr.op == "/":
                return "double"
            t_left = self.infer_type(expr.left)
            t_right = self.infer_type(expr.right)
            if "double" in (t_left, t_right) or "float" in (t_left, t_right):
                return "double"
            return t_left
        if isinstance(expr, Unary):
            if expr.op == "not":
                return "bool"
            return self.infer_type(expr.operand)
        if isinstance(expr, Call):
            if isinstance(expr.func, Name):
                fname = expr.func.id
                if fname in ("sqrt", "exp", "sin", "cos", "ln", "to_f64", "tanh", "mean"):
                    return "double"
                if fname in ("floor", "to_i64", "len", "argmax"):
                    return "int64_t"
                if fname in self.func_signatures:
                    return self.func_signatures[fname][0]
        return "double"

    def transpile(self) -> str:
        out = []
        out.append("/* ========================================================================= */")
        out.append("/* Gerado automaticamente pelo compilador Nativa (AOT C99 Transpiler)        */")
        out.append("/* ========================================================================= */")
        out.append("#include <stdio.h>")
        out.append("#include <stdlib.h>")
        out.append("#include <stdint.h>")
        out.append("#include <stdbool.h>")
        out.append("#include <math.h>")
        out.append("#include <string.h>")
        out.append("#ifndef NAT_PI")
        out.append("#define NAT_PI 3.14159265358979323846")
        out.append("#endif")
        out.append("#ifndef NAT_E")
        out.append("#define NAT_E  2.71828182845904523536")
        out.append("#endif")
        out.append("/* Primitivas de quantização ternária e empacotamento {-1, 0, +1} em C */")
        out.append("static inline int8_t* nat_quantize_ternary(const double* v, int64_t length, double threshold) {")
        out.append("    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));")
        out.append("    if (!out) return NULL;")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        if (v[i] > threshold) out[i] = 1;")
        out.append("        else if (v[i] < -threshold) out[i] = -1;")
        out.append("        else out[i] = 0;")
        out.append("    }")
        out.append("    return out;")
        out.append("}")
        out.append("static inline uint8_t* nat_pack_ternary(const int8_t* in, int64_t length) {")
        out.append("    int64_t num_bytes = (length + 3) / 4;")
        out.append("    uint8_t* out = (uint8_t*)calloc(num_bytes, sizeof(uint8_t));")
        out.append("    if (!out) return NULL;")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        uint8_t enc = (in[i] == 1) ? 1 : ((in[i] == -1) ? 2 : 0);")
        out.append("        int64_t b = i / 4;")
        out.append("        int shift = (i % 4) * 2;")
        out.append("        out[b] |= (enc << shift);")
        out.append("    }")
        out.append("    return out;")
        out.append("}")
        out.append("static inline int8_t* nat_unpack_ternary(const uint8_t* in, int64_t length) {")
        out.append("    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));")
        out.append("    if (!out) return NULL;")
        out.append("    static const int8_t map[4] = {0, 1, -1, 0};")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        int64_t b = i / 4;")
        out.append("        int shift = (i % 4) * 2;")
        out.append("        uint8_t code = (in[b] >> shift) & 0x03;")
        out.append("        out[i] = map[code];")
        out.append("    }")
        out.append("    return out;")
        out.append("}")
        out.append("/* Funções matemáticas e primitivas de IA */")
        out.append("static inline double nat_ln(double x) { return log(x); }")
        out.append("static inline double nat_relu(double x) { return x > 0.0 ? x : 0.0; }")
        out.append("static inline double nat_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }")
        out.append("static inline double nat_silu(double x) { return x / (1.0 + exp(-x)); }")
        out.append("")
        out.append("/* Kernel Add-Only Matmul Ternário {-1, 0, +1} em C */")
        out.append("static inline void nat_ternary_matmul_vec(const int8_t* W, const double* x, double* y, int64_t M, int64_t K) {")
        out.append("    for (int64_t i = 0; i < M; i++) {")
        out.append("        double acc = 0.0;")
        out.append("        const int8_t* row = W + i * K;")
        out.append("        for (int64_t k = 0; k < K; k++) {")
        out.append("            int8_t w = row[k];")
        out.append("            if (w == 1) acc += x[k];")
        out.append("            else if (w == -1) acc -= x[k];")
        out.append("        }")
        out.append("        y[i] = acc;")
        out.append("    }")
        out.append("}")
        out.append("")
        out.append("/* BitNet b1.58: Empacotamento base-3 denso (5 trits/byte = 1.60 bits/trit) */")
        out.append("static inline void nat_pack_5trits(const int8_t* in, uint8_t* out_bytes, int64_t length) {")
        out.append("    int64_t out_idx = 0;")
        out.append("    for (int64_t i = 0; i < length; i += 5) {")
        out.append("        uint16_t b_val = 0;")
        out.append("        uint16_t mul = 1;")
        out.append("        for (int j = 0; j < 5; j++) {")
        out.append("            int8_t t = (i + j < length) ? (in[i + j] + 1) : 1;")
        out.append("            b_val += (uint16_t)(t * mul);")
        out.append("            mul *= 3;")
        out.append("        }")
        out.append("        out_bytes[out_idx++] = (uint8_t)b_val;")
        out.append("    }")
        out.append("}")
        out.append("")
        out.append("static inline void nat_unpack_5trits(const uint8_t* in_bytes, int8_t* out, int64_t length) {")
        out.append("    int64_t out_idx = 0;")
        out.append("    int64_t num_bytes = (length + 4) / 5;")
        out.append("    for (int64_t b = 0; b < num_bytes && out_idx < length; b++) {")
        out.append("        uint8_t val = in_bytes[b];")
        out.append("        for (int step = 0; step < 5 && out_idx < length; step++) {")
        out.append("            out[out_idx++] = (int8_t)(val % 3) - 1;")
        out.append("            val /= 3;")
        out.append("        }")
        out.append("    }")
        out.append("}")
        out.append("")
        out.append("/* Fused Ternary GEMV: Decodifica 5 trits por byte em tempo real em registradores, streaming sem buffer uncompressed */")
        out.append("static inline void nat_fused_gemv_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {")
        out.append("    int64_t bytes_per_row = (K + 4) / 5;")
        out.append("    for (int64_t i = 0; i < M; i++) {")
        out.append("        double acc = 0.0;")
        out.append("        const uint8_t* row_packed = W_packed + i * bytes_per_row;")
        out.append("        int64_t k = 0;")
        out.append("        for (int64_t b = 0; b < bytes_per_row && k < K; b++) {")
        out.append("            uint8_t val = row_packed[b];")
        out.append("            for (int step = 0; step < 5 && k < K; step++) {")
        out.append("                int8_t trit = (int8_t)(val % 3) - 1;")
        out.append("                val /= 3;")
        out.append("                if (trit == 1) acc += x[k];")
        out.append("                else if (trit == -1) acc -= x[k];")
        out.append("                k++;")
        out.append("            }")
        out.append("        }")
        out.append("        y[i] = acc * gamma;")
        out.append("    }")
        out.append("}")
        out.append("")
        out.append("/* Vec-LUT Ternary GEMV: Look-Up Table por pares de trits com aceleração de streaming em C */")
        out.append("static inline void nat_gemv_vec_lut_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {")
        out.append("    int64_t num_groups = (K + 4) / 5;")
        out.append("    if (num_groups <= 0) return;")
        out.append("    double* lut = (double*)malloc(num_groups * 21 * sizeof(double));")
        out.append("    if (!lut) return;")
        out.append("    double* lut01 = lut;")
        out.append("    double* lut23 = lut + num_groups * 9;")
        out.append("    double* lut4  = lut + num_groups * 18;")
        out.append("    for (int64_t g = 0; g < num_groups; g++) {")
        out.append("        int64_t base = g * 5;")
        out.append("        double x0 = (base < K) ? x[base] : 0.0;")
        out.append("        double x1 = (base + 1 < K) ? x[base + 1] : 0.0;")
        out.append("        double x2 = (base + 2 < K) ? x[base + 2] : 0.0;")
        out.append("        double x3 = (base + 3 < K) ? x[base + 3] : 0.0;")
        out.append("        double x4 = (base + 4 < K) ? x[base + 4] : 0.0;")
        out.append("        for (int t1 = 0; t1 < 3; t1++) {")
        out.append("            for (int t0 = 0; t0 < 3; t0++) {")
        out.append("                lut01[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x0 + (t1 - 1) * x1;")
        out.append("                lut23[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x2 + (t1 - 1) * x3;")
        out.append("            }")
        out.append("        }")
        out.append("        for (int t4 = 0; t4 < 3; t4++) {")
        out.append("            lut4[g * 3 + t4] = (t4 - 1) * x4;")
        out.append("        }")
        out.append("    }")
        out.append("    for (int64_t i = 0; i < M; i++) {")
        out.append("        double acc = 0.0;")
        out.append("        const uint8_t* row = W_packed + i * num_groups;")
        out.append("        for (int64_t g = 0; g < num_groups; g++) {")
        out.append("            uint8_t b = row[g];")
        out.append("            uint8_t idx01 = b % 9;")
        out.append("            uint8_t idx23 = (b / 9) % 9;")
        out.append("            uint8_t idx4  = b / 81;")
        out.append("            acc += lut01[g * 9 + idx01] + lut23[g * 9 + idx23] + lut4[g * 3 + idx4];")
        out.append("        }")
        out.append("        y[i] = acc * gamma;")
        out.append("    }")
        out.append("    free(lut);")
        out.append("}")
        out.append("")
        out.append("/* Wrappers de Alto Nível de BitNet b1.58 para o Transpilador C */")
        out.append("static inline uint8_t* nat_pack_ternary_5trits(const int8_t* in, int64_t length) {")
        out.append("    int64_t num_bytes = (length + 4) / 5;")
        out.append("    uint8_t* out = (uint8_t*)malloc(num_bytes * sizeof(uint8_t));")
        out.append("    if (out) nat_pack_5trits(in, out, length);")
        out.append("    return out;")
        out.append("}")
        out.append("static inline int8_t* nat_unpack_ternary_5trits(const uint8_t* in, int64_t length) {")
        out.append("    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));")
        out.append("    if (out) nat_unpack_5trits(in, out, length);")
        out.append("    return out;")
        out.append("}")
        out.append("static inline double* nat_gemv_vec_lut(const uint8_t* W_packed, const double* x, int64_t M, int64_t K) {")
        out.append("    double* y = (double*)malloc(M * sizeof(double));")
        out.append("    if (y) nat_gemv_vec_lut_5trits(W_packed, x, y, M, K, 1.0);")
        out.append("    return y;")
        out.append("}")
        out.append("static inline double nat_tensor_absmean(const double* W, int64_t length) {")
        out.append("    if (length <= 0) return 0.0;")
        out.append("    double s = 0.0;")
        out.append("    for (int64_t i = 0; i < length; i++) s += fabs(W[i]);")
        out.append("    return s / (double)length;")
        out.append("}")
        out.append("static inline int8_t* nat_quantize_bitnet(const double* W, int64_t length) {")
        out.append("    double gamma = nat_tensor_absmean(W, length);")
        out.append("    double inv = 1.0 / (gamma + 1e-7);")
        out.append("    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));")
        out.append("    if (!out) return NULL;")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        double sc = W[i] * inv;")
        out.append("        double r = round(sc);")
        out.append("        if (r > 1.0) r = 1.0; else if (r < -1.0) r = -1.0;")
        out.append("        out[i] = (int8_t)r;")
        out.append("    }")
        out.append("    return out;")
        out.append("}")
        out.append("static inline double nat_mae(const double* a, const double* b, int64_t length) {")
        out.append("    if (length <= 0) return 0.0;")
        out.append("    double s = 0.0;")
        out.append("    for (int64_t i = 0; i < length; i++) s += fabs(a[i] - b[i]);")
        out.append("    return s / (double)length;")
        out.append("}")
        out.append("static inline double nat_relative_error(const double* a, const double* b, int64_t length) {")
        out.append("    double norm_diff = 0.0, norm_a = 0.0;")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        double d = a[i] - b[i];")
        out.append("        norm_diff += d * d; norm_a += a[i] * a[i];")
        out.append("    }")
        out.append("    return norm_a > 0.0 ? sqrt(norm_diff / norm_a) : 0.0;")
        out.append("}")
        out.append("static inline double nat_pearson_corr(const double* a, const double* b, int64_t length) {")
        out.append("    if (length < 2) return 1.0;")
        out.append("    double ma = 0.0, mb = 0.0;")
        out.append("    for (int64_t i = 0; i < length; i++) { ma += a[i]; mb += b[i]; }")
        out.append("    ma /= (double)length; mb /= (double)length;")
        out.append("    double num = 0.0, da = 0.0, db = 0.0;")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        double diff_a = a[i] - ma; double diff_b = b[i] - mb;")
        out.append("        num += diff_a * diff_b; da += diff_a * diff_a; db += diff_b * diff_b;")
        out.append("    }")
        out.append("    double den = sqrt(da * db);")
        out.append("    return den > 0.0 ? (num / den) : 0.0;")
        out.append("}")
        out.append("static inline double nat_cosine_sim(const double* a, const double* b, int64_t length) {")
        out.append("    double dot = 0.0, na = 0.0, nb = 0.0;")
        out.append("    for (int64_t i = 0; i < length; i++) {")
        out.append("        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];")
        out.append("    }")
        out.append("    double den = sqrt(na) * sqrt(nb);")
        out.append("    return den > 0.0 ? (dot / den) : 0.0;")
        out.append("}")
        out.append("")

        # Constantes de nível superior
        for it in self.program.items:
            if isinstance(it, ConstDecl):
                if it.name in ("PI", "E"):
                    continue
                ctype = self.c_type(it.type) if it.type else self.infer_type(it.value)
                val = self.emit_expr(it.value)
                out.append(f"static const {ctype} {it.name} = {val};")
                self.var_types[it.name] = ctype
        out.append("")

        # Declarações antecipadas de funções (Forward Declarations)
        for it in self.program.items:
            if isinstance(it, Function) and it.name != "main":
                ret_t, params = self.func_signatures[it.name]
                p_str = ", ".join(f"{pt} {pn}" for pn, pt in params) if params else "void"
                out.append(f"{ret_t} {it.name}({p_str});")
        out.append("")

        # Definições de funções
        for it in self.program.items:
            if isinstance(it, Function):
                out.append(self.emit_function(it))
                out.append("")

        return "\n".join(out)

    def emit_function(self, fn: Function) -> str:
        lines = []
        is_main = (fn.name == "main")
        ret_t = "int" if is_main else (self.c_type(fn.ret) if fn.ret else "void")
        p_str = "int argc, char** argv" if is_main else (", ".join(f"{self.c_type(p.type)} {p.name}" for p in fn.params) if fn.params else "void")

        old_vars = dict(self.var_types)
        for p in fn.params:
            self.var_types[p.name] = self.c_type(p.type)

        lines.append(f"{ret_t} {fn.name}({p_str}) {{")
        self.indent_level += 1
        for s in fn.body:
            code = self.emit_stmt(s)
            if code:
                lines.append(code)
        if is_main:
            lines.append(f"{self.indent()}return 0;")
        self.indent_level -= 1
        lines.append("}")

        self.var_types = old_vars
        return "\n".join(lines)

    def emit_stmt(self, s: Node) -> str:
        ind = self.indent()
        if isinstance(s, VarDecl):
            ctype = self.c_type(s.type)
            self.var_types[s.name] = ctype
            val = self.emit_expr(s.value)
            return f"{ind}{ctype} {s.name} = {val};"

        if isinstance(s, ConstDecl):
            ctype = self.c_type(s.type) if s.type else self.infer_type(s.value)
            self.var_types[s.name] = ctype
            val = self.emit_expr(s.value)
            return f"{ind}const {ctype} {s.name} = {val};"

        if isinstance(s, Assign):
            tgt = s.target
            if isinstance(tgt, Name):
                name = tgt.id
                if name not in self.var_types and s.op == "=":
                    ctype = self.infer_type(s.value)
                    self.var_types[name] = ctype
                    val = self.emit_expr(s.value)
                    return f"{ind}{ctype} {name} = {val};"
                val = self.emit_expr(s.value)
                return f"{ind}{name} {s.op} {val};"
            if isinstance(tgt, Index):
                arr = self.emit_expr(tgt.obj)
                idx = self.emit_expr(tgt.index)
                val = self.emit_expr(s.value)
                return f"{ind}{arr}[{idx}] {s.op} {val};"
            raise NatError(
                f"Atribuição a '{type(tgt).__name__}' não é suportada no backend C99 nativo",
                getattr(s, "line", 0),
                getattr(s, "col", 0)
            )

        if isinstance(s, ExprStmt):
            expr_code = self.emit_expr(s.expr)
            if expr_code.endswith(";"):
                return f"{ind}{expr_code}"
            return f"{ind}{expr_code};"

        if isinstance(s, If):
            cond = self.emit_expr(s.cond)
            res = [f"{ind}if ({cond}) {{"]
            self.indent_level += 1
            for st in s.body:
                c = self.emit_stmt(st)
                if c: res.append(c)
            self.indent_level -= 1
            if s.orelse:
                if len(s.orelse) == 1 and isinstance(s.orelse[0], If):
                    sub_if = self.emit_stmt(s.orelse[0]).lstrip()
                    res.append(f"{ind}}} else {sub_if}")
                    return "\n".join(res)
                else:
                    res.append(f"{ind}}} else {{")
                    self.indent_level += 1
                    for st in s.orelse:
                        c = self.emit_stmt(st)
                        if c: res.append(c)
                    self.indent_level -= 1
            res.append(f"{ind}}}")
            return "\n".join(res)

        if isinstance(s, While):
            cond = self.emit_expr(s.cond)
            res = [f"{ind}while ({cond}) {{"]
            self.indent_level += 1
            for st in s.body:
                c = self.emit_stmt(st)
                if c: res.append(c)
            self.indent_level -= 1
            res.append(f"{ind}}}")
            return "\n".join(res)

        if isinstance(s, For):
            if isinstance(s.iter, Range):
                start = self.emit_expr(s.iter.start) if s.iter.start else "0"
                end = self.emit_expr(s.iter.end) if s.iter.end else "0"
                op = "<=" if s.iter.inclusive else "<"
                self.var_types[s.var] = "int64_t"
                res = [f"{ind}for (int64_t {s.var} = {start}; {s.var} {op} {end}; {s.var}++) {{"]
                self.indent_level += 1
                for st in s.body:
                    c = self.emit_stmt(st)
                    if c: res.append(c)
                self.indent_level -= 1
                res.append(f"{ind}}}")
                return "\n".join(res)
            raise NatError(
                f"Backend C99 suporta iteração apenas sobre intervalos 'Range' (ex: '0..N'); iteração sobre '{type(s.iter).__name__}' não é suportada",
                getattr(s, "line", 0),
                getattr(s, "col", 0)
            )

        if isinstance(s, Return):
            if s.value is not None:
                val = self.emit_expr(s.value)
                return f"{ind}return {val};"
            return f"{ind}return;"

        if isinstance(s, Break):
            return f"{ind}break;"

        if isinstance(s, Continue):
            return f"{ind}continue;"

        if isinstance(s, Pass):
            return f"{ind}/* pass */;"

        raise NatError(
            f"Instrução do tipo '{type(s).__name__}' não é suportada no backend C99 nativo",
            getattr(s, "line", 0),
            getattr(s, "col", 0)
        )

    def emit_expr(self, e: Expr) -> str:
        if isinstance(e, Int):
            return f"{e.value}LL"
        if isinstance(e, Float):
            return f"{e.value}"
        if isinstance(e, Str):
            esc = e.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            return f'"{esc}"'
        if isinstance(e, Bool):
            return "true" if e.value else "false"
        if isinstance(e, Name):
            if e.id == "PI": return "NAT_PI"
            if e.id == "E": return "NAT_E"
            return e.id

        if isinstance(e, Unary):
            operand = self.emit_expr(e.operand)
            if e.op == "not":
                return f"(!({operand}))"
            return f"({e.op}({operand}))"

        if isinstance(e, Binary):
            l = self.emit_expr(e.left)
            r = self.emit_expr(e.right)
            if e.op == "and": return f"(({l}) && ({r}))"
            if e.op == "or": return f"(({l}) || ({r}))"
            if e.op == "**": return f"pow((double)({l}), (double)({r}))"
            if e.op == "//": return f"((int64_t)floor((double)({l}) / (double)({r})))"
            if e.op == "@":
                raise NatError(
                    "Operador de multiplicação matricial '@' requer operandos de tensor/matriz; multiplicação escalar não é suportada no backend C99 nativo (use kernels AOT como gemv_vec_lut)",
                    getattr(e, "line", 0),
                    getattr(e, "col", 0)
                )
            return f"(({l}) {e.op} ({r}))"

        if isinstance(e, Call):
            return self.emit_call(e)

        if isinstance(e, Index):
            obj = self.emit_expr(e.obj)
            idx = self.emit_expr(e.index)
            return f"({obj})[{idx}]"

        if isinstance(e, Attr):
            if isinstance(e.obj, Name) and e.obj.id == "math":
                if e.name == "PI": return "NAT_PI"
                if e.name == "E": return "NAT_E"
                return e.name
            return f"{self.emit_expr(e.obj)}.{e.name}"

        raise NatError(
            f"Expressão do tipo '{type(e).__name__}' não é suportada no backend C99 nativo",
            getattr(e, "line", 0),
            getattr(e, "col", 0)
        )

    def emit_call(self, c: Call) -> str:
        if isinstance(c.func, Name) and c.func.id == "print":
            if len(c.args) == 1 and isinstance(c.args[0], FString):
                return self.emit_print_fstring(c.args[0])
            elif len(c.args) == 1 and isinstance(c.args[0], Str):
                esc = c.args[0].value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                return f'printf("{esc}\\n");'
            else:
                parts = []
                for idx, a in enumerate(c.args):
                    sep = " " if idx < len(c.args) - 1 else "\\n"
                    itype = self.infer_type(a)
                    code = self.emit_expr(a)
                    if itype in ("int64_t", "int32_t", "int16_t", "int8_t"):
                        parts.append(f'printf("%lld{sep}", (long long)({code}));')
                    elif itype in ("double", "float"):
                        parts.append(f'printf("%f{sep}", (double)({code}));')
                    elif itype == "const char*":
                        parts.append(f'printf("%s{sep}", {code});')
                    elif itype == "bool":
                        parts.append(f'printf("%s{sep}", ({code}) ? "true" : "false");')
                    else:
                        parts.append(f'printf("%f{sep}", (double)({code}));')
                return " ".join(parts)

        fn_name = ""
        if isinstance(c.func, Name):
            fn_name = c.func.id
        elif isinstance(c.func, Attr) and isinstance(c.func.obj, Name) and c.func.obj.id == "math":
            fn_name = c.func.name

        if fn_name == "ln": fn_name = "nat_ln"
        elif fn_name == "relu": fn_name = "nat_relu"
        elif fn_name == "sigmoid": fn_name = "nat_sigmoid"
        elif fn_name == "silu": fn_name = "nat_silu"
        elif fn_name == "quantize_ternary": fn_name = "nat_quantize_ternary"
        elif fn_name == "pack_ternary": fn_name = "nat_pack_ternary"
        elif fn_name == "unpack_ternary": fn_name = "nat_unpack_ternary"
        elif fn_name == "pack_ternary_5trits": fn_name = "nat_pack_ternary_5trits"
        elif fn_name == "unpack_ternary_5trits": fn_name = "nat_unpack_ternary_5trits"
        elif fn_name == "quantize_bitnet": fn_name = "nat_quantize_bitnet"
        elif fn_name == "tensor_absmean": fn_name = "nat_tensor_absmean"
        elif fn_name == "gemv_vec_lut": fn_name = "nat_gemv_vec_lut"
        elif fn_name == "mae": fn_name = "nat_mae"
        elif fn_name == "relative_error": fn_name = "nat_relative_error"
        elif fn_name == "pearson_corr": fn_name = "nat_pearson_corr"
        elif fn_name == "cosine_sim": fn_name = "nat_cosine_sim"
        elif fn_name == "to_i64":
            return f"((int64_t)({self.emit_expr(c.args[0])}))"
        elif fn_name == "to_f64":
            return f"((double)({self.emit_expr(c.args[0])}))"

        args_code = ", ".join(self.emit_expr(a) for a in c.args)
        return f"{fn_name}({args_code})"

    def emit_print_fstring(self, fstr: FString) -> str:
        fmt_parts = []
        args = []
        for p in fstr.parts:
            if isinstance(p, str):
                fmt_parts.append(p.replace("%", "%%"))
            elif isinstance(p, FormatPart):
                itype = self.infer_type(p.expr)
                expr_code = self.emit_expr(p.expr)
                if p.spec:
                    spec = p.spec
                    if "d" in spec or "i" in spec:
                        spec = spec.replace("d", "lld").replace("i", "lld")
                        fmt_parts.append(f"%{spec}")
                        args.append(f"(long long)({expr_code})")
                    else:
                        fmt_parts.append(f"%{spec}")
                        args.append(f"(double)({expr_code})")
                else:
                    if itype in ("int64_t", "int32_t", "int16_t", "int8_t"):
                        fmt_parts.append("%lld")
                        args.append(f"(long long)({expr_code})")
                    elif itype in ("double", "float"):
                        fmt_parts.append("%f")
                        args.append(f"(double)({expr_code})")
                    elif itype == "bool":
                        fmt_parts.append("%s")
                        args.append(f"(({expr_code}) ? \"true\" : \"false\")")
                    elif itype == "const char*":
                        fmt_parts.append("%s")
                        args.append(expr_code)
                    else:
                        fmt_parts.append("%f")
                        args.append(f"(double)({expr_code})")
        fmt_str = "".join(fmt_parts) + "\\n"
        if args:
            return f'printf("{fmt_str}", {", ".join(args)});'
        return f'printf("{fmt_str}");'


def compile_c_to_binary(c_path: str, exe_path: str) -> bool:
    gcc = shutil.which("gcc")
    if gcc:
        cmd = [gcc, "-O3", "-std=c99", "-fopenmp", c_path, "-lm", "-o", exe_path]
        res = subprocess.run(cmd)
        return res.returncode == 0

    clang = shutil.which("clang")
    if clang:
        cmd = [clang, "-O3", "-std=c99", "-fopenmp", c_path, "-lm", "-o", exe_path]
        res = subprocess.run(cmd)
        return res.returncode == 0

    vcvars_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for vc in vcvars_candidates:
        if os.path.exists(vc):
            bat_dir = os.path.dirname(os.path.abspath(c_path))
            bat_path = os.path.join(bat_dir, "_nativa_build.bat")
            bat_content = f'@echo off\ncall "{vc}" >nul 2>&1\ncl.exe /O2 /openmp /nologo /W3 "{c_path}" /Fe:"{exe_path}"\n'
            with open(bat_path, "w", encoding="ascii", errors="ignore") as f:
                f.write(bat_content)
            try:
                res = subprocess.run([bat_path], capture_output=True, text=True)
            finally:
                if os.path.exists(bat_path):
                    try:
                        os.remove(bat_path)
                    except Exception:
                        pass
            obj_file = c_path.replace(".c", ".obj")
            if os.path.exists(obj_file):
                try:
                    os.remove(obj_file)
                except Exception:
                    pass
            if res.returncode != 0:
                print(res.stderr or res.stdout, file=sys.stderr)
            return res.returncode == 0

    cl = shutil.which("cl")
    if cl:
        cmd = [cl, "/O2", "/openmp", "/nologo", "/W3", c_path, f"/Fe:{exe_path}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr or res.stdout, file=sys.stderr)
        return res.returncode == 0

    print("Erro: Nenhum compilador C encontrado (instale GCC, Clang ou MSVC).", file=sys.stderr)
    return False


# ============================================================================
# TESTES DE ESCOPO, LIMITES DO BACKEND C E TESTES DIFERENCIAIS
# ============================================================================

class TesteEscopoInterpretador(unittest.TestCase):
    def test_escopo_bloco_if(self):
        # Variável definida dentro de if não deve vazar para fora
        code = "fn main():\n    if true:\n        x = 42\n    print(x)\n"
        with self.assertRaises(RuntimeError):
            Interpreter().run(parse(code))

    def test_escopo_bloco_for(self):
        # Variável de iteração do for não deve vazar para fora
        code = "fn main():\n    for i in 0..5:\n        pass\n    print(i)\n"
        with self.assertRaises(RuntimeError):
            Interpreter().run(parse(code))

    def test_imutabilidade_constante(self):
        # Reatribuição de constante em runtime deve falhar
        code = "fn main():\n    const c = 100\n    c = 200\n"
        with self.assertRaises(RuntimeError):
            Interpreter().run(parse(code))

    def test_mutacao_escopo_pai_em_loop(self):
        # Mutação de variável do escopo pai dentro de loop deve persistir
        code = "fn main():\n    s = 0\n    for i in 1..4:\n        s += i\n"
        interp = Interpreter()
        interp.run(parse(code))
        self.assertEqual(interp.last_scope.get("s"), 6)


class TesteLimitesCTranspiler(unittest.TestCase):
    def test_for_sobre_nao_range_falha_rapido(self):
        code = "fn main():\n    v = 10\n    for i in v:\n        print(i)\n"
        with self.assertRaises(NatError):
            CTranspiler(parse(code)).transpile()

    def test_matmul_escalar_falha_rapido(self):
        code = "fn main():\n    a = 2.0\n    b = 3.0\n    c = a @ b\n"
        with self.assertRaises(NatError):
            CTranspiler(parse(code)).transpile()

    def test_transpilacao_primitivas_ternarias(self):
        code_ok = "fn main():\n    print(relu(2.0))\n    print(sigmoid(0.0))\n    print(silu(1.0))\n"
        c_code = CTranspiler(parse(code_ok)).transpile()
        self.assertIn("nat_relu", c_code)
        self.assertIn("nat_sigmoid", c_code)
        self.assertIn("nat_silu", c_code)
        self.assertIn("nat_quantize_ternary", c_code)
        self.assertIn("nat_pack_ternary", c_code)


class TesteDiferencialC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vcvars_candidates = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        ]
        cls.compiler_available = bool(
            shutil.which("gcc") or shutil.which("clang") or shutil.which("cl")
            or any(os.path.exists(p) for p in vcvars_candidates)
        )

    def test_diferencial_fatorial_e_loops(self):
        if not self.compiler_available:
            self.skipTest("Nenhum compilador C disponível para teste diferencial")

        source = '''
fn fatorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * fatorial(n - 1)

fn main():
    acc = 0
    for i in 1..6:
        fat = fatorial(i)
        acc += fat
        print(fat)
    print(acc)
'''
        import io, tempfile
        prog = parse(source)
        interp = Interpreter()
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            interp.run(prog)
        finally:
            sys.stdout = old_stdout
        interp_out = buf.getvalue().strip()

        c_code = CTranspiler(prog).transpile()
        with tempfile.TemporaryDirectory() as tmpdir:
            c_path = os.path.join(tmpdir, "diff_fat.c")
            exe_path = os.path.join(tmpdir, "diff_fat.exe")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)
            ok = compile_c_to_binary(c_path, exe_path)
            self.assertTrue(ok, "Compilação C falhou")
            res = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            c_out = res.stdout.strip()

        self.assertEqual(interp_out, c_out)


# ============================================================================
# LINHA DE COMANDO
# ============================================================================

def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if res.wasSuccessful() else 1


def print_nativa_help() -> None:
    print("""======================================================================
  NATIVA CLI v0.3.0 - Linguagem Nativa (AI, Data & Ternary Edition)
======================================================================
Uso: nativa <comando> [opcoes]

Comandos principais:
  nativa run <arquivo.nat>           Executa o arquivo no interpretador Nativa
  nativa run <arquivo.nat> --native  Compila AOT para C99/Machine Code e executa
  nativa build <arquivo.nat>         Compila AOT gerando executavel nativo (.exe)
  nativa chat [modelo.natq4]         Inicia chat local Qwen2 em INT4
  nativa quantize <dir_hf> [out]     Converte Qwen2 SafeTensors para .natq4 Q4x4
  nativa test                        Executa a suite completa de 73 testes
  nativa version                     Exibe a versao instalada

Exemplos pre-carregados:
  nativa run @riemann                Zeros da funcao Zeta de Riemann
  nativa run @rede                   Rede Neural Perceptron multicamada
  nativa run @ia_tensor              Operacoes com Tensores e Autodiff
  nativa run @ternario               Computacao Ternaria BitNet / Trits
======================================================================""")


def print_detected_devices() -> int:
    print("====================================================================")
    print("  NATIVA - DISPOSITIVOS DE COMPUTAÇÃO DETECTADOS")
    print("====================================================================")
    print(f"  [CPU Host] SO: {sys.platform} | Python {sys.version.split()[0]}")

    try:
        import ctypes
        cl_name = "OpenCL.dll" if sys.platform == "win32" else "libOpenCL.so"
        cl = ctypes.cdll.LoadLibrary(cl_name)

        cl.clGetPlatformIDs.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)]
        cl.clGetPlatformIDs.restype = ctypes.c_int32
        cl.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        cl.clGetPlatformInfo.restype = ctypes.c_int32
        cl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)]
        cl.clGetDeviceIDs.restype = ctypes.c_int32
        cl.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        cl.clGetDeviceInfo.restype = ctypes.c_int32

        num_plat = ctypes.c_uint32(0)
        cl.clGetPlatformIDs(0, None, ctypes.byref(num_plat))
        if num_plat.value > 0:
            plats = (ctypes.c_void_p * num_plat.value)()
            cl.clGetPlatformIDs(num_plat.value, plats, None)
            total_devs = 0
            for p in plats:
                p_name_buf = ctypes.create_string_buffer(256)
                cl.clGetPlatformInfo(p, 0x0902, 256, p_name_buf, None)
                p_name = p_name_buf.value.decode("utf-8", errors="ignore")

                num_dev = ctypes.c_uint32(0)
                cl.clGetDeviceIDs(p, 0xFFFFFFFF, 0, None, ctypes.byref(num_dev))
                if num_dev.value > 0:
                    devs = (ctypes.c_void_p * num_dev.value)()
                    cl.clGetDeviceIDs(p, 0xFFFFFFFF, num_dev.value, devs, None)
                    for d in devs:
                        d_name_buf = ctypes.create_string_buffer(256)
                        d_type = ctypes.c_uint64(0)
                        cl.clGetDeviceInfo(d, 0x102B, 256, d_name_buf, None)
                        cl.clGetDeviceInfo(d, 0x1000, 8, ctypes.byref(d_type), None)
                        is_gpu = bool(d_type.value & 4)
                        d_name = d_name_buf.value.decode("utf-8", errors="ignore")
                        print(f"  [{total_devs}] [{'GPU' if is_gpu else 'CPU'}] {d_name} (Plataforma: {p_name})")
                        total_devs += 1
            if total_devs == 0:
                print("  Nenhum dispositivo OpenCL ativo.")
        else:
            print("  Nenhuma plataforma OpenCL ativa.")
    except Exception:
        print("  [OpenCL] Nenhuma biblioteca OpenCL ativa (modo CPU C99/OpenMP nativo padrão).")

    print("====================================================================")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    if not raw_args or (len(raw_args) == 1 and raw_args[0].lower() in ("help", "-h", "--help")):
        print_nativa_help()
        return 0

    # Subcomandos modernos estilo NodeJS / Cargo / Go / Python CLI
    subcmd = raw_args[0].lower()
    if subcmd in ("version", "--version", "-v"):
        print("Nativa v0.3.0 (AI, Data & Ternary Edition)")
        return 0
    elif subcmd == "run":
        if len(raw_args) == 1:
            print("Uso: nativa run <arquivo.nat|@exemplo> [--native]")
            return 1
        clean = [a for a in raw_args[1:] if a not in ("--native", "--aot")]
        if "--native" in raw_args or "--aot" in raw_args:
            return main([clean[0], "--build", "--run"] + clean[1:])
        return main([raw_args[1], "--run"] + raw_args[2:])
    elif subcmd == "build":
        if len(raw_args) == 1:
            print("Uso: nativa build <arquivo.nat|@exemplo>")
            return 1
        return main([raw_args[1], "--build"] + raw_args[2:])
    elif subcmd in ("test", "--test"):
        return run_tests()
    elif subcmd == "chat":
        model_target = raw_args[1] if len(raw_args) > 1 else os.path.join(os.path.dirname(__file__), "data", "qwen2.5_3b.natq4")
        chat_script = os.path.join(os.path.dirname(__file__), "tools", "chat_natq4.py")
        return subprocess.call([sys.executable, chat_script, model_target])
    elif subcmd == "quantize":
        quant_script = os.path.join(os.path.dirname(__file__), "tools", "nativa_quantize.py")
        return subprocess.call([sys.executable, quant_script] + raw_args[1:])
    elif subcmd in ("devices", "--devices", "opencl", "--opencl"):
        return print_detected_devices()

    ap = argparse.ArgumentParser(description="Nativa v0.3 (AI, Data & Ternary Edition)")
    ap.add_argument("arquivo", nargs="?", help="arquivo .nat ou @exemplo | @riemann | @rede | @ia_tensor | @ternario")
    ap.add_argument("--ast", action="store_true", help="imprime a AST")
    ap.add_argument("--tokens", action="store_true", help="imprime os tokens")
    ap.add_argument("--run", action="store_true", help="executa o programa (modo runtime)")
    ap.add_argument("--emit-c", metavar="ARQUIVO_C", nargs="?", const="", help="emite codigo C99 correspondente")
    ap.add_argument("--build", action="store_true", help="compila codigo C99 AOT para executavel nativo (.exe)")
    ap.add_argument("--test", action="store_true", help="roda a suite de testes")
    ap.add_argument("--devices", action="store_true", help="lista os dispositivos de computação disponíveis (CPU, OpenCL GPU/CPU)")
    ap.add_argument("--opencl", action="store_true", help="lista os dispositivos OpenCL disponíveis")
    ap.add_argument("-v", "--version", action="version", version="Nativa v0.3.0 (AI, Data & Ternary Edition)")
    args = ap.parse_args(argv)

    if args.devices or args.opencl:
        return print_detected_devices()

    if args.test:
        return run_tests()

    if not args.arquivo:
        print_nativa_help()
        return 0

    if args.arquivo.startswith("@"):
        nome = args.arquivo[1:]
        if nome not in EXEMPLOS:
            ap.error(f"exemplo desconhecido '{nome}' (opções: {', '.join(EXEMPLOS)})")
        source = EXEMPLOS[nome]
    elif args.arquivo in EXEMPLOS:
        source = EXEMPLOS[args.arquivo]
    else:
        with open(args.arquivo, encoding="utf-8") as fh:
            source = fh.read()

    try:
        if args.tokens:
            for t in tokenize(source):
                print(t)
            return 0
        prog = parse(source)
    except NatError as e:
        print(e.pretty(source, args.arquivo), file=sys.stderr)
        return 1

    if args.ast:
        print(dump(prog))
        return 0

    diags = SemanticAnalyzer().analyze(prog)
    for d in diags:
        print(f"{args.arquivo}:{d.line}: {d.level}: {d.msg}", file=sys.stderr)

    erros = sum(d.level == "erro" for d in diags)
    avisos = len(diags) - erros
    fns = sum(isinstance(i, Function) for i in prog.items)

    if erros:
        print(f"FALHOU: {fns} função(ões), {erros} erro(s), {avisos} aviso(s)")
        return 1

    if args.emit_c is not None:
        c_code = CTranspiler(prog).transpile()
        out_c = args.emit_c if args.emit_c else (args.arquivo.replace(".nat", "") + ".c" if not args.arquivo.startswith("@") else args.arquivo[1:] + ".c")
        with open(out_c, "w", encoding="utf-8") as f:
            f.write(c_code)
        print(f"[OK] Código C99 emitido com sucesso: {out_c}")
        return 0

    if args.build:
        base_name = args.arquivo.replace(".nat", "") if not args.arquivo.startswith("@") else args.arquivo[1:]
        out_c = base_name + ".c"
        out_exe = base_name + (".exe" if sys.platform == "win32" else "")
        c_code = CTranspiler(prog).transpile()
        with open(out_c, "w", encoding="utf-8") as f:
            f.write(c_code)
        print(f"[*] Código C99 emitido: {out_c}")
        t0 = time.perf_counter()
        ok = compile_c_to_binary(out_c, out_exe)
        t_comp = time.perf_counter() - t0
        if ok:
            print(f"[OK] Binário nativo compilado com sucesso em {t_comp:.3f}s: {out_exe}")
            if args.run:
                print(f"[*] Executando {out_exe} nativamente:\n")
                subprocess.run([os.path.abspath(out_exe)])
            return 0
        else:
            print(f"[ERRO] Falha ao compilar binário nativo.", file=sys.stderr)
            return 1

    if args.run:
        interp = Interpreter()
        interp.run(prog)
        return 0

    print(f"OK: {fns} função(ões), {erros} erro(s), {avisos} aviso(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
nativa.py — Nativa Language (v0.4 - AI, Data, Spectral & Ternary Edition)
DSL frontend, static semantic analyzer, prototyping interpreter (Python/NumPy),
and AOT transpiler to native C99 (GCC/Clang/MSVC).

Project Architecture:
    1. DSL Frontend: Pure Python lexer, parser, and static semantic analyzer.
    2. Prototyping Runtime (--run): Dynamic interpreter accelerated with NumPy
       and low-level kernels (BitNet b1.58, RMSNorm, SiLU, Softmax) via ctypes/AOT.
    3. Native AOT Compiler (--build): Transpiler to static C99, compiled directly
       via GCC, Clang, or MSVC to an executable binary (.exe / ELF) with zero external dependencies.

Usage:
    nativa run program.nat             # run via prototyping interpreter
    nativa build program.nat           # compile to native C99 executable binary
    nativa build program.nat --run     # compile to C99 binary and run natively
    nativa program.nat                 # syntax validation and semantic analysis
    nativa test                        # run the regression test suite (100 tests)
"""

from __future__ import annotations

import argparse
import copy
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
from typing import Any, Callable, Optional, Union

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
# Errors
# ----------------------------------------------------------------------------

class NatError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def pretty(self, source: str, filename: str = "<input>") -> str:
        lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        text = lines[self.line - 1] if 1 <= self.line <= len(lines) else ""
        caret = " " * max(self.col - 1, 0) + "^"
        return (f"{filename}:{self.line}:{self.col}: error: {self.message}\n"
                f"    {text}\n    {caret}")


class NatRuntimeError(RuntimeError, NatError):
    """Runtime error with accurate line and column tracking."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        formatted = f"[line {line}:{col}] {message}" if line else message
        RuntimeError.__init__(self, formatted)
        self.message = message
        self.line = line
        self.col = col



# ----------------------------------------------------------------------------
# Tokens and lexer
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
    "fn", "lambda", "struct", "const", "if", "elif", "else", "for", "in", "while", "return",
    "break", "continue", "pass", "import", "and", "or", "not", "true", "false",
}

OPS3 = {"..=", "**=", "//="}
OPS2 = {"->", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=",
        "**", "//", "..", "@="}
OPS1 = set("+-*/%<>=()[]{},:.@?")
CLOSERS = {")": "(", "]": "[", "}": "{"}

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0",
           "\\": "\\", '"': '"', "'": "'"}


def unescape(s: str, line: int, col: int) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            e = s[i + 1]
            if e not in ESCAPES:
                raise NatError(f"invalid escape sequence '\\{e}'", line, col)
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
                err("tabs are not allowed in indentation; use spaces", j)
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
                    err("inconsistent indentation does not match any outer block", j)
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
            err("tabs are only allowed inside strings and comments")

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
                        err("unterminated f-string", start)
                    if src[k] == "\\":
                        k += 1
                    k += 1
                if k >= n:
                    err("unterminated f-string", start)
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
                    err("unterminated string")
                if src[k] == "\\":
                    k += 1
                k += 1
            if k >= n:
                err("unterminated string")
            value = unescape(src[i + 1:k], line, i - line_start + 1)
            add("STRING", value, i)
            i = k + 1
            continue

        for size, table in ((3, OPS3), (2, OPS2), (1, OPS1)):
            op = src[i:i + size]
            if len(op) == size and op in table:
                break
        else:
            err(f"unexpected character {c!r}")

        if op in ("(", "[", "{"):
            brackets.append((op, line, i - line_start + 1))
        elif op in CLOSERS:
            if not brackets or brackets[-1][0] != CLOSERS[op]:
                err(f"'{op}' without matching opening bracket")
            brackets.pop()

        add("OP", op, i)
        i += size

    if brackets:
        ch, l, c_ = brackets[-1]
        raise NatError(f"'{ch}' was not closed", l, c_)

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
    col: int = field(default=0, compare=False, repr=False)

# --- types ---
@dataclass
class TypeName(Node):
    name: str

@dataclass
class TypeList(Node):
    elem: "TypeExpr"

@dataclass
class TypeTensor(Node):
    elem: "TypeExpr"

@dataclass
class TypeResult(Node):
    elem: "TypeExpr"

@dataclass
class TypeTuple(Node):
    elems: list      # list[TypeExpr]

TypeExpr = Union[TypeName, TypeList, TypeTensor, TypeResult, TypeTuple]

# --- expressions ---
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
class ListComp(Node):
    elt: "Expr"
    var: str
    iter: "Expr"
    cond: Optional["Expr"] = None

@dataclass
class Lambda(Node):
    params: list     # list[Param]
    ret: Optional[TypeExpr]
    body: "Expr"

@dataclass
class TupleLit(Node):
    items: list

@dataclass
class DictLit(Node):
    pairs: list      # list[tuple[Expr, Expr]]

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

@dataclass
class TryUnwrap(Node):
    expr: "Expr"

Expr = Union[Int, Float, Str, FString, Bool, Name, ListLit, ListComp, Lambda, TupleLit, DictLit, Unary, Binary,
             Range, Call, Index, Attr, TryUnwrap]

# --- statements ---
@dataclass
class Assign(Node):
    target: Expr     # Name | Index | Attr
    op: str          # '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=', '@='
    value: Expr

@dataclass
class TupleAssign(Node):
    targets: list    # list[Expr] (Name | Index | Attr)
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

# --- top level ---
@dataclass
class Param(Node):
    name: str
    type: TypeExpr

@dataclass
class StructDecl(Node):
    name: str
    fields: list                 # list[Param]

@dataclass
class Function(Node):
    name: str
    params: list
    ret: Optional[TypeExpr]
    body: list

@dataclass
class Import(Node):
    module: str                  # 'math' or 'a.b'

@dataclass
class Program(Node):
    items: list                  # Import | ConstDecl | StructDecl | Function


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "**=", "//=", "@="}
CMP_OPS = {"==", "!=", "<", "<=", ">", ">="}

def describe(t: Token) -> str:
    special = {"NEWLINE": "end of line", "INDENT": "unexpected indentation",
               "DEDENT": "end of block", "EOF": "end of file"}
    if t.kind in special:
        return special[t.kind]
    if t.kind == "FSTRING":
        return "an f-string"
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
            self.error(f"expected '{v}'{ctx}, found {describe(self.peek())}")
        return self.advance()

    def expect_kw(self, v: str) -> Token:
        if not self.is_kw(v):
            self.error(f"expected '{v}', found {describe(self.peek())}")
        return self.advance()

    def expect_name(self, what: str) -> Token:
        t = self.peek()
        if t.kind != "NAME":
            self.error(f"expected {what}, found {describe(t)}")
        return self.advance()

    def expect_newline(self):
        t = self.peek()
        if t.kind == "NEWLINE":
            self.advance()
        elif t.kind != "EOF":
            self.error(f"expected end of line, found {describe(t)}")

    def parse_program(self) -> Program:
        items = []
        while self.peek().kind != "EOF":
            t = self.peek()
            if t.kind == "NEWLINE":
                self.advance()
            elif t.kind == "INDENT":
                self.error("unexpected indentation")
            elif self.is_kw("import"):
                items.append(self.parse_import())
            elif self.is_kw("const"):
                items.append(self.parse_const())
            elif self.is_kw("struct"):
                items.append(self.parse_struct())
            elif self.is_kw("fn"):
                items.append(self.parse_function())
            else:
                self.error("only 'import', 'const', 'struct', and 'fn' are allowed "
                           f"at top level; found {describe(t)}")
        return Program(items, line=1, col=1)

    def parse_struct(self) -> StructDecl:
        kw = self.expect_kw("struct")
        name = self.expect_name("struct name").value
        self.expect_op(":", " after struct name")
        fields = []
        if self.peek().kind == "NEWLINE":
            self.advance()
            if self.peek().kind != "INDENT":
                self.error("expected an indented block in struct definition")
            self.advance()
            while self.peek().kind not in ("DEDENT", "EOF"):
                if self.peek().kind == "NEWLINE":
                    self.advance()
                    continue
                if self.is_kw("pass"):
                    self.advance()
                    self.expect_newline()
                    continue
                ft = self.expect_name("struct field name")
                self.expect_op(":", " (struct fields require a type)")
                fields.append(Param(ft.value, self.parse_type(), line=ft.line, col=ft.col))
                if self.is_op(","):
                    self.advance()
                if self.peek().kind in ("NEWLINE", "EOF"):
                    self.expect_newline()
            if self.peek().kind == "DEDENT":
                self.advance()
        else:
            while self.peek().kind not in ("NEWLINE", "EOF"):
                if self.is_kw("pass"):
                    self.advance()
                    break
                ft = self.expect_name("struct field name")
                self.expect_op(":", " (struct fields require a type)")
                fields.append(Param(ft.value, self.parse_type(), line=ft.line, col=ft.col))
                if self.is_op(","):
                    self.advance()
                elif self.peek().kind not in ("NEWLINE", "EOF"):
                    self.error(f"expected ',' or end of line in struct, found {describe(self.peek())}")
            self.expect_newline()
        return StructDecl(name, fields, line=kw.line, col=kw.col)

    def parse_import(self) -> Import:
        kw = self.expect_kw("import")
        parts = [self.expect_name("module name").value]
        while self.is_op("."):
            self.advance()
            parts.append(self.expect_name("module name").value)
        self.expect_newline()
        return Import(".".join(parts), line=kw.line, col=kw.col)

    def parse_const(self) -> ConstDecl:
        kw = self.expect_kw("const")
        name = self.expect_name("constant name").value
        ty = None
        if self.is_op(":"):
            self.advance()
            ty = self.parse_type()
        self.expect_op("=", " in constant declaration")
        value = self.parse_expr()
        self.expect_newline()
        return ConstDecl(name, ty, value, line=kw.line, col=kw.col)

    def parse_function(self) -> Function:
        kw = self.expect_kw("fn")
        name = self.expect_name("function name").value
        self.expect_op("(", " after function name")
        params = []
        while not self.is_op(")"):
            nt = self.expect_name("parameter name")
            self.expect_op(":", " (parameters require a type)")
            params.append(Param(nt.value, self.parse_type(), line=nt.line, col=nt.col))
            if self.is_op(","):
                self.advance()
            elif not self.is_op(")"):
                self.error(f"expected ',' or ')', found {describe(self.peek())}")
        self.expect_op(")")
        ret = None
        if self.is_op("->"):
            self.advance()
            ret = self.parse_type()
        body = self.parse_block()
        return Function(name, params, ret, body, line=kw.line, col=kw.col)

    def parse_type(self) -> TypeExpr:
        t = self.peek()
        if t.kind == "NAME":
            self.advance()
            if t.value in ("tensor", "Tensor") and self.is_op("["):
                self.advance()
                elem = self.parse_type()
                self.expect_op("]", " when closing tensor type")
                return TypeTensor(elem, line=t.line, col=t.col)
            if t.value in ("Result", "NatResult") and self.is_op("["):
                self.advance()
                elem = self.parse_type()
                self.expect_op("]", " when closing Result type")
                return TypeResult(elem, line=t.line, col=t.col)
            return TypeName(t.value, line=t.line, col=t.col)
        if self.is_op("["):
            self.advance()
            elem = self.parse_type()
            self.expect_op("]", " when closing list type")
            return TypeList(elem, line=t.line, col=t.col)
        if self.is_op("("):
            self.advance()
            elems = []
            while not self.is_op(")"):
                elems.append(self.parse_type())
                if self.is_op(","):
                    self.advance()
                elif not self.is_op(")"):
                    self.error(f"expected ',' or ')' in tuple type, found {describe(self.peek())}")
            self.expect_op(")", " when closing tuple type")
            return TypeTuple(elems, line=t.line, col=t.col)
        self.error(f"expected a type, found {describe(t)}")

    def parse_block(self) -> list:
        self.expect_op(":", " at end of header")
        if self.peek().kind == "NEWLINE":
            self.advance()
            if self.peek().kind != "INDENT":
                self.error("expected an indented block")
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
            self.error("unexpected indentation")
        if t.kind == "KW":
            if t.value == "if":
                return self.parse_if()
            if t.value == "while":
                return self.parse_while()
            if t.value == "for":
                return self.parse_for()
            if t.value in ("elif", "else"):
                self.error(f"'{t.value}' without matching 'if'")
            if t.value in ("fn", "import", "struct"):
                self.error(f"'{t.value}' is only allowed at top level")
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
        return If(cond, body, orelse, line=kw.line, col=kw.col)

    def parse_while(self) -> While:
        kw = self.expect_kw("while")
        cond = self.parse_expr()
        return While(cond, self.parse_block(), line=kw.line, col=kw.col)

    def parse_for(self) -> For:
        kw = self.expect_kw("for")
        var = self.expect_name("loop variable name").value
        self.expect_kw("in")
        it = self.parse_expr()
        return For(var, it, self.parse_block(), line=kw.line, col=kw.col)

    def parse_tuple_or_expr(self) -> Expr:
        first = self.parse_expr()
        if self.is_op(","):
            items = [first]
            while self.is_op(","):
                self.advance()
                if self.peek().kind in ("NEWLINE", "EOF"):
                    break
                items.append(self.parse_expr())
            return TupleLit(items, line=first.line, col=first.col)
        return first

    def parse_simple_statement(self):
        t = self.peek()
        if t.kind == "KW":
            if t.value == "return":
                self.advance()
                value = None
                if self.peek().kind not in ("NEWLINE", "EOF"):
                    value = self.parse_tuple_or_expr()
                return Return(value, line=t.line, col=t.col)
            if t.value == "break":
                self.advance()
                return Break(line=t.line, col=t.col)
            if t.value == "continue":
                self.advance()
                return Continue(line=t.line, col=t.col)
            if t.value == "pass":
                self.advance()
                return Pass(line=t.line, col=t.col)
            if t.value == "const":
                return self.parse_const_inline()

        expr = self.parse_expr()
        if self.is_op(","):
            targets = [expr]
            while self.is_op(","):
                self.advance()
                targets.append(self.parse_expr())
            for tgt in targets:
                if not isinstance(tgt, (Name, Index, Attr)):
                    self.error("the left-hand side of unpacking must be a name, "
                               "an index, or an attribute", t)
            self.expect_op("=", " in tuple unpacking")
            return TupleAssign(targets, self.parse_tuple_or_expr(), line=t.line, col=t.col)
        if isinstance(expr, TupleLit) and self.is_op("="):
            for tgt in expr.items:
                if not isinstance(tgt, (Name, Index, Attr)):
                    self.error("the left-hand side of unpacking must be a name, "
                               "an index, or an attribute", t)
            self.advance()
            return TupleAssign(expr.items, self.parse_tuple_or_expr(), line=t.line, col=t.col)
        if self.is_op(":"):
            if not isinstance(expr, Name):
                self.error("type annotations are only allowed on simple variable names")
            self.advance()
            ty = self.parse_type()
            self.expect_op("=", " in typed declaration")
            return VarDecl(expr.id, ty, self.parse_tuple_or_expr(), line=t.line, col=t.col)
        nt = self.peek()
        if nt.kind == "OP" and nt.value in ASSIGN_OPS:
            if not isinstance(expr, (Name, Index, Attr)):
                self.error("the left-hand side of an assignment must be a name, "
                           "an index, or an attribute", t)
            self.advance()
            val_expr = self.parse_tuple_or_expr() if nt.value == "=" else self.parse_expr()
            return Assign(expr, nt.value, val_expr, line=t.line, col=t.col)
        return ExprStmt(expr, line=t.line, col=t.col)

    def parse_const_inline(self) -> ConstDecl:
        kw = self.expect_kw("const")
        name = self.expect_name("constant name").value
        ty = None
        if self.is_op(":"):
            self.advance()
            ty = self.parse_type()
        self.expect_op("=", " in constant declaration")
        return ConstDecl(name, ty, self.parse_expr(), line=kw.line, col=kw.col)

    def parse_expr(self) -> Expr:
        left = self.parse_or()
        if self.is_op("..") or self.is_op("..="):
            op = self.advance()
            right = self.parse_or()
            if self.is_op("..") or self.is_op("..="):
                self.error("ranges cannot be chained")
            return Range(left, right, op.value == "..=", line=op.line, col=op.col)
        return left

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.is_kw("or"):
            op = self.advance()
            left = Binary("or", left, self.parse_and(), line=op.line, col=op.col)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_not()
        while self.is_kw("and"):
            op = self.advance()
            left = Binary("and", left, self.parse_not(), line=op.line, col=op.col)
        return left

    def parse_not(self) -> Expr:
        if self.is_kw("not"):
            op = self.advance()
            return Unary("not", self.parse_not(), line=op.line, col=op.col)
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        left = self.parse_arith()
        t = self.peek()
        if t.kind == "OP" and t.value in CMP_OPS:
            self.advance()
            left = Binary(t.value, left, self.parse_arith(), line=t.line, col=t.col)
            n = self.peek()
            if n.kind == "OP" and n.value in CMP_OPS:
                self.error("comparisons cannot be chained; "
                           "use 'and' (e.g., a < b and b < c)")
        return left

    def parse_arith(self) -> Expr:
        left = self.parse_term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.advance()
            left = Binary(op.value, left, self.parse_term(), line=op.line, col=op.col)
        return left

    def parse_term(self) -> Expr:
        left = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%", "//", "@"):
            op = self.advance()
            left = Binary(op.value, left, self.parse_unary(), line=op.line, col=op.col)
        return left

    def parse_unary(self) -> Expr:
        if self.is_op("-") or self.is_op("+"):
            op = self.advance()
            return Unary(op.value, self.parse_unary(), line=op.line, col=op.col)
        return self.parse_power()

    def parse_power(self) -> Expr:
        base = self.parse_postfix()
        if self.is_op("**"):
            op = self.advance()
            return Binary("**", base, self.parse_unary(), line=op.line, col=op.col)
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
                        self.error("expected ',' or ')' in argument list, "
                                   f"found {describe(self.peek())}")
                self.expect_op(")")
                e = Call(e, args, line=t.line, col=t.col)
            elif self.is_op("["):
                self.advance()
                idx = self.parse_expr()
                self.expect_op("]", " when closing index")
                e = Index(e, idx, line=t.line, col=t.col)
            elif self.is_op("."):
                self.advance()
                e = Attr(e, self.expect_name("attribute name").value, line=t.line, col=t.col)
            elif self.is_op("?"):
                self.advance()
                e = TryUnwrap(e, line=t.line, col=t.col)
            else:
                return e

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t.kind == "INT":
            self.advance()
            return Int(t.value, line=t.line, col=t.col)
        if t.kind == "FLOAT":
            self.advance()
            return Float(t.value, line=t.line, col=t.col)
        if t.kind == "STRING":
            self.advance()
            return Str(t.value, line=t.line, col=t.col)
        if t.kind == "FSTRING":
            self.advance()
            return self.parse_fstring(t)
        if t.kind == "KW" and t.value in ("true", "false"):
            self.advance()
            return Bool(t.value == "true", line=t.line, col=t.col)
        if self.is_kw("lambda") or self.is_kw("fn"):
            kw = self.advance()
            params = []
            if self.is_op("("):
                self.advance()
                while not self.is_op(")"):
                    pt = self.expect_name("lambda parameter name")
                    pty = None
                    if self.is_op(":"):
                        self.advance()
                        pty = self.parse_type()
                    params.append(Param(pt.value, pty, line=pt.line, col=pt.col))
                    if self.is_op(","):
                        self.advance()
                    elif not self.is_op(")"):
                        self.error(f"expected ',' or ')' in lambda parameters, found {describe(self.peek())}")
                self.expect_op(")")
            elif kw.value == "lambda":
                while not self.is_op(":") and not self.is_op("->"):
                    pt = self.expect_name("lambda parameter name")
                    params.append(Param(pt.value, None, line=pt.line, col=pt.col))
                    if self.is_op(","):
                        self.advance()
                    elif not self.is_op(":") and not self.is_op("->"):
                        self.error(f"expected ',' or ':' in lambda parameters, found {describe(self.peek())}")
            else:
                self.error(f"expected '(' after 'fn' in lambda expression, found {describe(self.peek())}")
            ret = None
            if self.is_op("->"):
                self.advance()
                ret = self.parse_type()
            self.expect_op(":", " in lambda body")
            body = self.parse_expr()
            return Lambda(params, ret, body, line=kw.line, col=kw.col)
        if t.kind == "NAME":
            self.advance()
            return Name(t.value, line=t.line, col=t.col)
        if self.is_op("("):
            self.advance()
            if self.is_op(")"):
                self.advance()
                return TupleLit([], line=t.line, col=t.col)
            e = self.parse_expr()
            if self.is_op(","):
                items = [e]
                while self.is_op(","):
                    self.advance()
                    if self.is_op(")"):
                        break
                    items.append(self.parse_expr())
                self.expect_op(")", " when closing tuple")
                return TupleLit(items, line=t.line, col=t.col)
            self.expect_op(")", " when closing parentheses")
            return e
        if self.is_op("["):
            self.advance()
            if self.is_op("]"):
                self.advance()
                return ListLit([], line=t.line, col=t.col)
            first = self.parse_expr()
            if self.is_kw("for"):
                self.advance()
                var = self.expect_name("list comprehension variable name").value
                self.expect_kw("in")
                it = self.parse_expr()
                cond = None
                if self.is_kw("if"):
                    self.advance()
                    cond = self.parse_expr()
                self.expect_op("]")
                return ListComp(first, var, it, cond, line=t.line, col=t.col)
            items = [first]
            if self.is_op(","):
                self.advance()
            elif not self.is_op("]"):
                self.error("expected ',' or ']' in list, "
                           f"found {describe(self.peek())}")
            while not self.is_op("]"):
                items.append(self.parse_expr())
                if self.is_op(","):
                    self.advance()
                elif not self.is_op("]"):
                    self.error("expected ',' or ']' in list, "
                               f"found {describe(self.peek())}")
            self.expect_op("]")
            return ListLit(items, line=t.line, col=t.col)
        if self.is_op("{"):
            self.advance()
            pairs = []
            while not self.is_op("}"):
                k_expr = self.parse_expr()
                self.expect_op(":", " between dictionary key and value")
                v_expr = self.parse_expr()
                pairs.append((k_expr, v_expr))
                if self.is_op(","):
                    self.advance()
                elif not self.is_op("}"):
                    self.error("expected ',' or '}' in dictionary, "
                               f"found {describe(self.peek())}")
            self.expect_op("}")
            return DictLit(pairs, line=t.line, col=t.col)
        self.error(f"expected an expression, found {describe(t)}")

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
                    self.error("'{' without matching '}' in f-string "
                               "(use '{{' for a literal '{')", tok)

                expr_text = raw[i + 1:colon if colon is not None else j]
                spec = raw[colon + 1:j] if colon is not None else None
                parts.append(FormatPart(self._sub_expr(expr_text, tok), spec,
                                        line=tok.line, col=tok.col))
                i = j + 1
            elif c == "}":
                if i + 1 < n and raw[i + 1] == "}":
                    buf.append("}")
                    i += 2
                else:
                    self.error("unmatched '}' in f-string "
                               "(use '}}' for a literal '}')", tok)
            else:
                buf.append(c)
                i += 1
        flush()
        return FString(parts, line=tok.line, col=tok.col)

    def _sub_expr(self, text: str, tok: Token) -> Expr:
        text = text.strip()
        if not text:
            self.error("empty expression inside '{}' in f-string", tok)
        try:
            sub = Parser(tokenize(text, first_line=tok.line))
            e = sub.parse_expr()
            if sub.peek().kind not in ("NEWLINE", "EOF"):
                sub.error(f"unexpected token {describe(sub.peek())}")
            return e
        except NatError as ex:
            raise NatError(f"in f-string: {ex.message}", tok.line, tok.col) from None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def parse(source: str) -> Program:
    return Parser(tokenize(source)).parse_program()


def parse_expression(source: str) -> Expr:
    p = Parser(tokenize(source.strip()))
    e = p.parse_expr()
    if p.peek().kind not in ("NEWLINE", "EOF"):
        p.error(f"unexpected token {describe(p.peek())}")
    return e


# ----------------------------------------------------------------------------
# AST pretty-printing
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
# SEMANTIC ANALYZER
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

# Expanded numeric types (including t2 / ternary for 1.58-bit AI and 256-bit fixed/interval)
I8, U8, I16, U16, I32, U32, I64, U64 = (Prim(n) for n in ("i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"))
F16, BF16, F32, F64 = (Prim(n) for n in ("f16", "bf16", "f32", "f64"))
C64 = Prim("c64")
COMPLEX = Prim("complex")
T2 = Prim("t2")   # Ternary: -1, 0, +1
FIX256 = Prim("fix256")
INTERVAL256 = Prim("interval256")
STR, BOOL, VOID, ANY = (Prim(n) for n in ("str", "bool", "void", "any"))

@dataclass(frozen=True)
class ResultT:
    elem: Any = ANY
    def __repr__(self) -> str:
        return f"Result[{self.elem}]"

@dataclass(frozen=True)
class TupleT:
    elems: tuple
    def __repr__(self) -> str:
        return "(" + ", ".join(str(e) for e in self.elems) + ")"

@dataclass(frozen=True)
class StructT:
    name: str
    fields: tuple   # tuple[tuple[str, Any], ...]
    def __repr__(self) -> str:
        return self.name
    def field_dict(self) -> dict[str, Any]:
        return dict(self.fields)

@dataclass(frozen=True)
class DictT:
    key: Any = ANY
    val: Any = ANY
    def __repr__(self) -> str:
        return f"dict[{self.key}, {self.val}]"

RESULT = ResultT(ANY)

FLOAT_TYPES = {"f16", "bf16", "f32", "f64"}
INT_TYPES = {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "t2", "ternary"}
COMPLEX_TYPES = {"c64", "complex"}

PRIM_TYPES = {
    "i8": I8, "u8": U8, "i16": I16, "u16": U16,
    "i32": I32, "u32": U32, "i64": I64, "u64": U64,
    "f16": F16, "bf16": BF16, "f32": F32, "f64": F64,
    "c64": C64, "complex": COMPLEX, "Complex": COMPLEX,
    "t2": T2, "ternary": T2,
    "fix256": FIX256, "Fix256": FIX256,
    "interval256": INTERVAL256, "Interval256": INTERVAL256,
    "str": STR, "bool": BOOL, "void": VOID, "any": ANY,
    "Result": RESULT, "NatResult": RESULT,
    "dict": DictT(ANY, ANY), "Dict": DictT(ANY, ANY),
    "KVCache": ANY, "Tokenizer": ANY,
}

@dataclass
class FuncSig:
    params: list
    ret: Any
    variadic: bool = False
    builtin: bool = False
    min_args: Optional[int] = None
    max_args: Optional[int] = None
    ret_builder: Optional[Callable[[list[Any]], Any]] = None
    arg_validator: Optional[Callable[[Any, list[Any], list[Any], Any], None]] = None
    impl: Optional[Callable] = None
    doc: str = ""


def _tensor_ret(ats: list[Any]) -> Any:
    if not ats:
        return TensorT(ANY)
    t = ats[0]
    if t == ANY:
        return TensorT(ANY)
    depth = 0
    c = t
    while isinstance(c, (ListT, TensorT)):
        depth += 1
        c = c.elem
    elem = c if is_num(c) else ANY
    res = elem
    for _ in range(max(1, depth)):
        res = TensorT(res)
    return res


def _quant_ret(ats: list[Any]) -> Any:
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


def _len_validator(analyzer: Any, ats: list[Any], args: list[Any], node: Any) -> None:
    if ats:
        t = ats[0]
        if not (isinstance(t, (ListT, TensorT, TupleT, DictT)) or t in (STR, ANY)):
            analyzer.report(f"argument 1 of 'len': expected list, tensor, tuple, dict, or str, got {t}", args[0])


BUILTINS: dict[str, FuncSig] = {
    "print":            FuncSig([ANY], VOID, variadic=True, builtin=True, min_args=0, max_args=None),
    "len":              FuncSig([ANY], I64, builtin=True, min_args=1, max_args=1, arg_validator=_len_validator),
    "append":           FuncSig([ANY, ANY], VOID, builtin=True, min_args=2, max_args=2),
    "push":             FuncSig([ANY, ANY], VOID, builtin=True, min_args=2, max_args=2),
    "Dict":             FuncSig([], DictT(ANY, ANY), builtin=True, min_args=0, max_args=0),
    "sqrt":             FuncSig([F64], F64, builtin=True, min_args=1, max_args=1),
    "exp":              FuncSig([F64], F64, builtin=True, min_args=1, max_args=1),
    "sin":              FuncSig([F64], F64, builtin=True, min_args=1, max_args=1),
    "cos":              FuncSig([F64], F64, builtin=True, min_args=1, max_args=1),
    "ln":               FuncSig([F64], F64, builtin=True, min_args=1, max_args=1),
    "floor":            FuncSig([F64], I64, builtin=True, min_args=1, max_args=1),
    "to_i64":           FuncSig([ANY], I64, builtin=True, min_args=1, max_args=1),
    "to_f64":           FuncSig([ANY], F64, builtin=True, min_args=1, max_args=1),
    "dot":              FuncSig([ListT(F64), ListT(F64)], F64, builtin=True, min_args=2, max_args=2),
    # Builtins added for AI and tensor manipulation
    "tanh":             FuncSig([F64], F64, builtin=True, min_args=1, max_args=1),
    "relu":             FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1, ret_builder=lambda ats: ats[0] if ats else ANY),
    "sigmoid":          FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1, ret_builder=lambda ats: ats[0] if ats else ANY),
    "matmul":           FuncSig([ANY, ANY], ANY, builtin=True, min_args=2, max_args=2),
    "sum":              FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    "mean":             FuncSig([ANY], F64, builtin=True, min_args=1, max_args=1),
    "max":              FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    "min":              FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    "argmax":           FuncSig([ANY], I64, builtin=True, min_args=1, max_args=1),
    "zeros":            FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    "ones":             FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    "shape":            FuncSig([ANY], ListT(I64), builtin=True, min_args=1, max_args=1),
    "reshape":          FuncSig([ANY, ANY], ANY, builtin=True, min_args=2, max_args=2),
    "transpose":        FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    "tensor":           FuncSig([ANY], TensorT(ANY), builtin=True, min_args=1, max_args=4, ret_builder=_tensor_ret),
    "device":           FuncSig([ANY], STR, builtin=True, min_args=1, max_args=1),
    # Ternary computing primitives
    "quantize_ternary": FuncSig([ANY], TensorT(T2), builtin=True, min_args=1, max_args=2, ret_builder=_quant_ret),
    "pack_ternary":     FuncSig([ANY], ListT(I64), builtin=True, min_args=1, max_args=1),
    "unpack_ternary":   FuncSig([ANY, I64], ListT(I64), builtin=True, min_args=2, max_args=2),
    "clock":            FuncSig([], F64, builtin=True, min_args=0, max_args=0),
    "random_ternary":   FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2),
    "randn":            FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2),
    "rotate_rht":       FuncSig([ANY], ListT(F64), builtin=True, min_args=1, max_args=2),
    "turbo_quant":      FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2),
    "turbo_dequant":    FuncSig([ANY], ListT(F64), builtin=True, min_args=1, max_args=1),
    "turbo_dot":        FuncSig([ANY, ANY], F64, builtin=True, min_args=2, max_args=2),
    "rmsnorm":          FuncSig([ANY, ANY], ANY, builtin=True, min_args=2, max_args=3, ret_builder=lambda ats: ats[0] if ats else ANY),
    "apply_rope":       FuncSig([ANY, I64], ListT(F64), builtin=True, min_args=2, max_args=3, ret_builder=lambda ats: ats[0] if ats else ANY),
    "silu":             FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1, ret_builder=lambda ats: ats[0] if ats else ANY),
    # SafeTensors I/O (Hugging Face standard)
    "load_safetensors": FuncSig([STR, STR], ANY, builtin=True, min_args=2, max_args=2),
    "safetensors_keys": FuncSig([STR], ListT(STR), builtin=True, min_args=1, max_args=1),
    "save_safetensors": FuncSig([STR, STR, ANY], VOID, builtin=True, min_args=3, max_args=3),
    # Canonical BitNet b1.58 (AbsMean Scaling & Dense Base-3 Packing 5 trits/byte)
    "quantize_bitnet":       FuncSig([ANY], TensorT(T2), builtin=True, min_args=1, max_args=1, ret_builder=_quant_ret),
    "tensor_absmean":        FuncSig([ANY], F64, builtin=True, min_args=1, max_args=1),
    "pack_ternary_5trits":   FuncSig([ANY], ListT(I64), builtin=True, min_args=1, max_args=1),
    "unpack_ternary_5trits": FuncSig([ANY, I64], ListT(I64), builtin=True, min_args=2, max_args=2),
    # Numerical Fidelity Validation & Vec-LUT Kernel
    "mae":                   FuncSig([ANY, ANY], F64, builtin=True, min_args=2, max_args=2),
    "relative_error":        FuncSig([ANY, ANY], F64, builtin=True, min_args=2, max_args=2),
    "pearson_corr":          FuncSig([ANY, ANY], F64, builtin=True, min_args=2, max_args=2),
    "cosine_sim":            FuncSig([ANY, ANY], F64, builtin=True, min_args=2, max_args=2),
    "gemv_vec_lut":          FuncSig([ANY, ANY, I64, I64], ANY, builtin=True, min_args=4, max_args=5),
    # Model Primitives (Attention, Tokenizer, KV Cache)
    "softmax":               FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2, ret_builder=lambda ats: ats[0] if ats else ANY),
    "causal_mask":           FuncSig([I64], ANY, builtin=True, min_args=1, max_args=1),
    "sdpa":                  FuncSig([ANY, ANY, ANY], TensorT(F32), builtin=True, min_args=3, max_args=5),
    "Tokenizer":             FuncSig([], ANY, builtin=True, min_args=0, max_args=1),
    "load_tokenizer":        FuncSig([], ANY, builtin=True, min_args=0, max_args=1),
    "encode":                FuncSig([ANY, STR], ANY, builtin=True, min_args=2, max_args=2),
    "decode":                FuncSig([ANY, ANY], STR, builtin=True, min_args=2, max_args=2),
    "KVCache":               FuncSig([I64, I64], ANY, builtin=True, min_args=0, max_args=2),
    "concat":                FuncSig([ANY], TensorT(F32), builtin=True, min_args=1, max_args=2),
    "sample":                FuncSig([ANY], I64, builtin=True, min_args=1, max_args=3),
    "read_file":             FuncSig([STR], STR, builtin=True, min_args=1, max_args=1),
    "read_file_safe":        FuncSig([STR], ResultT(STR), builtin=True, min_args=1, max_args=1),
    "exists_file":           FuncSig([STR], BOOL, builtin=True, min_args=1, max_args=1),
    "load_tokenizer_safe":   FuncSig([STR], ResultT(ANY), builtin=True, min_args=0, max_args=1),
    "ok":                    FuncSig([ANY], RESULT, builtin=True, min_args=1, max_args=1, ret_builder=lambda ats: ResultT(ats[0] if ats else ANY)),
    "err":                   FuncSig([STR], RESULT, builtin=True, min_args=1, max_args=1),
    "is_ok":                 FuncSig([ANY], BOOL, builtin=True, min_args=1, max_args=1),
    "is_err":                FuncSig([ANY], BOOL, builtin=True, min_args=1, max_args=1),
    "unwrap":                FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1, ret_builder=lambda ats: ats[0].elem if (ats and isinstance(ats[0], ResultT)) else (ats[0] if ats else ANY)),
    "unwrap_or":             FuncSig([ANY, ANY], ANY, builtin=True, min_args=2, max_args=2, ret_builder=lambda ats: ats[0].elem if (ats and isinstance(ats[0], ResultT)) else (ats[1] if len(ats) > 1 else ANY)),
    "error_message":         FuncSig([ANY], STR, builtin=True, min_args=1, max_args=1),
    "fill":                  FuncSig([I64, ANY], ListT(ANY), builtin=True, min_args=2, max_args=2, ret_builder=lambda ats: ListT(ats[1] if len(ats) > 1 else ANY)),
    "seed":                  FuncSig([I64], VOID, builtin=True, min_args=1, max_args=1),
    # Autodiff and Optimizers (Native Training)
    "requires_grad":         FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2, ret_builder=lambda ats: ats[0] if ats else ANY),
    "backward":              FuncSig([ANY], VOID, builtin=True, min_args=1, max_args=1),
    "zero_grad":             FuncSig([ANY], VOID, builtin=True, min_args=1, max_args=1),
    "cross_entropy":         FuncSig([ANY, ANY], ANY, builtin=True, min_args=2, max_args=2),
    "sgd":                   FuncSig([ANY, F64], VOID, builtin=True, min_args=2, max_args=2),
    "adam":                  FuncSig([ANY], VOID, builtin=True, min_args=1, max_args=5),
    # Spectral Primitives and Complex Numbers (Phase 1: Fourier)
    "Complex":               FuncSig([F64], COMPLEX, builtin=True, min_args=1, max_args=2),
    "rfft":                  FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2),
    "irfft":                 FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=2),
    "power_spectrum":        FuncSig([ANY], ANY, builtin=True, min_args=1, max_args=1),
    # 256-Bit Fixed-Point and Interval Arithmetic (High Altitude Steps)
    "Fix256":                FuncSig([ANY], FIX256, builtin=True, min_args=1, max_args=2),
    "fix256":                FuncSig([ANY], FIX256, builtin=True, min_args=1, max_args=2),
    "Interval256":           FuncSig([ANY, ANY], INTERVAL256, builtin=True, min_args=1, max_args=2),
    "interval256":           FuncSig([ANY, ANY], INTERVAL256, builtin=True, min_args=1, max_args=2),
    "interval256_step":      FuncSig([INTERVAL256, INTERVAL256, FIX256], INTERVAL256, builtin=True, min_args=3, max_args=3),
    "interval256_width":     FuncSig([INTERVAL256], FIX256, builtin=True, min_args=1, max_args=1),
    "interval256_mid":       FuncSig([INTERVAL256], FIX256, builtin=True, min_args=1, max_args=1),
    "interval256_is_coherent": FuncSig([INTERVAL256, FIX256], BOOL, builtin=True, min_args=2, max_args=2),
    # Higher-Order Functions (Lambda / Functional Programming)
    "map":                   FuncSig([ANY, ANY], ListT(ANY), builtin=True, min_args=2, max_args=2, ret_builder=lambda ats: ListT(ats[0].ret if (isinstance(ats[0], FuncSig) and ats[0].ret != ANY) else (ats[1].elem if (len(ats) > 1 and isinstance(ats[1], (ListT, TensorT))) else ANY))),
    "filter":                FuncSig([ANY, ANY], ListT(ANY), builtin=True, min_args=2, max_args=2, ret_builder=lambda ats: ListT(ats[1].elem if (len(ats) > 1 and isinstance(ats[1], (ListT, TensorT))) else ANY)),
}

KNOWN_MODULES = {"math", "time"}
MATH_FUNCS = {
    "sqrt", "exp", "sin", "cos", "ln", "floor", "tanh", "rfft", "irfft", "power_spectrum", "Complex",
    "Fix256", "fix256", "Interval256", "interval256", "interval256_step", "interval256_width", "interval256_mid", "interval256_is_coherent"
}
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
    level: str                  # "error" | "warning"
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
    return t if (is_num(t) or t == ANY) else None

def same(a, b) -> bool:
    if a == ANY or b == ANY:
        return True
    if isinstance(a, ListT) and isinstance(b, ListT):
        return same(a.elem, b.elem)
    if isinstance(a, TensorT) and isinstance(b, TensorT):
        return same(a.elem, b.elem)
    if isinstance(a, ResultT) and isinstance(b, ResultT):
        return same(a.elem, b.elem)
    if isinstance(a, TupleT) and isinstance(b, TupleT):
        return len(a.elems) == len(b.elems) and all(same(x, y) for x, y in zip(a.elems, b.elems))
    if isinstance(a, StructT) and isinstance(b, StructT):
        return a.name == b.name
    if isinstance(a, DictT) and isinstance(b, DictT):
        return same(a.key, b.key) and same(a.val, b.val)
    if isinstance(a, FuncSig) and isinstance(b, FuncSig):
        return len(a.params) == len(b.params) and all(same(x, y) for x, y in zip(a.params, b.params)) and same(a.ret, b.ret)
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
        return same(src, dst)   # Integer lists do not promote to float lists
    if isinstance(src, ResultT) and isinstance(dst, ResultT):
        return assignable(src.elem, dst.elem)
    if isinstance(src, TupleT) and isinstance(dst, TupleT):
        return len(src.elems) == len(dst.elems) and all(assignable(s, d) for s, d in zip(src.elems, dst.elems))
    if isinstance(src, DictT) and isinstance(dst, DictT):
        return assignable(src.key, dst.key) and assignable(src.val, dst.val)
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
    if isinstance(a, ResultT) and isinstance(b, ResultT):
        e = unify(a.elem, b.elem)
        return ResultT(e) if e is not None else None
    if isinstance(a, TupleT) and isinstance(b, TupleT) and len(a.elems) == len(b.elems):
        es = [unify(x, y) for x, y in zip(a.elems, b.elems)]
        return TupleT(tuple(es)) if all(e is not None for e in es) else None
    return None


def _resolve_local_module_path(mod_name: str) -> Optional[str]:
    candidates = [
        f"{mod_name}.nat",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{mod_name}.nat"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


class SemanticAnalyzer:
    def __init__(self):
        self.functions: dict[str, FuncSig] = dict(BUILTINS)
        self.structs: dict[str, StructT] = {}
        self.modules: set[str] = set()
        self.user_modules: dict[str, dict[str, Any]] = {}
        self.scopes: list[dict[str, Symbol]] = [{}]
        self.diagnostics: list[Diagnostic] = []
        self.cur_ret = None
        self.loop_depth = 0

    def report(self, msg: str, node, level: str = "error"):
        self.diagnostics.append(Diagnostic(level, msg, getattr(node, "line", 0)))

    @property
    def errors(self):
        return [d for d in self.diagnostics if d.level == "error"]

    @property
    def warnings(self):
        return [d for d in self.diagnostics if d.level == "warning"]

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        scope = self.scopes.pop()
        visible = [n for s in self.scopes for n in s] + list(scope)
        for sym in scope.values():
            if sym.implicit and not sym.used:
                msg = f"variable '{sym.name}' is assigned a value but never used"
                near = difflib.get_close_matches(
                    sym.name, [n for n in visible if n != sym.name], n=1, cutoff=0.75)
                if near:
                    msg += f" (possible typo for '{near[0]}'?)"
                self.diagnostics.append(Diagnostic("warning", msg, sym.line))

    def define(self, name: str, node, type_, *, const=False, implicit=False) -> bool:
        if name in self.modules:
            self.report(f"'{name}' is already a module name; choose another name", node)
            return False
        if name in self.functions:
            self.report(f"'{name}' is already a function name; choose another name", node)
            return False
        if name != "_":
            for scope in self.scopes:
                if name in scope:
                    self.report(f"'{name}' was already declared on line {scope[name].line} "
                                "(shadowing is not allowed)", node)
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
        return f"name '{name}' is not defined" + (f" (did you mean '{near[0]}'?)" if near else "")

    def resolve_type(self, t):
        if isinstance(t, TypeName):
            if t.name in self.structs:
                return self.structs[t.name]
            r = PRIM_TYPES.get(t.name)
            if r is None:
                self.report(f"unknown type '{t.name}'", t)
                return ANY
            return r
        if isinstance(t, TypeTensor):
            return TensorT(self.resolve_type(t.elem))
        if isinstance(t, TypeResult):
            return ResultT(self.resolve_type(t.elem))
        if isinstance(t, TypeTuple):
            return TupleT(tuple(self.resolve_type(e) for e in t.elems))
        return ListT(self.resolve_type(t.elem))

    def analyze(self, program: Program, _visited_modules: Optional[set[str]] = None) -> list[Diagnostic]:
        if _visited_modules is None:
            _visited_modules = set()
        for it in program.items:
            if isinstance(it, Import):
                if it.module in KNOWN_MODULES:
                    self.modules.add(it.module)
                elif it.module in _visited_modules or it.module in self.modules:
                    self.modules.add(it.module)
                else:
                    mod_path = _resolve_local_module_path(it.module)
                    if mod_path is not None:
                        try:
                            with open(mod_path, "r", encoding="utf-8") as mf:
                                mod_src = mf.read()
                            mod_prog = parse(mod_src)
                            sub_sem = SemanticAnalyzer()
                            sub_visited = set(_visited_modules)
                            sub_visited.add(it.module)
                            sub_sem.analyze(mod_prog, _visited_modules=sub_visited)
                            for d in sub_sem.errors:
                                self.report(f"in module '{it.module}': {d.msg}", it)
                            mod_funcs = {
                                k: v for k, v in sub_sem.functions.items() if k not in BUILTINS
                            }
                            mod_consts = {
                                k: sym.type for k, sym in sub_sem.scopes[0].items()
                            }
                            self.modules.add(it.module)
                            self.user_modules[it.module] = {
                                "functions": mod_funcs,
                                "consts": mod_consts,
                                "structs": dict(sub_sem.structs),
                            }
                            for sname, st in sub_sem.structs.items():
                                if sname not in self.structs:
                                    self.structs[sname] = st
                            for fname, fsig in mod_funcs.items():
                                if fname not in self.functions:
                                    self.functions[fname] = fsig
                        except Exception as ex:
                            self.report(f"error importing module '{it.module}': {ex}", it)
                    else:
                        self.report(f"unknown module '{it.module}'", it)

        # First pass for StructDecl (allows references in fields and functions)
        for it in program.items:
            if isinstance(it, StructDecl):
                if it.name in self.structs or it.name in PRIM_TYPES or it.name in self.functions:
                    self.report(f"struct '{it.name}' is already defined or collides with an existing name", it)
                    continue
                self.structs[it.name] = StructT(it.name, ())

        for it in program.items:
            if isinstance(it, StructDecl) and it.name in self.structs:
                seen_fields: set[str] = set()
                resolved_fields = []
                for f in it.fields:
                    if f.name in seen_fields:
                        self.report(f"duplicate field '{f.name}' in struct '{it.name}'", f)
                        continue
                    seen_fields.add(f.name)
                    resolved_fields.append((f.name, self.resolve_type(f.type)))
                st = StructT(it.name, tuple(resolved_fields))
                self.structs[it.name] = st
                self.functions[it.name] = FuncSig([ft for _, ft in resolved_fields], st)

        sigs: dict[int, FuncSig] = {}
        for it in program.items:
            if isinstance(it, Function):
                if it.name in self.functions:
                    why = ("is a built-in function" if self.functions[it.name].builtin
                           else "is already defined")
                    self.report(f"function '{it.name}' {why}", it)
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
            self.report(f"construct not supported by the analyzer: {type(node).__name__}", node)
            return ANY
        return m(node)

    def expr(self, node):
        t = self.visit(node)
        if t == VOID:
            self.report("this expression does not produce a value", node)
            return ANY
        return t

    def cond(self, node):
        t = self.expr(node)
        if t != BOOL and t != ANY:
            self.report(f"condition must be bool, got {t}", node)

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
            self.report(f"function '{node.name}' declares return type {sig.ret}, but not all "
                        "paths return a value", node)
        self.exit_scope()
        self.cur_ret = None

    def visit_ConstDecl(self, node: ConstDecl):
        vt = self.expr(node.value)
        dt = self.resolve_type(node.type) if node.type else vt
        if node.type and not assignable(vt, dt):
            self.report(f"constant '{node.name}' declared as {dt}, but value is {vt}", node)
        self.define(node.name, node, dt, const=True)

    def visit_VarDecl(self, node: VarDecl):
        vt = self.expr(node.value)
        dt = self.resolve_type(node.type)
        if not assignable(vt, dt):
            self.report(f"'{node.name}' declared as {dt}, but value is {vt}", node)
        self.define(node.name, node, dt)

    def _assign_target_type(self, tgt: Expr, vt: Any, op: str, node: Node):
        if isinstance(tgt, Name):
            name = tgt.id
            sym = self.lookup(name)
            if sym is None:
                if op != "=":
                    self.report(f"'{name}' used with '{op}' before being declared", node)
                else:
                    self.define(name, node, vt, implicit=True)
                return
            if sym.is_const:
                self.report(f"'{name}' is a constant and cannot be reassigned", node)
            elif op == "=":
                if not assignable(vt, sym.type):
                    self.report(f"'{name}' is {sym.type} and cannot be assigned {vt}", node)
            else:
                rt = self.binop(op[:-1], sym.type, vt, node)
                if not assignable(rt, sym.type):
                    self.report(f"'{name}' is {sym.type}, but '{op}' produces {rt}", node)
            return
        if isinstance(tgt, Index):
            elem = self.expr(tgt)
            rt = vt if op == "=" else self.binop(op[:-1], elem, vt, node)
            if not assignable(rt, elem):
                self.report(f"an element of type {elem} cannot be assigned {rt}", node)
            return
        if isinstance(tgt, Attr):
            ot = self.expr(tgt.obj)
            if isinstance(ot, StructT):
                fdict = ot.field_dict()
                if tgt.name not in fdict:
                    self.report(f"struct '{ot.name}' has no field '{tgt.name}'", node)
                    return
                ft = fdict[tgt.name]
                rt = vt if op == "=" else self.binop(op[:-1], ft, vt, node)
                if not assignable(rt, ft):
                    self.report(f"field '{tgt.name}' is {ft} and cannot be assigned {rt}", node)
                return
            if isinstance(ot, TensorT) and tgt.name in ("requires_grad", "grad", "data", "scale"):
                return
            if ot != ANY:
                self.report(f"cannot assign to attribute '{tgt.name}' of {ot}", node)
            return
        self.report("invalid assignment target", node)

    def visit_Assign(self, node: Assign):
        vt = self.expr(node.value)
        self._assign_target_type(node.target, vt, node.op, node)

    def visit_TupleAssign(self, node: TupleAssign):
        vt = self.expr(node.value)
        n_tgts = len(node.targets)
        if isinstance(vt, TupleT):
            if len(vt.elems) != n_tgts:
                self.report(
                    f"unpacking expects {n_tgts} values, but tuple has {len(vt.elems)}",
                    node,
                )
                elem_types = [ANY] * n_tgts
            else:
                elem_types = list(vt.elems)
        elif isinstance(vt, (ListT, TensorT)):
            elem_types = [vt.elem] * n_tgts
        elif vt == ANY:
            elem_types = [ANY] * n_tgts
        else:
            self.report(f"cannot unpack a value of type {vt}", node)
            elem_types = [ANY] * n_tgts

        for tgt, et in zip(node.targets, elem_types):
            self._assign_target_type(tgt, et, "=", node)

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
            self.report(f"cannot iterate over {it}", node.iter)
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
                self.report(f"this function must return {want}", node)
            return
        vt = self.expr(node.value)
        if want == VOID:
            self.report("a function without '-> type' cannot return a value", node)
        elif not assignable(vt, want):
            self.report(f"expected return type {want}, got {vt}", node)

    def visit_Break(self, node):
        if self.loop_depth == 0:
            self.report("'break' outside of a loop", node)

    def visit_Continue(self, node):
        if self.loop_depth == 0:
            self.report("'continue' outside of a loop", node)

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
            self.report(f"'{node.id}' is a module; access a member (e.g., {node.id}.sqrt)", node)
        elif node.id in self.functions:
            self.report(f"'{node.id}' is a function; call it with parentheses", node)
        else:
            self.report(self.undefined_msg(node.id), node)
        return ANY

    def visit_Attr(self, node: Attr):
        o = node.obj
        if isinstance(o, Name) and not self.lookup(o.id) and (o.id in KNOWN_MODULES or o.id in self.user_modules):
            if o.id not in self.modules:
                self.report(f"module '{o.id}' is not imported (use 'import {o.id}')", node)
            elif o.id in self.user_modules:
                umod = self.user_modules[o.id]
                if node.name in umod["consts"]:
                    return umod["consts"][node.name]
                if node.name in umod["functions"]:
                    self.report(f"'{o.id}.{node.name}' is a function; call it with parentheses", node)
                else:
                    self.report(f"module '{o.id}' has no member '{node.name}'", node)
            elif node.name in MATH_CONSTS:
                return MATH_CONSTS[node.name]
            elif node.name in MATH_FUNCS:
                self.report(f"'{o.id}.{node.name}' is a function; call it with parentheses", node)
            else:
                self.report(f"module '{o.id}' has no member '{node.name}'", node)
            return ANY
        ot = self.expr(o)
        if isinstance(ot, StructT):
            fdict = ot.field_dict()
            if node.name not in fdict:
                self.report(f"struct '{ot.name}' has no field '{node.name}'", node)
                return ANY
            return fdict[node.name]
        if isinstance(ot, TensorT):
            if node.name == "grad":
                return ot
            if node.name == "requires_grad":
                return BOOL
        return ANY

    def visit_FString(self, node: FString):
        for p in node.parts:
            if isinstance(p, FormatPart):
                self.expr(p.expr)
        return STR

    def visit_TupleLit(self, node: TupleLit):
        return TupleT(tuple(self.expr(it) for it in node.items))

    def visit_DictLit(self, node: DictLit):
        if not node.pairs:
            return DictT(ANY, ANY)
        kt = self.expr(node.pairs[0][0])
        vt = self.expr(node.pairs[0][1])
        for k_e, v_e in node.pairs[1:]:
            uk = unify(kt, self.expr(k_e))
            uv = unify(vt, self.expr(v_e))
            kt = uk if uk is not None else ANY
            vt = uv if uv is not None else ANY
        return DictT(kt, vt)

    def visit_ListLit(self, node: ListLit):
        if not node.items:
            return ListT(ANY)
        t = self.expr(node.items[0])
        for it in node.items[1:]:
            u = self.expr(it)
            n = unify(t, u)
            if n is None:
                self.report(f"list elements have incompatible types: {t} and {u}", it)
            else:
                t = n
        return ListT(t)

    def visit_ListComp(self, node: ListComp):
        it = self.expr(node.iter)
        et = it.elem if isinstance(it, (ListT, TensorT)) else ANY
        if it != ANY and not isinstance(it, (ListT, TensorT)):
            self.report(f"cannot iterate over {it}", node.iter)
        self.enter_scope()
        self.define(node.var, node, et)
        if node.cond is not None:
            self.cond(node.cond)
        elt_t = self.expr(node.elt)
        self.exit_scope()
        return ListT(elt_t)

    def visit_Lambda(self, node: Lambda):
        self.enter_scope()
        pts = []
        for p in node.params:
            pt = self.resolve_type(p.type) if p.type is not None else ANY
            pts.append(pt)
            self.define(p.name, p, pt)
        body_t = self.expr(node.body)
        self.exit_scope()
        if node.ret is not None:
            rt = self.resolve_type(node.ret)
            if not assignable(body_t, rt):
                self.report(f"expected return type {rt}, got {body_t}", node.body)
        else:
            rt = body_t
        return FuncSig(pts, rt, min_args=len(pts), max_args=len(pts))

    def visit_Range(self, node: Range):
        for b in (node.start, node.end):
            if b is not None:
                t = self.expr(b)
                if t != I64 and t != ANY and not (is_num(t) and is_int(t)):
                    self.report(f"range bounds must be i64, got {t}", b)
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
                self.report(f"a value of type {ot} cannot be sliced", node)
            return ANY
        it = self.expr(node.index)
        if isinstance(ot, DictT):
            if ot.key != ANY and not assignable(it, ot.key):
                self.report(f"dictionary key must be {ot.key}, got {it}", node.index)
            return ot.val
        if isinstance(ot, TensorT) and isinstance(it, (ListT, TensorT)) and (it.elem == ANY or (is_num(it.elem) and is_int(it.elem))):
            return ot
        if it != I64 and it != ANY and not (is_num(it) and is_int(it)):
            self.report(f"index must be i64, got {it}", node.index)
        if isinstance(ot, TupleT):
            if isinstance(node.index, Int):
                idx_v = node.index.value
                if -len(ot.elems) <= idx_v < len(ot.elems):
                    return ot.elems[idx_v]
                self.report(f"index {idx_v} out of bounds for tuple {ot}", node)
            return ANY
        if isinstance(ot, ListT):
            return ot.elem
        if isinstance(ot, TensorT):
            return ANY
        if ot == STR:
            return STR
        if ot != ANY:
            self.report(f"a value of type {ot} cannot be indexed", node)
        return ANY

    def visit_TryUnwrap(self, node: TryUnwrap):
        t = self.expr(node.expr)
        if isinstance(t, ResultT):
            return t.elem
        if t != ANY:
            self.report(f"the '?' operator requires a Result[T], got {t}", node)
        return ANY

    def visit_Unary(self, node: Unary):
        t = self.expr(node.operand)
        if node.op == "not":
            if t != BOOL and t != ANY:
                self.report(f"'not' requires bool, got {t}", node)
            return BOOL
        ok = t == ANY or is_num(t) or t in (FIX256, INTERVAL256) or (isinstance(t, (ListT, TensorT)) and base_num(t) is not None)
        if not ok:
            self.report(f"operator '{node.op}' is not defined for {t}", node)
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
                    self.report(f"'{op}' requires bool, got {t}", node)
            return BOOL
        if op in ("==", "!="):
            if l in (COMPLEX, C64) and r in (COMPLEX, C64):
                return BOOL
            if l in (FIX256, INTERVAL256) and r in (FIX256, INTERVAL256):
                return BOOL
            if not (same(l, r) or (is_num(l) and is_num(r))):
                self.report(f"cannot compare {l} with {r}", node)
            return BOOL
        if op in ("<", "<=", ">", ">="):
            ok = (l == ANY or r == ANY or (is_num(l) and is_num(r)) or (l == STR and r == STR) or (l == FIX256 and r == FIX256))
            if not ok:
                self.report(f"cannot compare {l} with {r}", node)
            return BOOL
        return self.arith(op, l, r, node)

    def arith(self, op, l, r, node):
        if l == ANY or r == ANY:
            return ANY
        if (l in (COMPLEX, C64) or r in (COMPLEX, C64)) and op in ("+", "-", "*", "/"):
            return COMPLEX
        if (l == FIX256 or r == FIX256) and op in ("+", "-", "*", "/"):
            if l in (FIX256, I64, F64) and r in (FIX256, I64, F64):
                return FIX256
        if (l == INTERVAL256 or r == INTERVAL256) and op in ("+", "-", "*", "/"):
            if l in (INTERVAL256, FIX256, I64, F64) and r in (INTERVAL256, FIX256, I64, F64):
                return INTERVAL256
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

        # Matrix multiplication with @ operator (supports ternary and continuous)
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

                # Resulting element type
                if F64 in (nl, nr):
                    res_elem = F64
                elif F32 in (nl, nr):
                    res_elem = F32
                elif is_float(nl) or is_float(nr):
                    res_elem = F32
                elif nl == ANY or nr == ANY:
                    res_elem = ANY if (nl == ANY and nr == ANY) else (nl if nr == ANY else nr)
                else:
                    res_elem = I64

                wrap = TensorT if (isinstance(l, TensorT) or isinstance(r, TensorT)) else ListT
                if dl == 1 and dr == 1:
                    if isinstance(l, TensorT) or isinstance(r, TensorT):
                        return wrap(res_elem)
                    return res_elem
                elif dl == 2 and dr == 1:
                    return wrap(res_elem)
                elif dl == 1 and dr == 2:
                    return wrap(res_elem)
                elif dl == 2 and dr == 2:
                    return wrap(wrap(res_elem))
                return wrap(wrap(res_elem))
            self.report(f"operator '@' is not defined for {l} and {r}", node)
            return ANY

        ll, rl = isinstance(l, (ListT, TensorT)), isinstance(r, (ListT, TensorT))
        numlike = lambda t: is_num(t) or t == ANY or (isinstance(t, (ListT, TensorT)) and base_num(t) is not None)
        if (ll or rl) and numlike(l) and numlike(r):          # broadcasting
            wrap = TensorT if (isinstance(l, TensorT) or isinstance(r, TensorT)) else ListT
            if ll and rl:
                return wrap(self.arith(op, l.elem, r.elem, node))
            if ll:
                return wrap(self.arith(op, l.elem, r, node))
            return wrap(self.arith(op, l, r.elem, node))

        self.report(f"operator '{op}' is not defined for {l} and {r}", node)
        return ANY

    def visit_Call(self, node: Call):
        f, sig, label = node.func, None, ""
        if isinstance(f, Name):
            sym = self.lookup(f.id)
            if sym:
                sym.used = True
                if isinstance(sym.type, FuncSig):
                    sig, label = sym.type, f.id
                elif sym.type == ANY:
                    sig, label = None, f.id
                else:
                    self.report(f"'{f.id}' is a variable, not a function", f)
            elif f.id in self.functions:
                sig, label = self.functions[f.id], f.id
            else:
                self.report(self.undefined_msg(f.id), f)
        elif isinstance(f, Lambda):
            sig, label = self.visit_Lambda(f), "lambda"
        elif isinstance(f, Attr) and isinstance(f.obj, Name) and not self.lookup(f.obj.id):
            mod = f.obj.id
            if mod not in KNOWN_MODULES and mod not in self.user_modules:
                self.report(f"'{mod}' is not a module", f)
            elif mod not in self.modules:
                self.report(f"module '{mod}' is not imported (use 'import {mod}')", f)
            elif mod in self.user_modules:
                umod = self.user_modules[mod]
                if f.name in umod["functions"]:
                    sig, label = umod["functions"][f.name], f"{mod}.{f.name}"
                else:
                    self.report(f"module '{mod}' has no member '{f.name}'", f)
            elif f.name in MATH_FUNCS:
                sig, label = BUILTINS[f.name], f"{mod}.{f.name}"
            else:
                self.report(f"module '{mod}' has no member '{f.name}'", f)
        elif isinstance(f, Attr):
            self.expr(f.obj)
            for a in node.args:
                self.expr(a)
            return ANY
        else:
            self.report("functions can only be called by name", f)

        ats = [self.expr(a) for a in node.args]
        if sig is None:
            return ANY

        # Uniform and generic arity check
        min_a = sig.min_args if sig.min_args is not None else (0 if sig.variadic else len(sig.params))
        max_a = sig.max_args if sig.max_args is not None else (None if sig.variadic else len(sig.params))

        if len(ats) < min_a or (max_a is not None and len(ats) > max_a):
            if min_a == max_a:
                self.report(f"'{label}' expects {min_a} argument(s), got {len(ats)}", node)
            elif max_a is None:
                self.report(f"'{label}' expects at least {min_a} argument(s), got {len(ats)}", node)
            else:
                self.report(f"'{label}' expects between {min_a} and {max_a} arguments, got {len(ats)}", node)

        # Custom validation if declared in the signature
        if sig.arg_validator:
            sig.arg_validator(self, ats, node.args, node)

        # Argument type check
        if sig.variadic:
            if sig.params and sig.params[0] != ANY:
                for i, (a, t) in enumerate(zip(node.args, ats)):
                    if not assignable(t, sig.params[0]):
                        self.report(f"argument {i + 1} of '{label}': expected {sig.params[0]}, got {t}", a)
        else:
            for i, (a, t, p) in enumerate(zip(node.args, ats, sig.params)):
                if p != ANY and not assignable(t, p):
                    self.report(f"argument {i + 1} of '{label}': expected {p}, got {t}", a)

        # Return type resolution (dynamic polymorphic or static)
        if sig.ret_builder:
            return sig.ret_builder(ats)
        return sig.ret


def analyze(program: Program) -> list[Diagnostic]:
    return SemanticAnalyzer().analyze(program)


# ============================================================================
# EXECUTION ENGINE / RUNTIME (NumPy SIMD + Ternary Add-Only Backend)
# ============================================================================

class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

class ErrPropagate(Exception):
    """Signal raised by '?' when a Result is Err.
    Caught by call_function: if the function declares a Result return type, the Err is returned
    to the caller (Rust-style propagation); otherwise it turns into a controlled panic."""
    def __init__(self, result: Any):
        self.result = result

def is_result_type_expr(t: Any) -> bool:
    """True if the AST type annotation is Result / Result[T] / NatResult."""
    if isinstance(t, TypeResult):
        return True
    return isinstance(t, TypeName) and t.name in ("Result", "NatResult")

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

    @staticmethod
    def Complex(*args, **kwargs):
        return complex_fn(*args, **kwargs)

    @staticmethod
    def rfft(*args, **kwargs):
        return rfft_fn(*args, **kwargs)

    @staticmethod
    def irfft(*args, **kwargs):
        return irfft_fn(*args, **kwargs)

    @staticmethod
    def power_spectrum(*args, **kwargs):
        return power_spectrum_fn(*args, **kwargs)

    @staticmethod
    def Fix256(*args, **kwargs):
        return fix256_fn(*args, **kwargs)

    @staticmethod
    def fix256(*args, **kwargs):
        return fix256_fn(*args, **kwargs)

    @staticmethod
    def Interval256(*args, **kwargs):
        return interval256_fn(*args, **kwargs)

    @staticmethod
    def interval256(*args, **kwargs):
        return interval256_fn(*args, **kwargs)


CURRENT_DEVICE = "cpu"

def device_fn(dev: Any = None) -> str:
    """Queries or sets the execution device of the Nativa runtime.
    Architectural note (v0.4): The execution pipeline operates natively on the CPU backend
    accelerated by OpenMP/AOT C99. The 'cuda' and 'opencl' tags operate as graph placement
    metadata for future compatibility with GPU backends.
    """
    global CURRENT_DEVICE
    if dev is not None:
        valid = ("cpu", "cuda", "aot", "opencl")
        dev_str = str(dev).lower().strip("\"'")
        if dev_str in valid:
            if dev_str in ("cuda", "opencl") and CURRENT_DEVICE != dev_str:
                sys.stderr.write(
                    f"[Nativa v0.4 Warning]: Device '{dev_str}' registered as graph metadata. "
                    f"The v0.4 runtime executes tensors on the optimized CPU backend (C99 AOT / OpenMP). "
                    f"Native hardware acceleration via '{dev_str}' is reserved for future versions.\n"
                )
            CURRENT_DEVICE = dev_str
        return dev_str
    return CURRENT_DEVICE


NATIVA_RT_HEADER_C = r"""/* ========================================================================= */
/* Unified C99 Runtime for the Nativa Language (nativa_rt.h)                 */
/* Shared between AOTKernelEngine (ctypes DLL) and CTranspiler (nativa)      */
/* ========================================================================= */
#ifndef NATIVA_RT_H
#define NATIVA_RT_H

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NAT_API
#ifdef NATIVA_BUILD_DLL
#ifdef _WIN32
#define NAT_API __declspec(dllexport)
#else
#define NAT_API __attribute__((visibility("default")))
#endif
#else
#define NAT_API static inline
#endif
#endif

#ifndef NAT_PI
#define NAT_PI 3.14159265358979323846
#endif
#ifndef NAT_E
#define NAT_E  2.71828182845904523536
#endif

/* ------------------------------------------------------------------------- */
/* 1. Float32 / BitNet b1.58 AOT Kernels (used via FFI and C99)              */
/* ------------------------------------------------------------------------- */

NAT_API void nat_c_gemv_ternary(const int8_t* W, const float* x, float* y, int64_t M, int64_t K) {
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

NAT_API void nat_c_gemv_veclut_5trits(const uint8_t* W_packed, const float* x, float* y, int64_t M, int64_t K, float gamma) {
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

NAT_API void nat_c_pack_5trits_2d(const int8_t* in, uint8_t* out_bytes, int64_t M, int64_t K) {
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

NAT_API void nat_c_rmsnorm(const float* x, const float* gamma, float* out, int64_t N, float eps) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)N + eps);
    float inv_rms = 1.0f / rms;
    for (int64_t i = 0; i < N; i++) out[i] = (x[i] * inv_rms) * gamma[i];
}

NAT_API void nat_c_silu(const float* x, float* out, int64_t N) {
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 0; i < N; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

NAT_API void nat_c_softmax(const float* x, float* out, int64_t N) {
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

/* ------------------------------------------------------------------------- */
/* 2. Complex Numbers and O(N log N) Cooley-Tukey Fast Fourier Transform     */
/* ------------------------------------------------------------------------- */

typedef struct {
    double real;
    double imag;
} NatComplex;

static inline NatComplex nat_complex(double re, double im) {
    NatComplex c; c.real = re; c.imag = im; return c;
}
static inline NatComplex nat_complex_add(NatComplex a, NatComplex b) {
    return nat_complex(a.real + b.real, a.imag + b.imag);
}
static inline NatComplex nat_complex_sub(NatComplex a, NatComplex b) {
    return nat_complex(a.real - b.real, a.imag - b.imag);
}
static inline NatComplex nat_complex_mul(NatComplex a, NatComplex b) {
    return nat_complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}
static inline double nat_complex_abs(NatComplex c) {
    return sqrt(c.real * c.real + c.imag * c.imag);
}
static inline double nat_complex_power(NatComplex c) {
    return c.real * c.real + c.imag * c.imag;
}

/* In-place O(N log N) Radix-2 Cooley-Tukey algorithm for powers of 2 */
static inline void nat_fft_cooley_tukey(NatComplex* a, int64_t N, bool inverse) {
    for (int64_t i = 1, j = 0; i < N; i++) {
        int64_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            NatComplex tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    for (int64_t len = 2; len <= N; len <<= 1) {
        double ang = 2.0 * NAT_PI / (double)len * (inverse ? 1.0 : -1.0);
        int64_t half = len >> 1;
        for (int64_t i = 0; i < N; i += len) {
            for (int64_t j = 0; j < half; j++) {
                double theta = ang * (double)j;
                NatComplex w = nat_complex(cos(theta), sin(theta));
                NatComplex u = a[i + j];
                NatComplex v = nat_complex_mul(a[i + j + half], w);
                a[i + j] = nat_complex_add(u, v);
                a[i + j + half] = nat_complex_sub(u, v);
            }
        }
    }
    if (inverse) {
        double inv_n = 1.0 / (double)N;
        for (int64_t i = 0; i < N; i++) {
            a[i].real *= inv_n;
            a[i].imag *= inv_n;
        }
    }
}

static inline void nat_rfft(const double* x, double* out_spec, int64_t N) {
    if (N <= 0) return;
    int64_t M = N / 2 + 1;
    if ((N & (N - 1)) == 0) {
        NatComplex* buf = (NatComplex*)malloc((size_t)N * sizeof(NatComplex));
        if (buf) {
            for (int64_t i = 0; i < N; i++) buf[i] = nat_complex(x[i], 0.0);
            nat_fft_cooley_tukey(buf, N, false);
            for (int64_t k = 0; k < M; k++) {
                out_spec[k * 2]     = buf[k].real;
                out_spec[k * 2 + 1] = buf[k].imag;
            }
            free(buf);
            return;
        }
    }
    for (int64_t k = 0; k < M; k++) {
        double re = 0.0, im = 0.0;
        for (int64_t n = 0; n < N; n++) {
            double angle = 2.0 * NAT_PI * (double)k * (double)n / (double)N;
            re += x[n] * cos(angle);
            im -= x[n] * sin(angle);
        }
        out_spec[k * 2]     = re;
        out_spec[k * 2 + 1] = im;
    }
}

static inline void nat_irfft(const double* spec, double* out_x, int64_t M, int64_t N) {
    if (N <= 0 || M <= 0) return;
    if ((N & (N - 1)) == 0) {
        NatComplex* buf = (NatComplex*)calloc((size_t)N, sizeof(NatComplex));
        if (buf) {
            for (int64_t k = 0; k < M && k < N; k++) {
                buf[k] = nat_complex(spec[k * 2], spec[k * 2 + 1]);
            }
            int64_t max_k = (N - 1) / 2;
            for (int64_t k = 1; k <= max_k && k < M; k++) {
                buf[N - k] = nat_complex(spec[k * 2], -spec[k * 2 + 1]);
            }
            nat_fft_cooley_tukey(buf, N, true);
            for (int64_t n = 0; n < N; n++) out_x[n] = buf[n].real;
            free(buf);
            return;
        }
    }
    double inv_N = 1.0 / (double)N;
    int64_t max_k = (N - 1) / 2;
    for (int64_t n = 0; n < N; n++) {
        double val = spec[0];
        if (N % 2 == 0 && (N / 2) < M) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            val += spec[(N / 2) * 2] * sign;
        }
        for (int64_t k = 1; k <= max_k && k < M; k++) {
            double angle = 2.0 * NAT_PI * (double)k * (double)n / (double)N;
            double re_k = spec[k * 2];
            double im_k = spec[k * 2 + 1];
            val += 2.0 * (re_k * cos(angle) - im_k * sin(angle));
        }
        out_x[n] = val * inv_N;
    }
}

static inline void nat_power_spectrum_spec(const double* spec, double* out_ps, int64_t M) {
    for (int64_t k = 0; k < M; k++) {
        double re = spec[k * 2];
        double im = spec[k * 2 + 1];
        out_ps[k] = re * re + im * im;
    }
}

static inline void nat_power_spectrum_signal(const double* x, double* out_ps, int64_t N) {
    if (N <= 0) return;
    int64_t M = N / 2 + 1;
    double* spec = (double*)malloc((size_t)(M * 2) * sizeof(double));
    if (spec) {
        nat_rfft(x, spec, N);
        nat_power_spectrum_spec(spec, out_ps, M);
        free(spec);
        return;
    }
    for (int64_t k = 0; k < M; k++) {
        double re = 0.0, im = 0.0;
        for (int64_t n = 0; n < N; n++) {
            double angle = 2.0 * NAT_PI * (double)k * (double)n / (double)N;
            re += x[n] * cos(angle);
            im -= x[n] * sin(angle);
        }
        out_ps[k] = re * re + im * im;
    }
}

/* ------------------------------------------------------------------------- */
/* 3. Ternary Quantization, BitNet b1.58 and F64 Math Primitives             */
/* ------------------------------------------------------------------------- */

static inline int8_t* nat_quantize_ternary(const double* v, int64_t length, double threshold) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        if (v[i] > threshold) out[i] = 1;
        else if (v[i] < -threshold) out[i] = -1;
        else out[i] = 0;
    }
    return out;
}
static inline uint8_t* nat_pack_ternary(const int8_t* in, int64_t length) {
    int64_t num_bytes = (length + 3) / 4;
    uint8_t* out = (uint8_t*)calloc(num_bytes, sizeof(uint8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        uint8_t enc = (in[i] == 1) ? 1 : ((in[i] == -1) ? 2 : 0);
        int64_t b = i / 4;
        int shift = (i % 4) * 2;
        out[b] |= (enc << shift);
    }
    return out;
}
static inline int8_t* nat_unpack_ternary(const uint8_t* in, int64_t length) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    static const int8_t map[4] = {0, 1, -1, 0};
    for (int64_t i = 0; i < length; i++) {
        int64_t b = i / 4;
        int shift = (i % 4) * 2;
        uint8_t code = (in[b] >> shift) & 0x03;
        out[i] = map[code];
    }
    return out;
}

static inline double nat_ln(double x) { return log(x); }
static inline double nat_relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double nat_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double nat_silu(double x) { return x / (1.0 + exp(-x)); }

static inline void nat_rmsnorm_f64(const double* x, const double* gamma, double* out, int64_t N, double eps) {
    if (N <= 0) return;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
    double inv_rms = 1.0 / sqrt(sum_sq / (double)N + eps);
    for (int64_t i = 0; i < N; i++) out[i] = (x[i] * inv_rms) * (gamma ? gamma[i] : 1.0);
}

static inline void nat_silu_vec_f64(const double* x, double* out, int64_t N) {
    for (int64_t i = 0; i < N; i++) out[i] = x[i] / (1.0 + exp(-x[i]));
}

static inline void nat_relu_vec_f64(const double* x, double* out, int64_t N) {
    for (int64_t i = 0; i < N; i++) out[i] = x[i] > 0.0 ? x[i] : 0.0;
}

static inline void nat_sigmoid_vec_f64(const double* x, double* out, int64_t N) {
    for (int64_t i = 0; i < N; i++) out[i] = 1.0 / (1.0 + exp(-x[i]));
}

static inline void nat_softmax_f64(const double* x, double* out, int64_t N) {
    if (N <= 0) return;
    double max_val = x[0];
    for (int64_t i = 1; i < N; i++) if (x[i] > max_val) max_val = x[i];
    double sum = 0.0;
    for (int64_t i = 0; i < N; i++) {
        out[i] = exp(x[i] - max_val);
        sum += out[i];
    }
    double inv = sum > 0.0 ? (1.0 / sum) : 0.0;
    for (int64_t i = 0; i < N; i++) out[i] *= inv;
}

static inline void nat_apply_rope_f64(const double* x, double* out, int64_t N, int64_t pos, double base) {
    int64_t half = N / 2;
    for (int64_t i = 0; i < half; i++) {
        double freq = 1.0 / pow(base, (double)i / (double)half);
        double angle = (double)pos * freq;
        double c = cos(angle);
        double s = sin(angle);
        double x1 = x[i];
        double x2 = x[i + half];
        out[i] = x1 * c - x2 * s;
        out[i + half] = x1 * s + x2 * c;
    }
    if (N % 2 != 0) out[N - 1] = x[N - 1];
}

static inline int64_t nat_argmax_f64(const double* x, int64_t N) {
    if (N <= 0) return 0;
    int64_t best_idx = 0;
    double best_val = x[0];
    for (int64_t i = 1; i < N; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static inline double* nat_dup_f64(const double* src, int64_t n) {
    double* dst = (double*)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (dst && src && n > 0) memcpy(dst, src, (size_t)n * sizeof(double));
    return dst;
}

static inline void nat_ternary_matmul_vec(const int8_t* W, const double* x, double* y, int64_t M, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc;
    }
}

static inline void nat_pack_5trits(const int8_t* in, uint8_t* out_bytes, int64_t length) {
    int64_t out_idx = 0;
    for (int64_t i = 0; i < length; i += 5) {
        uint16_t b_val = 0;
        uint16_t mul = 1;
        for (int j = 0; j < 5; j++) {
            int8_t t = (i + j < length) ? (in[i + j] + 1) : 1;
            b_val += (uint16_t)(t * mul);
            mul *= 3;
        }
        out_bytes[out_idx++] = (uint8_t)b_val;
    }
}

static inline void nat_unpack_5trits(const uint8_t* in_bytes, int8_t* out, int64_t length) {
    int64_t out_idx = 0;
    int64_t num_bytes = (length + 4) / 5;
    for (int64_t b = 0; b < num_bytes && out_idx < length; b++) {
        uint8_t val = in_bytes[b];
        for (int step = 0; step < 5 && out_idx < length; step++) {
            out[out_idx++] = (int8_t)(val % 3) - 1;
            val /= 3;
        }
    }
}

static inline void nat_fused_gemv_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {
    int64_t bytes_per_row = (K + 4) / 5;
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const uint8_t* row_packed = W_packed + i * bytes_per_row;
        int64_t k = 0;
        for (int64_t b = 0; b < bytes_per_row && k < K; b++) {
            uint8_t val = row_packed[b];
            for (int step = 0; step < 5 && k < K; step++) {
                int8_t trit = (int8_t)(val % 3) - 1;
                val /= 3;
                if (trit == 1) acc += x[k];
                else if (trit == -1) acc -= x[k];
                k++;
            }
        }
        y[i] = acc * gamma;
    }
}

static inline void nat_gemv_vec_lut_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {
    int64_t num_groups = (K + 4) / 5;
    if (num_groups <= 0) return;
    double* lut = (double*)malloc(num_groups * 21 * sizeof(double));
    if (!lut) return;
    double* lut01 = lut;
    double* lut23 = lut + num_groups * 9;
    double* lut4  = lut + num_groups * 18;
    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 5;
        double x0 = (base < K) ? x[base] : 0.0;
        double x1 = (base + 1 < K) ? x[base + 1] : 0.0;
        double x2 = (base + 2 < K) ? x[base + 2] : 0.0;
        double x3 = (base + 3 < K) ? x[base + 3] : 0.0;
        double x4 = (base + 4 < K) ? x[base + 4] : 0.0;
        for (int t1 = 0; t1 < 3; t1++) {
            for (int t0 = 0; t0 < 3; t0++) {
                lut01[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x0 + (t1 - 1) * x1;
                lut23[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x2 + (t1 - 1) * x3;
            }
        }
        for (int t4 = 0; t4 < 3; t4++) {
            lut4[g * 3 + t4] = (t4 - 1) * x4;
        }
    }
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const uint8_t* row = W_packed + i * num_groups;
        for (int64_t g = 0; g < num_groups; g++) {
            uint8_t b = row[g];
            uint8_t idx01 = b % 9;
            uint8_t idx23 = (b / 9) % 9;
            uint8_t idx4  = b / 81;
            acc += lut01[g * 9 + idx01] + lut23[g * 9 + idx23] + lut4[g * 3 + idx4];
        }
        y[i] = acc * gamma;
    }
    free(lut);
}

static inline uint8_t* nat_pack_ternary_5trits(const int8_t* in, int64_t length) {
    int64_t num_bytes = (length + 4) / 5;
    uint8_t* out = (uint8_t*)malloc(num_bytes * sizeof(uint8_t));
    if (out) nat_pack_5trits(in, out, length);
    return out;
}
static inline int8_t* nat_unpack_ternary_5trits(const uint8_t* in, int64_t length) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (out) nat_unpack_5trits(in, out, length);
    return out;
}
static inline double* nat_gemv_vec_lut(const uint8_t* W_packed, const double* x, int64_t M, int64_t K) {
    double* y = (double*)malloc(M * sizeof(double));
    if (y) nat_gemv_vec_lut_5trits(W_packed, x, y, M, K, 1.0);
    return y;
}
static inline double nat_tensor_absmean(const double* W, int64_t length) {
    if (length <= 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < length; i++) s += fabs(W[i]);
    return s / (double)length;
}
static inline int8_t* nat_quantize_bitnet(const double* W, int64_t length) {
    double gamma = nat_tensor_absmean(W, length);
    double inv = 1.0 / (gamma + 1e-7);
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        double sc = W[i] * inv;
        double r = round(sc);
        if (r > 1.0) r = 1.0; else if (r < -1.0) r = -1.0;
        out[i] = (int8_t)r;
    }
    return out;
}
static inline double nat_mae(const double* a, const double* b, int64_t length) {
    if (length <= 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < length; i++) s += fabs(a[i] - b[i]);
    return s / (double)length;
}
static inline double nat_relative_error(const double* a, const double* b, int64_t length) {
    double norm_diff = 0.0, norm_a = 0.0;
    for (int64_t i = 0; i < length; i++) {
        double d = a[i] - b[i];
        norm_diff += d * d; norm_a += a[i] * a[i];
    }
    return norm_a > 0.0 ? sqrt(norm_diff / norm_a) : 0.0;
}
static inline double nat_pearson_corr(const double* a, const double* b, int64_t length) {
    if (length < 2) return 1.0;
    double ma = 0.0, mb = 0.0;
    for (int64_t i = 0; i < length; i++) { ma += a[i]; mb += b[i]; }
    ma /= (double)length; mb /= (double)length;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int64_t i = 0; i < length; i++) {
        double diff_a = a[i] - ma; double diff_b = b[i] - mb;
        num += diff_a * diff_b; da += diff_a * diff_a; db += diff_b * diff_b;
    }
    double den = sqrt(da * db);
    return den > 0.0 ? (num / den) : 0.0;
}
static inline double nat_cosine_sim(const double* a, const double* b, int64_t length) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < length; i++) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    double den = sqrt(na) * sqrt(nb);
    return den > 0.0 ? (dot / den) : 0.0;
}

static inline double* nat_fill_f64(double* arr, int64_t n, double val) {
    for (int64_t i = 0; i < n; i++) arr[i] = val; return arr;
}
static inline int64_t* nat_fill_i64(int64_t* arr, int64_t n, int64_t val) {
    for (int64_t i = 0; i < n; i++) arr[i] = val; return arr;
}
static inline double* nat_zeros_f64(double* arr, int64_t n) {
    memset(arr, 0, n * sizeof(double)); return arr;
}
static inline double* nat_ones_f64(double* arr, int64_t n) {
    for (int64_t i = 0; i < n; i++) arr[i] = 1.0; return arr;
}
static inline double nat_dot(const double* a, const double* b, int64_t n) {
    double acc = 0.0; for (int64_t i = 0; i < n; i++) acc += a[i] * b[i]; return acc;
}
static inline double nat_sum(const double* a, int64_t n) {
    double acc = 0.0; for (int64_t i = 0; i < n; i++) acc += a[i]; return acc;
}
static inline double nat_mean(const double* a, int64_t n) {
    return n > 0 ? (nat_sum(a, n) / (double)n) : 0.0;
}
static inline double nat_min(const double* a, int64_t n) {
    if (n <= 0) return 0.0;
    double m = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
static inline double nat_max(const double* a, int64_t n) {
    if (n <= 0) return 0.0;
    double m = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}

static inline void nat_gemv_f64(const double* A, const double* x, double* y, int64_t M, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const double* row = A + i * K;
        for (int64_t k = 0; k < K; k++) acc += row[k] * x[k];
        y[i] = acc;
    }
}
static inline void nat_gemv_scaled_f64(const double* A, const double* x, double* y, int64_t M, int64_t K, double scale) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const double* row = A + i * K;
        for (int64_t k = 0; k < K; k++) acc += row[k] * x[k];
        y[i] = acc * scale;
    }
}
static inline void nat_ternary_matmul_vec_scaled(const int8_t* W, const double* x, double* y, int64_t M, int64_t K, double scale) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc * scale;
    }
}
static inline void nat_gemv_trans_f64(const double* A, const double* x, double* y, int64_t M, int64_t N) {
    for (int64_t j = 0; j < N; j++) {
        double acc = 0.0;
        for (int64_t i = 0; i < M; i++) acc += x[i] * A[i * N + j];
        y[j] = acc;
    }
}
static inline void nat_gemm_f64(const double* A, const double* B, double* C, int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}
static inline void nat_random_ternary(int8_t* out, int64_t length, uint64_t seed) {
    uint64_t state = (seed != 0) ? seed : 123456789ULL;
    for (int64_t i = 0; i < length; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int r = (int)((state >> 32) % 3);
        out[i] = (int8_t)(r - 1);
    }
}

/* ------------------------------------------------------------------------- */
/* 4. Contiguous C99 KVCache for Autoregressive LLM Inference                */
/* ------------------------------------------------------------------------- */

typedef struct {
    double* k;
    double* v;
    int64_t max_seq_len;
    int64_t dim;
    int64_t seq_len;
} NatKVCache;

static inline NatKVCache nat_kvcache_init(int64_t max_seq_len, int64_t dim) {
    NatKVCache c;
    c.max_seq_len = max_seq_len > 0 ? max_seq_len : 1;
    c.dim = dim > 0 ? dim : 1;
    c.seq_len = 0;
    c.k = (double*)calloc((size_t)(c.max_seq_len * c.dim), sizeof(double));
    c.v = (double*)calloc((size_t)(c.max_seq_len * c.dim), sizeof(double));
    return c;
}

static inline void nat_kvcache_update(NatKVCache* c, const double* k_new, const double* v_new, int64_t n_tokens) {
    if (!c || !c->k || !c->v) return;
    if (n_tokens <= 0) n_tokens = 1;
    if (c->seq_len + n_tokens <= c->max_seq_len) {
        memcpy(c->k + c->seq_len * c->dim, k_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        memcpy(c->v + c->seq_len * c->dim, v_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        c->seq_len += n_tokens;
    } else if (n_tokens >= c->max_seq_len) {
        int64_t offset = (n_tokens - c->max_seq_len) * c->dim;
        memcpy(c->k, k_new + offset, (size_t)(c->max_seq_len * c->dim) * sizeof(double));
        memcpy(c->v, v_new + offset, (size_t)(c->max_seq_len * c->dim) * sizeof(double));
        c->seq_len = c->max_seq_len;
    } else {
        int64_t shift = c->seq_len + n_tokens - c->max_seq_len;
        int64_t keep = c->seq_len - shift;
        memmove(c->k, c->k + shift * c->dim, (size_t)(keep * c->dim) * sizeof(double));
        memmove(c->v, c->v + shift * c->dim, (size_t)(keep * c->dim) * sizeof(double));
        memcpy(c->k + keep * c->dim, k_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        memcpy(c->v + keep * c->dim, v_new, (size_t)(n_tokens * c->dim) * sizeof(double));
        c->seq_len = c->max_seq_len;
    }
}

static inline void nat_kvcache_clear(NatKVCache* c) {
    if (c) c->seq_len = 0;
}

/* ------------------------------------------------------------------------- */
/* 5. Formatted Printing, Result[T] System and Safe C99 I/O                  */
/* ------------------------------------------------------------------------- */

static inline void nat_print_vec_f64(const double* v, int64_t n) {
    printf("[");
    for (int64_t i = 0; i < n; i++) printf("%g%s", v[i], (i < n - 1) ? ", " : "");
    printf("]");
}
static inline void nat_print_mat_f64(const double* A, int64_t M, int64_t N) {
    printf("[");
    for (int64_t i = 0; i < M; i++) {
        printf("[");
        for (int64_t j = 0; j < N; j++) printf("%g%s", A[i * N + j], (j < N - 1) ? ", " : "");
        printf("]%s", (i < M - 1) ? ", " : "");
    }
    printf("]");
}
static inline void nat_print_vec_i64(const int64_t* v, int64_t n) {
    printf("[");
    for (int64_t i = 0; i < n; i++) printf("%lld%s", (long long)v[i], (i < n - 1) ? ", " : "");
    printf("]");
}
static inline void nat_print_mat_i64(const int64_t* A, int64_t M, int64_t N) {
    printf("[");
    for (int64_t i = 0; i < M; i++) {
        printf("[");
        for (int64_t j = 0; j < N; j++) printf("%lld%s", (long long)A[i * N + j], (j < N - 1) ? ", " : "");
        printf("]%s", (i < M - 1) ? ", " : "");
    }
    printf("]");
}

typedef struct {
    const char* val_str;
    double val_f64;
    int64_t val_i64;
    const char* err;
    bool is_ok;
} NatResult;

static inline NatResult nat_ok_str(const char* val) {
    NatResult r; memset(&r, 0, sizeof(r)); r.val_str = val; r.is_ok = true; return r;
}
static inline NatResult nat_ok_i64(int64_t val) {
    NatResult r; memset(&r, 0, sizeof(r)); r.val_i64 = val; r.val_f64 = (double)val; r.is_ok = true; return r;
}
static inline NatResult nat_ok_f64(double val) {
    NatResult r; memset(&r, 0, sizeof(r)); r.val_f64 = val; r.val_i64 = (int64_t)val; r.is_ok = true; return r;
}
static inline NatResult nat_err(const char* err_msg) {
    NatResult r; memset(&r, 0, sizeof(r)); r.err = err_msg; r.is_ok = false; return r;
}
static inline bool nat_is_ok(NatResult r) { return r.is_ok; }
static inline bool nat_is_err(NatResult r) { return !r.is_ok; }
static inline const char* nat_unwrap_str(NatResult r) {
    if (!r.is_ok) {
        fprintf(stderr, "Runtime error: attempted to unwrap a Result containing an error: %s\n", r.err ? r.err : "unknown");
        exit(1);
    }
    return r.val_str ? r.val_str : "";
}
static inline int64_t nat_unwrap_i64(NatResult r) {
    if (!r.is_ok) {
        fprintf(stderr, "Runtime error: attempted to unwrap a Result containing an error: %s\n", r.err ? r.err : "unknown");
        exit(1);
    }
    return r.val_i64;
}
static inline double nat_unwrap_f64(NatResult r) {
    if (!r.is_ok) {
        fprintf(stderr, "Runtime error: attempted to unwrap a Result containing an error: %s\n", r.err ? r.err : "unknown");
        exit(1);
    }
    return r.val_f64;
}
static inline const char* nat_unwrap_or_str(NatResult r, const char* def) {
    return r.is_ok ? (r.val_str ? r.val_str : "") : def;
}
static inline int64_t nat_unwrap_or_i64(NatResult r, int64_t def) {
    return r.is_ok ? r.val_i64 : def;
}
static inline double nat_unwrap_or_f64(NatResult r, double def) {
    return r.is_ok ? r.val_f64 : def;
}
static inline const char* nat_error_message(NatResult r) {
    return (r.is_ok || !r.err) ? "" : r.err;
}
static inline bool nat_exists_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}
static inline NatResult nat_read_file_safe(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nat_err("file not found or inaccessible");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return nat_err("memory allocation failed for reading"); }
    size_t rd = fread(buf, 1, sz, f);
    buf[rd] = '\0';
    fclose(f);
    return nat_ok_str(buf);
}
static inline const char* nat_read_file(const char* path) {
    NatResult r = nat_read_file_safe(path);
    return nat_unwrap_str(r);
}

/* ------------------------------------------------------------------------- */
/* 6. 256-Bit Fixed-Point and Interval Arithmetic (High Altitude Steps)      */
/* ------------------------------------------------------------------------- */

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

#ifndef NAT_FIX256_DEFAULT_FRAC
#define NAT_FIX256_DEFAULT_FRAC 64
#endif

typedef struct {
    uint64_t limbs[4]; /* limbs[0]=least significant 64 bits, limbs[3]=most significant */
    int32_t frac_bits; /* Default 64: Q192.64 */
    int8_t sign;       /* +1 for >= 0, -1 for < 0 */
} nat_fix256_t;

typedef struct {
    nat_fix256_t lo;
    nat_fix256_t hi;
} nat_interval256_t;

static inline uint64_t nat_add_carry(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t* carry_out) {
    uint64_t s = a + carry_in;
    uint64_t c1 = (s < a) ? 1ULL : 0ULL;
    uint64_t res = s + b;
    uint64_t c2 = (res < s) ? 1ULL : 0ULL;
    *carry_out = c1 + c2;
    return res;
}

static inline uint64_t nat_sub_borrow(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t* borrow_out) {
    uint64_t s = a - borrow_in;
    uint64_t b1 = (a < borrow_in) ? 1ULL : 0ULL;
    uint64_t res = s - b;
    uint64_t b2 = (s < b) ? 1ULL : 0ULL;
    *borrow_out = b1 + b2;
    return res;
}

static inline uint64_t nat_mul64(uint64_t u, uint64_t v, uint64_t* hi) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return _umul128(u, v, hi);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 prod = (unsigned __int128)u * (unsigned __int128)v;
    *hi = (uint64_t)(prod >> 64);
    return (uint64_t)prod;
#else
    uint64_t u0 = u & 0xFFFFFFFFULL, u1 = u >> 32;
    uint64_t v0 = v & 0xFFFFFFFFULL, v1 = v >> 32;
    uint64_t p00 = u0 * v0;
    uint64_t p01 = u0 * v1;
    uint64_t p10 = u1 * v0;
    uint64_t p11 = u1 * v1;
    uint64_t mid = p01 + (p00 >> 32);
    uint64_t mid_c = (mid < p01) ? 1ULL : 0ULL;
    mid += p10;
    if (mid < p10) mid_c++;
    *hi = p11 + (mid_c << 32) + (mid >> 32);
    return (mid << 32) | (p00 & 0xFFFFFFFFULL);
#endif
}

static inline bool nat_u256_is_zero(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0ULL;
}

static inline int nat_u256_cmp(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

static inline uint64_t nat_u256_add(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t c = 0;
        out[i] = nat_add_carry(a[i], b[i], carry, &c);
        carry = c;
    }
    return carry;
}

static inline uint64_t nat_u256_sub(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t br = 0;
        out[i] = nat_sub_borrow(a[i], b[i], borrow, &br);
        borrow = br;
    }
    return borrow;
}

static inline void nat_u256_shift_left_1(uint64_t x[4]) {
    uint64_t c0 = x[0] >> 63;
    uint64_t c1 = x[1] >> 63;
    uint64_t c2 = x[2] >> 63;
    x[3] = (x[3] << 1) | c2;
    x[2] = (x[2] << 1) | c1;
    x[1] = (x[1] << 1) | c0;
    x[0] = (x[0] << 1);
}

static inline void nat_u256_shift_left(const uint64_t in[4], int shift, uint64_t out[4]) {
    if (shift <= 0) {
        memcpy(out, in, 4 * sizeof(uint64_t));
        return;
    }
    if (shift >= 256) {
        memset(out, 0, 4 * sizeof(uint64_t));
        return;
    }
    int w = shift / 64;
    int b = shift % 64;
    uint64_t tmp[4] = {0};
    for (int i = 3; i >= 0; i--) {
        int src = i - w;
        if (src >= 0) {
            uint64_t low = in[src];
            uint64_t prev = (src > 0) ? in[src - 1] : 0ULL;
            if (b == 0) {
                tmp[i] = low;
            } else {
                tmp[i] = (low << b) | (prev >> (64 - b));
            }
        }
    }
    memcpy(out, tmp, 4 * sizeof(uint64_t));
}

static inline void nat_u256_shift_right(const uint64_t in[4], int shift, uint64_t out[4]) {
    if (shift <= 0) {
        memcpy(out, in, 4 * sizeof(uint64_t));
        return;
    }
    if (shift >= 256) {
        memset(out, 0, 4 * sizeof(uint64_t));
        return;
    }
    int w = shift / 64;
    int b = shift % 64;
    uint64_t tmp[4] = {0};
    for (int i = 0; i < 4; i++) {
        int src = i + w;
        if (src < 4) {
            uint64_t low = in[src];
            uint64_t high = (src + 1 < 4) ? in[src + 1] : 0ULL;
            if (b == 0) {
                tmp[i] = low;
            } else {
                tmp[i] = (low >> b) | (high << (64 - b));
            }
        }
    }
    memcpy(out, tmp, 4 * sizeof(uint64_t));
}

static inline uint64_t nat_u256_mul_64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t hi = 0;
        uint64_t lo = nat_mul64(a[i], b, &hi);
        uint64_t c1 = 0;
        uint64_t s = nat_add_carry(lo, carry, 0, &c1);
        out[i] = s;
        carry = hi + c1;
    }
    return carry;
}

static inline uint64_t nat_u256_add_64(uint64_t a[4], uint64_t b) {
    uint64_t carry = b;
    for (int i = 0; i < 4 && carry > 0; i++) {
        uint64_t c = 0;
        a[i] = nat_add_carry(a[i], carry, 0, &c);
        carry = c;
    }
    return carry;
}

static inline void nat_u256_mul_full(const uint64_t a[4], const uint64_t b[4], uint64_t prod[8]) {
    memset(prod, 0, 8 * sizeof(uint64_t));
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi = 0;
            uint64_t lo = nat_mul64(a[i], b[j], &hi);
            uint64_t c1 = 0, c2 = 0;
            uint64_t s1 = nat_add_carry(prod[i + j], lo, 0, &c1);
            uint64_t s2 = nat_add_carry(s1, carry, 0, &c2);
            prod[i + j] = s2;
            carry = hi + c1 + c2;
        }
        for (int k = i + 4; k < 8 && carry > 0; k++) {
            uint64_t c = 0;
            prod[k] = nat_add_carry(prod[k], carry, 0, &c);
            carry = c;
        }
    }
}

static inline void nat_u512_shift_right(const uint64_t in[8], int shift_bits, uint64_t out[4]) {
    if (shift_bits <= 0) {
        memcpy(out, in, 4 * sizeof(uint64_t));
        return;
    }
    int w = shift_bits / 64;
    int b = shift_bits % 64;
    for (int i = 0; i < 4; i++) {
        int idx = i + w;
        uint64_t low = (idx < 8) ? in[idx] : 0ULL;
        uint64_t high = (idx + 1 < 8) ? in[idx + 1] : 0ULL;
        if (b == 0) {
            out[i] = low;
        } else {
            out[i] = (low >> b) | (high << (64 - b));
        }
    }
}

static inline void nat_u256_divmod(const uint64_t num[4], const uint64_t den[4], uint64_t quot[4], uint64_t rem[4]) {
    memset(quot, 0, 4 * sizeof(uint64_t));
    memset(rem, 0, 4 * sizeof(uint64_t));
    if (nat_u256_is_zero(den)) return;
    for (int i = 255; i >= 0; i--) {
        nat_u256_shift_left_1(rem);
        int bit = (int)((num[i / 64] >> (i % 64)) & 1ULL);
        rem[0] |= (uint64_t)bit;
        if (nat_u256_cmp(rem, den) >= 0) {
            nat_u256_sub(rem, den, rem);
            quot[i / 64] |= (1ULL << (i % 64));
        }
    }
}

static inline void nat_u256_shift_left_to_512(const uint64_t in[4], int shift, uint64_t out[8]) {
    memset(out, 0, 8 * sizeof(uint64_t));
    if (shift < 0 || shift >= 512) return;
    int w = shift / 64;
    int b = shift % 64;
    for (int i = 0; i < 4; i++) {
        int idx = i + w;
        if (idx < 8) {
            if (b == 0) {
                out[idx] |= in[i];
            } else {
                out[idx] |= (in[i] << b);
                if (idx + 1 < 8) {
                    out[idx + 1] |= (in[i] >> (64 - b));
                }
            }
        }
    }
}

static inline void nat_u512_div_256(const uint64_t num[8], const uint64_t den[4], uint64_t quot[4]) {
    memset(quot, 0, 4 * sizeof(uint64_t));
    if (nat_u256_is_zero(den)) return;
    int top = 7;
    while (top >= 0 && num[top] == 0) top--;
    if (top < 0) return;
    int top_bit = top * 64 + 63;
    while (top_bit >= 0 && !((num[top_bit / 64] >> (top_bit % 64)) & 1ULL)) {
        top_bit--;
    }
    uint64_t rem[4] = {0};
    for (int i = top_bit; i >= 0; i--) {
        nat_u256_shift_left_1(rem);
        int bit = (int)((num[i / 64] >> (i % 64)) & 1ULL);
        rem[0] |= (uint64_t)bit;
        if (nat_u256_cmp(rem, den) >= 0) {
            nat_u256_sub(rem, den, rem);
            if (i < 256) {
                quot[i / 64] |= (1ULL << (i % 64));
            }
        }
    }
}

static inline nat_fix256_t nat_fix256_zero(void) {
    nat_fix256_t r;
    memset(&r, 0, sizeof(r));
    r.frac_bits = NAT_FIX256_DEFAULT_FRAC;
    r.sign = 1;
    return r;
}

static inline nat_fix256_t nat_fix256_from_parts(int8_t sign, uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3, int32_t frac_bits) {
    nat_fix256_t r;
    r.limbs[0] = l0; r.limbs[1] = l1; r.limbs[2] = l2; r.limbs[3] = l3;
    r.frac_bits = frac_bits > 0 ? frac_bits : NAT_FIX256_DEFAULT_FRAC;
    r.sign = (sign >= 0) ? 1 : -1;
    if (nat_u256_is_zero(r.limbs)) r.sign = 1;
    return r;
}

static inline nat_fix256_t nat_fix256_from_i64(int64_t val, int32_t frac_bits) {
    nat_fix256_t r = nat_fix256_zero();
    if (frac_bits >= 0) r.frac_bits = frac_bits;
    if (val < 0) {
        r.sign = -1;
        val = -val;
    } else {
        r.sign = 1;
    }
    uint64_t uval = (uint64_t)val;
    uint64_t raw[4] = {uval, 0, 0, 0};
    nat_u256_shift_left(raw, r.frac_bits, r.limbs);
    return r;
}

static inline nat_fix256_t nat_fix256_from_f64(double val, int32_t frac_bits) {
    nat_fix256_t r = nat_fix256_zero();
    if (frac_bits >= 0) r.frac_bits = frac_bits;
    if (val < 0.0) {
        r.sign = -1;
        val = -val;
    } else {
        r.sign = 1;
    }
    double scale = ldexp(1.0, r.frac_bits);
    double scaled = val * scale;
    double int_part = 0.0;
    double frac_part = modf(scaled, &int_part);
    if (frac_part >= 0.5) int_part += 1.0;
    double cur = int_part;
    for (int i = 0; i < 4; i++) {
        double rem = fmod(cur, 18446744073709551616.0);
        r.limbs[i] = (uint64_t)rem;
        cur = floor(cur / 18446744073709551616.0);
    }
    return r;
}

static inline nat_fix256_t nat_fix256_from_str(const char* s, int32_t frac_bits) {
    nat_fix256_t r = nat_fix256_zero();
    if (frac_bits >= 0) r.frac_bits = frac_bits;
    if (!s) return r;
    while (*s == ' ' || *s == '\t') s++;
    int8_t sgn = 1;
    if (*s == '-') { sgn = -1; s++; }
    else if (*s == '+') { s++; }

    uint64_t int_val[4] = {0};
    while (*s >= '0' && *s <= '9') {
        nat_u256_mul_64(int_val, 10, int_val);
        nat_u256_add_64(int_val, (uint64_t)(*s - '0'));
        s++;
    }

    uint64_t frac_val[4] = {0};
    uint64_t frac_den[4] = {1, 0, 0, 0};
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            nat_u256_mul_64(frac_val, 10, frac_val);
            nat_u256_add_64(frac_val, (uint64_t)(*s - '0'));
            nat_u256_mul_64(frac_den, 10, frac_den);
            s++;
        }
    }

    nat_u256_shift_left(int_val, r.frac_bits, r.limbs);

    if (!nat_u256_is_zero(frac_val)) {
        uint64_t f_scaled[4] = {0};
        nat_u256_shift_left(frac_val, r.frac_bits, f_scaled);
        uint64_t f_part[4] = {0}, rem[4] = {0};
        nat_u256_divmod(f_scaled, frac_den, f_part, rem);
        nat_u256_add(r.limbs, f_part, r.limbs);
    }

    if (*s == 'e' || *s == 'E') {
        s++;
        int exp_sgn = 1;
        if (*s == '-') { exp_sgn = -1; s++; }
        else if (*s == '+') { s++; }
        int exp_val = 0;
        while (*s >= '0' && *s <= '9') {
            exp_val = exp_val * 10 + (*s - '0');
            s++;
        }
        if (exp_sgn > 0) {
            for (int k = 0; k < exp_val; k++) {
                nat_u256_mul_64(r.limbs, 10, r.limbs);
            }
        } else {
            uint64_t ten[4] = {10, 0, 0, 0};
            uint64_t q[4] = {0}, rem[4] = {0};
            for (int k = 0; k < exp_val; k++) {
                nat_u256_divmod(r.limbs, ten, q, rem);
                memcpy(r.limbs, q, 4 * sizeof(uint64_t));
            }
        }
    }

    r.sign = sgn;
    if (nat_u256_is_zero(r.limbs)) r.sign = 1;
    return r;
}

static inline bool nat_fix256_is_zero(nat_fix256_t a) {
    return nat_u256_is_zero(a.limbs);
}

static inline int nat_fix256_cmp(nat_fix256_t a, nat_fix256_t b) {
    if (nat_fix256_is_zero(a) && nat_fix256_is_zero(b)) return 0;
    if (a.sign != b.sign) return (a.sign > b.sign) ? 1 : -1;
    int c = nat_u256_cmp(a.limbs, b.limbs);
    return (a.sign > 0) ? c : -c;
}

static inline bool nat_fix256_eq(nat_fix256_t a, nat_fix256_t b) { return nat_fix256_cmp(a, b) == 0; }
static inline bool nat_fix256_lt(nat_fix256_t a, nat_fix256_t b) { return nat_fix256_cmp(a, b) < 0; }
static inline bool nat_fix256_le(nat_fix256_t a, nat_fix256_t b) { return nat_fix256_cmp(a, b) <= 0; }
static inline bool nat_fix256_gt(nat_fix256_t a, nat_fix256_t b) { return nat_fix256_cmp(a, b) > 0; }
static inline bool nat_fix256_ge(nat_fix256_t a, nat_fix256_t b) { return nat_fix256_cmp(a, b) >= 0; }

static inline nat_fix256_t nat_fix256_neg(nat_fix256_t a) {
    nat_fix256_t r = a;
    if (!nat_fix256_is_zero(r)) r.sign = -r.sign;
    return r;
}

static inline nat_fix256_t nat_fix256_abs(nat_fix256_t a) {
    nat_fix256_t r = a;
    r.sign = 1;
    return r;
}

static inline nat_fix256_t nat_fix256_add(nat_fix256_t a, nat_fix256_t b) {
    nat_fix256_t r = nat_fix256_zero();
    r.frac_bits = a.frac_bits;
    if (a.sign == b.sign) {
        r.sign = a.sign;
        nat_u256_add(a.limbs, b.limbs, r.limbs);
    } else {
        int c = nat_u256_cmp(a.limbs, b.limbs);
        if (c >= 0) {
            r.sign = a.sign;
            nat_u256_sub(a.limbs, b.limbs, r.limbs);
        } else {
            r.sign = b.sign;
            nat_u256_sub(b.limbs, a.limbs, r.limbs);
        }
    }
    if (nat_u256_is_zero(r.limbs)) r.sign = 1;
    return r;
}

static inline nat_fix256_t nat_fix256_sub(nat_fix256_t a, nat_fix256_t b) {
    return nat_fix256_add(a, nat_fix256_neg(b));
}

static inline nat_fix256_t nat_fix256_mul(nat_fix256_t a, nat_fix256_t b) {
    nat_fix256_t r = nat_fix256_zero();
    r.frac_bits = a.frac_bits;
    r.sign = a.sign * b.sign;
    uint64_t prod[8] = {0};
    nat_u256_mul_full(a.limbs, b.limbs, prod);
    nat_u512_shift_right(prod, r.frac_bits, r.limbs);
    if (nat_u256_is_zero(r.limbs)) r.sign = 1;
    return r;
}

static inline nat_fix256_t nat_fix256_div(nat_fix256_t a, nat_fix256_t b) {
    nat_fix256_t r = nat_fix256_zero();
    r.frac_bits = a.frac_bits;
    if (nat_fix256_is_zero(b)) return r;
    r.sign = a.sign * b.sign;
    uint64_t num[8];
    nat_u256_shift_left_to_512(a.limbs, r.frac_bits, num);
    nat_u512_div_256(num, b.limbs, r.limbs);
    if (nat_u256_is_zero(r.limbs)) r.sign = 1;
    return r;
}

static inline int64_t nat_fix256_to_i64(nat_fix256_t a) {
    uint64_t int_part[4] = {0};
    nat_u256_shift_right(a.limbs, a.frac_bits, int_part);
    int64_t val = (int64_t)int_part[0];
    return (a.sign < 0) ? -val : val;
}

static inline double nat_fix256_to_f64(nat_fix256_t a) {
    double scale = ldexp(1.0, -a.frac_bits);
    double total = 0.0;
    double limb_scale = 1.0;
    for (int i = 0; i < 4; i++) {
        total += (double)a.limbs[i] * limb_scale;
        limb_scale *= 18446744073709551616.0;
    }
    double res = total * scale;
    return (a.sign < 0) ? -res : res;
}

static inline void nat_fix256_to_str(nat_fix256_t a, char* buf, size_t buf_size) {
    if (!buf || buf_size < 4) return;
    if (nat_fix256_is_zero(a)) {
        snprintf(buf, buf_size, "0.000000");
        return;
    }
    uint64_t int_part[4] = {0};
    nat_u256_shift_right(a.limbs, a.frac_bits, int_part);

    uint64_t frac_part[4] = {0};
    uint64_t mask[4] = {0};
    nat_u256_shift_left((const uint64_t[]){1, 0, 0, 0}, a.frac_bits, mask);
    nat_u256_sub(mask, (const uint64_t[]){1, 0, 0, 0}, mask);
    for (int i = 0; i < 4; i++) frac_part[i] = a.limbs[i] & mask[i];

    char digits[128];
    int digit_count = 0;
    uint64_t ten[4] = {10, 0, 0, 0};

    uint64_t cur[4];
    memcpy(cur, int_part, 4 * sizeof(uint64_t));
    if (nat_u256_is_zero(cur)) {
        digits[digit_count++] = '0';
    } else {
        while (!nat_u256_is_zero(cur) && digit_count < 120) {
            uint64_t q[4] = {0}, rem[4] = {0};
            nat_u256_divmod(cur, ten, q, rem);
            digits[digit_count++] = (char)('0' + rem[0]);
            memcpy(cur, q, 4 * sizeof(uint64_t));
        }
    }

    size_t out_idx = 0;
    if (a.sign < 0 && out_idx + 1 < buf_size) buf[out_idx++] = '-';
    for (int i = digit_count - 1; i >= 0 && out_idx + 1 < buf_size; i--) {
        buf[out_idx++] = digits[i];
    }
    if (out_idx + 1 < buf_size) buf[out_idx++] = '.';

    uint64_t cur_f[4];
    memcpy(cur_f, frac_part, 4 * sizeof(uint64_t));
    for (int step = 0; step < 6 && out_idx + 1 < buf_size; step++) {
        nat_u256_mul_64(cur_f, 10, cur_f);
        uint64_t d[4] = {0};
        nat_u256_shift_right(cur_f, a.frac_bits, d);
        buf[out_idx++] = (char)('0' + (d[0] % 10));
        for (int i = 0; i < 4; i++) cur_f[i] &= mask[i];
    }
    buf[out_idx] = '\0';
}

static inline nat_interval256_t nat_interval256_create(nat_fix256_t lo, nat_fix256_t hi) {
    nat_interval256_t iv;
    if (nat_fix256_gt(lo, hi)) {
        iv.lo = hi; iv.hi = lo;
    } else {
        iv.lo = lo; iv.hi = hi;
    }
    return iv;
}

static inline bool nat_interval256_contains(nat_interval256_t a, nat_fix256_t x) {
    return nat_fix256_ge(x, a.lo) && nat_fix256_le(x, a.hi);
}

static inline nat_interval256_t nat_interval256_point(nat_fix256_t val) {
    return nat_interval256_create(val, val);
}

static inline nat_interval256_t nat_interval256_from_val(nat_fix256_t val, nat_fix256_t radius) {
    return nat_interval256_create(nat_fix256_sub(val, radius), nat_fix256_add(val, radius));
}

static inline nat_interval256_t nat_interval256_from_str(const char* lo_str, const char* hi_str, int32_t frac_bits) {
    nat_fix256_t lo = nat_fix256_from_str(lo_str, frac_bits);
    nat_fix256_t hi = nat_fix256_from_str(hi_str, frac_bits);
    return nat_interval256_create(lo, hi);
}

static inline nat_interval256_t nat_interval256_add(nat_interval256_t a, nat_interval256_t b) {
    return nat_interval256_create(nat_fix256_add(a.lo, b.lo), nat_fix256_add(a.hi, b.hi));
}

static inline nat_interval256_t nat_interval256_sub(nat_interval256_t a, nat_interval256_t b) {
    return nat_interval256_create(nat_fix256_sub(a.lo, b.hi), nat_fix256_sub(a.hi, b.lo));
}

static inline nat_interval256_t nat_interval256_mul(nat_interval256_t a, nat_interval256_t b) {
    nat_fix256_t p1 = nat_fix256_mul(a.lo, b.lo);
    nat_fix256_t p2 = nat_fix256_mul(a.lo, b.hi);
    nat_fix256_t p3 = nat_fix256_mul(a.hi, b.lo);
    nat_fix256_t p4 = nat_fix256_mul(a.hi, b.hi);
    nat_fix256_t min_p = p1, max_p = p1;
    if (nat_fix256_lt(p2, min_p)) min_p = p2;
    if (nat_fix256_gt(p2, max_p)) max_p = p2;
    if (nat_fix256_lt(p3, min_p)) min_p = p3;
    if (nat_fix256_gt(p3, max_p)) max_p = p3;
    if (nat_fix256_lt(p4, min_p)) min_p = p4;
    if (nat_fix256_gt(p4, max_p)) max_p = p4;
    return nat_interval256_create(min_p, max_p);
}

static inline nat_interval256_t nat_interval256_mul_scalar(nat_interval256_t a, nat_fix256_t s) {
    if (nat_fix256_ge(s, nat_fix256_zero())) {
        return nat_interval256_create(nat_fix256_mul(a.lo, s), nat_fix256_mul(a.hi, s));
    } else {
        return nat_interval256_create(nat_fix256_mul(a.hi, s), nat_fix256_mul(a.lo, s));
    }
}

static inline nat_interval256_t nat_interval256_div_scalar(nat_interval256_t a, nat_fix256_t s) {
    if (nat_fix256_is_zero(s)) {
        return nat_interval256_create(nat_fix256_zero(), nat_fix256_zero());
    }
    if (nat_fix256_ge(s, nat_fix256_zero())) {
        return nat_interval256_create(nat_fix256_div(a.lo, s), nat_fix256_div(a.hi, s));
    } else {
        return nat_interval256_create(nat_fix256_div(a.hi, s), nat_fix256_div(a.lo, s));
    }
}

static inline nat_interval256_t nat_interval256_div(nat_interval256_t a, nat_interval256_t b) {
    if (nat_interval256_contains(b, nat_fix256_zero())) {
        return nat_interval256_create(nat_fix256_zero(), nat_fix256_zero());
    }
    nat_fix256_t p1 = nat_fix256_div(a.lo, b.lo);
    nat_fix256_t p2 = nat_fix256_div(a.lo, b.hi);
    nat_fix256_t p3 = nat_fix256_div(a.hi, b.lo);
    nat_fix256_t p4 = nat_fix256_div(a.hi, b.hi);
    nat_fix256_t min_p = p1, max_p = p1;
    if (nat_fix256_lt(p2, min_p)) min_p = p2;
    if (nat_fix256_gt(p2, max_p)) max_p = p2;
    if (nat_fix256_lt(p3, min_p)) min_p = p3;
    if (nat_fix256_gt(p3, max_p)) max_p = p3;
    if (nat_fix256_lt(p4, min_p)) min_p = p4;
    if (nat_fix256_gt(p4, max_p)) max_p = p4;
    return nat_interval256_create(min_p, max_p);
}

static inline nat_interval256_t nat_interval256_neg(nat_interval256_t a) {
    return nat_interval256_create(nat_fix256_neg(a.hi), nat_fix256_neg(a.lo));
}

static inline nat_fix256_t nat_interval256_width(nat_interval256_t a) {
    return nat_fix256_sub(a.hi, a.lo);
}

static inline nat_fix256_t nat_interval256_mid(nat_interval256_t a) {
    nat_fix256_t sum = nat_fix256_add(a.lo, a.hi);
    nat_fix256_t two = nat_fix256_from_i64(2, a.lo.frac_bits);
    return nat_fix256_div(sum, two);
}

static inline bool nat_interval256_is_coherent(nat_interval256_t a, nat_fix256_t max_width) {
    nat_fix256_t w = nat_interval256_width(a);
    return nat_fix256_le(w, max_width);
}

static inline nat_interval256_t nat_interval256_step(nat_interval256_t pos, nat_interval256_t vel, nat_fix256_t dt) {
    nat_interval256_t delta = nat_interval256_mul_scalar(vel, dt);
    return nat_interval256_add(pos, delta);
}

static inline nat_interval256_t nat_interval256_simulate_steps_val(nat_interval256_t pos, nat_interval256_t vel, nat_fix256_t dt, int64_t num_steps) {
    nat_interval256_t delta = nat_interval256_mul_scalar(vel, dt);
    for (int64_t i = 0; i < num_steps; i++) {
        pos = nat_interval256_add(pos, delta);
    }
    return pos;
}

static inline void nat_interval256_to_str(nat_interval256_t a, char* buf, size_t buf_size) {
    if (!buf || buf_size < 8) return;
    char b_lo[128], b_hi[128];
    nat_fix256_to_str(a.lo, b_lo, sizeof(b_lo));
    nat_fix256_to_str(a.hi, b_hi, sizeof(b_hi));
    snprintf(buf, buf_size, "[%s, %s]", b_lo, b_hi);
}

static inline const char* nat_fix256_as_str(nat_fix256_t a) {
    static char bufs[8][128];
    static int idx = 0;
    char* b = bufs[(idx++) & 7];
    nat_fix256_to_str(a, b, 128);
    return b;
}

static inline const char* nat_interval256_as_str(nat_interval256_t a) {
    static char bufs[8][256];
    static int idx = 0;
    char* b = bufs[(idx++) & 7];
    nat_interval256_to_str(a, b, 256);
    return b;
}

static inline void nat_print_fix256(nat_fix256_t a) {
    printf("%s", nat_fix256_as_str(a));
}

static inline void nat_print_interval256(nat_interval256_t a) {
    printf("%s", nat_interval256_as_str(a));
}

/* NAT_API exported C kernels for ctypes FFI */
NAT_API void nat_c_fix256_from_str(const char* s, int32_t frac_bits, nat_fix256_t* out) {
    if (out) *out = nat_fix256_from_str(s, frac_bits);
}

NAT_API void nat_c_fix256_to_str(const nat_fix256_t* a, char* buf, size_t buf_size) {
    if (a && buf) nat_fix256_to_str(*a, buf, buf_size);
}

NAT_API void nat_c_fix256_from_i64(int64_t val, int32_t frac_bits, nat_fix256_t* out) {
    if (out) *out = nat_fix256_from_i64(val, frac_bits);
}

NAT_API void nat_c_fix256_from_f64(double val, int32_t frac_bits, nat_fix256_t* out) {
    if (out) *out = nat_fix256_from_f64(val, frac_bits);
}

NAT_API void nat_c_fix256_to_f64(const nat_fix256_t* a, double* out) {
    if (a && out) *out = nat_fix256_to_f64(*a);
}

NAT_API void nat_c_fix256_add(const nat_fix256_t* a, const nat_fix256_t* b, nat_fix256_t* out) {
    if (a && b && out) *out = nat_fix256_add(*a, *b);
}

NAT_API void nat_c_fix256_sub(const nat_fix256_t* a, const nat_fix256_t* b, nat_fix256_t* out) {
    if (a && b && out) *out = nat_fix256_sub(*a, *b);
}

NAT_API void nat_c_fix256_mul(const nat_fix256_t* a, const nat_fix256_t* b, nat_fix256_t* out) {
    if (a && b && out) *out = nat_fix256_mul(*a, *b);
}

NAT_API void nat_c_fix256_div(const nat_fix256_t* a, const nat_fix256_t* b, nat_fix256_t* out) {
    if (a && b && out) *out = nat_fix256_div(*a, *b);
}

NAT_API int32_t nat_c_fix256_cmp(const nat_fix256_t* a, const nat_fix256_t* b) {
    if (!a || !b) return 0;
    return (int32_t)nat_fix256_cmp(*a, *b);
}

NAT_API void nat_c_interval256_create(const nat_fix256_t* lo, const nat_fix256_t* hi, nat_interval256_t* out) {
    if (lo && hi && out) *out = nat_interval256_create(*lo, *hi);
}

NAT_API void nat_c_interval256_add(const nat_interval256_t* a, const nat_interval256_t* b, nat_interval256_t* out) {
    if (a && b && out) *out = nat_interval256_add(*a, *b);
}

NAT_API void nat_c_interval256_sub(const nat_interval256_t* a, const nat_interval256_t* b, nat_interval256_t* out) {
    if (a && b && out) *out = nat_interval256_sub(*a, *b);
}

NAT_API void nat_c_interval256_mul(const nat_interval256_t* a, const nat_interval256_t* b, nat_interval256_t* out) {
    if (a && b && out) *out = nat_interval256_mul(*a, *b);
}

NAT_API void nat_c_interval256_mul_scalar(const nat_interval256_t* a, const nat_fix256_t* s, nat_interval256_t* out) {
    if (a && s && out) *out = nat_interval256_mul_scalar(*a, *s);
}

NAT_API void nat_c_interval256_step(const nat_interval256_t* pos, const nat_interval256_t* vel, const nat_fix256_t* dt, nat_interval256_t* out) {
    if (pos && vel && dt && out) *out = nat_interval256_step(*pos, *vel, *dt);
}

NAT_API void nat_c_interval256_width(const nat_interval256_t* a, nat_fix256_t* out) {
    if (a && out) *out = nat_interval256_width(*a);
}

NAT_API void nat_c_interval256_mid(const nat_interval256_t* a, nat_fix256_t* out) {
    if (a && out) *out = nat_interval256_mid(*a);
}

NAT_API bool nat_c_interval256_contains(const nat_interval256_t* a, const nat_fix256_t* x) {
    if (!a || !x) return false;
    return nat_interval256_contains(*a, *x);
}

NAT_API bool nat_c_interval256_is_coherent(const nat_interval256_t* a, const nat_fix256_t* max_width) {
    if (!a || !max_width) return false;
    return nat_interval256_is_coherent(*a, *max_width);
}

NAT_API void nat_c_interval256_to_str(const nat_interval256_t* a, char* buf, size_t buf_size) {
    if (a && buf) nat_interval256_to_str(*a, buf, buf_size);
}

NAT_API void nat_c_interval256_simulate_steps(nat_interval256_t* pos, const nat_interval256_t* vel, const nat_fix256_t* dt, int64_t num_steps) {
    if (!pos || !vel || !dt || num_steps <= 0) return;
    nat_interval256_t delta = nat_interval256_mul_scalar(*vel, *dt);
    for (int64_t i = 0; i < num_steps; i++) {
        *pos = nat_interval256_add(*pos, delta);
    }
}

NAT_API void nat_c_interval256_div(const nat_interval256_t* a, const nat_interval256_t* b, nat_interval256_t* out) {
    if (a && b && out) *out = nat_interval256_div(*a, *b);
}

NAT_API void nat_c_interval256_div_scalar(const nat_interval256_t* a, const nat_fix256_t* s, nat_interval256_t* out) {
    if (a && s && out) *out = nat_interval256_div_scalar(*a, *s);
}

NAT_API void nat_c_interval256_neg(const nat_interval256_t* a, nat_interval256_t* out) {
    if (a && out) *out = nat_interval256_neg(*a);
}

#endif /* NATIVA_RT_H */
"""

AOT_C_SOURCE = "#define NATIVA_BUILD_DLL\n" + NATIVA_RT_HEADER_C


def ensure_nativa_rt_header() -> str:
    """Ensures the unified C99 header 'nativa_rt.h' is synchronized on disk."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rt_path = os.path.join(base_dir, "nativa_rt.h")
    try:
        need_write = True
        if os.path.exists(rt_path):
            with open(rt_path, "r", encoding="utf-8") as rf:
                if rf.read() == NATIVA_RT_HEADER_C:
                    need_write = False
        if need_write:
            with open(rt_path, "w", encoding="utf-8") as wf:
                wf.write(NATIVA_RT_HEADER_C)
    except Exception:
        pass
    return rt_path


class C_NatFix256(ctypes.Structure):
    _fields_ = [
        ("limbs", ctypes.c_uint64 * 4),
        ("frac_bits", ctypes.c_int32),
        ("sign", ctypes.c_int8),
    ]


class C_NatInterval256(ctypes.Structure):
    _fields_ = [
        ("lo", C_NatFix256),
        ("hi", C_NatFix256),
    ]


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
        ensure_nativa_rt_header()
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
                        obj_path = os.path.join(cache_dir, "aot_kernels.obj")
                        with open(bat, "w", encoding="utf-8") as bf:
                            bf.write(f'@echo off\ncall "{vc}" >nul 2>&1\ncl.exe /LD /O2 /openmp /nologo "{c_src_path}" /Fe:"{dll_path}" /Fo:"{obj_path}"\n')
                        try:
                            proc = subprocess.run([bat], capture_output=True, text=True)
                            if proc.returncode == 0 and os.path.exists(dll_path):
                                compiled_ok = True
                                break
                        finally:
                            if os.path.exists(bat):
                                try:
                                    os.remove(bat)
                                except OSError:
                                    pass
                            for extra_ext in (".obj", ".exp", ".lib"):
                                extra = os.path.splitext(dll_path)[0] + extra_ext
                                if os.path.exists(extra):
                                    try:
                                        os.remove(extra)
                                    except OSError:
                                        pass
                            if os.path.exists(obj_path):
                                try:
                                    os.remove(obj_path)
                                except OSError:
                                    pass

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
                sys.stderr.write("[Nativa v0.4 Warning]: C99 AOT recompilation failed. The cache was not updated.\n")

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

        if hasattr(dll, "nat_c_fix256_from_str"):
            dll.nat_c_fix256_from_str.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p]
            dll.nat_c_fix256_from_str.restype = None

            dll.nat_c_fix256_to_str.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            dll.nat_c_fix256_to_str.restype = None

            dll.nat_c_fix256_from_i64.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p]
            dll.nat_c_fix256_from_i64.restype = None

            dll.nat_c_fix256_from_f64.argtypes = [ctypes.c_double, ctypes.c_int32, ctypes.c_void_p]
            dll.nat_c_fix256_from_f64.restype = None

            dll.nat_c_fix256_to_f64.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
            dll.nat_c_fix256_to_f64.restype = None

            dll.nat_c_fix256_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_fix256_add.restype = None

            dll.nat_c_fix256_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_fix256_sub.restype = None

            dll.nat_c_fix256_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_fix256_mul.restype = None

            dll.nat_c_fix256_div.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_fix256_div.restype = None

            dll.nat_c_fix256_cmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_fix256_cmp.restype = ctypes.c_int32

            dll.nat_c_interval256_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_create.restype = None

            dll.nat_c_interval256_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_add.restype = None

            dll.nat_c_interval256_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_sub.restype = None

            dll.nat_c_interval256_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_mul.restype = None

            dll.nat_c_interval256_mul_scalar.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_mul_scalar.restype = None

            dll.nat_c_interval256_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_step.restype = None

            dll.nat_c_interval256_width.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_width.restype = None

            dll.nat_c_interval256_mid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_mid.restype = None

            dll.nat_c_interval256_contains.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_contains.restype = ctypes.c_bool

            dll.nat_c_interval256_is_coherent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            dll.nat_c_interval256_is_coherent.restype = ctypes.c_bool

            dll.nat_c_interval256_to_str.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            dll.nat_c_interval256_to_str.restype = None

            dll.nat_c_interval256_simulate_steps.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            dll.nat_c_interval256_simulate_steps.restype = None

            if hasattr(dll, "nat_c_interval256_div"):
                dll.nat_c_interval256_div.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                dll.nat_c_interval256_div.restype = None

                dll.nat_c_interval256_div_scalar.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                dll.nat_c_interval256_div_scalar.restype = None

                dll.nat_c_interval256_neg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                dll.nat_c_interval256_neg.restype = None


@dataclass
class NatResult:
    value: Any = None
    error: Optional[str] = None
    is_ok: bool = True

    def __repr__(self) -> str:
        if self.is_ok:
            return f"Ok({self.value!r})"
        return f"Err({self.error!r})"


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grad)
    if target_shape == ():
        return np.asarray(g.sum(), dtype=g.dtype)
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)
    for axis, (g_dim, t_dim) in enumerate(zip(g.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(target_shape)


def _matmul_backward_arrays(g: np.ndarray, a_data: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a_data.ndim == 1 and b_data.ndim == 1:
        return g * b_data, g * a_data
    if a_data.ndim == 2 and b_data.ndim == 1:
        return np.outer(g, b_data), a_data.T @ g
    if a_data.ndim == 1 and b_data.ndim == 2:
        return b_data @ g, np.outer(a_data, g)
    if a_data.ndim == 2 and b_data.ndim == 2:
        return g @ b_data.T, a_data.T @ g
    return np.matmul(g, np.swapaxes(b_data, -1, -2)), np.matmul(np.swapaxes(a_data, -1, -2), g)


class Tensor:
    """First-class dense data structure in the Nativa runtime.
    Maintains contiguous memory with shape, dtype, scale, device, and Autodiff graph metadata.
    """
    def __init__(self, data: Any, dtype: str = "f32", device: Optional[str] = None, requires_grad: bool = False):
        self.device = device if device is not None else CURRENT_DEVICE
        self.scale = 1.0
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional["Tensor"] = None
        self._prev: tuple["Tensor", ...] = ()
        self._backward: Optional[Callable[[], None]] = None

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
            if not requires_grad:
                self.requires_grad = data.requires_grad
        elif isinstance(data, np.ndarray):
            self.data = data if data.dtype == np_dt else data.astype(np_dt)
        else:
            self.data = np.asarray(data, dtype=np_dt)

    def requires_grad_(self, flag: bool = True) -> "Tensor":
        self.requires_grad = bool(flag)
        return self

    def zero_grad(self) -> None:
        self.grad = None
        self._prev = ()
        self._backward = None

    def _accum_grad(self, g: Any) -> None:
        grad_dt = np.float64 if self.dtype == "f64" else np.float32
        g_arr = _unbroadcast(np.asarray(g, dtype=grad_dt), self.data.shape)
        if self.grad is None:
            self.grad = Tensor(g_arr.copy(), dtype="f64" if self.dtype == "f64" else "f32", device=self.device)
        else:
            self.grad.data = self.grad.data + g_arr.astype(self.grad.data.dtype)

    def backward(self, grad: Any = None) -> None:
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(v: Tensor) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        grad_dt = np.float64 if self.dtype == "f64" else np.float32
        if grad is None:
            init_g = np.ones_like(self.data, dtype=grad_dt)
        elif isinstance(grad, Tensor):
            init_g = grad.data.astype(grad_dt)
        else:
            init_g = np.asarray(grad, dtype=grad_dt)
        self._accum_grad(init_g)
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            res = self.data
        else:
            res = self.data.astype(dtype)
        if copy:
            return res.copy()
        return res

    def __float__(self) -> float:
        return float(self.data.item())

    def __int__(self) -> int:
        return int(self.data.item())

    def __abs__(self) -> Any:
        if self.data.ndim == 0 or self.data.size == 1:
            return abs(float(self.data.item()))
        return Tensor(np.abs(self.data), dtype=self.dtype, device=self.device)

    def __round__(self, n: Optional[int] = None) -> Any:
        return round(float(self.data.item()), n)

    def __lt__(self, other: Any) -> Any:
        val = other.data if isinstance(other, Tensor) else other
        res = self.data < val
        return bool(res.item()) if res.ndim == 0 or res.size == 1 else res

    def __le__(self, other: Any) -> Any:
        val = other.data if isinstance(other, Tensor) else other
        res = self.data <= val
        return bool(res.item()) if res.ndim == 0 or res.size == 1 else res

    def __gt__(self, other: Any) -> Any:
        val = other.data if isinstance(other, Tensor) else other
        res = self.data > val
        return bool(res.item()) if res.ndim == 0 or res.size == 1 else res

    def __ge__(self, other: Any) -> Any:
        val = other.data if isinstance(other, Tensor) else other
        res = self.data >= val
        return bool(res.item()) if res.ndim == 0 or res.size == 1 else res

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Tensor):
            return bool(np.array_equal(self.data, other.data))
        if self.data.ndim == 0 and isinstance(other, (int, float, np.number)):
            return bool(self.data.item() == other)
        return False

    def __format__(self, format_spec: str) -> str:
        if self.data.ndim == 0 or self.data.size == 1:
            return format(self.data.item(), format_spec)
        return format(str(self), format_spec)

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
        t = Tensor(self.data, dtype=self.dtype, device=device, requires_grad=self.requires_grad)
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
        if isinstance(res, np.ndarray) or self.requires_grad:
            t = Tensor(res, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
            t.scale = self.scale
            t._prev = (self,)
            src = self
            def _bw():
                if t.grad is None:
                    return
                grad_dt = np.float64 if src.dtype == "f64" else np.float32
                dx = np.zeros_like(src.data, dtype=grad_dt)
                np.add.at(dx, idx, t.grad.data.astype(grad_dt))
                src._accum_grad(dx)
            t._backward = _bw
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
                t = Tensor(item, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
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

        out_dtype = "f64" if (self.dtype == "f64" or getattr(other, "dtype", "") == "f64") else "f32"
        res = ternary_add_only_matmul(self, other)
        out = res if isinstance(res, Tensor) else Tensor(res, dtype=out_dtype, device=self.device)
        if out_dtype == "f64" and out.dtype != "f64":
            out = Tensor(np.matmul(self.data.astype(np.float64), (other.data if isinstance(other, Tensor) else np.asarray(other)).astype(np.float64)), dtype="f64", device=self.device)
        a_t = self
        b_t = other if isinstance(other, Tensor) else None
        a_data = self.data
        b_data = other.data if isinstance(other, Tensor) else np.asarray(other)
        out.requires_grad = self.requires_grad or (b_t.requires_grad if b_t is not None else False)
        out._prev = (a_t, b_t) if b_t is not None else (a_t,)
        def _bw():
            if out.grad is None:
                return
            work_dt = np.float64 if out_dtype == "f64" else np.float32
            da, db = _matmul_backward_arrays(out.grad.data.astype(work_dt), a_data.astype(work_dt), b_data.astype(work_dt))
            a_t._accum_grad(da)
            if b_t is not None:
                b_t._accum_grad(db)
        out._backward = _bw
        return out

    def __rmatmul__(self, other: Any) -> Any:
        lhs = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype, device=self.device)
        return lhs.__matmul__(self)

    def __neg__(self) -> "Tensor":
        t = Tensor(-self.data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        t._prev = (self,)
        src = self
        def _bw():
            if t.grad is not None:
                src._accum_grad(-t.grad.data)
        t._backward = _bw
        return t

    def __pos__(self) -> "Tensor":
        return self

    def __mul__(self, other: Any) -> "Tensor":
        b_t = other if isinstance(other, Tensor) else None
        val = b_t.data if b_t is not None else other
        out_dtype = "f64" if (self.dtype == "f64" or getattr(b_t, "dtype", "") == "f64") else self.dtype
        res = self.data * val
        t = Tensor(res, dtype=out_dtype, device=self.device, requires_grad=self.requires_grad or (b_t.requires_grad if b_t else False))
        t.scale = self.scale
        a_t = self
        a_data = self.data
        b_data = val
        t._prev = (a_t, b_t) if b_t is not None else (a_t,)
        def _bw():
            if t.grad is None:
                return
            g = t.grad.data
            a_t._accum_grad(g * b_data)
            if b_t is not None:
                b_t._accum_grad(g * a_data)
        t._backward = _bw
        return t

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __add__(self, other: Any) -> "Tensor":
        b_t = other if isinstance(other, Tensor) else None
        val = b_t.data if b_t is not None else other
        out_dtype = "f64" if (self.dtype == "f64" or getattr(b_t, "dtype", "") == "f64") else self.dtype
        res = self.data + val
        t = Tensor(res, dtype=out_dtype, device=self.device, requires_grad=self.requires_grad or (b_t.requires_grad if b_t else False))
        a_t = self
        t._prev = (a_t, b_t) if b_t is not None else (a_t,)
        def _bw():
            if t.grad is None:
                return
            g = t.grad.data
            a_t._accum_grad(g)
            if b_t is not None:
                b_t._accum_grad(g)
        t._backward = _bw
        return t

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        b_t = other if isinstance(other, Tensor) else None
        val = b_t.data if b_t is not None else other
        out_dtype = "f64" if (self.dtype == "f64" or getattr(b_t, "dtype", "") == "f64") else self.dtype
        res = self.data - val
        t = Tensor(res, dtype=out_dtype, device=self.device, requires_grad=self.requires_grad or (b_t.requires_grad if b_t else False))
        a_t = self
        t._prev = (a_t, b_t) if b_t is not None else (a_t,)
        def _bw():
            if t.grad is None:
                return
            g = t.grad.data
            a_t._accum_grad(g)
            if b_t is not None:
                b_t._accum_grad(-g)
        t._backward = _bw
        return t

    def __rsub__(self, other: Any) -> "Tensor":
        b_t = other if isinstance(other, Tensor) else None
        val = b_t.data if b_t is not None else other
        out_dtype = "f64" if (self.dtype == "f64" or getattr(b_t, "dtype", "") == "f64") else self.dtype
        res = val - self.data
        t = Tensor(res, dtype=out_dtype, device=self.device, requires_grad=self.requires_grad or (b_t.requires_grad if b_t else False))
        a_t = self
        t._prev = (b_t, a_t) if b_t is not None else (a_t,)
        def _bw():
            if t.grad is None:
                return
            g = t.grad.data
            if b_t is not None:
                b_t._accum_grad(g)
            a_t._accum_grad(-g)
        t._backward = _bw
        return t

    def __truediv__(self, other: Any) -> "Tensor":
        b_t = other if isinstance(other, Tensor) else None
        val = b_t.data if b_t is not None else other
        out_dtype = "f64" if (self.dtype == "f64" or getattr(b_t, "dtype", "") == "f64") else self.dtype
        res = self.data / val
        t = Tensor(res, dtype=out_dtype, device=self.device, requires_grad=self.requires_grad or (b_t.requires_grad if b_t else False))
        a_t = self
        a_data = self.data
        b_data = val
        t._prev = (a_t, b_t) if b_t is not None else (a_t,)
        def _bw():
            if t.grad is None:
                return
            g = t.grad.data
            a_t._accum_grad(g / b_data)
            if b_t is not None:
                b_t._accum_grad(-g * a_data / (b_data ** 2))
        t._backward = _bw
        return t

    def __rtruediv__(self, other: Any) -> "Tensor":
        b_t = other if isinstance(other, Tensor) else None
        val = b_t.data if b_t is not None else other
        out_dtype = "f64" if (self.dtype == "f64" or getattr(b_t, "dtype", "") == "f64") else self.dtype
        res = val / self.data
        t = Tensor(res, dtype=out_dtype, device=self.device, requires_grad=self.requires_grad or (b_t.requires_grad if b_t else False))
        a_t = self
        a_data = self.data
        b_data = val
        t._prev = (b_t, a_t) if b_t is not None else (a_t,)
        def _bw():
            if t.grad is None:
                return
            g = t.grad.data
            if b_t is not None:
                b_t._accum_grad(g / a_data)
            a_t._accum_grad(-g * b_data / (a_data ** 2))
        t._backward = _bw
        return t

    def __repr__(self) -> str:
        if self.data.ndim == 0:
            return repr(self.data.item())
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
    """Canonical BitNet b1.58 quantizer: Clip(Round(W / (gamma + eps)), -1, 1)."""
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
    """Dense base-3 packing: 5 trits per byte (3^5 = 243 <= 256), achieving 1.60 bits/trit."""
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
    """Dense base-3 unpacking: extracts trits {0, 1, 2} and maps to {-1, 0, +1}."""
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
    """Mean Absolute Error (MAE): 1/N * sum(|a - b|)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    return float(np.mean(np.abs(a_arr - b_arr)))


def relative_error_fn(a: Any, b: Any) -> float:
    """L2 Norm Relative Error: ||a - b|| / ||a||."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    norm_a = float(np.linalg.norm(a_arr))
    if norm_a == 0.0:
        return 0.0
    return float(np.linalg.norm(a_arr - b_arr) / norm_a)


def pearson_corr_fn(a: Any, b: Any) -> float:
    """Pearson Linear Correlation Coefficient between two tensors."""
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
    """Cosine Similarity between two vectors: (a . b) / (||a|| * ||b||)."""
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
    """Multiplication-free matrix multiplication for ternary weights {-1, 0, 1}."""
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
        a = a.reshape((-1, 2, h))
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
    has_tensor = isinstance(x, Tensor) or isinstance(gamma, Tensor)
    out_dtype = "f64" if (getattr(x, "dtype", "") == "f64" or getattr(gamma, "dtype", "") == "f64") else "f32"
    work_dt = np.float64 if out_dtype == "f64" else np.float32
    dll = AOTKernelEngine.get_dll()
    if dll and out_dtype == "f32" and x_arr.ndim == 1 and len(x_arr) == len(g_arr) and not has_tensor:
        x_c = np.ascontiguousarray(x_arr, dtype=np.float32)
        g_c = np.ascontiguousarray(g_arr, dtype=np.float32)
        out = np.empty_like(x_c)
        dll.nat_c_rmsnorm(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x_c), eps
        )
        return out if isinstance(x, np.ndarray) else out.tolist()
    x_w = x_arr.astype(work_dt)
    g_w = g_arr.astype(work_dt)
    rms = np.sqrt(np.mean(x_w ** 2, axis=-1, keepdims=True) + eps)
    x_hat = x_w / rms
    res = x_hat * g_w
    if has_tensor:
        parents = tuple(p for p in (x, gamma) if isinstance(p, Tensor))
        req_g = any(p.requires_grad for p in parents)
        out_t = Tensor(res, dtype=out_dtype, requires_grad=req_g)
        out_t._prev = parents
        def _bw():
            if out_t.grad is None:
                return
            dy = out_t.grad.data.astype(work_dt)
            if isinstance(gamma, Tensor):
                gamma._accum_grad(dy * x_hat)
            if isinstance(x, Tensor):
                u = dy * g_w
                dx = (u - x_hat * np.mean(u * x_hat, axis=-1, keepdims=True)) / rms
                x._accum_grad(dx)
        out_t._backward = _bw
        return out_t
    return res if isinstance(x, np.ndarray) else res.tolist()


def apply_rope_fn(x: Any, pos: Any, base: float = 10000.0) -> Any:
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    dim = len(x_arr)
    half = dim // 2
    x1 = x_arr[:half]
    x2 = x_arr[half:]
    freqs = 1.0 / (base ** (np.arange(0, half, dtype=float) / half))
    angles = float(pos) * freqs
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    out1 = x1 * cos_a - x2 * sin_a
    out2 = x1 * sin_a + x2 * cos_a
    res = np.concatenate([out1, out2])
    if isinstance(x, Tensor):
        out_t = Tensor(res, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is None:
                return
            dy = out_t.grad.data
            dy1, dy2 = dy[:half], dy[half:]
            dx1 = dy1 * cos_a + dy2 * sin_a
            dx2 = -dy1 * sin_a + dy2 * cos_a
            x._accum_grad(np.concatenate([dx1, dx2]))
        out_t._backward = _bw
        return out_t
    return res if isinstance(x, np.ndarray) else res.tolist()


def silu_fn(x: Any) -> Any:
    if isinstance(x, (int, float)):
        return float(x / (1.0 + math.exp(-x)))
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    if isinstance(x, Tensor):
        out_dtype = "f64" if x.dtype == "f64" else "f32"
        work_dt = np.float64 if out_dtype == "f64" else np.float32
        x_w = x_arr.astype(work_dt)
        sig = 1.0 / (1.0 + np.exp(-x_w))
        res = x_w * sig
        out_t = Tensor(res, dtype=out_dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is None:
                return
            dy = out_t.grad.data.astype(work_dt)
            dx = dy * sig * (1.0 + x_w * (1.0 - sig))
            x._accum_grad(dx)
        out_t._backward = _bw
        return out_t
    dll = AOTKernelEngine.get_dll()
    if dll and x_arr.ndim == 1:
        x_c = np.ascontiguousarray(x_arr, dtype=np.float32)
        out = np.empty_like(x_c)
        dll.nat_c_silu(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x_c)
        )
        return out if isinstance(x, np.ndarray) else out.tolist()
    res = x_arr / (1.0 + np.exp(-x_arr))
    return res if isinstance(x, np.ndarray) else res.tolist()


def softmax_fn(x: Any, axis: int = -1) -> Any:
    x_arr = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    if isinstance(x, Tensor):
        out_dtype = "f64" if x.dtype == "f64" else "f32"
        work_dt = np.float64 if out_dtype == "f64" else np.float32
        x_w = x_arr.astype(work_dt)
        max_v = np.max(x_w, axis=axis, keepdims=True)
        exp_v = np.exp(x_w - max_v)
        s = exp_v / np.sum(exp_v, axis=axis, keepdims=True)
        out_t = Tensor(s, dtype=out_dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is None:
                return
            dy = out_t.grad.data.astype(work_dt)
            dx = s * (dy - np.sum(dy * s, axis=axis, keepdims=True))
            x._accum_grad(dx)
        out_t._backward = _bw
        return out_t
    dll = AOTKernelEngine.get_dll()
    if dll and x_arr.ndim == 1:
        x_c = np.ascontiguousarray(x_arr, dtype=np.float32)
        out = np.empty_like(x_c)
        dll.nat_c_softmax(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x_c)
        )
        return out if isinstance(x, np.ndarray) else out.tolist()
    max_v = np.max(x_arr, axis=axis, keepdims=True)
    exp_v = np.exp(x_arr - max_v)
    sum_v = np.sum(exp_v, axis=axis, keepdims=True)
    res = exp_v / sum_v
    return res if isinstance(x, np.ndarray) else res.tolist()


def causal_mask_fn(seq_len: int) -> Tensor:
    mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
    mask = np.triu(mask, k=1)
    return Tensor(mask, dtype="f32")


def sdpa_fn(q: Any, k: Any, v: Any, mask: Any = None, scale: Optional[float] = None) -> Tensor:
    out_dtype = "f64" if any(getattr(t, "dtype", "") == "f64" for t in (q, k, v)) else "f32"
    work_dt = np.float64 if out_dtype == "f64" else np.float32
    Q = (q.data if isinstance(q, Tensor) else np.asarray(q)).astype(work_dt)
    K = (k.data if isinstance(k, Tensor) else np.asarray(k)).astype(work_dt)
    V = (v.data if isinstance(v, Tensor) else np.asarray(v)).astype(work_dt)

    d_k = Q.shape[-1]
    inv_sqrt = (1.0 / math.sqrt(d_k)) if scale is None else float(scale)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * inv_sqrt

    if mask is not None:
        M = (mask.data if isinstance(mask, Tensor) else np.asarray(mask)).astype(work_dt)
        scores = scores + M

    max_s = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    out = np.matmul(probs, V)
    parents = tuple(p for p in (q, k, v) if isinstance(p, Tensor))
    out_t = Tensor(out, dtype=out_dtype, requires_grad=any(p.requires_grad for p in parents))
    out_t._prev = parents
    def _bw():
        if out_t.grad is None:
            return
        dout = out_t.grad.data.astype(work_dt)
        dV = np.matmul(np.swapaxes(probs, -1, -2), dout)
        dprobs = np.matmul(dout, np.swapaxes(V, -1, -2))
        dscores = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
        dQ = np.matmul(dscores, K) * inv_sqrt
        dK = np.matmul(np.swapaxes(dscores, -1, -2), Q) * inv_sqrt
        if isinstance(q, Tensor):
            q._accum_grad(dQ)
        if isinstance(k, Tensor):
            k._accum_grad(dK)
        if isinstance(v, Tensor):
            v._accum_grad(dV)
    out_t._backward = _bw
    return out_t


def concat_fn(tensors: list[Any], axis: int = 0) -> Tensor:
    """Concatenates a list of tensors along an axis."""
    arrs = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    res = np.concatenate(arrs, axis=axis)
    parents = tuple(t for t in tensors if isinstance(t, Tensor))
    out_dtype = "f64" if any(t.dtype == "f64" for t in parents) else "f32"
    out_t = Tensor(res, dtype=out_dtype, requires_grad=any(t.requires_grad for t in parents))
    out_t._prev = parents
    sizes = [a.shape[axis] for a in arrs]
    def _bw():
        if out_t.grad is None:
            return
        splits = np.cumsum(sizes[:-1])
        grads = np.split(out_t.grad.data, splits, axis=axis)
        for t_in, g_slice in zip(tensors, grads):
            if isinstance(t_in, Tensor):
                t_in._accum_grad(g_slice)
    out_t._backward = _bw
    return out_t


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
    """Key-Value Cache for autoregressive token-by-token inference with Memory Arena (llama2.c style).
    Pre-allocates total contiguous memory upfront at session start, eliminating dynamic reallocations,
    redundant copies, and memory fragmentation on each generated token.
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
        """Inserts new tokens into the Memory Arena without memory reallocation."""
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

        # If dimension does not match and no tokens inserted, resize arena
        if token_dim != self.dim and self.seq_len == 0:
            self.dim = token_dim
            self._arena_k = np.zeros((self.max_seq_len, self.dim), dtype=np.float32)
            self._arena_v = np.zeros((self.max_seq_len, self.dim), dtype=np.float32)

        # If exceeding maximum capacity, expand arena (double capacity as a safeguard)
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
        """Resets sequence pointer without deallocating the memory arena."""
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
    """Canonical byte -> unicode character mapping table (GPT-2 / HuggingFace / Qwen style)."""
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
    """BPE Tokenizer with complete Merge and Byte-Fallback rules (LLaMA/Mistral/Qwen/GPT style).
    Supports:
      1. Vocabulary loaded from tokenizer.json files (HuggingFace/Qwen/LLaMA) or internal default.
      2. Canonical BPE Merge rules with rank priorities.
      3. Added tokens / special tokens (<|endoftext|>, etc.).
      4. GPT-2/Qwen byte-level mapping and robust fallback for <0xXX>.
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

        if vocab_file is not None:
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(f"Tokenizer vocabulary file not found: '{vocab_file}'")
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
        """Applies BPE merge rules over the character sequence."""
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

        # Fallback: Greedy Substring Match with Byte-Fallback (Default vocabulary without merges)
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

        # Extract normalizer (e.g., NFC)
        if "normalizer" in data and isinstance(data["normalizer"], dict):
            self.normalizer_type = data["normalizer"].get("type")

        # Extract pattern from official pre-tokenizer if declared in JSON
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
                    # Robust fallback for Python's standard 're' module (without nested character classes)
                    clean_pat = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
                    try:
                        self._pre_regex = re.compile(clean_pat)
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
    """Closes open mmap memory mapping for the given path or all open mappings."""
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
        raise FileNotFoundError(f"Safetensors file not found: '{resolved}'")
    file_size = os.path.getsize(resolved)
    if file_size < 8:
        raise ValueError(f"Invalid or truncated safetensors file: '{resolved}' (size {file_size} bytes < 8-byte header)")

    fh = open(resolved, "rb")
    try:
        header_size_bytes = fh.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"Safetensors file truncated while reading header size: '{resolved}'")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size <= 0:
            raise ValueError(f"Corrupted safetensors file: header_size = {header_size} in '{resolved}'")
        if header_size > file_size - 8:
            raise ValueError(
                f"Corrupted safetensors file in '{resolved}': header_size ({header_size} bytes) "
                f"exceeds total file size ({file_size} bytes)"
            )
        if header_size > 100 * 1024 * 1024:
            raise ValueError(f"Safetensors header exceeds 100MB safety limit ({header_size} bytes) in '{resolved}'")

        header_json_bytes = fh.read(header_size)
        if len(header_json_bytes) != header_size:
            raise ValueError(f"Truncated safetensors file: expected {header_size} header bytes, read {len(header_json_bytes)} in '{resolved}'")
        try:
            header = json.loads(header_json_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Malformed safetensors JSON header in '{resolved}': {e}") from e
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header (expected JSON object) in '{resolved}'")

        data_start = 8 + header_size
        tensor_intervals: list[tuple[int, int, str]] = []
        for tname, tinfo in header.items():
            if tname == "__metadata__":
                continue
            if not isinstance(tinfo, dict):
                raise ValueError(f"Invalid metadata for tensor '{tname}' in '{resolved}': expected dictionary")
            for req in ("dtype", "shape", "data_offsets"):
                if req not in tinfo:
                    raise ValueError(f"Incomplete metadata for tensor '{tname}' in '{resolved}': missing required field '{req}'")
            offsets = tinfo["data_offsets"]
            if not (isinstance(offsets, list) and len(offsets) == 2 and isinstance(offsets[0], int) and isinstance(offsets[1], int)):
                raise ValueError(f"Invalid offsets for tensor '{tname}' in '{resolved}': {offsets}")
            if offsets[0] < 0 or offsets[1] < offsets[0]:
                raise ValueError(f"Invalid offset range for tensor '{tname}' in '{resolved}': {offsets}")
            if data_start + offsets[1] > file_size:
                raise ValueError(
                    f"Truncated safetensors file in '{resolved}': tensor '{tname}' requires end offset "
                    f"{data_start + offsets[1]}, but file only has {file_size} bytes"
                )
            tensor_intervals.append((offsets[0], offsets[1], tname))

        # Canonical SafeTensors validation: overlap and contiguous coverage check
        if tensor_intervals:
            tensor_intervals.sort(key=lambda x: (x[0], x[1]))
            if tensor_intervals[0][0] != 0:
                raise ValueError(
                    f"Invalid safetensors file in '{resolved}': first tensor starts at offset {tensor_intervals[0][0]} "
                    f"(expected offset 0, unreferenced bytes before first tensor)."
                )
            for idx in range(len(tensor_intervals) - 1):
                cur_start, cur_end, cur_name = tensor_intervals[idx]
                nxt_start, nxt_end, nxt_name = tensor_intervals[idx + 1]
                if nxt_start < cur_end:
                    raise ValueError(
                        f"Illegal tensor overlap in '{resolved}': tensor '{cur_name}' [{cur_start}:{cur_end}] "
                        f"overlaps tensor '{nxt_name}' [{nxt_start}:{nxt_end}]."
                    )
                if nxt_start > cur_end:
                    raise ValueError(
                        f"Illegal byte gap in '{resolved}': range [{cur_end}:{nxt_start}] "
                        f"between '{cur_name}' and '{nxt_name}' does not belong to any tensor."
                    )
            last_end = tensor_intervals[-1][1]
            total_payload = file_size - data_start
            if last_end < total_payload:
                raise ValueError(
                    f"Unreferenced trailing bytes in '{resolved}': last tensor ends at {last_end}, "
                    f"but file payload has {total_payload} bytes ({total_payload - last_end} excess bytes at end)."
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
        raise KeyError(f"Tensor '{tensor_name}' not found in {path}. Available tensors: {available}")
    info = header[tensor_name]
    dtype_str = info.get("dtype", "F32")
    shape = info.get("shape", [])
    start_offset, end_offset = info["data_offsets"]

    if dtype_str not in DTYPE_MAP and dtype_str != "BF16":
        supported = list(DTYPE_MAP.keys()) + ["BF16"]
        raise ValueError(f"Unsupported data type '{dtype_str}' for tensor '{tensor_name}'. Supported types: {supported}")

    # Strict validation of dimensions and byte count
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
            f"Out-of-bounds read in SafeTensors for tensor '{tensor_name}': "
            f"final offset {byte_end} exceeds mapped size {len(mm)}"
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


def read_file_fn(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def read_file_safe_fn(path: str) -> NatResult:
    try:
        if not os.path.exists(path):
            return NatResult(error=f"File not found: '{path}'", is_ok=False)
        with open(path, "r", encoding="utf-8", newline="") as f:
            return NatResult(value=f.read(), is_ok=True)
    except Exception as e:
        return NatResult(error=str(e), is_ok=False)


def load_tokenizer_safe_fn(path: Optional[str] = None) -> NatResult:
    try:
        return NatResult(value=Tokenizer(path), is_ok=True)
    except Exception as e:
        return NatResult(error=str(e), is_ok=False)


@dataclass
class NatComplex:
    real: float = 0.0
    imag: float = 0.0

    def __repr__(self) -> str:
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real}{sign}{abs(self.imag)}i"

    def __add__(self, other: Any) -> "NatComplex":
        if isinstance(other, NatComplex):
            return NatComplex(self.real + other.real, self.imag + other.imag)
        if isinstance(other, (int, float)):
            return NatComplex(self.real + float(other), self.imag)
        return NotImplemented

    def __radd__(self, other: Any) -> "NatComplex":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "NatComplex":
        if isinstance(other, NatComplex):
            return NatComplex(self.real - other.real, self.imag - other.imag)
        if isinstance(other, (int, float)):
            return NatComplex(self.real - float(other), self.imag)
        return NotImplemented

    def __rsub__(self, other: Any) -> "NatComplex":
        if isinstance(other, (int, float)):
            return NatComplex(float(other) - self.real, -self.imag)
        return NotImplemented

    def __mul__(self, other: Any) -> "NatComplex":
        if isinstance(other, NatComplex):
            return NatComplex(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real
            )
        if isinstance(other, (int, float)):
            return NatComplex(self.real * float(other), self.imag * float(other))
        return NotImplemented

    def __rmul__(self, other: Any) -> "NatComplex":
        return self.__mul__(other)

    def __abs__(self) -> float:
        return math.hypot(self.real, self.imag)

    @property
    def magnitude(self) -> float:
        return math.hypot(self.real, self.imag)

    @property
    def power(self) -> float:
        return self.real * self.real + self.imag * self.imag


def complex_fn(real: Any = 0.0, imag: Any = 0.0) -> NatComplex:
    return NatComplex(float(real), float(imag))


class NatFix256:
    """256-bit fixed-point arithmetic type (default Q192.64) for high-altitude/cosmic scale steps."""
    def __init__(self, val: Any = 0, frac_bits: int = 64):
        self.frac_bits = int(frac_bits) if frac_bits is not None else 64
        if isinstance(val, NatFix256):
            self.limbs = list(val.limbs)
            self.sign = val.sign
            self.frac_bits = val.frac_bits
        elif isinstance(val, C_NatFix256):
            self.limbs = [val.limbs[0], val.limbs[1], val.limbs[2], val.limbs[3]]
            self.sign = int(val.sign)
            self.frac_bits = int(val.frac_bits)
        elif isinstance(val, str):
            dll = AOTKernelEngine.get_dll()
            if dll and hasattr(dll, "nat_c_fix256_from_str"):
                c_out = C_NatFix256()
                dll.nat_c_fix256_from_str(val.encode("utf-8"), self.frac_bits, ctypes.byref(c_out))
                self.limbs = [c_out.limbs[0], c_out.limbs[1], c_out.limbs[2], c_out.limbs[3]]
                self.sign = int(c_out.sign)
            else:
                self._from_str_py(val)
        elif isinstance(val, int):
            dll = AOTKernelEngine.get_dll()
            if dll and hasattr(dll, "nat_c_fix256_from_i64") and -0x8000000000000000 <= val <= 0x7FFFFFFFFFFFFFFF:
                c_out = C_NatFix256()
                dll.nat_c_fix256_from_i64(ctypes.c_int64(val), self.frac_bits, ctypes.byref(c_out))
                self.limbs = [c_out.limbs[0], c_out.limbs[1], c_out.limbs[2], c_out.limbs[3]]
                self.sign = int(c_out.sign)
            else:
                self._from_int_py(val)
        elif isinstance(val, float):
            dll = AOTKernelEngine.get_dll()
            if dll and hasattr(dll, "nat_c_fix256_from_f64"):
                c_out = C_NatFix256()
                dll.nat_c_fix256_from_f64(ctypes.c_double(val), self.frac_bits, ctypes.byref(c_out))
                self.limbs = [c_out.limbs[0], c_out.limbs[1], c_out.limbs[2], c_out.limbs[3]]
                self.sign = int(c_out.sign)
            else:
                self._from_float_py(val)
        else:
            self.limbs = [0, 0, 0, 0]
            self.sign = 1

    def _from_int_py(self, val: int):
        self.sign = -1 if val < 0 else 1
        mag = (abs(val) << self.frac_bits)
        self.limbs = [
            mag & 0xFFFFFFFFFFFFFFFF,
            (mag >> 64) & 0xFFFFFFFFFFFFFFFF,
            (mag >> 128) & 0xFFFFFFFFFFFFFFFF,
            (mag >> 192) & 0xFFFFFFFFFFFFFFFF,
        ]

    def _from_float_py(self, val: float):
        self.sign = -1 if val < 0 else 1
        scaled = int(round(abs(val) * (1 << self.frac_bits)))
        self.limbs = [
            scaled & 0xFFFFFFFFFFFFFFFF,
            (scaled >> 64) & 0xFFFFFFFFFFFFFFFF,
            (scaled >> 128) & 0xFFFFFFFFFFFFFFFF,
            (scaled >> 192) & 0xFFFFFFFFFFFFFFFF,
        ]

    def _from_str_py(self, s: str):
        s = s.strip()
        self.sign = 1
        if s.startswith("-"):
            self.sign = -1
            s = s[1:]
        elif s.startswith("+"):
            s = s[1:]
        from decimal import Decimal
        d = Decimal(s)
        scaled = int(round(d * (Decimal(1 << self.frac_bits))))
        mag = abs(scaled)
        self.limbs = [
            mag & 0xFFFFFFFFFFFFFFFF,
            (mag >> 64) & 0xFFFFFFFFFFFFFFFF,
            (mag >> 128) & 0xFFFFFFFFFFFFFFFF,
            (mag >> 192) & 0xFFFFFFFFFFFFFFFF,
        ]

    def to_c(self) -> C_NatFix256:
        c = C_NatFix256()
        c.limbs[0] = self.limbs[0]
        c.limbs[1] = self.limbs[1]
        c.limbs[2] = self.limbs[2]
        c.limbs[3] = self.limbs[3]
        c.frac_bits = self.frac_bits
        c.sign = self.sign
        return c

    @classmethod
    def from_c(cls, c: C_NatFix256) -> "NatFix256":
        r = cls(0, c.frac_bits)
        r.limbs = [c.limbs[0], c.limbs[1], c.limbs[2], c.limbs[3]]
        r.sign = int(c.sign)
        return r

    @property
    def raw_magnitude(self) -> int:
        return self.limbs[0] | (self.limbs[1] << 64) | (self.limbs[2] << 128) | (self.limbs[3] << 192)

    @property
    def raw_value(self) -> int:
        return self.sign * self.raw_magnitude

    def to_str(self) -> str:
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_to_str"):
            buf = ctypes.create_string_buffer(128)
            c = self.to_c()
            dll.nat_c_fix256_to_str(ctypes.byref(c), buf, 128)
            return buf.value.decode("utf-8")
        int_part = self.raw_magnitude >> self.frac_bits
        mask = (1 << self.frac_bits) - 1
        frac_part = self.raw_magnitude & mask
        res = f"-{int_part}" if self.sign < 0 else f"{int_part}"
        if self.frac_bits > 0:
            frac_dec = int(round((frac_part * 1000000) / (1 << self.frac_bits)))
            res += f".{frac_dec:06d}"
        return res

    def to_f64(self) -> float:
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_to_f64"):
            out = ctypes.c_double(0.0)
            c = self.to_c()
            dll.nat_c_fix256_to_f64(ctypes.byref(c), ctypes.byref(out))
            return out.value
        return self.raw_value / (1 << self.frac_bits)

    def to_i64(self) -> int:
        return self.sign * (self.raw_magnitude >> self.frac_bits)

    def __repr__(self) -> str:
        return self.to_str()

    def __str__(self) -> str:
        return self.to_str()

    def __neg__(self) -> "NatFix256":
        r = NatFix256(self)
        if r.raw_magnitude != 0:
            r.sign = -r.sign
        return r

    def __abs__(self) -> "NatFix256":
        r = NatFix256(self)
        r.sign = 1
        return r

    def __add__(self, other: Any) -> "NatFix256":
        b = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_add"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatFix256()
            dll.nat_c_fix256_add(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatFix256.from_c(cout)
        val = self.raw_value + b.raw_value
        r = NatFix256(0, self.frac_bits)
        r.sign = -1 if val < 0 else 1
        mag = abs(val)
        r.limbs = [mag & 0xFFFFFFFFFFFFFFFF, (mag >> 64) & 0xFFFFFFFFFFFFFFFF, (mag >> 128) & 0xFFFFFFFFFFFFFFFF, (mag >> 192) & 0xFFFFFFFFFFFFFFFF]
        return r

    def __radd__(self, other: Any) -> "NatFix256":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "NatFix256":
        b = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_sub"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatFix256()
            dll.nat_c_fix256_sub(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatFix256.from_c(cout)
        val = self.raw_value - b.raw_value
        r = NatFix256(0, self.frac_bits)
        r.sign = -1 if val < 0 else 1
        mag = abs(val)
        r.limbs = [mag & 0xFFFFFFFFFFFFFFFF, (mag >> 64) & 0xFFFFFFFFFFFFFFFF, (mag >> 128) & 0xFFFFFFFFFFFFFFFF, (mag >> 192) & 0xFFFFFFFFFFFFFFFF]
        return r

    def __rsub__(self, other: Any) -> "NatFix256":
        a = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        return a.__sub__(self)

    def __mul__(self, other: Any) -> "NatFix256":
        b = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_mul"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatFix256()
            dll.nat_c_fix256_mul(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatFix256.from_c(cout)
        val = (self.raw_value * b.raw_value) >> self.frac_bits
        r = NatFix256(0, self.frac_bits)
        r.sign = -1 if val < 0 else 1
        mag = abs(val)
        r.limbs = [mag & 0xFFFFFFFFFFFFFFFF, (mag >> 64) & 0xFFFFFFFFFFFFFFFF, (mag >> 128) & 0xFFFFFFFFFFFFFFFF, (mag >> 192) & 0xFFFFFFFFFFFFFFFF]
        return r

    def __rmul__(self, other: Any) -> "NatFix256":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "NatFix256":
        b = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_div"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatFix256()
            dll.nat_c_fix256_div(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatFix256.from_c(cout)
        if b.raw_magnitude == 0:
            return NatFix256(0, self.frac_bits)
        val = (self.raw_value << self.frac_bits) // b.raw_value
        r = NatFix256(0, self.frac_bits)
        r.sign = -1 if val < 0 else 1
        mag = abs(val)
        r.limbs = [mag & 0xFFFFFFFFFFFFFFFF, (mag >> 64) & 0xFFFFFFFFFFFFFFFF, (mag >> 128) & 0xFFFFFFFFFFFFFFFF, (mag >> 192) & 0xFFFFFFFFFFFFFFFF]
        return r

    def __rtruediv__(self, other: Any) -> "NatFix256":
        a = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        return a.__truediv__(self)

    def _cmp(self, other: Any) -> int:
        b = other if isinstance(other, NatFix256) else NatFix256(other, self.frac_bits)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_fix256_cmp"):
            ca, cb = self.to_c(), b.to_c()
            return int(dll.nat_c_fix256_cmp(ctypes.byref(ca), ctypes.byref(cb)))
        va, vb = self.raw_value, b.raw_value
        return 1 if va > vb else (-1 if va < vb else 0)

    def __eq__(self, other: Any) -> bool:
        return self._cmp(other) == 0

    def __lt__(self, other: Any) -> bool:
        return self._cmp(other) < 0

    def __le__(self, other: Any) -> bool:
        return self._cmp(other) <= 0

    def __gt__(self, other: Any) -> bool:
        return self._cmp(other) > 0

    def __ge__(self, other: Any) -> bool:
        return self._cmp(other) >= 0


class NatInterval256:
    """Rigorous 256-bit interval arithmetic [lo, hi] for coherence verification in extreme steps."""
    def __init__(self, lo: Any = 0, hi: Any = None):
        if isinstance(lo, NatInterval256):
            self.lo = NatFix256(lo.lo)
            self.hi = NatFix256(lo.hi)
        elif isinstance(lo, C_NatInterval256):
            self.lo = NatFix256.from_c(lo.lo)
            self.hi = NatFix256.from_c(lo.hi)
        else:
            self.lo = lo if isinstance(lo, NatFix256) else NatFix256(lo)
            self.hi = self.lo if hi is None else (hi if isinstance(hi, NatFix256) else NatFix256(hi))
            if self.lo > self.hi:
                self.lo, self.hi = self.hi, self.lo

    def to_c(self) -> C_NatInterval256:
        c = C_NatInterval256()
        c.lo = self.lo.to_c()
        c.hi = self.hi.to_c()
        return c

    @classmethod
    def from_c(cls, c: C_NatInterval256) -> "NatInterval256":
        return cls(NatFix256.from_c(c.lo), NatFix256.from_c(c.hi))

    def to_str(self) -> str:
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_to_str"):
            buf = ctypes.create_string_buffer(256)
            c = self.to_c()
            dll.nat_c_interval256_to_str(ctypes.byref(c), buf, 256)
            return buf.value.decode("utf-8")
        return f"[{self.lo.to_str()}, {self.hi.to_str()}]"

    def __repr__(self) -> str:
        return self.to_str()

    def __str__(self) -> str:
        return self.to_str()

    def __add__(self, other: Any) -> "NatInterval256":
        b = other if isinstance(other, NatInterval256) else NatInterval256(other)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_add"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatInterval256()
            dll.nat_c_interval256_add(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatInterval256.from_c(cout)
        return NatInterval256(self.lo + b.lo, self.hi + b.hi)

    def __sub__(self, other: Any) -> "NatInterval256":
        b = other if isinstance(other, NatInterval256) else NatInterval256(other)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_sub"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatInterval256()
            dll.nat_c_interval256_sub(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatInterval256.from_c(cout)
        return NatInterval256(self.lo - b.hi, self.hi - b.lo)

    def __mul__(self, other: Any) -> "NatInterval256":
        if isinstance(other, (int, float, NatFix256)):
            s = other if isinstance(other, NatFix256) else NatFix256(other)
            dll = AOTKernelEngine.get_dll()
            if dll and hasattr(dll, "nat_c_interval256_mul_scalar"):
                ca, cs, cout = self.to_c(), s.to_c(), C_NatInterval256()
                dll.nat_c_interval256_mul_scalar(ctypes.byref(ca), ctypes.byref(cs), ctypes.byref(cout))
                return NatInterval256.from_c(cout)
            return NatInterval256(self.lo * s, self.hi * s) if s >= 0 else NatInterval256(self.hi * s, self.lo * s)
        b = other if isinstance(other, NatInterval256) else NatInterval256(other)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_mul"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatInterval256()
            dll.nat_c_interval256_mul(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatInterval256.from_c(cout)
        p = [self.lo * b.lo, self.lo * b.hi, self.hi * b.lo, self.hi * b.hi]
        return NatInterval256(min(p), max(p))

    def __radd__(self, other: Any) -> "NatInterval256":
        return self.__add__(other)

    def __rsub__(self, other: Any) -> "NatInterval256":
        a = other if isinstance(other, NatInterval256) else NatInterval256(other)
        return a.__sub__(self)

    def __rmul__(self, other: Any) -> "NatInterval256":
        return self.__mul__(other)

    def __neg__(self) -> "NatInterval256":
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_neg"):
            ca, cout = self.to_c(), C_NatInterval256()
            dll.nat_c_interval256_neg(ctypes.byref(ca), ctypes.byref(cout))
            return NatInterval256.from_c(cout)
        return NatInterval256(-self.hi, -self.lo)

    def __truediv__(self, other: Any) -> "NatInterval256":
        if isinstance(other, (int, float, NatFix256)):
            s = other if isinstance(other, NatFix256) else NatFix256(other)
            dll = AOTKernelEngine.get_dll()
            if dll and hasattr(dll, "nat_c_interval256_div_scalar"):
                ca, cs, cout = self.to_c(), s.to_c(), C_NatInterval256()
                dll.nat_c_interval256_div_scalar(ctypes.byref(ca), ctypes.byref(cs), ctypes.byref(cout))
                return NatInterval256.from_c(cout)
            if s.raw_magnitude == 0:
                return NatInterval256(0, 0)
            return NatInterval256(self.lo / s, self.hi / s) if s >= 0 else NatInterval256(self.hi / s, self.lo / s)
        b = other if isinstance(other, NatInterval256) else NatInterval256(other)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_div"):
            ca, cb, cout = self.to_c(), b.to_c(), C_NatInterval256()
            dll.nat_c_interval256_div(ctypes.byref(ca), ctypes.byref(cb), ctypes.byref(cout))
            return NatInterval256.from_c(cout)
        if b.contains(0):
            return NatInterval256(0, 0)
        p = [self.lo / b.lo, self.lo / b.hi, self.hi / b.lo, self.hi / b.hi]
        return NatInterval256(min(p), max(p))

    def __rtruediv__(self, other: Any) -> "NatInterval256":
        a = other if isinstance(other, NatInterval256) else NatInterval256(other)
        return a.__truediv__(self)

    def width(self) -> NatFix256:
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_width"):
            ca, cout = self.to_c(), C_NatFix256()
            dll.nat_c_interval256_width(ctypes.byref(ca), ctypes.byref(cout))
            return NatFix256.from_c(cout)
        return self.hi - self.lo

    def mid(self) -> NatFix256:
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_mid"):
            ca, cout = self.to_c(), C_NatFix256()
            dll.nat_c_interval256_mid(ctypes.byref(ca), ctypes.byref(cout))
            return NatFix256.from_c(cout)
        return (self.lo + self.hi) / 2

    def contains(self, x: Any) -> bool:
        fx = x if isinstance(x, NatFix256) else NatFix256(x)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_contains"):
            ca, cx = self.to_c(), fx.to_c()
            return bool(dll.nat_c_interval256_contains(ctypes.byref(ca), ctypes.byref(cx)))
        return self.lo <= fx <= self.hi

    def is_coherent(self, max_width: Any) -> bool:
        mw = max_width if isinstance(max_width, NatFix256) else NatFix256(max_width)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_is_coherent"):
            ca, cmw = self.to_c(), mw.to_c()
            return bool(dll.nat_c_interval256_is_coherent(ctypes.byref(ca), ctypes.byref(cmw)))
        return self.width() <= mw

    def step(self, vel: Any, dt: Any) -> "NatInterval256":
        v = vel if isinstance(vel, NatInterval256) else NatInterval256(vel)
        t = dt if isinstance(dt, NatFix256) else NatFix256(dt)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_step"):
            cp, cv, ct, cout = self.to_c(), v.to_c(), t.to_c(), C_NatInterval256()
            dll.nat_c_interval256_step(ctypes.byref(cp), ctypes.byref(cv), ctypes.byref(ct), ctypes.byref(cout))
            return NatInterval256.from_c(cout)
        return self + (v * t)

    def simulate_steps(self, vel: Any, dt: Any, num_steps: int) -> "NatInterval256":
        v = vel if isinstance(vel, NatInterval256) else NatInterval256(vel)
        t = dt if isinstance(dt, NatFix256) else NatFix256(dt)
        dll = AOTKernelEngine.get_dll()
        if dll and hasattr(dll, "nat_c_interval256_simulate_steps"):
            cp, cv, ct = self.to_c(), v.to_c(), t.to_c()
            dll.nat_c_interval256_simulate_steps(ctypes.byref(cp), ctypes.byref(cv), ctypes.byref(ct), ctypes.c_int64(num_steps))
            return NatInterval256.from_c(cp)
        cur = self
        step_delta = v * t
        for _ in range(num_steps):
            cur = cur + step_delta
        return cur


def fix256_fn(val: Any = 0, frac_bits: int = 64) -> NatFix256:
    return NatFix256(val, frac_bits)


def interval256_fn(lo: Any = 0, hi: Any = None) -> NatInterval256:
    return NatInterval256(lo, hi)


def interval256_step_fn(pos: Any, vel: Any, dt: Any) -> NatInterval256:
    iv_pos = pos if isinstance(pos, NatInterval256) else NatInterval256(pos)
    return iv_pos.step(vel, dt)


def interval256_width_fn(iv: Any) -> NatFix256:
    iv_obj = iv if isinstance(iv, NatInterval256) else NatInterval256(iv)
    return iv_obj.width()


def interval256_mid_fn(iv: Any) -> NatFix256:
    iv_obj = iv if isinstance(iv, NatInterval256) else NatInterval256(iv)
    return iv_obj.mid()


def interval256_is_coherent_fn(iv: Any, max_width: Any) -> bool:
    iv_obj = iv if isinstance(iv, NatInterval256) else NatInterval256(iv)
    return iv_obj.is_coherent(max_width)


def rfft_fn(x: Any, n: Optional[int] = None) -> Any:
    if isinstance(x, Tensor):
        arr = np.asarray(x.data, dtype=np.float64)
        is_tensor = True
    else:
        arr = np.asarray(x, dtype=np.float64)
        is_tensor = False

    N = int(n) if n is not None else arr.shape[0]
    spec_complex = np.fft.rfft(arr, n=N)
    spec_2d = np.column_stack((spec_complex.real, spec_complex.imag)).astype(np.float64)
    if is_tensor:
        return Tensor(spec_2d, dtype="f64")
    return spec_2d.tolist()


def irfft_fn(freqs: Any, n: Optional[int] = None) -> Any:
    if isinstance(freqs, list) and freqs and isinstance(freqs[0], NatComplex):
        complex_arr = np.array([complex(c.real, c.imag) for c in freqs], dtype=np.complex128)
        N = int(n) if n is not None else (len(complex_arr) - 1) * 2
        rec = np.fft.irfft(complex_arr, n=N).astype(np.float64)
        return rec.tolist()

    if isinstance(freqs, Tensor):
        arr = np.asarray(freqs.data, dtype=np.float64)
        is_tensor = True
    else:
        arr = np.asarray(freqs, dtype=np.float64)
        is_tensor = False

    if arr.ndim == 2 and arr.shape[1] == 2:
        complex_arr = arr[:, 0] + 1j * arr[:, 1]
    elif arr.dtype in (np.complex128, np.complex64):
        complex_arr = arr
    else:
        complex_arr = arr.astype(np.complex128)

    N = int(n) if n is not None else (len(complex_arr) - 1) * 2
    rec = np.fft.irfft(complex_arr, n=N).astype(np.float64)
    if is_tensor:
        return Tensor(rec, dtype="f64")
    return rec.tolist()


def power_spectrum_fn(x: Any) -> Any:
    if isinstance(x, list) and x and isinstance(x[0], NatComplex):
        return [c.real ** 2 + c.imag ** 2 for c in x]

    if isinstance(x, Tensor):
        arr = np.asarray(x.data, dtype=np.float64)
        is_tensor = True
    else:
        arr = np.asarray(x, dtype=np.float64)
        is_tensor = False

    if arr.ndim == 2 and arr.shape[1] == 2:
        ps = arr[:, 0] ** 2 + arr[:, 1] ** 2
    else:
        spec_c = np.fft.rfft(arr)
        ps = spec_c.real ** 2 + spec_c.imag ** 2

    ps = ps.astype(np.float64)
    if is_tensor:
        return Tensor(ps, dtype="f64")
    return ps.tolist()


def _iter_tensors(obj: Any):
    if isinstance(obj, Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for val in obj.values():
            yield from _iter_tensors(val)
    elif hasattr(obj, "_fields") and isinstance(object.__getattribute__(obj, "_fields"), dict):
        for val in object.__getattribute__(obj, "_fields").values():
            yield from _iter_tensors(val)


def requires_grad_fn(x: Any, flag: bool = True) -> Any:
    for t in _iter_tensors(x):
        t.requires_grad_(flag)
    return x


def backward_fn(loss: Any, grad: Any = None) -> None:
    if isinstance(loss, Tensor):
        loss.backward(grad)
    else:
        raise RuntimeError(f"backward() requires a Tensor, but received {type(loss).__name__}")


def zero_grad_fn(params: Any) -> None:
    for t in _iter_tensors(params):
        t.zero_grad()


def sgd_fn(params: Any, lr: float = 0.01) -> None:
    lr_f = float(lr)
    for p in _iter_tensors(params):
        if p.grad is not None:
            p.data = p.data - (lr_f * p.grad.data).astype(p.data.dtype)


def adam_fn(params: Any, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    lr_f = float(lr)
    b1 = float(beta1)
    b2 = float(beta2)
    eps_f = float(eps)
    for p in _iter_tensors(params):
        if p.grad is None:
            continue
        g = p.grad.data.astype(np.float64)
        if not hasattr(p, "_adam_m") or p._adam_m is None or p._adam_m.shape != p.data.shape:
            p._adam_m = np.zeros_like(p.data, dtype=np.float64)
            p._adam_v = np.zeros_like(p.data, dtype=np.float64)
            p._adam_t = 0
        p._adam_t += 1
        t_step = p._adam_t
        p._adam_m = b1 * p._adam_m + (1.0 - b1) * g
        p._adam_v = b2 * p._adam_v + (1.0 - b2) * (g * g)
        m_hat = p._adam_m / (1.0 - b1 ** t_step)
        v_hat = p._adam_v / (1.0 - b2 ** t_step)
        step = lr_f * m_hat / (np.sqrt(v_hat) + eps_f)
        p.data = p.data - step.astype(p.data.dtype)


def cross_entropy_fn(logits: Any, target: Any) -> Tensor:
    l_arr = logits.data if isinstance(logits, Tensor) else np.asarray(logits, dtype=np.float32)
    out_dtype = "f64" if getattr(logits, "dtype", "") == "f64" else "f32"
    work_dt = np.float64 if out_dtype == "f64" else np.float32
    l_w = l_arr.astype(work_dt)

    if l_w.ndim == 1:
        tgt_idx = int(target.item() if hasattr(target, "item") else target)
        max_v = np.max(l_w)
        exp_v = np.exp(l_w - max_v)
        probs = exp_v / np.sum(exp_v)
        loss_val = -np.log(np.maximum(probs[tgt_idx], 1e-12))
        out_t = Tensor(np.asarray(loss_val, dtype=work_dt), dtype=out_dtype, requires_grad=isinstance(logits, Tensor))
        if isinstance(logits, Tensor):
            out_t._prev = (logits,)
            def _bw():
                if out_t.grad is None:
                    return
                dout = float(out_t.grad.data.item())
                dlogits = probs.copy()
                dlogits[tgt_idx] -= 1.0
                logits._accum_grad(dlogits * dout)
            out_t._backward = _bw
        return out_t
    else:
        tgt_arr = (target.data if isinstance(target, Tensor) else np.asarray(target)).astype(np.int64).ravel()
        n = l_w.shape[0]
        max_v = np.max(l_w, axis=-1, keepdims=True)
        exp_v = np.exp(l_w - max_v)
        probs = exp_v / np.sum(exp_v, axis=-1, keepdims=True)
        picked = probs[np.arange(n), tgt_arr]
        loss_val = -np.mean(np.log(np.maximum(picked, 1e-12)))
        out_t = Tensor(np.asarray(loss_val, dtype=work_dt), dtype=out_dtype, requires_grad=isinstance(logits, Tensor))
        if isinstance(logits, Tensor):
            out_t._prev = (logits,)
            def _bw():
                if out_t.grad is None:
                    return
                dout = float(out_t.grad.data.item())
                dlogits = probs.copy()
                dlogits[np.arange(n), tgt_arr] -= 1.0
                logits._accum_grad((dlogits / float(n)) * dout)
            out_t._backward = _bw
        return out_t


def matmul_fn(a: Any, b: Any) -> Any:
    if isinstance(a, Tensor):
        return a @ b
    if isinstance(b, Tensor):
        return Tensor(a, dtype=b.dtype, device=b.device) @ b
    return (np.matmul(np.array(a), np.array(b))).tolist()


def relu_fn(x: Any) -> Any:
    if isinstance(x, Tensor):
        res = np.maximum(0, x.data)
        out_t = Tensor(res, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is not None:
                x._accum_grad(out_t.grad.data * (x.data > 0))
        out_t._backward = _bw
        return out_t
    if isinstance(x, list):
        return (np.maximum(0, np.array(x))).tolist()
    return max(0, x)


def sigmoid_fn(x: Any) -> Any:
    if isinstance(x, Tensor):
        out_dtype = "f64" if x.dtype == "f64" else "f32"
        work_dt = np.float64 if out_dtype == "f64" else np.float32
        sig = 1.0 / (1.0 + np.exp(-x.data.astype(work_dt)))
        out_t = Tensor(sig, dtype=out_dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is not None:
                x._accum_grad(out_t.grad.data.astype(work_dt) * sig * (1.0 - sig))
        out_t._backward = _bw
        return out_t
    if isinstance(x, list):
        return (1.0 / (1.0 + np.exp(-np.array(x, dtype=float)))).tolist()
    return 1.0 / (1.0 + math.exp(-x))


def sum_fn(x: Any) -> Any:
    if isinstance(x, Tensor):
        out_dtype = "f64" if x.dtype == "f64" else "f32"
        work_dt = np.float64 if out_dtype == "f64" else np.float32
        res = np.asarray(np.sum(x.data.astype(work_dt)), dtype=work_dt)
        out_t = Tensor(res, dtype=out_dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is not None:
                g_val = float(out_t.grad.data.item())
                x._accum_grad(np.full_like(x.data, g_val, dtype=work_dt))
        out_t._backward = _bw
        return out_t
    if isinstance(x, (list, np.ndarray)):
        return (np.sum(np.asarray(x))).item()
    return x


def mean_fn(x: Any) -> Any:
    if isinstance(x, Tensor):
        out_dtype = "f64" if x.dtype == "f64" else "f32"
        work_dt = np.float64 if out_dtype == "f64" else np.float32
        res = np.asarray(np.mean(x.data.astype(work_dt)), dtype=work_dt)
        out_t = Tensor(res, dtype=out_dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        n_el = max(1, int(x.data.size))
        def _bw():
            if out_t.grad is not None:
                g_val = float(out_t.grad.data.item()) / float(n_el)
                x._accum_grad(np.full_like(x.data, g_val, dtype=work_dt))
        out_t._backward = _bw
        return out_t
    return float(np.mean(np.asarray(x)))


def reshape_fn(x: Any, s: Any) -> Any:
    if isinstance(x, Tensor):
        orig_shape = x.data.shape
        out_t = Tensor(np.reshape(x.data, s), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is not None:
                x._accum_grad(np.reshape(out_t.grad.data, orig_shape))
        out_t._backward = _bw
        return out_t
    return (np.reshape(np.array(x), s)).tolist()


def transpose_fn(x: Any) -> Any:
    if isinstance(x, Tensor):
        out_t = Tensor(np.transpose(x.data), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        out_t._prev = (x,)
        def _bw():
            if out_t.grad is not None:
                x._accum_grad(np.transpose(out_t.grad.data))
        out_t._backward = _bw
        return out_t
    return (np.transpose(np.array(x))).tolist()


BUILTIN_IMPLS: dict[str, Callable] = {
    "print": print,
    "len": len,
    "append": lambda lst, x: lst.append(x),
    "push": lambda lst, x: lst.append(x),
    "Dict": dict,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "ln": math.log,
    "floor": lambda x: int(math.floor(x)),
    "to_i64": int,
    "to_f64": float,
    "dot": lambda a, b: float(np.dot(np.array(a), np.array(b))),
    "matmul": matmul_fn,
    "tanh": math.tanh,
    "relu": relu_fn,
    "sigmoid": sigmoid_fn,
    "sum": sum_fn,
    "mean": mean_fn,
    "max": lambda x: (np.max(x.data if isinstance(x, Tensor) else np.asarray(x))).item() if isinstance(x, (list, Tensor, np.ndarray)) else x,
    "min": lambda x: (np.min(x.data if isinstance(x, Tensor) else np.asarray(x))).item() if isinstance(x, (list, Tensor, np.ndarray)) else x,
    "argmax": lambda x: int(np.argmax(x.data if isinstance(x, Tensor) else np.asarray(x))),
    "zeros": lambda s: Tensor(np.zeros(s, dtype=np.float32), dtype="f32"),
    "ones": lambda s: Tensor(np.ones(s, dtype=np.float32), dtype="f32"),
    "shape": lambda x: list(x.shape) if hasattr(x, "shape") else list(np.shape(x)),
    "reshape": reshape_fn,
    "transpose": transpose_fn,
    "tensor": lambda x, dtype="f32", dev=None, requires_grad=False: Tensor(x, dtype=dtype, device=dev, requires_grad=requires_grad),
    "device": device_fn,
    "quantize_ternary": quantize_ternary_fn,
    "pack_ternary": pack_ternary_fn,
    "unpack_ternary": unpack_ternary_fn,
    "clock": time.perf_counter,
    "random_ternary": lambda s, seed=None: (
        Tensor(np.random.RandomState(int(seed)).choice([-1, 0, 1], size=s).astype(np.int8), dtype="t2")
        if seed is not None else
        Tensor(np.random.choice([-1, 0, 1], size=s).astype(np.int8), dtype="t2")
    ),
    "randn": lambda s, seed=None: (
        Tensor(np.random.RandomState(int(seed)).randn(*(s if isinstance(s, (list, tuple)) else (s,))).astype(np.float32), dtype="f32")
        if seed is not None else
        Tensor(np.random.randn(*(s if isinstance(s, (list, tuple)) else (s,))).astype(np.float32), dtype="f32")
    ),
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
    "read_file": read_file_fn,
    "read_file_safe": read_file_safe_fn,
    "exists_file": lambda p: os.path.exists(p),
    "load_tokenizer_safe": load_tokenizer_safe_fn,
    "ok": lambda v: NatResult(value=v, is_ok=True),
    "err": lambda m: NatResult(error=str(m), is_ok=False),
    "is_ok": lambda r: getattr(r, "is_ok", True) if isinstance(r, NatResult) else True,
    "is_err": lambda r: not getattr(r, "is_ok", True) if isinstance(r, NatResult) else False,
    "unwrap": lambda r: (r.value if r.is_ok else (_ for _ in ()).throw(RuntimeError(f"unwrap called on Err Result: {r.error}"))) if isinstance(r, NatResult) else r,
    "unwrap_or": lambda r, d: (r.value if r.is_ok else d) if isinstance(r, NatResult) else r,
    "error_message": lambda r: (r.error if (isinstance(r, NatResult) and not r.is_ok) else ""),
    "fill": lambda n, v: [copy.deepcopy(v) if isinstance(v, (dict, list)) else v for _ in range(int(n))],
    "seed": lambda s: np.random.seed(int(s)),
    "requires_grad": requires_grad_fn,
    "backward": backward_fn,
    "zero_grad": zero_grad_fn,
    "cross_entropy": cross_entropy_fn,
    "sgd": sgd_fn,
    "adam": adam_fn,
    "Complex": complex_fn,
    "rfft": rfft_fn,
    "irfft": irfft_fn,
    "power_spectrum": power_spectrum_fn,
    "Fix256": fix256_fn,
    "fix256": fix256_fn,
    "Interval256": interval256_fn,
    "interval256": interval256_fn,
    "interval256_step": interval256_step_fn,
    "interval256_width": interval256_width_fn,
    "interval256_mid": interval256_mid_fn,
    "interval256_is_coherent": interval256_is_coherent_fn,
    "map": lambda fn, it: [fn(x.item() if hasattr(x, "item") else x) for x in it],
    "filter": lambda fn, it: [x.item() if hasattr(x, "item") else x for x in it if fn(x.item() if hasattr(x, "item") else x)],
}


def _bind_builtin_implementations():
    """Garante coerência bidirecional estrita entre BUILTINS (semântico) e BUILTIN_IMPLS (runtime)."""
    missing_impls = set(BUILTINS.keys()) - set(BUILTIN_IMPLS.keys())
    missing_decls = set(BUILTIN_IMPLS.keys()) - set(BUILTINS.keys())
    if missing_impls:
        raise RuntimeError(f"Coerência semântica/runtime violada: builtins em BUILTINS sem implementação: {missing_impls}")
    if missing_decls:
        raise RuntimeError(f"Coerência semântica/runtime violada: builtins em BUILTIN_IMPLS sem assinatura: {missing_decls}")

    for name, impl in BUILTIN_IMPLS.items():
        BUILTINS[name].impl = impl


_bind_builtin_implementations()


class NatLambda:
    """Anonymous function (lexical closure) at runtime."""
    def __init__(self, node: Lambda, captured_scopes: list[dict[str, Any]], interp: "Interpreter"):
        self.node = node
        self.captured_scopes = captured_scopes
        self.interp = interp

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.node.params):
            raise self.interp._runtime_error(
                f"lambda expects {len(self.node.params)} argument(s), but received {len(args)}",
                self.node,
            )
        base = len(self.interp.scopes)
        for sc in self.captured_scopes:
            self.interp.scopes.append(sc)
            self.interp.const_scopes.append(set())
        local_scope = {p.name: arg for p, arg in zip(self.node.params, args)}
        self.interp.scopes.append(local_scope)
        self.interp.const_scopes.append(set())
        self.interp.frame_bases.append(base)
        try:
            return self.interp.eval_expr(self.node.body)
        finally:
            self.interp.frame_bases.pop()
            for _ in range(len(self.captured_scopes) + 1):
                self.interp.const_scopes.pop()
                self.interp.scopes.pop()

    def __repr__(self) -> str:
        params_str = ", ".join(p.name for p in self.node.params)
        return f"<lambda ({params_str})>"


class NatStructInstance:
    def __init__(self, struct_name: str, field_values: dict[str, Any]):
        object.__setattr__(self, "_struct_name", struct_name)
        object.__setattr__(self, "_fields", dict(field_values))

    def __getattr__(self, name: str) -> Any:
        fields_dict = object.__getattribute__(self, "_fields")
        if name in fields_dict:
            return fields_dict[name]
        sname = object.__getattribute__(self, "_struct_name")
        raise AttributeError(f"Struct '{sname}' has no field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        fields_dict = object.__getattribute__(self, "_fields")
        if name in fields_dict:
            fields_dict[name] = value
        else:
            sname = object.__getattribute__(self, "_struct_name")
            raise RuntimeError(f"Struct '{sname}' has no field '{name}'")

    def __repr__(self) -> str:
        sname = object.__getattribute__(self, "_struct_name")
        fields_dict = object.__getattribute__(self, "_fields")
        inner = ", ".join(f"{k}={v!r}" for k, v in fields_dict.items())
        return f"{sname}({inner})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NatStructInstance):
            return (
                object.__getattribute__(self, "_struct_name") == object.__getattribute__(other, "_struct_name")
                and object.__getattribute__(self, "_fields") == object.__getattribute__(other, "_fields")
            )
        return False


class NatUserModule:
    def __init__(self, mod_name: str, exports: dict[str, Any]):
        object.__setattr__(self, "_mod_name", mod_name)
        object.__setattr__(self, "_exports", dict(exports))

    def __getattr__(self, name: str) -> Any:
        exports = object.__getattribute__(self, "_exports")
        if name in exports:
            return exports[name]
        mname = object.__getattribute__(self, "_mod_name")
        raise AttributeError(f"Module '{mname}' has no attribute '{name}'")


class Interpreter:
    def __init__(self):
        self.globals: dict[str, Any] = {}
        self.const_globals: set[str] = set()
        self.scopes: list[dict[str, Any]] = [{}]
        self.const_scopes: list[set[str]] = [set()]
        self.frame_bases: list[int] = [0]
        self.last_scope: dict[str, Any] = {}
        self.cur_node: Optional[Node] = None
        self._loading_modules: set[str] = set()
        self.init_builtins()

    def _runtime_error(self, message: str, node: Optional[Node] = None) -> NatRuntimeError:
        n = node if node is not None else self.cur_node
        line = getattr(n, "line", 0) if n else 0
        col = getattr(n, "col", 0) if n else 0
        return NatRuntimeError(message, line=line, col=col)

    def init_builtins(self):
        # Single Source of Truth: populate global functions directly from the BUILTINS table
        self.globals = {name: sig.impl for name, sig in BUILTINS.items()}
        self.globals["Tensor"] = Tensor
        self.globals["math"] = MathModule()

    @staticmethod
    def _make_struct_ctor(struct_decl: StructDecl):
        field_names = [f.name for f in struct_decl.fields]
        sname = struct_decl.name
        def _ctor(*args):
            if len(args) != len(field_names):
                raise RuntimeError(f"Struct '{sname}' expects {len(field_names)} argument(s), but received {len(args)}")
            return NatStructInstance(sname, dict(zip(field_names, args)))
        return _ctor

    def _load_user_module(self, mod_name: str) -> None:
        if mod_name in self.globals and isinstance(self.globals[mod_name], NatUserModule):
            return
        if mod_name in self._loading_modules:
            return
        self._loading_modules.add(mod_name)
        try:
            mod_path = _resolve_local_module_path(mod_name)
            if mod_path is None:
                raise RuntimeError(f"Module not found: '{mod_name}'")
            with open(mod_path, "r", encoding="utf-8") as mf:
                mod_src = mf.read()
            mod_prog = parse(mod_src)
            exports: dict[str, Any] = {}
            for it in mod_prog.items:
                if isinstance(it, Import) and it.module not in KNOWN_MODULES:
                    self._load_user_module(it.module)
                elif isinstance(it, StructDecl):
                    ctor = self._make_struct_ctor(it)
                    exports[it.name] = ctor
                    if it.name not in self.globals:
                        self.globals[it.name] = ctor
                elif isinstance(it, Function) and it.name != "main":
                    exports[it.name] = it
                    if it.name not in self.globals:
                        self.globals[it.name] = it
                elif isinstance(it, ConstDecl):
                    val = self.eval_expr(it.value)
                    exports[it.name] = val
                    if it.name not in self.globals:
                        self.globals[it.name] = val
                        self.const_globals.add(it.name)
            self.globals[mod_name] = NatUserModule(mod_name, exports)
        finally:
            self._loading_modules.remove(mod_name)

    def run(self, program: Program):
        self.const_globals.clear()
        for it in program.items:
            if isinstance(it, Import) and it.module not in KNOWN_MODULES:
                self._load_user_module(it.module)
            elif isinstance(it, StructDecl):
                self.globals[it.name] = self._make_struct_ctor(it)
            elif isinstance(it, Function):
                self.globals[it.name] = it
            elif isinstance(it, ConstDecl):
                val = self.eval_expr(it.value)
                self.globals[it.name] = val
                self.const_globals.add(it.name)
        if "main" in self.globals and isinstance(self.globals["main"], Function):
            return self.call_function(self.globals["main"], [])
        return None

    def call_function(self, fn: Function, args: list[Any]) -> Any:
        if len(args) != len(fn.params):
            raise self._runtime_error(
                f"function '{fn.name}' expects {len(fn.params)} argument(s), but received {len(args)}",
                fn,
            )
        local_scope = {}
        for p, arg in zip(fn.params, args):
            local_scope[p.name] = arg
        base = len(self.scopes)
        self.scopes.append(local_scope)
        self.const_scopes.append(set())
        self.frame_bases.append(base)
        try:
            for stmt in fn.body:
                self.eval_stmt(stmt)
        except ReturnSignal as ret:
            return ret.value
        except ErrPropagate as prop:
            if is_result_type_expr(fn.ret):
                return prop.result          # propagation: return Err to caller
            raise self._runtime_error(
                f"panic: error propagated by '?' in '{fn.name}' (which does not return Result): "
                f"{prop.result.error}",
                fn
            ) from None
        finally:
            self.frame_bases.pop()
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
        prev_node = self.cur_node
        self.cur_node = s
        try:
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
            elif isinstance(s, TupleAssign):
                self.eval_tuple_assign(s)
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
        except (ReturnSignal, BreakSignal, ContinueSignal, ErrPropagate, NatRuntimeError):
            raise
        except RuntimeError as exc:
            raise self._runtime_error(str(exc), s) from exc
        except ZeroDivisionError as exc:
            raise self._runtime_error("Division by zero", s) from exc
        except IndexError as exc:
            raise self._runtime_error(f"Index out of bounds: {exc}", s) from exc
        finally:
            self.cur_node = prev_node

    def _assign_single_target(self, tgt: Expr, op: str, val: Any):
        if isinstance(tgt, Name):
            name = tgt.id
            base = self.frame_bases[-1]
            target_scope = None
            for sc, c_sc in zip(reversed(self.scopes[base:]), reversed(self.const_scopes[base:])):
                if name in sc:
                    if name in c_sc:
                        raise self._runtime_error(f"Cannot reassign constant '{name}'", tgt)
                    target_scope = sc
                    break
            else:
                if name in self.const_globals:
                    raise self._runtime_error(f"Cannot reassign constant '{name}'", tgt)

            if target_scope is None:
                target_scope = self.scopes[-1]

            if op == "=":
                target_scope[name] = val
            else:
                if name not in target_scope:
                    raise self._runtime_error(f"Undefined variable: {name}", tgt)
                base_op = op[:-1]
                target_scope[name] = self.eval_binary_op(base_op, target_scope[name], val)
        elif isinstance(tgt, Index):
            obj = self.eval_expr(tgt.obj)
            idx = self.eval_index(tgt.index)
            if op == "=":
                obj[idx] = val
            else:
                base_op = op[:-1]
                obj[idx] = self.eval_binary_op(base_op, obj[idx], val)
        elif isinstance(tgt, Attr):
            obj = self.eval_expr(tgt.obj)
            try:
                if op == "=":
                    setattr(obj, tgt.name, val)
                else:
                    base_op = op[:-1]
                    cur = getattr(obj, tgt.name)
                    setattr(obj, tgt.name, self.eval_binary_op(base_op, cur, val))
            except AttributeError as ae:
                raise self._runtime_error(str(ae), tgt) from ae
            except RuntimeError as re:
                raise self._runtime_error(str(re), tgt) from re
        else:
            raise self._runtime_error(f"Invalid assignment target: {tgt}", tgt)

    def eval_assign(self, node: Assign):
        val = self.eval_expr(node.value)
        self._assign_single_target(node.target, node.op, val)

    def eval_tuple_assign(self, node: TupleAssign):
        val = self.eval_expr(node.value)
        try:
            items = list(val)
        except TypeError:
            raise self._runtime_error(f"Value is not iterable for unpacking: {val!r}", node) from None
        if len(items) != len(node.targets):
            raise self._runtime_error(
                f"Unpacking expects {len(node.targets)} values, but received {len(items)}",
                node
            )
        for tgt, item_val in zip(node.targets, items):
            self._assign_single_target(tgt, "=", item_val)

    def eval_index(self, node: Expr):
        if isinstance(node, Range):
            start = self.eval_expr(node.start) if node.start is not None else None
            end = self.eval_expr(node.end) if node.end is not None else None
            if end is not None and node.inclusive:
                end += 1
            return slice(start, end)
        return self.eval_expr(node)

    def eval_expr(self, e: Expr) -> Any:
        prev_node = self.cur_node
        self.cur_node = e
        try:
            if isinstance(e, Int) or isinstance(e, Float) or isinstance(e, Str) or isinstance(e, Bool):
                return e.value
            if isinstance(e, Name):
                base = self.frame_bases[-1]
                for sc in reversed(self.scopes[base:]):
                    if e.id in sc:
                        return sc[e.id]
                if e.id in self.globals:
                    return self.globals[e.id]
                raise self._runtime_error(f"Undefined variable: {e.id}", e)
            if isinstance(e, TupleLit):
                return tuple(self.eval_expr(it) for it in e.items)
            if isinstance(e, DictLit):
                return {self.eval_expr(k): self.eval_expr(v) for k, v in e.pairs}
            if isinstance(e, ListLit):
                return [self.eval_expr(it) for it in e.items]
            if isinstance(e, ListComp):
                iterable = self.eval_expr(e.iter)
                res = []
                self.scopes.append({})
                try:
                    for item in iterable:
                        self.scopes[-1][e.var] = item
                        if e.cond is None or self.eval_expr(e.cond):
                            res.append(self.eval_expr(e.elt))
                finally:
                    self.scopes.pop()
                return res
            if isinstance(e, Lambda):
                base = self.frame_bases[-1]
                captured = [dict(sc) for sc in self.scopes[base:]]
                return NatLambda(e, captured, self)
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
                try:
                    return getattr(obj, e.name)
                except AttributeError as ae:
                    raise self._runtime_error(str(ae), e) from ae
            if isinstance(e, TryUnwrap):
                val = self.eval_expr(e.expr)
                if isinstance(val, NatResult):
                    if not val.is_ok:
                        raise ErrPropagate(val)
                    return val.value
                return val
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
                fn = self.eval_expr(e.func)
                args = [self.eval_expr(a) for a in e.args]
                if isinstance(fn, Function):
                    return self.call_function(fn, args)
                if callable(fn):
                    return fn(*args)
                raise self._runtime_error(f"'{fn}' is not callable", e)
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
            raise self._runtime_error(f"Unknown expression: {e}", e)
        except (ReturnSignal, BreakSignal, ContinueSignal, ErrPropagate, NatRuntimeError):
            raise
        except RuntimeError as exc:
            raise self._runtime_error(str(exc), e) from exc
        except ZeroDivisionError as exc:
            raise self._runtime_error("Division by zero", e) from exc
        except IndexError as exc:
            raise self._runtime_error(f"Index out of bounds: {exc}", e) from exc
        finally:
            self.cur_node = prev_node

    def eval_unary_op(self, op: str, operand: Any) -> Any:
        if op == "not":
            return not operand
        if op == "-":
            if isinstance(operand, list):
                return (-np.array(operand)).tolist()
            return -operand
        if op == "+":
            return operand
        raise self._runtime_error(f"Unknown unary operator: {op}")

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
            else: raise self._runtime_error(f"Unknown operator: {op}")

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
        raise self._runtime_error(f"Unsupported operator: {op}")


# ============================================================================
# C99 AHEAD-OF-TIME (AOT) TRANSPILER & NATIVE BACKEND
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
            "fix256": "nat_fix256_t", "Fix256": "nat_fix256_t", "NatFix256": "nat_fix256_t",
            "interval256": "nat_interval256_t", "Interval256": "nat_interval256_t", "NatInterval256": "nat_interval256_t",
            "c64": "NatComplex", "complex": "NatComplex", "Complex": "NatComplex",
            "Result": "NatResult", "NatResult": "NatResult",
            "KVCache": "NatKVCache", "NatKVCache": "NatKVCache",
            "bool": "bool", "str": "const char*", "void": "void"
        }
        self.struct_decls: dict[str, StructDecl] = {}
        self.struct_field_types: dict[tuple[str, str], str] = {}
        self.struct_field_shapes: dict[tuple[str, str], tuple[int, ...]] = {}
        self.var_types: dict[str, str] = {}
        self.var_lengths: dict[str, int] = {}
        self.var_shapes: dict[str, tuple[int, ...]] = {}
        self.var_result_elem: dict[str, str] = {}
        self.kvcache_dims: dict[str, tuple[int, int]] = {}
        self.lambda_vars: dict[str, Lambda] = {}
        self.declared_vars: set[str] = set()
        self.func_signatures: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        self.func_result_elem: dict[str, str] = {}
        self.func_param_shapes: dict[tuple[str, str], tuple[int, ...]] = {}
        self.func_ret_shapes: dict[str, tuple[int, ...]] = {}
        self.cur_fn_returns_result: bool = False
        self.cur_fn_ret_type: str = "void"
        self.pending_pre_stmts: list[str] = []
        self._temp_counter: int = 0
        self._collect_signatures()

    def _new_temp(self, prefix: str = "_tmp") -> str:
        self._temp_counter += 1
        return f"{prefix}_{self._temp_counter}"

    def _subst_expr(self, expr: Expr, mapping: dict[str, Expr]) -> Expr:
        if not mapping:
            return expr
        if isinstance(expr, (Int, Float, Str, Bool)):
            return expr
        if isinstance(expr, Name):
            return mapping.get(expr.id, expr)
        if isinstance(expr, Unary):
            return Unary(expr.op, self._subst_expr(expr.operand, mapping))
        if isinstance(expr, Binary):
            return Binary(expr.op, self._subst_expr(expr.left, mapping), self._subst_expr(expr.right, mapping))
        if isinstance(expr, Call):
            fn = self._subst_expr(expr.func, mapping)
            args = [self._subst_expr(a, mapping) for a in expr.args]
            new_call = Call(fn, args)
            if getattr(expr, "keywords", None):
                new_call.keywords = [(k, self._subst_expr(v, mapping)) for k, v in expr.keywords]
            return new_call
        if isinstance(expr, Index):
            return Index(self._subst_expr(expr.obj, mapping), self._subst_expr(expr.index, mapping))
        if isinstance(expr, Attr):
            return Attr(self._subst_expr(expr.obj, mapping), expr.name)
        if isinstance(expr, Range):
            return Range(
                self._subst_expr(expr.start, mapping) if expr.start else None,
                self._subst_expr(expr.end, mapping) if expr.end else None,
                expr.inclusive,
            )
        if isinstance(expr, ListLit):
            return ListLit([self._subst_expr(it, mapping) for it in expr.items])
        if isinstance(expr, TupleLit):
            return TupleLit([self._subst_expr(it, mapping) for it in expr.items])
        if isinstance(expr, DictLit):
            return DictLit([(self._subst_expr(k, mapping), self._subst_expr(v, mapping)) for k, v in expr.pairs])
        if isinstance(expr, ListComp):
            sub_map = {k: v for k, v in mapping.items() if k != expr.var}
            return ListComp(
                self._subst_expr(expr.elt, sub_map),
                expr.var,
                self._subst_expr(expr.iter, mapping),
                self._subst_expr(expr.cond, sub_map) if expr.cond else None,
            )
        if isinstance(expr, Lambda):
            pnames = {p.name for p in expr.params}
            sub_map = {k: v for k, v in mapping.items() if k not in pnames}
            return Lambda(expr.params, expr.ret, self._subst_expr(expr.body, sub_map))
        if isinstance(expr, TryUnwrap):
            return TryUnwrap(self._subst_expr(expr.expr, mapping))
        if isinstance(expr, FString):
            return FString([FormatPart(self._subst_expr(p.expr, mapping), p.spec) if isinstance(p, FormatPart) else p for p in expr.parts])
        return expr

    def _resolve_lambda_call(self, c: Call) -> Optional[Expr]:
        lam = None
        if isinstance(c.func, Lambda):
            lam = c.func
        elif isinstance(c.func, Name) and c.func.id in self.lambda_vars:
            lam = self.lambda_vars[c.func.id]
        if lam is not None:
            mapping = {p.name: arg for p, arg in zip(lam.params, c.args)}
            return self._subst_expr(lam.body, mapping)
        return None

    def _desugar_hof(self, e: Expr) -> Expr:
        if isinstance(e, Call) and isinstance(e.func, Name) and e.func.id in ("map", "filter") and len(e.args) == 2:
            fn_arg, iter_arg = e.args[0], e.args[1]
            var_name = "_m_x"
            if isinstance(fn_arg, Lambda) and fn_arg.params:
                var_name = fn_arg.params[0].name
                body_expr = fn_arg.body
            elif isinstance(fn_arg, Name) and fn_arg.id in self.lambda_vars and self.lambda_vars[fn_arg.id].params:
                lam = self.lambda_vars[fn_arg.id]
                var_name = lam.params[0].name
                body_expr = lam.body
            else:
                body_expr = Call(fn_arg, [Name(var_name)])
            if e.func.id == "map":
                return ListComp(elt=body_expr, var=var_name, iter=iter_arg, cond=None)
            return ListComp(elt=Name(var_name), var=var_name, iter=iter_arg, cond=body_expr)
        return e

    def _collect_signatures(self):
        for it in self.program.items:
            if isinstance(it, StructDecl):
                self.struct_decls[it.name] = it
                self.types_map[it.name] = it.name
        for it in self.program.items:
            if isinstance(it, StructDecl):
                params = []
                for f in it.fields:
                    ft_c = self.c_type(f.type)
                    params.append((f.name, ft_c))
                    self.struct_field_types[(it.name, f.name)] = ft_c
                self.func_signatures[it.name] = (it.name, params)
            elif isinstance(it, Function):
                ret_t = self.c_type(it.ret) if it.ret else "void"
                params = [(p.name, self.c_type(p.type)) for p in it.params]
                self.func_signatures[it.name] = (ret_t, params)
                if isinstance(it.ret, TypeResult):
                    self.func_result_elem[it.name] = self.c_type(it.ret.elem)
                elif isinstance(it.ret, ResultT):
                    self.func_result_elem[it.name] = self.c_type(it.ret.elem)

    def indent(self) -> str:
        return "    " * self.indent_level

    def c_type(self, t: Any) -> str:
        if isinstance(t, TypeName):
            return self.types_map.get(t.name, "double")
        if isinstance(t, Prim):
            return self.types_map.get(t.name, "double")
        if isinstance(t, StructT):
            return t.name
        if isinstance(t, TypeResult):
            return "NatResult"
        if isinstance(t, ResultT):
            return "NatResult"
        if isinstance(t, (TypeList, TypeTensor)):
            elem_t = self.c_type(t.elem)
            return f"{elem_t}*"
        if isinstance(t, (ListT, TensorT)):
            elem_t = self.c_type(t.elem)
            return f"{elem_t}*"
        return "double"

    def _infer_result_elem(self, expr: Expr) -> str:
        if isinstance(expr, Name) and expr.id in self.var_result_elem:
            return self.var_result_elem[expr.id]
        if isinstance(expr, Call) and isinstance(expr.func, Name):
            fn = expr.func.id
            if fn == "read_file_safe":
                return "const char*"
            if fn == "ok" and expr.args:
                return self.infer_type(expr.args[0])
            if fn in self.func_result_elem:
                return self.func_result_elem[fn]
        return "const char*"

    def _expr_shape(self, e: Expr) -> Optional[tuple[int, ...]]:
        if isinstance(e, Name):
            if e.id in self.var_shapes:
                return self.var_shapes[e.id]
            if e.id in self.var_lengths:
                return (self.var_lengths[e.id],)
            return None
        if isinstance(e, Attr) and isinstance(e.obj, Name):
            key = f"{e.obj.id}.{e.name}"
            if key in self.var_shapes:
                return self.var_shapes[key]
            obj_t = self.var_types.get(e.obj.id, "")
            if (obj_t, e.name) in self.struct_field_shapes:
                return self.struct_field_shapes[(obj_t, e.name)]
            if obj_t == "NatKVCache" and e.obj.id in self.kvcache_dims and e.name in ("k", "v"):
                return self.kvcache_dims[e.obj.id]
            return None
        if isinstance(e, Attr) and isinstance(e.obj, Index) and isinstance(e.obj.obj, Name):
            arr_t = self.var_types.get(e.obj.obj.id, "")
            if arr_t.endswith("*"):
                elem_t = arr_t[:-1]
                if (elem_t, e.name) in self.struct_field_shapes:
                    return self.struct_field_shapes[(elem_t, e.name)]
                if elem_t == "NatKVCache" and e.obj.obj.id in self.kvcache_dims and e.name in ("k", "v"):
                    return self.kvcache_dims[e.obj.obj.id]
        if isinstance(e, ListLit):
            if e.items and isinstance(e.items[0], ListLit):
                return (len(e.items), len(e.items[0].items))
            return (len(e.items),)
        if isinstance(e, ListComp):
            if e.cond is None and isinstance(e.iter, Range):
                r = e.iter
                s_val = r.start.value if isinstance(r.start, Int) and r.start.value is not None else 0
                if isinstance(r.end, Int) and r.end.value is not None:
                    n = (r.end.value - s_val) + (1 if r.inclusive else 0)
                    return (max(n, 0),)
            elif e.cond is None:
                it_len = self._expr_length(e.iter)
                if it_len is not None:
                    return (it_len,)
        if isinstance(e, Index):
            if isinstance(e.index, Range):
                r = e.index
                if isinstance(r.start, Int) and isinstance(r.end, Int) and r.start.value is not None and r.end.value is not None:
                    return (r.end.value - r.start.value,)
                if r.start is None and isinstance(r.end, Int) and r.end.value is not None:
                    return (r.end.value,)
                return None
            obj_sh = self._expr_shape(e.obj)
            if obj_sh and len(obj_sh) == 2:
                return (obj_sh[1],)
            return None
        if isinstance(e, Binary):
            if e.op == "@":
                l_sh = self._expr_shape(e.left)
                r_sh = self._expr_shape(e.right)
                if l_sh and r_sh:
                    if len(l_sh) == 2 and len(r_sh) == 1:
                        return (l_sh[0],)
                    if len(l_sh) == 2 and len(r_sh) == 2:
                        return (l_sh[0], r_sh[1])
                    if len(l_sh) == 1 and len(r_sh) == 2:
                        return (r_sh[1],)
                return None
            if e.op in ("+", "-", "*", "/"):
                l_sh = self._expr_shape(e.left)
                r_sh = self._expr_shape(e.right)
                return l_sh or r_sh
        if isinstance(e, Call):
            resolved = self._resolve_lambda_call(e)
            if resolved is not None:
                return self._expr_shape(resolved)
            desugared = self._desugar_hof(e)
            if desugared is not e:
                return self._expr_shape(desugared)
            fn = ""
            if isinstance(e.func, Name):
                fn = e.func.id
            elif isinstance(e.func, Attr) and isinstance(e.func.obj, Name) and e.func.obj.id == "math":
                fn = e.func.name
            if fn in ("rmsnorm", "silu", "relu", "sigmoid", "softmax", "apply_rope") and e.args:
                return self._expr_shape(e.args[0])
            if fn == "fill" and e.args and isinstance(e.args[0], Int) and e.args[0].value is not None:
                return (e.args[0].value,)
            if fn in ("zeros", "ones", "random_ternary") and e.args:
                d_arg = e.args[0]
                if isinstance(d_arg, ListLit) and len(d_arg.items) == 2 and isinstance(d_arg.items[0], Int) and isinstance(d_arg.items[1], Int):
                    return (d_arg.items[0].value, d_arg.items[1].value)
                if isinstance(d_arg, ListLit) and len(d_arg.items) == 1 and isinstance(d_arg.items[0], Int):
                    return (d_arg.items[0].value,)
                if isinstance(d_arg, Int) and d_arg.value is not None:
                    return (d_arg.value,)
            if fn == "matmul" and len(e.args) >= 2:
                return self._expr_shape(Binary(e.args[0], "@", e.args[1]))
            if fn in self.func_ret_shapes:
                return self.func_ret_shapes[fn]
        return None

    def _expr_length(self, e: Expr) -> Optional[int]:
        if isinstance(e, Name) and e.id in self.var_lengths:
            return self.var_lengths[e.id]
        sh = self._expr_shape(e)
        if sh:
            prod = 1
            for d in sh:
                prod *= d
            return prod
        return None

    def infer_type(self, expr: Expr) -> str:
        if isinstance(expr, Int):
            return "int64_t"
        if isinstance(expr, Float):
            return "double"
        if isinstance(expr, Bool):
            return "bool"
        if isinstance(expr, Str):
            return "const char*"
        if isinstance(expr, Attr):
            if expr.name in ("real", "imag", "magnitude", "power"):
                return "double"
            obj_inferred = self.var_types.get(expr.obj.id, "") if isinstance(expr.obj, Name) else self.infer_type(expr.obj)
            if obj_inferred in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                if expr.name in ("lo", "hi"):
                    return "nat_fix256_t"
            if obj_inferred in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                if expr.name in ("sign", "frac_bits"):
                    return "int32_t"
            if isinstance(expr.obj, Name):
                key = f"{expr.obj.id}.{expr.name}"
                if key in self.var_types:
                    return self.var_types[key]
                obj_t = self.var_types.get(expr.obj.id, "")
                if (obj_t, expr.name) in self.struct_field_types:
                    return self.struct_field_types[(obj_t, expr.name)]
                if obj_t == "NatKVCache":
                    if expr.name in ("k", "v"):
                        return "double*"
                    if expr.name in ("seq_len", "max_seq_len", "dim", "capacity"):
                        return "int64_t"
            if isinstance(expr.obj, Index) and isinstance(expr.obj.obj, Name):
                arr_t = self.var_types.get(expr.obj.obj.id, "")
                if arr_t.endswith("*"):
                    elem_t = arr_t[:-1]
                    if (elem_t, expr.name) in self.struct_field_types:
                        return self.struct_field_types[(elem_t, expr.name)]
                    if elem_t == "NatKVCache":
                        if expr.name in ("k", "v"):
                            return "double*"
                        if expr.name in ("seq_len", "max_seq_len", "dim", "capacity"):
                            return "int64_t"
            return "double"
        if isinstance(expr, Index):
            obj_t = self.infer_type(expr.obj)
            sh = self._expr_shape(expr.obj)
            if sh and len(sh) == 2 and not isinstance(expr.index, Range):
                return obj_t if obj_t.endswith("*") else "double*"
            if isinstance(expr.index, Range):
                return obj_t if obj_t.endswith("*") else "double*"
            if obj_t.endswith("*"):
                return obj_t[:-1]
            return "double"
        if isinstance(expr, ListLit):
            if not expr.items:
                return "double*"
            it = self.infer_type(expr.items[0])
            if it in self.struct_decls or it in ("NatKVCache", "NatComplex", "const char*"):
                return f"{it}*"
            if "int" in it:
                return "int64_t*"
            return "double*"
        if isinstance(expr, ListComp):
            old_v = self.var_types.get(expr.var)
            if isinstance(expr.iter, Range):
                self.var_types[expr.var] = "int64_t"
            else:
                it_iter = self.infer_type(expr.iter)
                self.var_types[expr.var] = it_iter[:-1] if it_iter.endswith("*") else "double"
            it = self.infer_type(expr.elt)
            if old_v is not None:
                self.var_types[expr.var] = old_v
            else:
                self.var_types.pop(expr.var, None)
            if it in self.struct_decls or it in ("NatKVCache", "NatComplex", "const char*"):
                return f"{it}*"
            if "int" in it:
                return "int64_t*"
            return "double*"
        if isinstance(expr, Name):
            return self.var_types.get(expr.id, "double")
        if isinstance(expr, Binary):
            if expr.op in ("<", "<=", ">", ">=", "==", "!=", "and", "or"):
                return "bool"
            if expr.op == "@":
                l_sh = self._expr_shape(expr.left)
                r_sh = self._expr_shape(expr.right)
                if l_sh and r_sh and len(l_sh) == 1 and len(r_sh) == 1:
                    return "double"
                return "double*"
            if self._expr_length(expr) is not None and expr.op in ("+", "-", "*", "/"):
                return "double*"
            if expr.op == "/":
                return "double"
            t_left = self.infer_type(expr.left)
            t_right = self.infer_type(expr.right)
            if t_left == "NatComplex" or t_right == "NatComplex":
                return "NatComplex"
            if t_left in ("nat_fix256_t", "NatFix256") or t_right in ("nat_fix256_t", "NatFix256"):
                return "nat_fix256_t"
            if t_left in ("nat_interval256_t", "NatInterval256") or t_right in ("nat_interval256_t", "NatInterval256"):
                return "nat_interval256_t"
            if "double" in (t_left, t_right) or "float" in (t_left, t_right):
                return "double"
            return t_left
        if isinstance(expr, TryUnwrap):
            inner_t = self.infer_type(expr.expr)
            if inner_t == "NatResult":
                return self._infer_result_elem(expr.expr)
            return inner_t
        if isinstance(expr, Unary):
            if expr.op == "not":
                return "bool"
            return self.infer_type(expr.operand)
        if isinstance(expr, Call):
            resolved = self._resolve_lambda_call(expr)
            if resolved is not None:
                return self.infer_type(resolved)
            desugared = self._desugar_hof(expr)
            if desugared is not expr:
                return self.infer_type(desugared)
            if isinstance(expr.func, Attr) and not (isinstance(expr.func.obj, Name) and expr.func.obj.id == "math"):
                obj_t = self.infer_type(expr.func.obj)
                mname = expr.func.name
                if obj_t in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                    if mname in ("step", "simulate_steps"):
                        return "nat_interval256_t"
                    if mname in ("width", "mid"):
                        return "nat_fix256_t"
                    if mname in ("is_coherent", "contains"):
                        return "bool"
                if obj_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                    if mname == "to_f64":
                        return "double"
                    if mname == "to_i64":
                        return "int64_t"
                    if mname == "to_str":
                        return "const char*"

            fname = ""
            if isinstance(expr.func, Name):
                fname = expr.func.id
            elif isinstance(expr.func, Attr) and isinstance(expr.func.obj, Name) and expr.func.obj.id == "math":
                fname = expr.func.name

            if fname:
                if fname in self.struct_decls:
                    return fname
                if fname == "KVCache":
                    return "NatKVCache"
                if fname == "Complex":
                    return "NatComplex"
                if fname in ("Fix256", "fix256", "interval256_width", "interval256_mid"):
                    return "nat_fix256_t"
                if fname in ("Interval256", "interval256", "interval256_step"):
                    return "nat_interval256_t"
                if fname == "interval256_is_coherent":
                    return "bool"
                if fname in ("rmsnorm", "softmax", "apply_rope"):
                    return "double*"
                if fname in ("silu", "relu", "sigmoid") and expr.args and self._expr_length(expr.args[0]) is not None:
                    return "double*"
                if fname in ("sqrt", "exp", "sin", "cos", "ln", "to_f64", "tanh", "mean", "sum", "dot", "min", "max", "silu", "relu", "sigmoid"):
                    return "double"
                if fname in ("floor", "to_i64", "len", "argmax"):
                    return "int64_t"
                if fname == "read_file" or fname == "error_message":
                    return "const char*"
                if fname == "unwrap" and expr.args:
                    return self._infer_result_elem(expr.args[0])
                if fname == "unwrap_or" and len(expr.args) > 1:
                    return self.infer_type(expr.args[1])
                if fname in ("is_ok", "is_err", "exists_file"):
                    return "bool"
                if fname in ("read_file_safe", "ok", "err"):
                    return "NatResult"
                if fname == "fill":
                    v_t = self.infer_type(expr.args[1]) if len(expr.args) > 1 else "double"
                    return "int64_t*" if "int" in v_t else "double*"
                if fname in ("zeros", "ones"):
                    return "double*"
                if fname == "random_ternary":
                    return "int8_t*"
                if fname in self.func_signatures:
                    return self.func_signatures[fname][0]
        return "double"

    def _prescan_shapes(self):
        """Propagates static tensor shapes to struct fields and function parameters."""
        all_stmts: list[Node] = []
        fn_map: dict[str, Function] = {}
        for it in self.program.items:
            if isinstance(it, Function):
                fn_map[it.name] = it
                all_stmts.extend(it.body)

        # Run 2 passes to propagate shapes from main -> structs/functions -> returns
        for _ in range(2):
            for fn_name in (["main"] + [k for k in fn_map if k != "main"]):
                if fn_name not in fn_map:
                    continue
                fn = fn_map[fn_name]
                for p in fn.params:
                    p_t = self.c_type(p.type)
                    self.var_types[p.name] = p_t
                    if (fn.name, p.name) in self.func_param_shapes:
                        sh = self.func_param_shapes[(fn.name, p.name)]
                        self.var_shapes[p.name] = sh
                        prod = 1
                        for d in sh:
                            prod *= d
                        self.var_lengths[p.name] = prod
                self._prescan_block(fn.body, fn.name)

        self.var_types.clear()
        self.var_lengths.clear()
        self.var_shapes.clear()
        self.kvcache_dims.clear()
        self.lambda_vars.clear()

    def _prescan_block(self, stmts: list[Node], cur_fn: str):
        for s in stmts:
            if isinstance(s, (VarDecl, ConstDecl)):
                self._prescan_assign(s.name, s.value)
            elif isinstance(s, Assign) and isinstance(s.target, Name) and s.op == "=":
                self._prescan_assign(s.target.id, s.value)
            elif isinstance(s, TupleAssign):
                if isinstance(s.value, Call) and isinstance(s.value.func, Attr) and s.value.func.name == "update":
                    obj = s.value.func.obj
                    dims = None
                    if isinstance(obj, Name) and obj.id in self.kvcache_dims:
                        dims = self.kvcache_dims[obj.id]
                    elif isinstance(obj, Index) and isinstance(obj.obj, Name) and obj.obj.id in self.kvcache_dims:
                        dims = self.kvcache_dims[obj.obj.id]
                    if dims is not None:
                        for t in s.targets:
                            if isinstance(t, Name):
                                self.var_types[t.id] = "double*"
                                self.var_shapes[t.id] = dims
                                self.var_lengths[t.id] = dims[0] * dims[1]
            elif isinstance(s, If):
                self._prescan_block(s.body, cur_fn)
                if s.orelse:
                    self._prescan_block(s.orelse, cur_fn)
            elif isinstance(s, (While, For)):
                if isinstance(s, For) and not isinstance(s.iter, Range):
                    it_t = self.infer_type(s.iter)
                    if it_t.endswith("*"):
                        self.var_types[s.var] = it_t[:-1]
                self._prescan_block(s.body, cur_fn)
            elif isinstance(s, Return) and s.value is not None:
                sh = self._expr_shape(s.value)
                if sh:
                    self.func_ret_shapes[cur_fn] = sh
            elif isinstance(s, ExprStmt):
                self._prescan_expr(s.expr)

    def _prescan_assign(self, name: str, val: Expr):
        if isinstance(val, Lambda):
            self.lambda_vars[name] = val
            return
        val = self._desugar_hof(val)
        self._prescan_expr(val)
        t = self.infer_type(val)
        self.var_types[name] = t
        sh = self._expr_shape(val)
        if sh:
            self.var_shapes[name] = sh
            prod = 1
            for d in sh:
                prod *= d
            self.var_lengths[name] = prod
        if isinstance(val, Call) and isinstance(val.func, Name):
            fn = val.func.id
            if fn == "KVCache" and len(val.args) >= 2:
                if isinstance(val.args[0], Int) and isinstance(val.args[1], Int):
                    ms, dm = val.args[0].value, val.args[1].value
                    self.kvcache_dims[name] = (ms, dm)
                    self.var_shapes[f"{name}.k"] = (ms, dm)
                    self.var_shapes[f"{name}.v"] = (ms, dm)
        elif isinstance(val, ListComp) and isinstance(val.elt, Call) and isinstance(val.elt.func, Name):
            if val.elt.func.id == "KVCache" and len(val.elt.args) >= 2:
                if isinstance(val.elt.args[0], Int) and isinstance(val.elt.args[1], Int):
                    ms, dm = val.elt.args[0].value, val.elt.args[1].value
                    self.kvcache_dims[name] = (ms, dm)

    def _prescan_expr(self, e: Expr):
        if isinstance(e, ListComp):
            self._prescan_expr(e.iter)
            old_v = self.var_types.get(e.var)
            if isinstance(e.iter, Range):
                self.var_types[e.var] = "int64_t"
            else:
                it_iter = self.infer_type(e.iter)
                self.var_types[e.var] = it_iter[:-1] if it_iter.endswith("*") else "double"
            self._prescan_expr(e.elt)
            if e.cond is not None:
                self._prescan_expr(e.cond)
            if old_v is not None:
                self.var_types[e.var] = old_v
            else:
                self.var_types.pop(e.var, None)
            return
        if isinstance(e, Call) and isinstance(e.func, Name):
            fn = e.func.id
            for a in e.args:
                self._prescan_expr(a)
            for _, ka in getattr(e, "keywords", []) or []:
                self._prescan_expr(ka)
            if fn in self.struct_decls:
                sdecl = self.struct_decls[fn]
                ordered_args = self._order_struct_args(sdecl, e)
                for f, arg_expr in zip(sdecl.fields, ordered_args):
                    sh = self._expr_shape(arg_expr)
                    if sh:
                        self.struct_field_shapes[(fn, f.name)] = sh
                    self.struct_field_types[(fn, f.name)] = self.infer_type(arg_expr)
            elif fn in self.func_signatures:
                _, params = self.func_signatures[fn]
                for (pname, _), arg_expr in zip(params, e.args):
                    sh = self._expr_shape(arg_expr)
                    if sh:
                        self.func_param_shapes[(fn, pname)] = sh
                    if isinstance(arg_expr, Name) and arg_expr.id in self.kvcache_dims:
                        self.kvcache_dims[pname] = self.kvcache_dims[arg_expr.id]
                    elif isinstance(arg_expr, Index) and isinstance(arg_expr.obj, Name) and arg_expr.obj.id in self.kvcache_dims:
                        self.kvcache_dims[pname] = self.kvcache_dims[arg_expr.obj.id]
        elif isinstance(e, Binary):
            self._prescan_expr(e.left)
            self._prescan_expr(e.right)

    def _order_struct_args(self, sdecl: StructDecl, c: Call) -> list[Expr]:
        ordered: list[Optional[Expr]] = [None] * len(sdecl.fields)
        for idx, a in enumerate(c.args):
            if idx < len(ordered):
                ordered[idx] = a
        field_idx = {f.name: i for i, f in enumerate(sdecl.fields)}
        for kname, kval in (getattr(c, "keywords", None) or []):
            if kname in field_idx:
                ordered[field_idx[kname]] = kval
        return [a if a is not None else Int(0) for a in ordered]

    def transpile(self) -> str:
        ensure_nativa_rt_header()
        self._prescan_shapes()
        out = []
        out.append("/* ========================================================================= */")
        out.append("/* Automatically generated by Nativa compiler (AOT C99 Transpiler)          */")
        out.append(NATIVA_RT_HEADER_C)
        out.append("")

        # Definições de Structs do usuário e construtores inline
        for it in self.program.items:
            if isinstance(it, StructDecl):
                out.append(f"typedef struct {{")
                for f in it.fields:
                    ft_c = self.c_type(f.type)
                    out.append(f"    {ft_c} {f.name};")
                out.append(f"}} {it.name};")
                params_str = ", ".join(f"{self.c_type(f.type)} {f.name}" for f in it.fields) if it.fields else "void"
                out.append(f"static inline {it.name} {it.name}_ctor({params_str}) {{")
                out.append(f"    {it.name} _s;")
                for f in it.fields:
                    out.append(f"    _s.{f.name} = {f.name};")
                out.append(f"    return _s;")
                out.append(f"}}")
                out.append("")

        # Constantes de nível superior
        for it in self.program.items:
            if isinstance(it, ConstDecl):
                if it.name in ("PI", "E"):
                    continue
                if isinstance(it.value, Lambda):
                    self.lambda_vars[it.name] = it.value
                    continue
                ctype = self.c_type(it.type) if it.type else self.infer_type(it.value)
                val = self.emit_expr(it.value)
                out.append(f"static const {ctype} {it.name} = {val};")
                self.var_types[it.name] = ctype
        out.append("")

        # Forward Declarations
        for it in self.program.items:
            if isinstance(it, Function) and it.name != "main":
                ret_t, params = self.func_signatures[it.name]
                p_str = ", ".join(f"{pt} {pn}" for pn, pt in params) if params else "void"
                out.append(f"{ret_t} {it.name}({p_str});")
        out.append("")

        # Function Definitions
        for it in self.program.items:
            if isinstance(it, Function):
                out.append(self.emit_function(it))
                out.append("")

        return "\n".join(out)

    def emit_function(self, fn: Function) -> str:
        lines = []
        is_main = (fn.name == "main")
        self.cur_fn_returns_result = (not is_main) and is_result_type_expr(fn.ret)
        ret_t = "int" if is_main else (self.c_type(fn.ret) if fn.ret else "void")
        self.cur_fn_ret_type = ret_t
        p_str = "int argc, char** argv" if is_main else (", ".join(f"{self.c_type(p.type)} {p.name}" for p in fn.params) if fn.params else "void")

        old_vars = dict(self.var_types)
        old_lens = dict(self.var_lengths)
        old_shapes = dict(self.var_shapes)
        old_res_elem = dict(self.var_result_elem)
        old_lambdas = dict(self.lambda_vars)
        old_declared = set(self.declared_vars)
        self.declared_vars = set()

        for p in fn.params:
            p_c = self.c_type(p.type)
            self.var_types[p.name] = p_c
            self.declared_vars.add(p.name)
            if isinstance(p.type, TypeResult):
                self.var_result_elem[p.name] = self.c_type(p.type.elem)
            if (fn.name, p.name) in self.func_param_shapes:
                sh = self.func_param_shapes[(fn.name, p.name)]
                self.var_shapes[p.name] = sh
                prod = 1
                for d in sh:
                    prod *= d
                self.var_lengths[p.name] = prod

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
        self.var_lengths = old_lens
        self.var_shapes = old_shapes
        self.var_result_elem = old_res_elem
        self.lambda_vars = old_lambdas
        self.declared_vars = old_declared
        return "\n".join(lines)

    def _ensure_vec_operand(self, expr: Expr, ind: str) -> tuple[str, str]:
        if isinstance(expr, (Name, Attr)):
            return "", self.emit_expr(expr)
        if self._expr_length(expr) is not None:
            tmp = self._new_temp("_vec")
            pre = self._emit_buffer_init(tmp, expr, ind, is_const=False)
            if pre is not None:
                return pre + "\n", tmp
        return "", self.emit_expr(expr)

    def _wrap_buffer_target(self, name: str, elem_t: str, total: int, shape: tuple[int, ...], const_p: str, ind: str, body_builder) -> str:
        already_decl = (name in self.declared_vars) and (self.var_lengths.get(name) == total)
        self.var_types[name] = f"{elem_t}*"
        self.var_lengths[name] = total
        self.var_shapes[name] = shape
        if already_decl:
            tmp = self._new_temp("_buf")
            stmt = body_builder(tmp, "")
            return f"{stmt} memcpy({name}, {tmp}, {total} * sizeof({elem_t}));"
        else:
            self.declared_vars.add(name)
            return body_builder(name, const_p)

    def _emit_buffer_init(self, name: str, val_expr: Expr, ind: str, is_const: bool = False) -> Optional[str]:
        const_p = "const " if is_const else ""
        if isinstance(val_expr, Call):
            resolved = self._resolve_lambda_call(val_expr)
            if resolved is not None:
                val_expr = resolved
            else:
                val_expr = self._desugar_hof(val_expr)

        # 0. KVCache(max_seq, dim)
        if isinstance(val_expr, Call) and isinstance(val_expr.func, Name) and val_expr.func.id == "KVCache":
            ms_arg = val_expr.args[0] if len(val_expr.args) > 0 else Int(64)
            dm_arg = val_expr.args[1] if len(val_expr.args) > 1 else Int(4)
            ms_code = self.emit_expr(ms_arg)
            dm_code = self.emit_expr(dm_arg)
            if isinstance(ms_arg, Int) and isinstance(dm_arg, Int) and ms_arg.value is not None and dm_arg.value is not None:
                ms_v, dm_v = ms_arg.value, dm_arg.value
                self.kvcache_dims[name] = (ms_v, dm_v)
                self.var_shapes[f"{name}.k"] = (ms_v, dm_v)
                self.var_shapes[f"{name}.v"] = (ms_v, dm_v)
                self.var_lengths[f"{name}.k"] = ms_v * dm_v
                self.var_lengths[f"{name}.v"] = ms_v * dm_v
                self.var_types[f"{name}.k"] = "double*"
                self.var_types[f"{name}.v"] = "double*"
            self.var_types[name] = "NatKVCache"
            if name in self.declared_vars:
                return f"{ind}{name} = nat_kvcache_init({ms_code}, {dm_code});"
            self.declared_vars.add(name)
            return f"{ind}NatKVCache {name} = nat_kvcache_init({ms_code}, {dm_code});"

        # 1. fill(n, val)
        if isinstance(val_expr, Call) and isinstance(val_expr.func, Name) and val_expr.func.id == "fill":
            n_arg = val_expr.args[0]
            v_arg = val_expr.args[1]
            val_code = self.emit_expr(v_arg)
            val_t = self.infer_type(v_arg)
            elem_t = "int64_t" if "int" in val_t else "double"
            fill_fn = "nat_fill_i64" if "int" in val_t else "nat_fill_f64"
            if isinstance(n_arg, Int) and n_arg.value is not None:
                n_int = n_arg.value
                return self._wrap_buffer_target(
                    name, elem_t, n_int, (n_int,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}{elem_t} {tgt}[{n_int}]; {fill_fn}({tgt}, {n_int}, {val_code});"
                )
            else:
                n_code = self.emit_expr(n_arg)
                self.var_types[name] = f"{elem_t}*"
                self.declared_vars.add(name)
                return f"{ind}{const_p}{elem_t}* {name} = ({elem_t}*)malloc(({n_code}) * sizeof({elem_t})); if ({name}) {fill_fn}({name}, {n_code}, {val_code});"

        # 2. zeros(shape) ou ones(shape)
        if isinstance(val_expr, Call) and isinstance(val_expr.func, Name) and val_expr.func.id in ("zeros", "ones"):
            fn = val_expr.func.id
            d_arg = val_expr.args[0]
            if isinstance(d_arg, ListLit) and len(d_arg.items) == 2 and isinstance(d_arg.items[0], Int) and isinstance(d_arg.items[1], Int):
                M = d_arg.items[0].value
                K = d_arg.items[1].value
                total = M * K
                return self._wrap_buffer_target(
                    name, "double", total, (M, K), const_p, ind,
                    lambda tgt, cp: (
                        f"{ind}{cp}double {tgt}[{total}]; memset({tgt}, 0, sizeof({tgt}));"
                        if fn == "zeros" else
                        f"{ind}{cp}double {tgt}[{total}]; nat_ones_f64({tgt}, {total});"
                    )
                )
            elif isinstance(d_arg, ListLit) and len(d_arg.items) == 1 and isinstance(d_arg.items[0], Int):
                n_int = d_arg.items[0].value
                return self._wrap_buffer_target(
                    name, "double", n_int, (n_int,), const_p, ind,
                    lambda tgt, cp: (
                        f"{ind}{cp}double {tgt}[{n_int}]; memset({tgt}, 0, sizeof({tgt}));"
                        if fn == "zeros" else
                        f"{ind}{cp}double {tgt}[{n_int}]; nat_ones_f64({tgt}, {n_int});"
                    )
                )
            elif isinstance(d_arg, Int) and d_arg.value is not None:
                n_int = d_arg.value
                return self._wrap_buffer_target(
                    name, "double", n_int, (n_int,), const_p, ind,
                    lambda tgt, cp: (
                        f"{ind}{cp}double {tgt}[{n_int}]; memset({tgt}, 0, sizeof({tgt}));"
                        if fn == "zeros" else
                        f"{ind}{cp}double {tgt}[{n_int}]; nat_ones_f64({tgt}, {n_int});"
                    )
                )
            else:
                d_code = self.emit_expr(d_arg)
                self.var_types[name] = "double*"
                self.declared_vars.add(name)
                if fn == "zeros":
                    return f"{ind}{const_p}double* {name} = (double*)calloc({d_code}, sizeof(double));"
                else:
                    return f"{ind}{const_p}double* {name} = (double*)malloc(({d_code}) * sizeof(double)); if ({name}) nat_ones_f64({name}, {d_code});"

        # 3. random_ternary(shape, seed=None)
        if isinstance(val_expr, Call) and isinstance(val_expr.func, Name) and val_expr.func.id == "random_ternary":
            s_arg = val_expr.args[0]
            seed_code = self.emit_expr(val_expr.args[1]) if len(val_expr.args) > 1 else "12345LL"
            if isinstance(s_arg, ListLit) and len(s_arg.items) == 2 and isinstance(s_arg.items[0], Int) and isinstance(s_arg.items[1], Int):
                M = s_arg.items[0].value
                K = s_arg.items[1].value
                total = M * K
                return self._wrap_buffer_target(
                    name, "int8_t", total, (M, K), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}int8_t {tgt}[{total}]; nat_random_ternary({tgt}, {total}, (uint64_t)({seed_code}));"
                )
            elif isinstance(s_arg, ListLit) and len(s_arg.items) == 1 and isinstance(s_arg.items[0], Int):
                n_int = s_arg.items[0].value
                return self._wrap_buffer_target(
                    name, "int8_t", n_int, (n_int,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}int8_t {tgt}[{n_int}]; nat_random_ternary({tgt}, {n_int}, (uint64_t)({seed_code}));"
                )
            elif isinstance(s_arg, Int) and s_arg.value is not None:
                n_int = s_arg.value
                return self._wrap_buffer_target(
                    name, "int8_t", n_int, (n_int,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}int8_t {tgt}[{n_int}]; nat_random_ternary({tgt}, {n_int}, (uint64_t)({seed_code}));"
                )

        # 4. ListLit (1D e 2D, incluindo listas de structs e KVCaches)
        if isinstance(val_expr, ListLit):
            if val_expr.items and isinstance(val_expr.items[0], ListLit):
                M = len(val_expr.items)
                K = len(val_expr.items[0].items)
                total = M * K
                flat_items = []
                for row in val_expr.items:
                    if isinstance(row, ListLit):
                        flat_items.extend(row.items)
                it_t = self.infer_type(flat_items[0]) if flat_items else "double"
                elem_t = "int64_t" if "int" in it_t else "double"
                items_code = ", ".join(self.emit_expr(it) for it in flat_items)
                return self._wrap_buffer_target(
                    name, elem_t, total, (M, K), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}{elem_t} {tgt}[{total}] = {{ {items_code} }};"
                )
            else:
                n_elems = len(val_expr.items)
                it_t = self.infer_type(val_expr.items[0]) if val_expr.items else "double"
                if it_t in self.struct_decls or it_t in ("NatKVCache", "NatComplex", "const char*"):
                    elem_t = it_t
                elif "int" in it_t:
                    elem_t = "int64_t"
                else:
                    elem_t = "double"
                items_code = ", ".join(self.emit_expr(it) for it in val_expr.items)
                if val_expr.items and it_t == "NatKVCache" and isinstance(val_expr.items[0], Name) and val_expr.items[0].id in self.kvcache_dims:
                    self.kvcache_dims[name] = self.kvcache_dims[val_expr.items[0].id]
                return self._wrap_buffer_target(
                    name, elem_t, max(n_elems, 1), (n_elems,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}{elem_t} {tgt}[{max(n_elems, 1)}] = {{ {items_code} }};"
                )

        # 4b. ListComp ([elt for var in iter (if cond)])
        if isinstance(val_expr, ListComp):
            old_v = self.var_types.get(val_expr.var)
            if isinstance(val_expr.iter, Range):
                iter_elem_t = "int64_t"
            else:
                it_iter = self.infer_type(val_expr.iter)
                iter_elem_t = it_iter[:-1] if it_iter.endswith("*") else "double"
            self.var_types[val_expr.var] = iter_elem_t
            it_t = self.infer_type(val_expr.elt)
            if it_t in self.struct_decls or it_t in ("NatKVCache", "NatComplex", "const char*"):
                elem_t = it_t
            elif "int" in it_t:
                elem_t = "int64_t"
            else:
                elem_t = "double"
            if it_t == "NatKVCache" and isinstance(val_expr.elt, Call) and len(val_expr.elt.args) >= 2:
                if isinstance(val_expr.elt.args[0], Int) and isinstance(val_expr.elt.args[1], Int):
                    self.kvcache_dims[name] = (val_expr.elt.args[0].value, val_expr.elt.args[1].value)
            elt_code = self.emit_expr(val_expr.elt)
            cond_code = self.emit_expr(val_expr.cond) if val_expr.cond is not None else None
            if old_v is not None:
                self.var_types[val_expr.var] = old_v
            else:
                self.var_types.pop(val_expr.var, None)

            if isinstance(val_expr.iter, Range):
                r = val_expr.iter
                start_code = self.emit_expr(r.start) if r.start else "0LL"
                end_code = self.emit_expr(r.end) if r.end else "0LL"
                cmp_op = "<=" if r.inclusive else "<"
                s_val = r.start.value if isinstance(r.start, Int) and r.start.value is not None else 0
                n_elems = max((r.end.value - s_val) + (1 if r.inclusive else 0), 1) if isinstance(r.end, Int) and r.end.value is not None else 64
                if cond_code is None:
                    loop_tpl = f"for (int64_t {val_expr.var} = {start_code}; {val_expr.var} {cmp_op} {end_code}; {val_expr.var}++) {{ {{tgt}}[{val_expr.var} - ({start_code})] = {elt_code}; }}"
                else:
                    cnt_v = self._new_temp("_lc_n")
                    loop_tpl = f"int64_t {cnt_v} = 0; for (int64_t {val_expr.var} = {start_code}; {val_expr.var} {cmp_op} {end_code}; {val_expr.var}++) {{ if ({cond_code}) {{ {{tgt}}[{cnt_v}++] = {elt_code}; }} }}"
            else:
                iter_len = self._expr_length(val_expr.iter) or 64
                n_elems = max(iter_len, 1)
                pre_it, iter_code = self._ensure_vec_operand(val_expr.iter, ind)
                idx_v = self._new_temp("_lc_i")
                if cond_code is None:
                    loop_tpl = f"{pre_it}for (int64_t {idx_v} = 0; {idx_v} < {iter_len}; {idx_v}++) {{ {iter_elem_t} {val_expr.var} = ({iter_code})[{idx_v}]; {{tgt}}[{idx_v}] = {elt_code}; }}"
                else:
                    cnt_v = self._new_temp("_lc_n")
                    loop_tpl = f"{pre_it}int64_t {cnt_v} = 0; for (int64_t {idx_v} = 0; {idx_v} < {iter_len}; {idx_v}++) {{ {iter_elem_t} {val_expr.var} = ({iter_code})[{idx_v}]; if ({cond_code}) {{ {{tgt}}[{cnt_v}++] = {elt_code}; }} }}"

            return self._wrap_buffer_target(
                name, elem_t, n_elems, (n_elems,), "", ind,
                lambda tgt, cp: f"{ind}{elem_t} {tgt}[{n_elems}]; {loop_tpl.replace('{tgt}', tgt)}"
            )

        # 5. Fatiamento de array obj[start..end]
        if isinstance(val_expr, Index) and isinstance(val_expr.index, Range):
            arr_code = self.emit_expr(val_expr.obj)
            r = val_expr.index
            start_code = self.emit_expr(r.start) if r.start else "0"
            elem_t = self.infer_type(val_expr.obj)
            if not elem_t.endswith("*"):
                elem_t = "double*"
            slice_len = None
            if isinstance(r.start, Int) and isinstance(r.end, Int) and r.start.value is not None and r.end.value is not None:
                slice_len = r.end.value - r.start.value
            elif r.start is None and isinstance(r.end, Int) and r.end.value is not None:
                slice_len = r.end.value
            if slice_len is not None:
                self.var_lengths[name] = slice_len
                self.var_shapes[name] = (slice_len,)
            self.var_types[name] = elem_t
            self.declared_vars.add(name)
            return f"{ind}{const_p}{elem_t} {name} = ({arr_code}) + ({start_code});"

        # 6. Indexação de linha em Matriz 2D: row = A[i] (ou cache.k[i], c.wq[i])
        if isinstance(val_expr, Index) and not isinstance(val_expr.index, Range):
            sh = self._expr_shape(val_expr.obj)
            if sh and len(sh) == 2:
                _M, K = sh
                obj_code = self.emit_expr(val_expr.obj)
                idx_code = self.emit_expr(val_expr.index)
                obj_t = self.infer_type(val_expr.obj)
                base_t = obj_t[:-1] if obj_t.endswith("*") else "double"
                return self._wrap_buffer_target(
                    name, base_t, K, (K,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}{base_t} {tgt}[{K}]; memcpy({tgt}, ({obj_code}) + ({idx_code}) * {K}, {K} * sizeof({base_t}));"
                )

        # 6b. Primitivas vetoriais de IA: rmsnorm, silu, relu, sigmoid, softmax, apply_rope
        if isinstance(val_expr, Call):
            fn = ""
            if isinstance(val_expr.func, Name):
                fn = val_expr.func.id
            elif isinstance(val_expr.func, Attr) and isinstance(val_expr.func.obj, Name) and val_expr.func.obj.id == "math":
                fn = val_expr.func.name

            if fn == "rmsnorm" and len(val_expr.args) >= 2:
                x_len = self._expr_length(val_expr.args[0])
                x_sh = self._expr_shape(val_expr.args[0])
                if x_len is not None:
                    pre1, x_code = self._ensure_vec_operand(val_expr.args[0], ind)
                    pre2, g_code = self._ensure_vec_operand(val_expr.args[1], ind)
                    eps_code = self.emit_expr(val_expr.args[2]) if len(val_expr.args) > 2 else "1e-6"
                    stmt = self._wrap_buffer_target(
                        name, "double", x_len, x_sh or (x_len,), const_p, ind,
                        lambda tgt, cp: f"{ind}{cp}double {tgt}[{x_len}]; nat_rmsnorm_f64({x_code}, {g_code}, {tgt}, {x_len}LL, (double)({eps_code}));"
                    )
                    return f"{pre1}{pre2}{stmt}"

            if fn in ("silu", "relu", "sigmoid", "softmax") and len(val_expr.args) >= 1:
                x_len = self._expr_length(val_expr.args[0])
                x_sh = self._expr_shape(val_expr.args[0])
                if x_len is not None:
                    pre1, x_code = self._ensure_vec_operand(val_expr.args[0], ind)
                    c_fn = f"nat_{fn}_f64" if fn == "softmax" else f"nat_{fn}_vec_f64"
                    stmt = self._wrap_buffer_target(
                        name, "double", x_len, x_sh or (x_len,), const_p, ind,
                        lambda tgt, cp: f"{ind}{cp}double {tgt}[{x_len}]; {c_fn}({x_code}, {tgt}, {x_len}LL);"
                    )
                    return f"{pre1}{stmt}"

            if fn == "apply_rope" and len(val_expr.args) >= 2:
                x_len = self._expr_length(val_expr.args[0])
                x_sh = self._expr_shape(val_expr.args[0])
                if x_len is not None:
                    pre1, x_code = self._ensure_vec_operand(val_expr.args[0], ind)
                    pos_code = self.emit_expr(val_expr.args[1])
                    base_code = self.emit_expr(val_expr.args[2]) if len(val_expr.args) > 2 else "10000.0"
                    stmt = self._wrap_buffer_target(
                        name, "double", x_len, x_sh or (x_len,), const_p, ind,
                        lambda tgt, cp: f"{ind}{cp}double {tgt}[{x_len}]; nat_apply_rope_f64({x_code}, {tgt}, {x_len}LL, (int64_t)({pos_code}), (double)({base_code}));"
                    )
                    return f"{pre1}{stmt}"

            if fn in self.func_ret_shapes:
                sh = self.func_ret_shapes[fn]
                prod = 1
                for d in sh:
                    prod *= d
                call_code = self.emit_call(val_expr)
                return self._wrap_buffer_target(
                    name, "double", prod, sh, const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{prod}]; double* _ret_{tgt} = {call_code}; if (_ret_{tgt}) {{ memcpy({tgt}, _ret_{tgt}, {prod} * sizeof(double)); free(_ret_{tgt}); }}"
                )

        # 7. Multiplicação Matricial: y = W @ x, C = A @ B, y = (W @ x) * gamma, etc.
        is_matmul = False
        scale_expr = None
        left_expr, right_expr = None, None

        if isinstance(val_expr, Binary) and val_expr.op == "@":
            is_matmul = True
            left_expr = val_expr.left
            right_expr = val_expr.right
        elif isinstance(val_expr, Call) and isinstance(val_expr.func, Name) and val_expr.func.id == "matmul":
            is_matmul = True
            left_expr = val_expr.args[0]
            right_expr = val_expr.args[1]
        elif isinstance(val_expr, Binary) and val_expr.op == "*" and isinstance(val_expr.left, Binary) and val_expr.left.op == "@" and self._expr_length(val_expr.right) is None:
            is_matmul = True
            left_expr = val_expr.left.left
            right_expr = val_expr.left.right
            scale_expr = val_expr.right
        elif isinstance(val_expr, Binary) and val_expr.op == "*" and isinstance(val_expr.right, Binary) and val_expr.right.op == "@" and self._expr_length(val_expr.left) is None:
            is_matmul = True
            left_expr = val_expr.right.left
            right_expr = val_expr.right.right
            scale_expr = val_expr.left

        if is_matmul and left_expr is not None and right_expr is not None:
            l_shape = self._expr_shape(left_expr)
            r_shape = self._expr_shape(right_expr)
            pre_l, l_code = self._ensure_vec_operand(left_expr, ind)
            pre_r, r_code = self._ensure_vec_operand(right_expr, ind)
            l_type = self.infer_type(left_expr)
            sc_code = self.emit_expr(scale_expr) if scale_expr is not None else None
            pre_all = f"{pre_l}{pre_r}"

            # Caso 1: 1D @ 1D (Produto Escalar / Dot Product)
            if l_shape and r_shape and len(l_shape) == 1 and len(r_shape) == 1:
                K = l_shape[0]
                self.var_types[name] = "double"
                decl_prefix = "" if name in self.declared_vars else f"{const_p}double "
                self.declared_vars.add(name)
                if sc_code:
                    return f"{pre_all}{ind}{decl_prefix}{name} = nat_dot({l_code}, {r_code}, {K}LL) * ({sc_code});"
                return f"{pre_all}{ind}{decl_prefix}{name} = nat_dot({l_code}, {r_code}, {K}LL);"

            # Caso 2: 2D @ 1D (GEMV - Matriz por Vetor)
            if l_shape and r_shape and len(l_shape) == 2 and len(r_shape) == 1:
                M, K = l_shape
                if "int8" in l_type:
                    stmt = self._wrap_buffer_target(
                        name, "double", M, (M,), const_p, ind,
                        lambda tgt, cp: (
                            f"{ind}{cp}double {tgt}[{M}]; nat_ternary_matmul_vec_scaled({l_code}, {r_code}, {tgt}, {M}LL, {K}LL, (double)({sc_code}));"
                            if sc_code else
                            f"{ind}{cp}double {tgt}[{M}]; nat_ternary_matmul_vec({l_code}, {r_code}, {tgt}, {M}LL, {K}LL);"
                        )
                    )
                else:
                    stmt = self._wrap_buffer_target(
                        name, "double", M, (M,), const_p, ind,
                        lambda tgt, cp: (
                            f"{ind}{cp}double {tgt}[{M}]; nat_gemv_scaled_f64({l_code}, {r_code}, {tgt}, {M}LL, {K}LL, (double)({sc_code}));"
                            if sc_code else
                            f"{ind}{cp}double {tgt}[{M}]; nat_gemv_f64({l_code}, {r_code}, {tgt}, {M}LL, {K}LL);"
                        )
                    )
                return f"{pre_all}{stmt}"

            # Caso 3: 2D @ 2D (GEMM - Matriz por Matriz)
            if l_shape and r_shape and len(l_shape) == 2 and len(r_shape) == 2:
                M, K = l_shape
                _K2, N = r_shape
                total = M * N
                stmt = self._wrap_buffer_target(
                    name, "double", total, (M, N), const_p, ind,
                    lambda tgt, cp: (
                        f"{ind}{cp}double {tgt}[{total}]; nat_gemm_f64({l_code}, {r_code}, {tgt}, {M}LL, {K}LL, {N}LL); for (int64_t _i = 0; _i < {total}; _i++) {tgt}[_i] *= ({sc_code});"
                        if sc_code else
                        f"{ind}{cp}double {tgt}[{total}]; nat_gemm_f64({l_code}, {r_code}, {tgt}, {M}LL, {K}LL, {N}LL);"
                    )
                )
                return f"{pre_all}{stmt}"

            # Caso 4: 1D @ 2D (GEMV Transposto - Vetor por Matriz)
            if l_shape and r_shape and len(l_shape) == 1 and len(r_shape) == 2:
                M = l_shape[0]
                _M2, N = r_shape
                stmt = self._wrap_buffer_target(
                    name, "double", N, (N,), const_p, ind,
                    lambda tgt, cp: (
                        f"{ind}{cp}double {tgt}[{N}]; nat_gemv_trans_f64({r_code}, {l_code}, {tgt}, {M}LL, {N}LL); for (int64_t _i = 0; _i < {N}; _i++) {tgt}[_i] *= ({sc_code});"
                        if sc_code else
                        f"{ind}{cp}double {tgt}[{N}]; nat_gemv_trans_f64({r_code}, {l_code}, {tgt}, {M}LL, {N}LL);"
                    )
                )
                return f"{pre_all}{stmt}"

        # 8. Operações Aritméticas Elemento a Elemento (Vetor-Vetor e Vetor-Escalar, incluindo subexpressões)
        if isinstance(val_expr, Binary) and val_expr.op in ("+", "-", "*", "/"):
            l_len = self._expr_length(val_expr.left)
            r_len = self._expr_length(val_expr.right)
            if l_len is not None and r_len is not None:
                total = l_len
                shape = self._expr_shape(val_expr.left) or (total,)
                pre_l, l_code = self._ensure_vec_operand(val_expr.left, ind)
                pre_r, r_code = self._ensure_vec_operand(val_expr.right, ind)
                stmt = self._wrap_buffer_target(
                    name, "double", total, shape, const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{total}]; for (int64_t _i = 0; _i < {total}; _i++) {tgt}[_i] = ({l_code})[_i] {val_expr.op} ({r_code})[_i];"
                )
                return f"{pre_l}{pre_r}{stmt}"
            elif l_len is not None and r_len is None:
                total = l_len
                shape = self._expr_shape(val_expr.left) or (total,)
                pre_l, l_code = self._ensure_vec_operand(val_expr.left, ind)
                r_code = self.emit_expr(val_expr.right)
                stmt = self._wrap_buffer_target(
                    name, "double", total, shape, const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{total}]; for (int64_t _i = 0; _i < {total}; _i++) {tgt}[_i] = ({l_code})[_i] {val_expr.op} ({r_code});"
                )
                return f"{pre_l}{stmt}"
            elif r_len is not None and l_len is None:
                total = r_len
                shape = self._expr_shape(val_expr.right) or (total,)
                l_code = self.emit_expr(val_expr.left)
                pre_r, r_code = self._ensure_vec_operand(val_expr.right, ind)
                stmt = self._wrap_buffer_target(
                    name, "double", total, shape, const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{total}]; for (int64_t _i = 0; _i < {total}; _i++) {tgt}[_i] = ({l_code}) {val_expr.op} ({r_code})[_i];"
                )
                return f"{pre_r}{stmt}"

        # 9. rfft(signal, [n])
        is_rfft = False
        if isinstance(val_expr, Call):
            fn = ""
            if isinstance(val_expr.func, Name):
                fn = val_expr.func.id
            elif isinstance(val_expr.func, Attr) and isinstance(val_expr.func.obj, Name) and val_expr.func.obj.id == "math":
                fn = val_expr.func.name
            if fn == "rfft":
                is_rfft = True

        if is_rfft and val_expr.args:
            sig_arg = val_expr.args[0]
            sig_code = self.emit_expr(sig_arg)
            n_val = None
            if len(val_expr.args) > 1 and isinstance(val_expr.args[1], Int) and val_expr.args[1].value is not None:
                n_val = val_expr.args[1].value
            elif isinstance(sig_arg, Name):
                if sig_arg.id in self.var_shapes and len(self.var_shapes[sig_arg.id]) == 1:
                    n_val = self.var_shapes[sig_arg.id][0]
                elif sig_arg.id in self.var_lengths:
                    n_val = self.var_lengths[sig_arg.id]
            elif isinstance(sig_arg, ListLit):
                n_val = len(sig_arg.items)

            if n_val is not None:
                M = n_val // 2 + 1
                total = M * 2
                return self._wrap_buffer_target(
                    name, "double", total, (M, 2), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{total}]; nat_rfft({sig_code}, {tgt}, {n_val}LL);"
                )
            else:
                n_code = self.emit_expr(val_expr.args[1]) if len(val_expr.args) > 1 else f"((int64_t)(sizeof({sig_code}) / sizeof(({sig_code})[0])))"
                self.var_types[name] = "double*"
                self.declared_vars.add(name)
                return f"{ind}int64_t _n_{name} = (int64_t)({n_code}); int64_t _m_{name} = _n_{name} / 2 + 1; {const_p}double* {name} = (double*)malloc(_m_{name} * 2 * sizeof(double)); if ({name}) nat_rfft({sig_code}, {name}, _n_{name});"

        # 10. irfft(spec, [n])
        is_irfft = False
        if isinstance(val_expr, Call):
            fn = ""
            if isinstance(val_expr.func, Name):
                fn = val_expr.func.id
            elif isinstance(val_expr.func, Attr) and isinstance(val_expr.func.obj, Name) and val_expr.func.obj.id == "math":
                fn = val_expr.func.name
            if fn == "irfft":
                is_irfft = True

        if is_irfft and val_expr.args:
            spec_arg = val_expr.args[0]
            spec_code = self.emit_expr(spec_arg)
            M_val = None
            if isinstance(spec_arg, Name):
                if spec_arg.id in self.var_shapes and len(self.var_shapes[spec_arg.id]) == 2:
                    M_val = self.var_shapes[spec_arg.id][0]
                elif spec_arg.id in self.var_lengths:
                    M_val = self.var_lengths[spec_arg.id] // 2

            n_val = None
            if len(val_expr.args) > 1 and isinstance(val_expr.args[1], Int) and val_expr.args[1].value is not None:
                n_val = val_expr.args[1].value
            elif M_val is not None:
                n_val = (M_val - 1) * 2

            if n_val is not None and M_val is not None:
                return self._wrap_buffer_target(
                    name, "double", n_val, (n_val,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{n_val}]; nat_irfft({spec_code}, {tgt}, {M_val}LL, {n_val}LL);"
                )
            else:
                m_code = f"{M_val}LL" if M_val is not None else f"((int64_t)(sizeof({spec_code}) / (2 * sizeof(double))))"
                n_code = self.emit_expr(val_expr.args[1]) if len(val_expr.args) > 1 else f"(({m_code} - 1) * 2)"
                self.var_types[name] = "double*"
                self.declared_vars.add(name)
                return f"{ind}int64_t _m_{name} = (int64_t)({m_code}); int64_t _n_{name} = (int64_t)({n_code}); {const_p}double* {name} = (double*)malloc(_n_{name} * sizeof(double)); if ({name}) nat_irfft({spec_code}, {name}, _m_{name}, _n_{name});"

        # 11. power_spectrum(x)
        is_ps = False
        if isinstance(val_expr, Call):
            fn = ""
            if isinstance(val_expr.func, Name):
                fn = val_expr.func.id
            elif isinstance(val_expr.func, Attr) and isinstance(val_expr.func.obj, Name) and val_expr.func.obj.id == "math":
                fn = val_expr.func.name
            if fn == "power_spectrum":
                is_ps = True

        if is_ps and val_expr.args:
            x_arg = val_expr.args[0]
            x_code = self.emit_expr(x_arg)
            is_spec_2d = False
            M_val = None
            N_val = None
            if isinstance(x_arg, Name):
                if x_arg.id in self.var_shapes and len(self.var_shapes[x_arg.id]) == 2:
                    is_spec_2d = True
                    M_val = self.var_shapes[x_arg.id][0]
                elif x_arg.id in self.var_shapes and len(self.var_shapes[x_arg.id]) == 1:
                    N_val = self.var_shapes[x_arg.id][0]
                    M_val = N_val // 2 + 1
                elif x_arg.id in self.var_lengths:
                    N_val = self.var_lengths[x_arg.id]
                    M_val = N_val // 2 + 1
            elif isinstance(x_arg, ListLit):
                N_val = len(x_arg.items)
                M_val = N_val // 2 + 1

            if is_spec_2d and M_val is not None:
                return self._wrap_buffer_target(
                    name, "double", M_val, (M_val,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{M_val}]; nat_power_spectrum_spec({x_code}, {tgt}, {M_val}LL);"
                )
            elif N_val is not None and M_val is not None:
                return self._wrap_buffer_target(
                    name, "double", M_val, (M_val,), const_p, ind,
                    lambda tgt, cp: f"{ind}{cp}double {tgt}[{M_val}]; nat_power_spectrum_signal({x_code}, {tgt}, {N_val}LL);"
                )
            else:
                self.var_types[name] = "double*"
                self.declared_vars.add(name)
                return f"{ind}int64_t _n_{name} = ((int64_t)(sizeof({x_code}) / sizeof(({x_code})[0]))); int64_t _m_{name} = _n_{name} / 2 + 1; {const_p}double* {name} = (double*)malloc(_m_{name} * sizeof(double)); if ({name}) nat_power_spectrum_signal({x_code}, {name}, _n_{name});"

        return None

    def _record_struct_var_fields(self, var_name: str, val_expr: Expr):
        if isinstance(val_expr, Call) and isinstance(val_expr.func, Name) and val_expr.func.id in self.struct_decls:
            sname = val_expr.func.id
            sdecl = self.struct_decls[sname]
            ordered = self._order_struct_args(sdecl, val_expr)
            for f, arg_expr in zip(sdecl.fields, ordered):
                fname = f.name
                sh = self._expr_shape(arg_expr)
                ft = self.infer_type(arg_expr)
                self.var_types[f"{var_name}.{fname}"] = ft
                self.struct_field_types[(sname, fname)] = ft
                if sh:
                    self.var_shapes[f"{var_name}.{fname}"] = sh
                    prod = 1
                    for d in sh:
                        prod *= d
                    self.var_lengths[f"{var_name}.{fname}"] = prod
                    self.struct_field_shapes[(sname, fname)] = sh

    def _with_pre_stmts(self, stmt_code: str) -> str:
        if self.pending_pre_stmts:
            pre = "\n".join(self.pending_pre_stmts)
            self.pending_pre_stmts = []
            return f"{pre}\n{stmt_code}"
        return stmt_code

    def emit_stmt(self, s: Node) -> str:
        ind = self.indent()
        if isinstance(s, VarDecl):
            self.pending_pre_stmts = []
            if isinstance(s.value, Lambda):
                self.lambda_vars[s.name] = s.value
                return f"{ind}/* lambda {s.name} */"
            buf = self._emit_buffer_init(s.name, s.value, ind, is_const=False)
            if buf is not None:
                return self._with_pre_stmts(buf)
            ctype = self.c_type(s.type) if s.type else self.infer_type(s.value)
            self.var_types[s.name] = ctype
            self.declared_vars.add(s.name)
            if isinstance(s.type, TypeResult):
                self.var_result_elem[s.name] = self.c_type(s.type.elem)
            elif ctype == "NatResult":
                self.var_result_elem[s.name] = self._infer_result_elem(s.value)
            self._record_struct_var_fields(s.name, s.value)
            val = self.emit_expr(s.value)
            return self._with_pre_stmts(f"{ind}{ctype} {s.name} = {val};")

        if isinstance(s, ConstDecl):
            self.pending_pre_stmts = []
            if isinstance(s.value, Lambda):
                self.lambda_vars[s.name] = s.value
                return f"{ind}/* lambda {s.name} */"
            buf = self._emit_buffer_init(s.name, s.value, ind, is_const=True)
            if buf is not None:
                return self._with_pre_stmts(buf)
            ctype = self.c_type(s.type) if s.type else self.infer_type(s.value)
            self.var_types[s.name] = ctype
            self.declared_vars.add(s.name)
            if isinstance(s.type, TypeResult):
                self.var_result_elem[s.name] = self.c_type(s.type.elem)
            elif ctype == "NatResult":
                self.var_result_elem[s.name] = self._infer_result_elem(s.value)
            self._record_struct_var_fields(s.name, s.value)
            val = self.emit_expr(s.value)
            return self._with_pre_stmts(f"{ind}const {ctype} {s.name} = {val};")

        if isinstance(s, TupleAssign):
            self.pending_pre_stmts = []
            if isinstance(s.value, Call) and isinstance(s.value.func, Attr) and s.value.func.name == "update":
                cache_expr = s.value.func.obj
                cache_code = self.emit_expr(cache_expr)
                pre_k, k_code = self._ensure_vec_operand(s.value.args[0], ind)
                pre_v, v_code = self._ensure_vec_operand(s.value.args[1], ind)
                lines = []
                if pre_k: lines.append(pre_k.rstrip("\n"))
                if pre_v: lines.append(pre_v.rstrip("\n"))
                lines.append(f"{ind}nat_kvcache_update(&({cache_code}), {k_code}, {v_code}, 1LL);")
                dims = None
                if isinstance(cache_expr, Name) and cache_expr.id in self.kvcache_dims:
                    dims = self.kvcache_dims[cache_expr.id]
                elif isinstance(cache_expr, Index) and isinstance(cache_expr.obj, Name) and cache_expr.obj.id in self.kvcache_dims:
                    dims = self.kvcache_dims[cache_expr.obj.id]
                for idx, tgt in enumerate(s.targets):
                    if isinstance(tgt, Name):
                        field = "k" if idx == 0 else "v"
                        self.var_types[tgt.id] = "double*"
                        if dims:
                            self.var_shapes[tgt.id] = dims
                            self.var_lengths[tgt.id] = dims[0] * dims[1]
                        if tgt.id in self.declared_vars:
                            lines.append(f"{ind}{tgt.id} = ({cache_code}).{field};")
                        else:
                            self.declared_vars.add(tgt.id)
                            lines.append(f"{ind}double* {tgt.id} = ({cache_code}).{field};")
                return self._with_pre_stmts("\n".join(lines))
            if isinstance(s.value, TupleLit) and len(s.value.items) == len(s.targets):
                lines = []
                temps = []
                for item in s.value.items:
                    t_type = self.infer_type(item)
                    t_name = self._new_temp("_tup")
                    t_val = self.emit_expr(item)
                    lines.append(f"{ind}{t_type} {t_name} = {t_val};")
                    temps.append((t_name, t_type, item))
                for tgt, (t_name, t_type, _) in zip(s.targets, temps):
                    if isinstance(tgt, Name):
                        self.var_types[tgt.id] = t_type
                        if tgt.id in self.declared_vars:
                            lines.append(f"{ind}{tgt.id} = {t_name};")
                        else:
                            self.declared_vars.add(tgt.id)
                            lines.append(f"{ind}{t_type} {tgt.id} = {t_name};")
                return self._with_pre_stmts("\n".join(lines))

        if isinstance(s, Assign):
            self.pending_pre_stmts = []
            tgt = s.target
            if isinstance(tgt, Name):
                name = tgt.id
                if s.op == "=" and isinstance(s.value, Lambda):
                    self.lambda_vars[name] = s.value
                    return f"{ind}/* lambda {name} */"
                if s.op == "=":
                    buf = self._emit_buffer_init(name, s.value, ind, is_const=False)
                    if buf is not None:
                        return self._with_pre_stmts(buf)
                if name not in self.declared_vars and s.op == "=":
                    ctype = self.infer_type(s.value)
                    self.var_types[name] = ctype
                    self.declared_vars.add(name)
                    if ctype == "NatResult":
                        self.var_result_elem[name] = self._infer_result_elem(s.value)
                    self._record_struct_var_fields(name, s.value)
                    val = self.emit_expr(s.value)
                    return self._with_pre_stmts(f"{ind}{ctype} {name} = {val};")
                val = self.emit_expr(s.value)
                var_t = self.var_types.get(name, "")
                if var_t in ("nat_fix256_t", "NatFix256"):
                    if s.op == "+=": return self._with_pre_stmts(f"{ind}{name} = nat_fix256_add({name}, {val});")
                    if s.op == "-=": return self._with_pre_stmts(f"{ind}{name} = nat_fix256_sub({name}, {val});")
                    if s.op == "*=": return self._with_pre_stmts(f"{ind}{name} = nat_fix256_mul({name}, {val});")
                    if s.op == "/=": return self._with_pre_stmts(f"{ind}{name} = nat_fix256_div({name}, {val});")
                elif var_t in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                    if s.op == "+=": return self._with_pre_stmts(f"{ind}{name} = nat_interval256_add({name}, {val});")
                    if s.op == "-=": return self._with_pre_stmts(f"{ind}{name} = nat_interval256_sub({name}, {val});")
                    if s.op == "*=":
                        val_t = self.infer_type(s.value)
                        if val_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256", "double", "float", "int64_t", "int32_t"):
                            r_fix = val if val_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256") else f"nat_fix256_from_f64((double)({val}), 64)"
                            return self._with_pre_stmts(f"{ind}{name} = nat_interval256_mul_scalar({name}, {r_fix});")
                        return self._with_pre_stmts(f"{ind}{name} = nat_interval256_mul({name}, {val});")
                    if s.op == "/=":
                        val_t = self.infer_type(s.value)
                        if val_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256", "double", "float", "int64_t", "int32_t"):
                            r_fix = val if val_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256") else f"nat_fix256_from_f64((double)({val}), 64)"
                            return self._with_pre_stmts(f"{ind}{name} = nat_interval256_div_scalar({name}, {r_fix});")
                        return self._with_pre_stmts(f"{ind}{name} = nat_interval256_div({name}, {val});")
                elif var_t == "NatComplex":
                    if s.op == "+=": return self._with_pre_stmts(f"{ind}{name} = nat_complex_add({name}, {val});")
                    if s.op == "-=": return self._with_pre_stmts(f"{ind}{name} = nat_complex_sub({name}, {val});")
                    if s.op == "*=": return self._with_pre_stmts(f"{ind}{name} = nat_complex_mul({name}, {val});")
                return self._with_pre_stmts(f"{ind}{name} {s.op} {val};")
            if isinstance(tgt, Index):
                arr = self.emit_expr(tgt.obj)
                idx = self.emit_expr(tgt.index)
                val = self.emit_expr(s.value)
                return self._with_pre_stmts(f"{ind}{arr}[{idx}] {s.op} {val};")
            if isinstance(tgt, Attr):
                obj_code = self.emit_expr(tgt.obj)
                val = self.emit_expr(s.value)
                return self._with_pre_stmts(f"{ind}{obj_code}.{tgt.name} {s.op} {val};")
            raise NatError(
                f"Assignment to '{type(tgt).__name__}' is not supported in native C99 backend",
                getattr(s, "line", 0),
                getattr(s, "col", 0)
            )

        if isinstance(s, ExprStmt):
            self.pending_pre_stmts = []
            expr_code = self.emit_expr(s.expr)
            if expr_code.endswith(";"):
                return self._with_pre_stmts(f"{ind}{expr_code}")
            return self._with_pre_stmts(f"{ind}{expr_code};")

        if isinstance(s, If):
            self.pending_pre_stmts = []
            cond = self.emit_expr(s.cond)
            pre = self._with_pre_stmts(f"{ind}if ({cond}) {{")
            res = [pre]
            old_decl = set(self.declared_vars)
            self.indent_level += 1
            for st in s.body:
                c = self.emit_stmt(st)
                if c: res.append(c)
            self.indent_level -= 1
            self.declared_vars = set(old_decl)
            if s.orelse:
                if len(s.orelse) == 1 and isinstance(s.orelse[0], If):
                    sub_if = self.emit_stmt(s.orelse[0]).lstrip()
                    res.append(f"{ind}}} else {sub_if}")
                    self.declared_vars = set(old_decl)
                    return "\n".join(res)
                else:
                    res.append(f"{ind}}} else {{")
                    self.indent_level += 1
                    for st in s.orelse:
                        c = self.emit_stmt(st)
                        if c: res.append(c)
                    self.indent_level -= 1
                    self.declared_vars = set(old_decl)
            res.append(f"{ind}}}")
            return "\n".join(res)

        if isinstance(s, While):
            self.pending_pre_stmts = []
            cond = self.emit_expr(s.cond)
            res = [self._with_pre_stmts(f"{ind}while ({cond}) {{")]
            old_decl = set(self.declared_vars)
            self.indent_level += 1
            for st in s.body:
                c = self.emit_stmt(st)
                if c: res.append(c)
            self.indent_level -= 1
            self.declared_vars = old_decl
            res.append(f"{ind}}}")
            return "\n".join(res)

        if isinstance(s, For):
            if isinstance(s.iter, Range):
                start = self.emit_expr(s.iter.start) if s.iter.start else "0"
                end = self.emit_expr(s.iter.end) if s.iter.end else "0"
                op = "<=" if s.iter.inclusive else "<"
                old_decl = set(self.declared_vars)
                self.var_types[s.var] = "int64_t"
                self.declared_vars.add(s.var)
                res = [f"{ind}for (int64_t {s.var} = {start}; {s.var} {op} {end}; {s.var}++) {{"]
                self.indent_level += 1
                for st in s.body:
                    c = self.emit_stmt(st)
                    if c: res.append(c)
                self.indent_level -= 1
                self.declared_vars = old_decl
                res.append(f"{ind}}}")
                return "\n".join(res)

            iter_len = self._expr_length(s.iter)
            iter_t = self.infer_type(s.iter)
            if iter_len is not None and (iter_t.endswith("*") or isinstance(s.iter, (ListLit, ListComp))):
                elem_t = iter_t[:-1] if iter_t.endswith("*") else "double"
                idx_var = self._new_temp(f"_idx_{s.var}")
                old_decl = set(self.declared_vars)
                self.var_types[s.var] = elem_t
                self.declared_vars.add(s.var)
                pre_it, iter_code = self._ensure_vec_operand(s.iter, ind)
                res = []
                if pre_it:
                    res.append(pre_it.rstrip("\n"))
                res.append(f"{ind}for (int64_t {idx_var} = 0; {idx_var} < {iter_len}; {idx_var}++) {{")
                self.indent_level += 1
                res.append(f"{self.indent()}{elem_t} {s.var} = ({iter_code})[{idx_var}];")
                for st in s.body:
                    c = self.emit_stmt(st)
                    if c: res.append(c)
                self.indent_level -= 1
                self.declared_vars = old_decl
                res.append(f"{ind}}}")
                return "\n".join(res)

            raise NatError(
                f"C99 backend supports iteration only over 'Range' intervals (e.g. '0..N') or lists/tensors; iteration over '{type(s.iter).__name__}' is not supported",
                getattr(s, "line", 0),
                getattr(s, "col", 0)
            )

        if isinstance(s, Return):
            self.pending_pre_stmts = []
            if s.value is not None:
                if self.cur_fn_ret_type == "double*" and self._expr_length(s.value) is not None:
                    n_len = self._expr_length(s.value)
                    pre_v, v_code = self._ensure_vec_operand(s.value, ind)
                    return self._with_pre_stmts(f"{pre_v}{ind}return nat_dup_f64({v_code}, {n_len}LL);")
                val = self.emit_expr(s.value)
                return self._with_pre_stmts(f"{ind}return {val};")
            return f"{ind}return;"

        if isinstance(s, Break):
            return f"{ind}break;"

        if isinstance(s, Continue):
            return f"{ind}continue;"

        if isinstance(s, Pass):
            return f"{ind}/* pass */;"

        raise NatError(
            f"Statement of type '{type(s).__name__}' is not supported in native C99 backend",
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
            op_t = self.infer_type(e.operand)
            if op_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256") and e.op == "-":
                return f"nat_fix256_neg({operand})"
            if op_t in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256") and e.op == "-":
                return f"nat_interval256_neg({operand})"
            if e.op == "not":
                return f"(!({operand}))"
            return f"({e.op}({operand}))"

        if isinstance(e, Binary):
            l = self.emit_expr(e.left)
            r = self.emit_expr(e.right)
            tl = self.infer_type(e.left)
            tr = self.infer_type(e.right)
            if tl == "NatComplex" or tr == "NatComplex":
                l_c = l if tl == "NatComplex" else f"nat_complex((double)({l}), 0.0)"
                r_c = r if tr == "NatComplex" else f"nat_complex((double)({r}), 0.0)"
                if e.op == "+": return f"nat_complex_add({l_c}, {r_c})"
                if e.op == "-": return f"nat_complex_sub({l_c}, {r_c})"
                if e.op == "*": return f"nat_complex_mul({l_c}, {r_c})"
            if tl in ("nat_fix256_t", "NatFix256") or tr in ("nat_fix256_t", "NatFix256"):
                l_f = l if tl in ("nat_fix256_t", "NatFix256") else f"nat_fix256_from_f64((double)({l}), 64)"
                r_f = r if tr in ("nat_fix256_t", "NatFix256") else f"nat_fix256_from_f64((double)({r}), 64)"
                if e.op == "+": return f"nat_fix256_add({l_f}, {r_f})"
                if e.op == "-": return f"nat_fix256_sub({l_f}, {r_f})"
                if e.op == "*": return f"nat_fix256_mul({l_f}, {r_f})"
                if e.op == "/": return f"nat_fix256_div({l_f}, {r_f})"
                if e.op == "==": return f"nat_fix256_eq({l_f}, {r_f})"
                if e.op == "!=": return f"(!nat_fix256_eq({l_f}, {r_f}))"
                if e.op == "<": return f"nat_fix256_lt({l_f}, {r_f})"
                if e.op == "<=": return f"nat_fix256_le({l_f}, {r_f})"
                if e.op == ">": return f"nat_fix256_gt({l_f}, {r_f})"
                if e.op == ">=": return f"nat_fix256_ge({l_f}, {r_f})"
            if tl in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256") or tr in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                l_i = l if tl in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256") else f"nat_interval256_point(nat_fix256_from_f64((double)({l}), 64))"
                r_i = r if tr in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256") else f"nat_interval256_point(nat_fix256_from_f64((double)({r}), 64))"
                if e.op == "+": return f"nat_interval256_add({l_i}, {r_i})"
                if e.op == "-": return f"nat_interval256_sub({l_i}, {r_i})"
                if e.op == "*":
                    if tr in ("nat_fix256_t", "NatFix256", "Fix256", "fix256", "double", "float", "int64_t", "int32_t"):
                        r_fix = r if tr in ("nat_fix256_t", "NatFix256", "Fix256", "fix256") else f"nat_fix256_from_f64((double)({r}), 64)"
                        return f"nat_interval256_mul_scalar({l_i}, {r_fix})"
                    if tl in ("nat_fix256_t", "NatFix256", "Fix256", "fix256", "double", "float", "int64_t", "int32_t"):
                        l_fix = l if tl in ("nat_fix256_t", "NatFix256", "Fix256", "fix256") else f"nat_fix256_from_f64((double)({l}), 64)"
                        return f"nat_interval256_mul_scalar({r_i}, {l_fix})"
                    return f"nat_interval256_mul({l_i}, {r_i})"
                if e.op == "/":
                    if tr in ("nat_fix256_t", "NatFix256", "Fix256", "fix256", "double", "float", "int64_t", "int32_t"):
                        r_fix = r if tr in ("nat_fix256_t", "NatFix256", "Fix256", "fix256") else f"nat_fix256_from_f64((double)({r}), 64)"
                        return f"nat_interval256_div_scalar({l_i}, {r_fix})"
                    return f"nat_interval256_div({l_i}, {r_i})"
            if e.op == "and": return f"(({l}) && ({r}))"
            if e.op == "or": return f"(({l}) || ({r}))"
            if e.op == "**": return f"pow((double)({l}), (double)({r}))"
            if e.op == "//": return f"((int64_t)floor((double)({l}) / (double)({r})))"
            if e.op == "@":
                raise NatError(
                    "Matrix multiplication operator '@' requires tensor/matrix operands; scalar multiplication is not supported in native C99 backend (use AOT kernels such as gemv_vec_lut)",
                    getattr(e, "line", 0),
                    getattr(e, "col", 0)
                )
            return f"(({l}) {e.op} ({r}))"

        if isinstance(e, ListLit):
            elem_type = "int64_t" if self.infer_type(e) == "int64_t*" else "double"
            items_code = ", ".join(self.emit_expr(it) for it in e.items)
            return f"(({elem_type}[]){{ {items_code} }})"

        if isinstance(e, ListComp):
            tmp = self._new_temp("_lc")
            pre = self._emit_buffer_init(tmp, e, self.indent(), is_const=False)
            if pre is not None:
                self.pending_pre_stmts.append(pre)
                return tmp

        if isinstance(e, Call):
            return self.emit_call(e)

        if isinstance(e, Index):
            sh = self._expr_shape(e.obj)
            if sh and len(sh) == 2 and not isinstance(e.index, Range):
                _M, K = sh
                obj_code = self.emit_expr(e.obj)
                idx = self.emit_expr(e.index)
                return f"(({obj_code}) + ({idx}) * {K})"
            obj = self.emit_expr(e.obj)
            idx = self.emit_expr(e.index)
            return f"({obj})[{idx}]"

        if isinstance(e, TryUnwrap):
            inner = self.emit_expr(e.expr)
            elem_t = self._infer_result_elem(e.expr)
            if getattr(self, "cur_fn_returns_result", False):
                tmp_r = self._new_temp("_try")
                self.pending_pre_stmts.append(
                    f"{self.indent()}NatResult {tmp_r} = {inner}; if (!{tmp_r}.is_ok) return {tmp_r};"
                )
                if "int" in elem_t or elem_t == "bool":
                    return f"{tmp_r}.val_i64"
                if "double" in elem_t or "float" in elem_t:
                    return f"{tmp_r}.val_f64"
                return f"{tmp_r}.val_str"
            if self.infer_type(e.expr) == "NatResult":
                if "int" in elem_t or elem_t == "bool":
                    return f"nat_unwrap_i64({inner})"
                if "double" in elem_t or "float" in elem_t:
                    return f"nat_unwrap_f64({inner})"
                return f"nat_unwrap_str({inner})"
            return inner

        if isinstance(e, Attr):
            if isinstance(e.obj, Name) and e.obj.id == "math":
                if e.name == "PI": return "NAT_PI"
                if e.name == "E": return "NAT_E"
                return e.name
            if self.infer_type(e.obj) == "NatKVCache" and e.name == "capacity":
                return f"({self.emit_expr(e.obj)}).max_seq_len"
            return f"{self.emit_expr(e.obj)}.{e.name}"

        raise NatError(
            f"Expression of type '{type(e).__name__}' is not supported in native C99 backend",
            getattr(e, "line", 0),
            getattr(e, "col", 0)
        )

    def emit_call(self, c: Call) -> str:
        resolved = self._resolve_lambda_call(c)
        if resolved is not None:
            return self.emit_expr(resolved)
        desugared = self._desugar_hof(c)
        if desugared is not c:
            return self.emit_expr(desugared)
        # Métodos de objetos (ex: cache.update(k, v), cache.clear(), cache.reset())
        if isinstance(c.func, Attr) and not (isinstance(c.func.obj, Name) and c.func.obj.id == "math"):
            obj_t = self.infer_type(c.func.obj)
            obj_code = self.emit_expr(c.func.obj)
            mname = c.func.name
            if obj_t == "NatKVCache":
                if mname == "update" and len(c.args) >= 2:
                    k_code = self.emit_expr(c.args[0])
                    v_code = self.emit_expr(c.args[1])
                    return f"nat_kvcache_update(&({obj_code}), {k_code}, {v_code}, 1LL)"
                if mname in ("clear", "reset"):
                    return f"nat_kvcache_clear(&({obj_code}))"
            if obj_t in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                if mname == "step" and len(c.args) >= 2:
                    v_code = self.emit_expr(c.args[0])
                    dt_code = self.emit_expr(c.args[1])
                    dt_t = self.infer_type(c.args[1])
                    if dt_t not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                        dt_code = f"nat_fix256_from_f64((double)({dt_code}), 64)"
                    return f"nat_interval256_step({obj_code}, {v_code}, {dt_code})"
                if mname == "simulate_steps" and len(c.args) >= 3:
                    v_code = self.emit_expr(c.args[0])
                    dt_code = self.emit_expr(c.args[1])
                    dt_t = self.infer_type(c.args[1])
                    if dt_t not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                        dt_code = f"nat_fix256_from_f64((double)({dt_code}), 64)"
                    n_code = self.emit_expr(c.args[2])
                    return f"nat_interval256_simulate_steps_val({obj_code}, {v_code}, {dt_code}, (int64_t)({n_code}))"
                if mname == "width":
                    return f"nat_interval256_width({obj_code})"
                if mname == "mid":
                    return f"nat_interval256_mid({obj_code})"
                if mname == "is_coherent" and len(c.args) >= 1:
                    mw_code = self.emit_expr(c.args[0])
                    mw_t = self.infer_type(c.args[0])
                    if mw_t not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                        mw_code = f"nat_fix256_from_f64((double)({mw_code}), 64)"
                    return f"nat_interval256_is_coherent({obj_code}, {mw_code})"
                if mname == "contains" and len(c.args) >= 1:
                    x_code = self.emit_expr(c.args[0])
                    x_t = self.infer_type(c.args[0])
                    if x_t not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                        x_code = f"nat_fix256_from_f64((double)({x_code}), 64)"
                    return f"nat_interval256_contains({obj_code}, {x_code})"
            if obj_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                if mname == "to_f64":
                    return f"nat_fix256_to_f64({obj_code})"
                if mname == "to_i64":
                    return f"nat_fix256_to_i64({obj_code})"

        if isinstance(c.func, Name) and c.func.id == "print":
            if len(c.args) == 1:
                arg = c.args[0]
                v_name = arg.id if isinstance(arg, Name) else None
                sh = self.var_shapes.get(v_name) if v_name and v_name in self.var_shapes else self._expr_shape(arg)
                if sh:
                    arg_code = self.emit_expr(arg)
                    arg_t = self.var_types.get(v_name, "") if v_name and v_name in self.var_types else self.infer_type(arg)
                    is_int = "int" in arg_t
                    if len(sh) == 2:
                        fn = "nat_print_mat_i64" if is_int else "nat_print_mat_f64"
                        return f'{fn}({arg_code}, {sh[0]}LL, {sh[1]}LL); printf("\\n");'
                    elif len(sh) == 1:
                        fn = "nat_print_vec_i64" if is_int else "nat_print_vec_f64"
                        return f'{fn}({arg_code}, {sh[0]}LL); printf("\\n");'
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
                    elif itype == "NatComplex":
                        parts.append(f'printf("%f%+fi{sep}", ({code}).real, ({code}).imag);')
                    elif itype in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                        parts.append(f'printf("%s{sep}", nat_fix256_as_str({code}));')
                    elif itype in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                        parts.append(f'printf("%s{sep}", nat_interval256_as_str({code}));')
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

        if fn_name in self.struct_decls:
            sdecl = self.struct_decls[fn_name]
            ordered = self._order_struct_args(sdecl, c)
            args_code = ", ".join(self.emit_expr(a) for a in ordered)
            return f"{fn_name}_ctor({args_code})"
        if fn_name == "KVCache":
            ms = self.emit_expr(c.args[0]) if len(c.args) > 0 else "64LL"
            dm = self.emit_expr(c.args[1]) if len(c.args) > 1 else "4LL"
            return f"nat_kvcache_init({ms}, {dm})"
        if fn_name == "Complex":
            re = self.emit_expr(c.args[0]) if len(c.args) > 0 else "0.0"
            im = self.emit_expr(c.args[1]) if len(c.args) > 1 else "0.0"
            return f"nat_complex((double)({re}), (double)({im}))"
        if fn_name in ("Fix256", "fix256"):
            if not c.args:
                return "nat_fix256_zero()"
            arg0 = c.args[0]
            fb = self.emit_expr(c.args[1]) if len(c.args) > 1 else "64"
            arg_t = self.infer_type(arg0)
            arg_code = self.emit_expr(arg0)
            if isinstance(arg0, Str) or arg_t == "const char*":
                return f"nat_fix256_from_str({arg_code}, {fb})"
            elif "int" in arg_t:
                return f"nat_fix256_from_i64((int64_t)({arg_code}), {fb})"
            else:
                return f"nat_fix256_from_f64((double)({arg_code}), {fb})"
        if fn_name in ("Interval256", "interval256"):
            if not c.args:
                return "nat_interval256_create(nat_fix256_zero(), nat_fix256_zero())"
            if len(c.args) == 1:
                arg0 = c.args[0]
                t0 = self.infer_type(arg0)
                c0 = self.emit_expr(arg0)
                if t0 not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                    if isinstance(arg0, Str) or t0 == "const char*":
                        c0 = f"nat_fix256_from_str({c0}, 64)"
                    elif "int" in t0:
                        c0 = f"nat_fix256_from_i64((int64_t)({c0}), 64)"
                    else:
                        c0 = f"nat_fix256_from_f64((double)({c0}), 64)"
                return f"nat_interval256_point({c0})"
            else:
                arg0, arg1 = c.args[0], c.args[1]
                t0, t1 = self.infer_type(arg0), self.infer_type(arg1)
                c0, c1 = self.emit_expr(arg0), self.emit_expr(arg1)
                if t0 not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                    if isinstance(arg0, Str) or t0 == "const char*":
                        c0 = f"nat_fix256_from_str({c0}, 64)"
                    elif "int" in t0:
                        c0 = f"nat_fix256_from_i64((int64_t)({c0}), 64)"
                    else:
                        c0 = f"nat_fix256_from_f64((double)({c0}), 64)"
                if t1 not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                    if isinstance(arg1, Str) or t1 == "const char*":
                        c1 = f"nat_fix256_from_str({c1}, 64)"
                    elif "int" in t1:
                        c1 = f"nat_fix256_from_i64((int64_t)({c1}), 64)"
                    else:
                        c1 = f"nat_fix256_from_f64((double)({c1}), 64)"
                return f"nat_interval256_create({c0}, {c1})"
        elif fn_name == "interval256_step":
            pos_c = self.emit_expr(c.args[0])
            vel_c = self.emit_expr(c.args[1])
            dt_c = self.emit_expr(c.args[2])
            dt_t = self.infer_type(c.args[2])
            if dt_t not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                dt_c = f"nat_fix256_from_f64((double)({dt_c}), 64)"
            return f"nat_interval256_step({pos_c}, {vel_c}, {dt_c})"
        elif fn_name == "interval256_width":
            return f"nat_interval256_width({self.emit_expr(c.args[0])})"
        elif fn_name == "interval256_mid":
            return f"nat_interval256_mid({self.emit_expr(c.args[0])})"
        elif fn_name == "interval256_is_coherent":
            iv_c = self.emit_expr(c.args[0])
            mw_c = self.emit_expr(c.args[1])
            mw_t = self.infer_type(c.args[1])
            if mw_t not in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                mw_c = f"nat_fix256_from_f64((double)({mw_c}), 64)"
            return f"nat_interval256_is_coherent({iv_c}, {mw_c})"
        elif fn_name == "ln": fn_name = "nat_ln"
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
        elif fn_name == "argmax":
            a_code = self.emit_expr(c.args[0])
            a_len = self._expr_length(c.args[0])
            len_code = f"{a_len}LL" if a_len is not None else f"((int64_t)(sizeof({a_code}) / sizeof(({a_code})[0])))"
            return f"nat_argmax_f64({a_code}, {len_code})"
        elif fn_name == "to_i64":
            arg_t = self.infer_type(c.args[0])
            arg_code = self.emit_expr(c.args[0])
            if arg_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                return f"nat_fix256_to_i64({arg_code})"
            return f"((int64_t)({arg_code}))"
        elif fn_name == "to_f64":
            arg_t = self.infer_type(c.args[0])
            arg_code = self.emit_expr(c.args[0])
            if arg_t in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                return f"nat_fix256_to_f64({arg_code})"
            return f"((double)({arg_code}))"
        elif fn_name == "len":
            if c.args:
                arg = c.args[0]
                if isinstance(arg, Str):
                    return f"((int64_t)({len(arg.value)}LL))"
                sh = self._expr_shape(arg)
                if sh:
                    return f"((int64_t)({sh[0]}LL))"
                a_len = self._expr_length(arg)
                if a_len is not None:
                    return f"((int64_t)({a_len}LL))"
                if isinstance(arg, Name) and self.var_types.get(arg.id) == "const char*":
                    return f"((int64_t)strlen({arg.id}))"
                arg_type = self.infer_type(arg)
                arg_code = self.emit_expr(arg)
                if arg_type == "const char*":
                    return f"((int64_t)strlen({arg_code}))"
                return f"((int64_t)(sizeof({arg_code}) / sizeof(({arg_code})[0])))"
        elif fn_name == "dot":
            a_code = self.emit_expr(c.args[0])
            b_code = self.emit_expr(c.args[1])
            l0 = self._expr_length(c.args[0])
            l1 = self._expr_length(c.args[1]) if len(c.args) > 1 else None
            if l0 is not None:
                len_code = f"{l0}LL"
            elif l1 is not None:
                len_code = f"{l1}LL"
            else:
                len_code = f"((int64_t)(sizeof({a_code}) / sizeof(({a_code})[0])))"
            return f"nat_dot({a_code}, {b_code}, {len_code})"
        elif fn_name in ("sum", "mean", "min", "max"):
            a_code = self.emit_expr(c.args[0])
            l0 = self._expr_length(c.args[0])
            if l0 is not None:
                len_code = f"{l0}LL"
            else:
                len_code = f"((int64_t)(sizeof({a_code}) / sizeof(({a_code})[0])))"
            return f"nat_{fn_name}({a_code}, {len_code})"
        elif fn_name == "exists_file":
            return f"nat_exists_file({self.emit_expr(c.args[0])})"
        elif fn_name == "read_file":
            return f"nat_read_file({self.emit_expr(c.args[0])})"
        elif fn_name == "read_file_safe":
            return f"nat_read_file_safe({self.emit_expr(c.args[0])})"
        elif fn_name == "ok":
            arg_t = self.infer_type(c.args[0])
            arg_code = self.emit_expr(c.args[0])
            if arg_t == "const char*":
                return f"nat_ok_str({arg_code})"
            if "int" in arg_t or arg_t == "bool":
                return f"nat_ok_i64((int64_t)({arg_code}))"
            return f"nat_ok_f64((double)({arg_code}))"
        elif fn_name == "err":
            return f"nat_err({self.emit_expr(c.args[0])})"
        elif fn_name == "is_ok":
            return f"nat_is_ok({self.emit_expr(c.args[0])})"
        elif fn_name == "is_err":
            return f"nat_is_err({self.emit_expr(c.args[0])})"
        elif fn_name == "unwrap":
            elem_t = self._infer_result_elem(c.args[0])
            r_code = self.emit_expr(c.args[0])
            if "int" in elem_t or elem_t == "bool":
                return f"nat_unwrap_i64({r_code})"
            if "double" in elem_t or "float" in elem_t:
                return f"nat_unwrap_f64({r_code})"
            return f"nat_unwrap_str({r_code})"
        elif fn_name == "unwrap_or":
            def_t = self.infer_type(c.args[1])
            r_code = self.emit_expr(c.args[0])
            d_code = self.emit_expr(c.args[1])
            if "int" in def_t or def_t == "bool":
                return f"nat_unwrap_or_i64({r_code}, (int64_t)({d_code}))"
            if "double" in def_t or "float" in def_t:
                return f"nat_unwrap_or_f64({r_code}, (double)({d_code}))"
            return f"nat_unwrap_or_str({r_code}, {d_code})"
        elif fn_name == "error_message":
            return f"nat_error_message({self.emit_expr(c.args[0])})"

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
                    elif itype in ("nat_fix256_t", "NatFix256", "Fix256", "fix256"):
                        fmt_parts.append("%s")
                        args.append(f"nat_fix256_as_str({expr_code})")
                    elif itype in ("nat_interval256_t", "NatInterval256", "Interval256", "interval256"):
                        fmt_parts.append("%s")
                        args.append(f"nat_interval256_as_str({expr_code})")
                    elif itype in ("double", "float"):
                        fmt_parts.append("%f")
                        args.append(f"(double)({expr_code})")
                    elif itype == "bool":
                        fmt_parts.append("%s")
                        args.append(f"(({expr_code}) ? \"True\" : \"False\")")
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
            obj_file = os.path.splitext(os.path.abspath(c_path))[0] + ".obj"
            bat_content = f'@echo off\ncall "{vc}" >nul 2>&1\ncl.exe /O2 /openmp /nologo /W3 "{c_path}" /Fe:"{exe_path}" /Fo:"{obj_file}"\n'
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(bat_content)
            try:
                res = subprocess.run([bat_path], capture_output=True, text=True)
            finally:
                if os.path.exists(bat_path):
                    try:
                        os.remove(bat_path)
                    except OSError:
                        pass
                if os.path.exists(obj_file):
                    try:
                        os.remove(obj_file)
                    except OSError:
                        pass
            if res.returncode != 0:
                print(res.stderr or res.stdout, file=sys.stderr)
            return res.returncode == 0

    cl = shutil.which("cl")
    if cl:
        obj_file = os.path.splitext(os.path.abspath(c_path))[0] + ".obj"
        cmd = [cl, "/O2", "/openmp", "/nologo", "/W3", c_path, f"/Fe:{exe_path}", f"/Fo:{obj_file}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            if os.path.exists(obj_file):
                try:
                    os.remove(obj_file)
                except OSError:
                    pass
        if res.returncode != 0:
            print(res.stderr or res.stdout, file=sys.stderr)
        return res.returncode == 0

    print("Error: No C compiler found (install GCC, Clang, or MSVC).", file=sys.stderr)
    return False


# ============================================================================
# BUILT-IN EXAMPLES
# ============================================================================

EXEMPLOS = {
'exemplo': r'''# File: exemplo.nat
fn soma(valores: [i64]) -> i64:
    total = 0
    for valor in valores:
        total += valor
    return total

fn main():
    numeros = [10, 20, 30, 40, 50]
    resultado = soma(numeros)
    print(f"Result: {resultado}")
''',

'riemann': r'''# riemann.nat
# Experimental prototype: Riemann-Siegel formula main sum.
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

    print(f"Final weights: {w}")
    for i in 0..4:
        soma = entradas[i] @ w
        pred = 1.0 / (1.0 + math.exp(-soma))
        print(f"Input {entradas[i]} -> Pred: {pred:.3f} (Target: {targets[i]})")
''',

'ia_tensor': r'''import math
import time

fn main():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[2.0, 0.0], [1.0, 2.0]]
    C = A @ B
    print(f"Matrix Multiplication C = A @ B: {C}")

    s = sum(C)
    m = mean(C)
    print(f"Element sum: {s}, Mean: {m}")

    ativ = relu(C - 5.0)
    sig = sigmoid(C * 0.1)
    print(f"ReLU(C - 5.0): {ativ}")
    print(f"Sigmoid(C * 0.1): {sig}")

    v = [10, 20, 30, 40, 50]
    fatia = v[1..4]
    print(f"Vector slice v[1..4]: {fatia}")
''',

'ternario': r'''# Ternary Computing (1.58 bits) and Ternary Matrices Example
fn main():
    # 1. Ternary Weights Matrix {-1, 0, +1} inspired by BitNet
    W: tensor[t2] = [
        [1,  0, -1,  1],
        [-1, 1,  0, -1],
        [0, -1,  1,  0]
    ]

    # Continuous f32 activation vector
    x: [f32] = [0.8, -1.2, 2.5, 0.4]

    # 2. Native Ternary Matrix Multiplication (Multiplication-free Add-only GEMM)
    y = W @ x
    print(f"Input activations: {x}")
    print(f"Ternary Matmul Result (W @ x): {y}")

    # 3. Dynamic ternary quantization of activations
    ativacoes = [1.8, -0.3, 0.1, -1.5, 0.9, -0.1]
    trits = quantize_ternary(ativacoes, 0.5)
    print(f"Original activations: {ativacoes}")
    print(f"Activations quantized to {{-1, 0, 1}}: {trits}")

    # 4. Memory compression (Bit Packing: 2 bits per trit, 4 trits per byte)
    compactado = pack_ternary(trits)
    descompactado = unpack_ternary(compactado, len(trits))
    print(f"Packed bytes (4 trits/byte): {compactado}")
    print(f"Unpacked values: {descompactado}")
''',
}


# ============================================================================
# REGRESSION TEST SUITES: FRONTEND & RUNTIME
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
        parse_fn = parse_expression if expr else parse
        with self.assertRaises(NatError) as cm:
            parse_fn(codigo)
        self.assertIn(trecho, cm.exception.message)
        if linha: self.assertEqual(cm.exception.line, linha)

    def test_erros(self):
        self.falha("fn f(x):\n    pass\n", "require a type", 1)
        self.falha("fn f():\nx = 1\n", "expected an indented block", 2)
        self.falha("fn f():\n    x = (1 + 2\n", "was not closed", 2)
        self.falha("fn f():\n    x = 1)\n", "without matching opening bracket", 2)
        self.falha("fn f():\n    a = 1\n  b = 2\n", "inconsistent indentation", 3)
        self.falha("fn f():\n    a = 1\n        b = 2\n", "unexpected indentation", 3)
        self.falha("fn f():\n\tx = 1\n", "tab", 2)
        self.falha("fn f():\n    x = 1 2\n", "end of line", 2)
        self.falha("fn f():\n    else:\n        pass\n", "without matching 'if'", 2)
        self.falha("fn f():\n    f() = 1\n", "left-hand side", 2)
        self.falha("fn f():\n    x = \n", "expected an expression", 2)
        self.falha("fn f():\n    x = $\n", "unexpected character", 2)
        self.falha("x = 1\n", "allowed at top level", 1)
        self.falha('fn f():\n    s = "abc\n', "unterminated string", 2)
        self.falha("fn f():\n    fn g():\n        pass\n", "only allowed at top level", 2)

    def test_encadeamentos_proibidos(self):
        self.falha("a < b < c", "cannot be chained", expr=True)
        self.falha("1..2..3", "cannot be chained", expr=True)

    def test_fstring_erros(self):
        self.falha('f"{}"', "empty expression", expr=True)
        self.falha('f"{x"', "without matching '}'", expr=True)
        self.falha('f"x}"', "unmatched '}'", expr=True)
        self.falha('f"{1 +}"', "in f-string", expr=True)

    def test_mensagem_formatada(self):
        s = "fn f():\n    x = (1\n"
        with self.assertRaises(NatError) as cm: parse(s)
        out = cm.exception.pretty(s, "t.nat")
        self.assertIn("t.nat:2:9: error:", out)
        self.assertIn("x = (1", out)


def analisar(fonte):
    d = SemanticAnalyzer().analyze(parse(fonte))
    return ([x.msg for x in d if x.level in ("error", "erro")], [x.msg for x in d if x.level in ("warning", "aviso")])

def M(corpo, antes=""):
    linhas = "\n".join(("    " + l) if l.strip() else l for l in corpo.strip("\n").split("\n"))
    return antes + "fn main():\n" + linhas + "\n"


class SemBase(unittest.TestCase):
    def ok(self, fonte):
        e, _ = analisar(fonte)
        self.assertEqual(e, [], e)

    def erro(self, fonte, trecho):
        e, _ = analisar(fonte)
        self.assertTrue(any(trecho in x for x in e), f"expected '{trecho}' in {e}")

    def aviso(self, fonte, trecho):
        e, w = analisar(fonte)
        self.assertEqual(e, [], e)
        self.assertTrue(any(trecho in x for x in w), f"expected '{trecho}' in {w}")


class SemExemplos(SemBase):
    def test_exemplos_limpos(self):
        for nome in ("exemplo", "riemann", "rede", "ia_tensor", "ternario"):
            self.assertEqual(analisar(EXEMPLOS[nome]), ([], []), nome)


class SemTipos(SemBase):
    def test_tipo_estavel(self):
        self.erro(M('x = 1\nx = "a"\nprint(x)'), "cannot be assigned")
        self.erro(M("x = 1\nx = 2.5\nprint(x)"), "cannot be assigned")
        self.ok(M("x = 1.0\nx = 2\nprint(x)"))

    def test_divisao(self):
        self.erro(M("n: i64 = 4 / 2\nprint(n)"), "declared as")
        self.ok(M("a = 4 / 2\na = 1.5\nprint(a)"))
        self.ok(M("n: i64 = 7 // 2\nprint(n)"))

    def test_operadores(self):
        self.erro(M('a = 1 + "x"\nprint(a)'), "is not defined for")
        self.ok(M('s = "a" + "b"\nprint(s)'))
        self.erro(M('print(1 < "a")'), "cannot compare")
        self.erro(M('print(1 == "a")'), "cannot compare")

    def test_logicos(self):
        self.erro(M("print(1 and true)"), "'and' requires bool")
        self.erro(M("print(not 1)"), "'not' requires bool")
        self.erro(M("if 1:\n    pass"), "condition must be bool")
        self.erro(M("while 1.5:\n    pass"), "condition must be bool")
        self.ok(M("a = 1 < 2 and not false\nprint(a)"))

    def test_listas(self):
        self.ok(M("a = [1, 2.5]\nb: [f64] = a\nprint(b)"))
        self.erro(M('a = [1, "x"]\nprint(a)'), "incompatible")
        self.erro(M("a = [1.0]\nprint(a[0.5])"), "index must be i64")
        self.erro(M("x = 5\nprint(x[0])"), "cannot be indexed")
        self.erro(M("for i in 0..2.5:\n    pass"), "range bounds")
        self.erro(M("for i in 5:\n    pass"), "cannot iterate over")
        self.erro(M('v = [1.0, 2.0]\nv[1] = "a"\nprint(v)'), "cannot be assigned")
        self.erro(M("a: [f64] = [1, 2]\nprint(a)"), "declared as")

    def test_broadcasting(self):
        self.ok(M("w = [1.0, 2.0]\nw += 2.0\nw = w * w + w\nw = 2.0 * w\nprint(w)"))
        self.erro(M('w = [1.0]\nw += "a"\nprint(w)'), "is not defined for")

    def test_composta(self):
        self.erro(M("x = 1\nx += 2.5\nprint(x)"), "produces")
        self.erro(M("y += 1"), "before being declared")

    def test_sem_valor(self):
        self.erro(M("x = print(1)"), "does not produce a value")

    def test_result_unwrap(self):
        self.ok(M('r = ok(42)\nx: i64 = r?\nprint(x)'))
        self.ok(M('r = ok("teste")\ns: str = unwrap(r)\nprint(s)'))
        self.ok(M('r = err("falha")\ns: str = unwrap_or(r, "padrao")\nprint(s)'))
        self.ok(M('r: Result[str] = ok("ola")\ns: str = r?\nprint(s)'))
        self.erro(M('r: Result[str] = ok("ola")\nx: i64 = r?\nprint(x)'), "declared as")
        self.erro(M('x = 1\nr: Result[str] = ok("ola")\nx = r?\nprint(x)'), "cannot be assigned")
        self.erro(M('s = read_file("a.txt")?\nprint(s)'), "requires a Result[T]")
        self.erro(M('n = 5\nm = n?\nprint(m)'), "requires a Result[T]")

    def test_tensor_matmul_broadcasting(self):
        # Multiplicação matricial e broadcasting com tensores e ativações
        self.ok(M("A = tensor([[1.0, 2.0], [3.0, 4.0]])\nx = tensor([0.5, -0.5])\ny = relu(A @ x + tensor([0.1, -0.1]))\nprint(y)"))
        self.ok(M("W: tensor[t2] = random_ternary([4, 4], 1)\nx: [f32] = [1.0, 2.0, 3.0, 4.0]\ny: [f32] = sigmoid((W @ x) * 0.5)\nprint(y)"))


class SemFuncoes(SemBase):
    F = "fn f(a: i64) -> i64:\n    return a\n\n"

    def test_aridade(self):
        self.erro(M("print(sqrt(1.0, 2.0))"), "expects 1")
        self.erro(M("print(sqrt())"), "expects 1")
        self.ok(M("print(1, 2, 3)"))
        self.ok(M("print()"))

    def test_argumentos(self):
        self.erro(M("print(f(2.5))", self.F), "argument 1")
        self.ok(M("print(f(2))", self.F))
        self.ok(M("print(g(2))", "fn g(a: f64) -> f64:\n    return a\n\n"))
        self.erro(M("print(dot([1, 2], [1.0]))"), "argument 1")

    def test_retorno(self):
        self.erro("fn f() -> i64:\n    pass\n", "not all")
        self.erro("fn f() -> i64:\n    if true:\n        return 1\n", "not all")
        self.ok("fn f(x: bool) -> i64:\n    if x:\n        return 1\n    else:\n        return 2\n")
        self.ok("fn f(x: bool) -> i64:\n    if x:\n        return 1\n    elif true:\n        return 2\n    else:\n        return 3\n")
        self.erro("fn f():\n    return 1\n", "cannot return a value")
        self.erro("fn f() -> i64:\n    return 1.5\n", "expected return type")
        self.erro("fn f() -> i64:\n    return\n", "must return")
        self.ok("fn f() -> f64:\n    return 1\n")

    def test_hoisting(self):
        self.ok("fn main():\n    print(f())\n\nfn f() -> i64:\n    return 1\n")

    def test_nomes(self):
        self.erro("fn sqrt():\n    pass\n", "is a built-in function")
        self.erro("fn f():\n    pass\nfn f():\n    pass\n", "already defined")
        self.erro(M("print = 1"), "already a function name")
        self.erro(M("x = 1\nx()\nprint(x)"), "is a variable, not a function")
        self.erro(M("print(f)", "fn f():\n    pass\n\n"), "is a function; call it with parentheses")
        self.erro("fn f(x: foo):\n    pass\n", "unknown type")

    def test_math(self):
        self.ok(M("print(math.sqrt(4.0) + math.PI)", "import math\n\n"))
        self.erro(M("print(math.sqrt(4.0))"), "is not imported")
        self.erro(M("print(math.foo(1.0))", "import math\n\n"), "has no member")
        self.erro("import xyz\n", "unknown module")

    def test_break_continue(self):
        self.erro(M("break"), "outside of a loop")
        self.erro(M("continue"), "outside of a loop")
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
        self.erro(M("if true:\n    t = 5\nprint(t)"), "not defined")
        self.ok(M("t = 0\nif true:\n    t = 5\nprint(t)"))
        self.erro(M("if true:\n    pass\nelse:\n    z = 1\n    print(z)\nprint(z)"), "not defined")
        self.erro(M("for i in 0..3:\n    pass\nprint(i)"), "not defined")

    def test_constantes(self):
        self.erro(M("const k = 1\nk = 2"), "constant")
        self.erro("const PI = 3.14\nfn main():\n    PI = 3.0\n", "constant")
        self.ok(M("const x: f64 = 3\nprint(x)"))
        self.erro(M('const x: i64 = "a"\nprint(x)'), "declared as")
        self.ok("fn main():\n    print(PI)\n\nconst PI = 3.14\n")

    def test_nao_definido(self):
        self.erro(M("while true:\n    y = naoexiste\n    print(y)\n    break"), "not defined")
        self.erro(M('print(f"{naoexiste}")'), "not defined")
        self.erro(M("print(-naoexiste)"), "not defined")
        self.erro(M("total = 1\nprint(totl)"), "did you mean 'total'")
        self.erro(M("x = x + 1\nprint(x)"), "not defined")


class SemAvisos(SemBase):
    def test_typo(self):
        self.aviso(M("total = 0\ntotl = total + 1\nprint(total)"), "possible typo for 'total'")

    def test_nunca_usada(self):
        self.aviso(M("x = 1"), "never used")
        self.aviso(M("x = 1\nx += 1"), "never used")

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
        self.erro(M("x = 1.0 @ 2.0"), "is not defined for")

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
    def _analyze_and_run(self, fonte: str) -> Interpreter:
        prog = parse(fonte)
        erros = [d.msg for d in SemanticAnalyzer().analyze(prog) if d.level in ("error", "erro")]
        self.assertEqual(erros, [], f"Unexpected semantic errors: {erros}")
        interp = Interpreter()
        interp.run(prog)
        return interp

    def test_run_exemplo(self):
        self._analyze_and_run(src("exemplo.nat"))

    def test_run_rede_neural(self):
        interp = self._analyze_and_run(src("rede.nat"))
        w = interp.last_scope.get("w")
        self.assertIsNotNone(w)
        self.assertGreater(w[0], 2.0)
        self.assertGreater(w[1], 2.0)

    def test_run_ia_tensor(self):
        self._analyze_and_run(src("ia_tensor.nat"))

    def test_run_ternario(self):
        self._analyze_and_run(src("ternario.nat"))

    def test_rmsnorm_2d_por_linha(self):
        x_np = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        gamma_np = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32)
        eps = 1e-6
        esperado = (x_np / np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + eps)) * gamma_np
        obtido = rmsnorm_fn(Tensor(x_np, dtype="f32"), Tensor(gamma_np, dtype="f32"), eps)
        np.testing.assert_allclose(obtido.data, esperado, rtol=1e-5, atol=1e-6)
        # Como a linha 1 é 10x a linha 0, após RMSNorm por linha ambas devem ser quase idênticas
        np.testing.assert_allclose(obtido.data[0], obtido.data[1], rtol=1e-5, atol=1e-5)

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
# REGRESSION TEST SUITES: SCOPE, C99 AOT BACKEND & DIFFERENTIAL
# ============================================================================

class TesteCoerenciaSemanticoRuntime(unittest.TestCase):
    def test_correspondencia_biunivoca_builtins(self):
        """Critique #2: Ensures that no function declared in BUILTINS is missing from runtime and vice versa."""
        self.assertEqual(set(BUILTINS.keys()), set(BUILTIN_IMPLS.keys()))
        for name, sig in BUILTINS.items():
            self.assertIsNotNone(sig.impl, f"Builtin '{name}' has no linked implementation!")
            self.assertTrue(callable(sig.impl), f"Implementation of '{name}' is not callable!")
            self.assertIsNotNone(sig.min_args, f"Builtin '{name}' has no min_args defined!")
            if sig.max_args is not None:
                self.assertGreaterEqual(sig.max_args, sig.min_args, f"Builtin '{name}' has max_args < min_args!")

    def test_interpretador_inicializa_de_builtins(self):
        """Critique #2: The interpreter must derive its builtins strictly from BUILTINS."""
        interp = Interpreter()
        for name in BUILTINS:
            self.assertIn(name, interp.globals, f"Builtin '{name}' missing from Interpreter.globals!")
            self.assertIs(interp.globals[name], BUILTINS[name].impl)


class TestePropagacaoResult(unittest.TestCase):
    """Runtime semantics of the '?' operator on Result[T]."""
    FN = ('fn ler(p: str) -> Result[str]:\n'
          '    s = read_file_safe(p)?\n'
          '    return ok(s)\n\n')

    def _run(self, code):
        import io, contextlib
        prog = parse(code)
        erros = [d.msg for d in SemanticAnalyzer().analyze(prog) if d.level in ("error", "erro")]
        self.assertEqual(erros, [], f"Unexpected semantic errors: {erros}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Interpreter().run(prog)
        return buf.getvalue().strip()

    def test_propaga_err_em_funcao_result(self):
        code = self.FN + ('fn main():\n'
                          '    r = ler("nao_existe_xyz_123.txt")\n'
                          '    print(is_err(r))\n'
                          '    print(error_message(r) != "")\n')
        self.assertEqual(analisar(code)[0], [])
        self.assertEqual(self._run(code).splitlines(), ["True", "True"])

    def test_ok_desempacota(self):
        code = 'fn main():\n    r = ok("abc")\n    s = r?\n    print(s)\n'
        self.assertEqual(self._run(code), "abc")

    def test_panic_controlado_fora_de_result(self):
        code = 'fn main():\n    s = read_file_safe("nao_existe_xyz_123.txt")?\n    print(s)\n'
        with self.assertRaises(RuntimeError) as cm:
            self._run(code)
        self.assertIn("panic", str(cm.exception))

    def test_c99_suporta_propagacao_e_result_tipado(self):
        code_prop = self.FN + 'fn main():\n    r = ler("nao_existe_xyz_123.txt")\n    print(is_err(r))\n'
        c_code = CTranspiler(parse(code_prop)).transpile()
        self.assertIn("if (!_try_", c_code)

        code_i64 = (
            'fn dobro_positivo(x: i64) -> Result[i64]:\n'
            '    if x < 0:\n'
            '        return err("negativo")\n'
            '    return ok(x * 2)\n\n'
            'fn somar_dobros(a: i64, b: i64) -> Result[i64]:\n'
            '    da = dobro_positivo(a)?\n'
            '    db = dobro_positivo(b)?\n'
            '    return ok(da + db)\n\n'
            'fn main():\n'
            '    r1 = somar_dobros(3, 4)\n'
            '    r2 = somar_dobros(3, -1)\n'
            '    print(unwrap(r1))\n'
            '    print(is_err(r2))\n'
        )
        self.assertEqual(analisar(code_i64)[0], [])
        self.assertEqual(self._run(code_i64).splitlines(), ["14", "True"])
        c_i64 = CTranspiler(parse(code_i64)).transpile()
        self.assertIn("nat_ok_i64", c_i64)
        self.assertIn("nat_unwrap_i64", c_i64)


class TesteEscopoInterpretador(unittest.TestCase):
    def test_escopo_bloco_if(self):
        # Variable defined inside if must not leak outside
        code = "fn main():\n    if true:\n        x = 42\n    print(x)\n"
        with self.assertRaises(RuntimeError):
            Interpreter().run(parse(code))

    def test_escopo_bloco_for(self):
        # For loop iteration variable must not leak outside
        code = "fn main():\n    for i in 0..5:\n        pass\n    print(i)\n"
        with self.assertRaises(RuntimeError):
            Interpreter().run(parse(code))

    def test_imutabilidade_constante(self):
        # Reassigning constant at runtime must fail
        code = "fn main():\n    const c = 100\n    c = 200\n"
        with self.assertRaises(RuntimeError):
            Interpreter().run(parse(code))

    def test_mutacao_escopo_pai_em_loop(self):
        # Mutating parent scope variable inside loop must persist
        code = "fn main():\n    s = 0\n    for i in 1..4:\n        s += i\n"
        prog = parse(code)
        self.assertEqual([d.msg for d in SemanticAnalyzer().analyze(prog) if d.level in ("error", "erro")], [])
        interp = Interpreter()
        interp.run(prog)
        self.assertEqual(interp.last_scope.get("s"), 6)

    def test_isolamento_escopo_funcao_atribuicao(self):
        # Function assigning 'x' must not alter caller's 'x' (lexical scope)
        code = (
            "fn alterar():\n"
            "    x = 999\n"
            "    print(x)\n\n"
            "fn main():\n"
            "    x = 10\n"
            "    alterar()\n"
        )
        prog = parse(code)
        self.assertEqual([d.msg for d in SemanticAnalyzer().analyze(prog) if d.level in ("error", "erro")], [])
        interp = Interpreter()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            interp.run(prog)
        self.assertEqual(interp.last_scope.get("x"), 10)

    def test_isolamento_escopo_funcao_leitura(self):
        # Called function must not see caller's local variables
        code = (
            "fn ler_x() -> i64:\n"
            "    return x\n\n"
            "fn main():\n"
            "    x = 42\n"
            "    print(ler_x())\n"
        )
        prog = parse(code)
        erros = [d.msg for d in SemanticAnalyzer().analyze(prog) if d.level in ("error", "erro")]
        self.assertGreater(len(erros), 0)
        interp = Interpreter()
        with self.assertRaises(RuntimeError):
            interp.run(prog)

    def test_erro_runtime_rastreia_linha_e_coluna(self):
        # Runtime error reports accurate line and column
        code = (
            "fn main():\n"
            "    if true:\n"
            "        x = 42\n"
            "    print(x)\n"
        )
        prog = parse(code)
        interp = Interpreter()
        with self.assertRaises(NatRuntimeError) as cm:
            interp.run(prog)
        err = cm.exception
        self.assertEqual(err.line, 4)
        self.assertEqual(err.col, 11)
        self.assertIn("line 4:11", str(err))
        pretty = err.pretty(code, "teste.nat")
        self.assertIn("teste.nat:4:11: error: Undefined variable: x", pretty)
        self.assertIn("^", pretty)

    def test_funcao_aridade_incorreta(self):
        code = "fn soma2(a: i64, b: i64) -> i64:\n    return a + b\n\nfn main():\n    x = soma2(1)\n"
        prog = parse(code)
        interp = Interpreter()
        with self.assertRaises(NatRuntimeError) as cm:
            interp.run(prog)
        self.assertIn("expects 2 argument(s), but received 1", str(cm.exception))


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

    def _run_differential(self, source: str, basename: str = "diff_test"):
        import io, tempfile
        prog = parse(source)
        erros = [d.msg for d in SemanticAnalyzer().analyze(prog) if d.level in ("error", "erro")]
        self.assertEqual(erros, [], f"Unexpected semantic errors: {erros}")

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
        kwargs = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
        with tempfile.TemporaryDirectory(**kwargs) as tmpdir:
            c_path = os.path.join(tmpdir, f"{basename}.c")
            exe_path = os.path.join(tmpdir, f"{basename}.exe")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)
            ok = compile_c_to_binary(c_path, exe_path)
            self.assertTrue(ok, f"Compilação C falhou para {basename}")
            res = subprocess.run([exe_path], capture_output=True, text=True, timeout=15)
            c_out = res.stdout.strip()

        return interp_out, c_out, c_code

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
        interp_out, c_out, _ = self._run_differential(source, "diff_fat")
        self.assertEqual(interp_out, c_out)

    def test_header_unificado_e_cooley_tukey_fft(self):
        rt_path = ensure_nativa_rt_header()
        self.assertTrue(os.path.exists(rt_path))
        self.assertIn("nat_fft_cooley_tukey", NATIVA_RT_HEADER_C)
        self.assertIn(NATIVA_RT_HEADER_C, AOT_C_SOURCE)

    def test_diferencial_mini_llm_c99(self):
        """Acceptance target: full Mini-LLM with Structs, KVCache, RoPE,
        RMSNorm, SiLU, Softmax, layer lists, and tuple unpacking compiled
        to native C99 and matching the Python interpreter."""
        if not self.compiler_available:
            self.skipTest("No C compiler available for differential test")

        source = '''
struct CamadaLLM:
    rms_att: tensor[f64]
    wq: tensor[f64]
    wk: tensor[f64]
    wv: tensor[f64]
    wo: tensor[f64]
    rms_ffn: tensor[f64]
    w1: tensor[f64]
    w2: tensor[f64]
    w3: tensor[f64]

fn main():
    tok_emb = [
        [0.5, -0.2, 0.8, 0.1],
        [-0.4, 0.7, 0.2, -0.5],
        [0.3, 0.3, -0.6, 0.9],
        [0.9, -0.8, 0.1, 0.4]
    ]
    rms_att0 = [1.0, 1.0, 1.0, 1.0]
    wq0 = [[0.4, 0.1, 0.0, -0.1], [0.0, 0.5, 0.1, 0.0], [-0.1, 0.0, 0.4, 0.2], [0.2, -0.1, 0.0, 0.3]]
    wk0 = [[0.3, 0.0, 0.1, 0.0], [0.1, 0.4, 0.0, -0.1], [0.0, 0.1, 0.5, 0.0], [-0.1, 0.0, 0.1, 0.4]]
    wv0 = [[0.5, -0.1, 0.0, 0.1], [0.0, 0.3, 0.2, 0.0], [0.1, 0.0, 0.4, -0.1], [0.0, 0.2, -0.1, 0.5]]
    wo0 = [[0.4, 0.0, 0.1, 0.1], [0.1, 0.4, -0.1, 0.0], [0.0, 0.1, 0.3, 0.2], [-0.1, 0.1, 0.0, 0.4]]
    rms_ffn0 = [1.0, 1.0, 1.0, 1.0]
    w1_0 = [[0.3, -0.2, 0.1, 0.0], [0.1, 0.4, 0.2, -0.1], [-0.2, 0.1, 0.5, 0.1], [0.0, -0.1, 0.2, 0.4]]
    w2_0 = [[0.4, 0.1, -0.1, 0.2], [0.0, 0.3, 0.2, 0.1], [0.2, -0.1, 0.4, 0.0], [-0.1, 0.2, 0.1, 0.3]]
    w3_0 = [[0.2, 0.3, 0.0, -0.1], [-0.1, 0.2, 0.3, 0.1], [0.1, 0.0, 0.2, 0.3], [0.3, -0.1, 0.1, 0.2]]

    c0 = CamadaLLM(rms_att0, wq0, wk0, wv0, wo0, rms_ffn0, w1_0, w2_0, w3_0)
    c1 = CamadaLLM(rms_att0, wq0, wk0, wv0, wo0, rms_ffn0, w1_0, w2_0, w3_0)
    camadas = [c0, c1]

    kv0 = KVCache(8, 4)
    kv1 = KVCache(8, 4)
    caches = [kv0, kv1]

    rms_final = [1.0, 1.0, 1.0, 1.0]
    lm_head = [
        [0.6, -0.2, 0.1, 0.3],
        [-0.3, 0.7, 0.2, -0.1],
        [0.1, -0.1, 0.8, 0.4],
        [0.2, 0.4, -0.3, 0.7]
    ]

    prompt = [0, 1, 2]
    next_tok = 0
    for pos in 0..3:
        tok = prompt[pos]
        x = tok_emb[tok]
        for l in 0..2:
            cam = camadas[l]
            xn = rmsnorm(x, cam.rms_att)
            q = apply_rope(cam.wq @ xn, pos)
            k = apply_rope(cam.wk @ xn, pos)
            v = cam.wv @ xn
            k_all, v_all = caches[l].update(k, v)

            att_scores = fill(3, -1000000000.0)
            for t in 0..(pos + 1):
                kt = k_all[t]
                att_scores[t] = dot(q, kt) * 0.5
            pesos = softmax(att_scores)

            contexto = zeros(4)
            for t in 0..(pos + 1):
                vt = v_all[t]
                p_t = pesos[t]
                contexto = contexto + (vt * p_t)

            x = x + (cam.wo @ contexto)
            xf = rmsnorm(x, cam.rms_ffn)
            h = silu(cam.w1 @ xf) * (cam.w3 @ xf)
            x = x + (cam.w2 @ h)

        xfinal = rmsnorm(x, rms_final)
        logits = lm_head @ xfinal
        next_tok = argmax(logits)
        print(f"pos={pos} tok={next_tok} logit0={logits[0]:.5f} logit1={logits[1]:.5f}")
'''
        interp_out, c_out, _ = self._run_differential(source, "diff_mini_llm")
        self.assertEqual(interp_out, c_out)

    def test_c99_print_vetor_matriz_inteiros(self):
        source = """
fn main():
    v = [1, 2, 3, 4]
    print(v)
    m = [[10, 20], [30, 40]]
    print(m)
"""
        interp_out, c_out, _ = self._run_differential(source, "diff_print_int_arrays")
        self.assertEqual(interp_out, c_out)



class TesteEspectralFourier(unittest.TestCase):
    def test_complex_aritmetica(self):
        c1 = complex_fn(3.0, 4.0)
        c2 = complex_fn(1.0, 2.0)
        self.assertEqual(c1.real, 3.0)
        self.assertEqual(c1.imag, 4.0)
        self.assertAlmostEqual(c1.magnitude, 5.0)
        self.assertAlmostEqual(c1.power, 25.0)

        c_soma = c1 + c2
        self.assertEqual((c_soma.real, c_soma.imag), (4.0, 6.0))

        c_sub = c1 - c2
        self.assertEqual((c_sub.real, c_sub.imag), (2.0, 2.0))

        c_mul = c1 * c2
        self.assertEqual((c_mul.real, c_mul.imag), (-5.0, 10.0))

    def test_rfft_irfft_reconstrucao(self):
        sinal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        spec = rfft_fn(sinal)
        self.assertEqual(len(spec), 5)
        self.assertEqual(len(spec[0]), 2)

        rec = irfft_fn(spec, n=8)
        self.assertEqual(len(rec), 8)
        for orig, r in zip(sinal, rec):
            self.assertAlmostEqual(orig, r, places=6)

    def test_power_spectrum(self):
        sinal = [1.0, 0.0, -1.0, 0.0]
        ps = power_spectrum_fn(sinal)
        self.assertEqual(len(ps), 3)
        for p in ps:
            self.assertGreaterEqual(p, 0.0)

    def test_interpretador_fourier_nat(self):
        code = '''
fn main():
    c = Complex(3.0, 4.0)
    print(c.real)
    print(c.imag)
    sig = [1.0, 2.0, 3.0, 4.0]
    spec = rfft(sig)
    ps = power_spectrum(sig)
    rec = irfft(spec)
    print(len(spec))
    print(len(rec))
'''
        import io
        prog = parse(code)
        interp = Interpreter()
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            interp.run(prog)
        finally:
            sys.stdout = old_stdout
        lines = [l.strip() for l in buf.getvalue().strip().splitlines()]
        self.assertEqual(lines[0], "3.0")
        self.assertEqual(lines[1], "4.0")
        self.assertEqual(lines[2], "3")
        self.assertEqual(lines[3], "4")

    def test_diferencial_fourier_c99(self):
        TesteDiferencialC.setUpClass()
        if not TesteDiferencialC.compiler_available:
            self.skipTest("Nenhum compilador C disponível para teste diferencial")

        source = '''
fn main():
    c = Complex(3.0, 4.0)
    c2 = Complex(1.0, -2.0)
    c3 = c + c2
    print(c3.real)
    print(c3.imag)
    x = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    spec = rfft(x)
    ps = power_spectrum(spec)
    rec = irfft(spec)
    print(len(spec))
    print(len(rec))
    print(ps[0])
    print(rec[0])
    print(rec[3])
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
        kwargs = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
        with tempfile.TemporaryDirectory(**kwargs) as tmpdir:
            c_path = os.path.join(tmpdir, "diff_fourier.c")
            exe_path = os.path.join(tmpdir, "diff_fourier.exe")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)
            ok = compile_c_to_binary(c_path, exe_path)
            self.assertTrue(ok, "Compilação C falhou para diff_fourier")
            res = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            c_out = res.stdout.strip()

        interp_lines = interp_out.splitlines()
        c_lines = c_out.splitlines()
        self.assertEqual(len(interp_lines), len(c_lines))
        for il, cl in zip(interp_lines, c_lines):
            try:
                self.assertAlmostEqual(float(il), float(cl), places=4)
            except ValueError:
                self.assertEqual(il, cl)



class TesteStructsTuplasModulos(unittest.TestCase):
    """Tests structs, tuples/unpacking, attribute assignment, append/push/Dict, and local module imports."""

    def _run_checked(self, src: str) -> Interpreter:
        prog = parse(src)
        sem = SemanticAnalyzer()
        sem.analyze(prog)
        self.assertEqual(sem.errors, [], f"Unexpected semantic errors: {sem.errors}")
        interp = Interpreter()
        interp.run(prog)
        return interp

    def test_struct_declaracao_acesso_e_mutacao(self):
        src = """
struct Ponto:
    x: f64
    y: f64

struct Layer: w: [f64], escala: f64

fn deslocar(p: Ponto, dx: f64, dy: f64) -> Ponto:
    p.x += dx
    p.y = p.y + dy
    return p

fn main():
    p = Ponto(1.5, 2.5)
    p2 = deslocar(p, 0.5, 1.5)
    cam = Layer([1.0, 2.0, 3.0], 2.0)
    cam.escala = 4.0
    res_x = p2.x
    res_y = p2.y
    res_esc = cam.escala * cam.w[1]
"""
        interp = self._run_checked(src)
        self.assertAlmostEqual(interp.last_scope["res_x"], 2.0)
        self.assertAlmostEqual(interp.last_scope["res_y"], 4.0)
        self.assertAlmostEqual(interp.last_scope["res_esc"], 8.0)

    def test_struct_erros_semanticos(self):
        src_campo_inexistente = """
struct Ponto:
    x: f64
    y: f64

fn main():
    p = Ponto(1.0, 2.0)
    p.z = 3.0
"""
        s1 = SemanticAnalyzer()
        s1.analyze(parse(src_campo_inexistente))
        self.assertTrue(any("has no field 'z'" in d.msg for d in s1.errors))

        src_tipo_incompativel = """
struct Ponto:
    x: f64
    y: f64

fn main():
    p = Ponto(1.0, 2.0)
    p.x = "texto"
"""
        s2 = SemanticAnalyzer()
        s2.analyze(parse(src_tipo_incompativel))
        self.assertTrue(any("cannot be assigned" in d.msg for d in s2.errors))

    def test_tuplas_retorno_multiplo_e_desempacotamento(self):
        src = """
fn min_max(a: i64, b: i64) -> (i64, i64):
    if a < b:
        return a, b
    return (b, a)

fn main():
    menor, maior = min_max(42, 7)
    t = (10, 2.5, "ok")
    v0 = t[0]
    cache = KVCache(4, 2)
    k_novo = ones([1, 2])
    v_novo = ones([1, 2]) * 3.0
    k, v = cache.update(k_novo, v_novo)
    s_v = sum(v)
"""
        interp = self._run_checked(src)
        self.assertEqual(interp.last_scope["menor"], 7)
        self.assertEqual(interp.last_scope["maior"], 42)
        self.assertEqual(interp.last_scope["v0"], 10)
        self.assertAlmostEqual(interp.last_scope["s_v"], 6.0)

    def test_append_push_e_dicionarios(self):
        src = """
fn main():
    nums: [i64] = []
    append(nums, 10)
    push(nums, 20)
    append(nums, 30)
    total_nums = len(nums)
    soma_nums = nums[0] + nums[1] + nums[2]

    mapa = {"a": 100, "b": 200}
    mapa["c"] = 300
    mapa["a"] += 50
    val_a = mapa["a"]
    val_c = mapa["c"]

    d2 = Dict()
    d2["chave"] = 42
    val_d2 = d2["chave"]
"""
        interp = self._run_checked(src)
        self.assertEqual(interp.last_scope["total_nums"], 3)
        self.assertEqual(interp.last_scope["soma_nums"], 60)
        self.assertEqual(interp.last_scope["val_a"], 150)
        self.assertEqual(interp.last_scope["val_c"], 300)
        self.assertEqual(interp.last_scope["val_d2"], 42)

    def test_import_modulo_local_nat(self):
        mod_name = "_temp_mod_teste_nativa"
        mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{mod_name}.nat")
        try:
            with open(mod_path, "w", encoding="utf-8") as f:
                f.write(
                    "const FATOR: i64 = 3\n\n"
                    "fn triplo(x: i64) -> i64:\n"
                    "    return x * FATOR\n"
                )
            src = f"""
import {mod_name}

fn main():
    a = {mod_name}.triplo(7)
    b = triplo(10)
    c = {mod_name}.FATOR
    res = a + b + c
"""
            interp = self._run_checked(src)
            self.assertEqual(interp.last_scope["res"], 21 + 30 + 3)
        finally:
            if os.path.exists(mod_path):
                os.remove(mod_path)

    def test_import_modulos_circulares(self):
        mod_a = "_temp_mod_circ_a"
        mod_b = "_temp_mod_circ_b"
        path_a = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{mod_a}.nat")
        path_b = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{mod_b}.nat")
        try:
            with open(path_a, "w", encoding="utf-8") as f:
                f.write(f"import {mod_b}\nfn func_a() -> i64:\n    return 42\n")
            with open(path_b, "w", encoding="utf-8") as f:
                f.write(f"import {mod_a}\nfn func_b() -> i64:\n    return 99\n")
            src = f"""
import {mod_a}
import {mod_b}

fn main():
    a = func_a()
    b = func_b()
    res = a + b
"""
            interp = self._run_checked(src)
            self.assertEqual(interp.last_scope["res"], 141)
        finally:
            if os.path.exists(path_a):
                os.remove(path_a)
            if os.path.exists(path_b):
                os.remove(path_b)

    def test_list_comprehensions_basico_e_aninhado(self):
        src = """
struct Ponto:
    x: f64
    y: f64

fn main():
    dobros = [i * 2 for i in 0..5]
    pares = [i for i in 0..10 if i % 2 == 0]
    pontos = [Ponto(i * 1.0, i * 2.0) for i in 0..4]
    caches = [[KVCache(8, 4) for _ in 0..2] for _ in 0..3]
    k_in = ones([1, 4])
    v_in = ones([1, 4]) * 2.0
    caches[2][1].update(k_in, v_in)
    seq_c = caches[2][1].seq_len
    soma_p = pontos[3].x + pontos[3].y
"""
        interp = self._run_checked(src)
        self.assertEqual(interp.last_scope["dobros"], [0, 2, 4, 6, 8])
        self.assertEqual(interp.last_scope["pares"], [0, 2, 4, 6, 8])
        self.assertEqual(interp.last_scope["seq_c"], 1)
        self.assertAlmostEqual(interp.last_scope["soma_p"], 9.0)

    def test_list_comprehensions_diferencial_c99(self):
        TesteDiferencialC.setUpClass()
        if not TesteDiferencialC.compiler_available:
            self.skipTest("Nenhum compilador C disponível para teste diferencial")

        source = """
struct Ponto:
    x: f64
    y: f64

fn main():
    quads = [i * i for i in 0..6]
    impares = [i for i in 0..8 if i % 2 == 1]
    pts = [Ponto(to_f64(i) * 1.5, to_f64(i) * 2.5) for i in 0..4]
    caches = [KVCache(8, 4) for _ in 0..4]
    k_vec = [1.0, 2.0, 3.0, 4.0]
    v_vec = [2.0, 4.0, 6.0, 8.0]
    k_out, v_out = caches[2].update(k_vec, v_vec)
    print(quads[4])
    print(impares[2])
    print(pts[3].x + pts[3].y)
    print(caches[2].seq_len)
    print(sum(v_out))
"""
        import io, tempfile
        prog = parse(source)
        sem = SemanticAnalyzer()
        sem.analyze(prog)
        self.assertEqual(sem.errors, [], f"Erros semânticos: {sem.errors}")

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
        kwargs = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
        with tempfile.TemporaryDirectory(**kwargs) as tmpdir:
            c_path = os.path.join(tmpdir, "diff_listcomp.c")
            exe_path = os.path.join(tmpdir, "diff_listcomp.exe")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)
            ok = compile_c_to_binary(c_path, exe_path)
            self.assertTrue(ok, f"Compilação C falhou para diff_listcomp:\n{c_code}")
            res = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            c_out = res.stdout.strip()

        interp_lines = interp_out.splitlines()
        c_lines = c_out.splitlines()
        self.assertEqual(len(interp_lines), len(c_lines))
        for il, cl in zip(interp_lines, c_lines):
            try:
                self.assertAlmostEqual(float(il), float(cl), places=4)
            except ValueError:
                self.assertEqual(il, cl)

    def test_lambda_closures_map_filter(self):
        src = """
fn main():
    fator = 3
    escalar = lambda x: x * fator
    soma2 = fn(a: i64, b: i64) -> i64: a + b
    r1 = escalar(7)
    r2 = soma2(10, 25)
    r3 = (lambda u, v: u * v + 1)(4, 5)
    dobrados = map(lambda x: x * 2, [1, 2, 3, 4])
    maiores = filter(fn(x: i64) -> bool: x >= 3, [1, 2, 3, 4, 5])
"""
        interp = self._run_checked(src)
        self.assertEqual(interp.last_scope["r1"], 21)
        self.assertEqual(interp.last_scope["r2"], 35)
        self.assertEqual(interp.last_scope["r3"], 21)
        self.assertEqual(interp.last_scope["dobrados"], [2, 4, 6, 8])
        self.assertEqual(interp.last_scope["maiores"], [3, 4, 5])

    def test_lambda_diferencial_c99(self):
        TesteDiferencialC.setUpClass()
        if not TesteDiferencialC.compiler_available:
            self.skipTest("Nenhum compilador C disponível para teste diferencial")

        source = """
fn main():
    k = 10
    soma_k = lambda x: x + k
    mult = fn(a: i64, b: i64) -> i64: a * b
    v1 = soma_k(5)
    v2 = mult(6, 7)
    v3 = (lambda x, y: x * x + y)(4, 3)
    mapeados = map(lambda x: x * 3 + k, 0..5)
    pares = filter(fn(x: i64) -> bool: x % 2 == 0, 0..8)
    print(v1)
    print(v2)
    print(v3)
    print(mapeados[3])
    print(pares[2])
"""
        import io, tempfile
        prog = parse(source)
        sem = SemanticAnalyzer()
        sem.analyze(prog)
        self.assertEqual(sem.errors, [], f"Erros semânticos: {sem.errors}")

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
        kwargs = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
        with tempfile.TemporaryDirectory(**kwargs) as tmpdir:
            c_path = os.path.join(tmpdir, "diff_lambda.c")
            exe_path = os.path.join(tmpdir, "diff_lambda.exe")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)
            ok = compile_c_to_binary(c_path, exe_path)
            self.assertTrue(ok, f"Compilação C falhou para diff_lambda:\n{c_code}")
            res = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            c_out = res.stdout.strip()

        interp_lines = interp_out.splitlines()
        c_lines = c_out.splitlines()
        self.assertEqual(len(interp_lines), len(c_lines))
        for il, cl in zip(interp_lines, c_lines):
            try:
                self.assertAlmostEqual(float(il), float(cl), places=4)
            except ValueError:
                self.assertEqual(il, cl)


class TesteAutodiff(unittest.TestCase):
    """Validates the Autodiff engine (graph in Tensor, backward against finite differences, sgd/adam, cross_entropy, and training a 5-layer mini-LLM)."""

    def _check_grad_finite_diff(self, fn: Callable[[], Tensor], params: list[Tensor], eps: float = 1e-6, tol: float = 1e-5):
        zero_grad_fn(params)
        loss = fn()
        loss.backward()
        analytic_grads = [p.grad.data.copy() for p in params]

        for p, g_ana in zip(params, analytic_grads):
            orig = p.data.copy()
            g_num = np.zeros_like(orig, dtype=np.float64)
            it = np.nditer(orig, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                p.data[idx] = orig[idx] + eps
                loss_plus = float(fn().data.item())
                p.data[idx] = orig[idx] - eps
                loss_minus = float(fn().data.item())
                p.data[idx] = orig[idx]
                g_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)
                it.iternext()
            np.testing.assert_allclose(g_ana, g_num, rtol=tol, atol=tol)

    def test_gradientes_contra_diferencas_finitas(self):
        rng = np.random.RandomState(123)
        a = Tensor(rng.randn(3, 4), dtype="f64", requires_grad=True)
        b = Tensor(rng.randn(4), dtype="f64", requires_grad=True)
        w = Tensor(rng.randn(4, 3), dtype="f64", requires_grad=True)
        gamma = Tensor(np.abs(rng.randn(3)) + 0.5, dtype="f64", requires_grad=True)

        # 1. +, -, *, /, @ com broadcasting
        self._check_grad_finite_diff(
            lambda: sum_fn(((a + b) * (a - 0.5 * b) / 1.5) @ w),
            [a, b, w],
        )

        # 2. silu, relu, sigmoid
        self._check_grad_finite_diff(
            lambda: mean_fn(silu_fn(a @ w) + relu_fn(a @ w) * sigmoid_fn(a @ w)),
            [a, w],
        )

        # 3. rmsnorm (2D e 1D) + softmax
        self._check_grad_finite_diff(
            lambda: sum_fn(softmax_fn(rmsnorm_fn(a @ w, gamma)) * Tensor([[0.2, -0.5, 0.8], [0.9, 0.1, -0.3], [-0.4, 0.7, 0.2]], dtype="f64")),
            [a, w, gamma],
        )

        # 4. sdpa (Scaled Dot-Product Attention)
        q = Tensor(rng.randn(3, 4), dtype="f64", requires_grad=True)
        k = Tensor(rng.randn(3, 4), dtype="f64", requires_grad=True)
        v = Tensor(rng.randn(3, 4), dtype="f64", requires_grad=True)
        up_sdpa = Tensor(rng.randn(3, 4), dtype="f64")
        self._check_grad_finite_diff(
            lambda: sum_fn(sdpa_fn(q, k, v) * up_sdpa),
            [q, k, v],
        )

        # 5. cross_entropy (1D e 2D)
        logits_1d = Tensor(rng.randn(5), dtype="f64", requires_grad=True)
        self._check_grad_finite_diff(
            lambda: cross_entropy_fn(logits_1d, 2),
            [logits_1d],
        )
        logits_2d = Tensor(rng.randn(3, 5), dtype="f64", requires_grad=True)
        self._check_grad_finite_diff(
            lambda: cross_entropy_fn(logits_2d, [1, 4, 0]),
            [logits_2d],
        )

    def test_otimizadores_sgd_e_adam_em_nativa(self):
        src = """
fn main():
    w = tensor([3.0, -4.0], "f32", "cpu", true)
    alvo = tensor([1.0, 2.0])
    loss_ini = sum((w - alvo) * (w - alvo))
    for step in 0..80:
        zero_grad(w)
        diff = w - alvo
        loss = sum(diff * diff)
        backward(loss)
        sgd(w, 0.05)
    loss_fim = sum((w - alvo) * (w - alvo))
"""
        prog = parse(src)
        sem = SemanticAnalyzer()
        sem.analyze(prog)
        self.assertEqual(sem.errors, [])
        interp = Interpreter()
        interp.run(prog)
        self.assertLess(float(interp.last_scope["loss_fim"]), float(interp.last_scope["loss_ini"]) * 1e-4)

    def test_treinamento_mini_llm_5_camadas(self):
        src = """
struct Layer:
    wq: tensor[f32]
    wk: tensor[f32]
    wv: tensor[f32]
    wo: tensor[f32]
    w1: tensor[f32]
    w2: tensor[f32]
    rms_att: tensor[f32]
    rms_ffn: tensor[f32]

fn criar_camada(dim: i64, s: i64) -> Layer:
    wq = randn([dim, dim], s) * 0.15
    wk = randn([dim, dim], s + 1) * 0.15
    wv = randn([dim, dim], s + 2) * 0.15
    wo = randn([dim, dim], s + 3) * 0.15
    w1 = randn([dim, dim], s + 4) * 0.15
    w2 = randn([dim, dim], s + 5) * 0.15
    ra = ones([dim])
    rf = ones([dim])
    return Layer(wq, wk, wv, wo, w1, w2, ra, rf)

fn forward_llm(tokens_in: [i64], tok_emb: tensor[f32], camadas: [Layer], rms_f: tensor[f32], w_out: tensor[f32], mask: tensor[f32]) -> tensor[f32]:
    x = tok_emb[tokens_in]
    for c in camadas:
        xn = rmsnorm(x, c.rms_att)
        q = xn @ c.wq
        k = xn @ c.wk
        v = xn @ c.wv
        att = sdpa(q, k, v, mask)
        x = x + (att @ c.wo)
        xf = rmsnorm(x, c.rms_ffn)
        ffn = silu(xf @ c.w1) @ c.w2
        x = x + ffn
    x_norm = rmsnorm(x, rms_f)
    return x_norm @ w_out

fn main():
    vocab_size = 6
    dim = 8
    tok_emb = randn([vocab_size, dim], 10) * 0.2
    c0 = criar_camada(dim, 100)
    c1 = criar_camada(dim, 200)
    c2 = criar_camada(dim, 300)
    c3 = criar_camada(dim, 400)
    c4 = criar_camada(dim, 500)
    camadas = [c0, c1, c2, c3, c4]
    rms_f = ones([dim])
    w_out = randn([dim, vocab_size], 900) * 0.2

    params = [tok_emb, camadas, rms_f, w_out]
    requires_grad(params)

    # Texto pequeno tokenizado: sequência [0, 1, 2, 3, 4] prevendo [1, 2, 3, 4, 5]
    entrada = [0, 1, 2, 3, 4]
    alvo = [1, 2, 3, 4, 5]
    mask = causal_mask(5)

    logits_0 = forward_llm(entrada, tok_emb, camadas, rms_f, w_out, mask)
    loss_inicial = to_f64(cross_entropy(logits_0, alvo))

    for epoca in 0..45:
        zero_grad(params)
        logits = forward_llm(entrada, tok_emb, camadas, rms_f, w_out, mask)
        loss = cross_entropy(logits, alvo)
        backward(loss)
        adam(params, 0.03)

    logits_f = forward_llm(entrada, tok_emb, camadas, rms_f, w_out, mask)
    loss_final = to_f64(cross_entropy(logits_f, alvo))
"""
        prog = parse(src)
        sem = SemanticAnalyzer()
        sem.analyze(prog)
        self.assertEqual(sem.errors, [], f"Semantic errors: {sem.errors}")
        interp = Interpreter()
        interp.run(prog)
        loss_ini = interp.last_scope["loss_inicial"]
        loss_fim = interp.last_scope["loss_final"]
        self.assertLess(loss_fim, loss_ini * 0.15, f"The 5-layer mini-LLM loss did not decrease enough: {loss_ini:.4f} -> {loss_fim:.4f}")
        self.assertLess(loss_fim, 0.15)


class TesteAritmeticaFix256Intervalar(unittest.TestCase):
    """Testes para aritmética de ponto fixo de 256 bits e aritmética intervalar em coordenadas extremas (ex: 10^50)."""

    def test_perda_coerencia_f64_vs_fix256(self):
        # Em IEEE 754 f64 (double), na altitude x = 10^50, passos delta < 10^34 sofrem perda total de coerência
        f_pos = 1e50
        delta = 10.0
        f_next = f_pos + delta
        # Comprovação de falha no double (perda de coerência catastrófica)
        self.assertEqual(f_next - f_pos, 0.0, "Em double float f64, 1e50 + 10.0 - 1e50 deve arredondar para 0.0")

        # Em NatFix256 (256-bit fixed point Q192.64):
        pos = NatFix256("100000000000000000000000000000000000000000000000000")
        step_val = NatFix256(10)
        curr = pos
        for _ in range(1000):
            curr = curr + step_val
        diff = curr - pos
        self.assertEqual(diff.to_i64(), 10000)
        self.assertEqual(diff.to_f64(), 10000.0)
        self.assertEqual(curr.to_str(), "100000000000000000000000000000000000000000000010000.000000")

    def test_aritmetica_intervalar_coerencia(self):
        pos_init = NatFix256("100000000000000000000000000000000000000000000000000")
        pos_iv = NatInterval256(pos_init - NatFix256(1), pos_init + NatFix256(1))
        self.assertEqual(pos_iv.width().to_f64(), 2.0)

        vel_iv = NatInterval256(NatFix256(10) - NatFix256("0.1"), NatFix256(10) + NatFix256("0.1"))
        self.assertEqual(vel_iv.width().to_f64(), 0.2)

        dt = NatFix256(1)
        sim_pos = pos_iv.simulate_steps(vel_iv, dt, 100)
        # Largura final esperada: 2.0 + 100 * 0.2 = 22.0
        self.assertAlmostEqual(sim_pos.width().to_f64(), 22.0, places=4)
        self.assertTrue(sim_pos.is_coherent(NatFix256(25)))
        self.assertFalse(sim_pos.is_coherent(NatFix256(20)))

    def test_aot_ctypes_signatures(self):
        dll = AOTKernelEngine.get_dll()
        self.assertIsNotNone(dll, "AOT DLL do Nativa runtime deve estar disponível e compilada")
        self.assertTrue(hasattr(dll, "nat_c_fix256_add"))
        self.assertTrue(hasattr(dll, "nat_c_interval256_step"))

        c_a = C_NatFix256()
        c_b = C_NatFix256()
        c_out = C_NatFix256()
        dll.nat_c_fix256_from_str(b"100000000000000000000000000000000000000000000000000", 64, ctypes.byref(c_a))
        dll.nat_c_fix256_from_i64(50, 64, ctypes.byref(c_b))
        dll.nat_c_fix256_add(ctypes.byref(c_a), ctypes.byref(c_b), ctypes.byref(c_out))

        buf = ctypes.create_string_buffer(128)
        dll.nat_c_fix256_to_str(ctypes.byref(c_out), buf, 128)
        self.assertEqual(buf.value.decode("utf-8"), "100000000000000000000000000000000000000000000000050.000000")

    def test_transpilacao_c_e2e_fix256(self):
        src = """fn main():
    pos: fix256 = Fix256("100000000000000000000000000000000000000000000000000")
    delta: fix256 = Fix256(25)
    pos = pos + delta
    half_pos: fix256 = pos / Fix256(2)
    iv: interval256 = Interval256(pos, pos + Fix256(10))
    w: fix256 = iv.width()
    coh: bool = iv.is_coherent(Fix256(15))
    iv_small: interval256 = Interval256(Fix256(10), Fix256(20))
    iv_half: interval256 = iv_small / 2
    iv_neg: interval256 = -iv_half
    print("w:", w)
    print("coh:", coh)
    print("half_pos:", half_pos)
    print("iv_half:", iv_half)
    print("iv_neg:", iv_neg)
"""
        prog = parse(src)
        sem = SemanticAnalyzer()
        sem.analyze(prog)
        self.assertEqual(sem.errors, [], f"Erros semânticos: {sem.errors}")

        interp = Interpreter()
        interp.run(prog)

        transpiler = CTranspiler(prog)
        c_code = transpiler.transpile()
        self.assertIn("nat_fix256_add", c_code)
        self.assertIn("nat_interval256_create", c_code)
        self.assertIn("nat_interval256_div_scalar", c_code)
        self.assertIn("nat_interval256_neg", c_code)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = os.path.join(tmpdir, "test_fix256.c")
            exe_file = os.path.join(tmpdir, "test_fix256.exe")
            with open(c_file, "w", encoding="utf-8") as f:
                f.write(c_code)
            ok = compile_c_to_binary(c_file, exe_file)
            self.assertTrue(ok, "Compilação C99 do programa com Fix256/Interval256 deve ser bem-sucedida")
            res = subprocess.run([exe_file], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            self.assertIn("w: 10.000000", res.stdout)
            self.assertIn("coh: true", res.stdout)
            self.assertIn("half_pos: 50000000000000000000000000000000000000000000000012.500000", res.stdout)
            self.assertIn("iv_half: [5.000000, 10.000000]", res.stdout)
            self.assertIn("iv_neg: [-10.000000, -5.000000]", res.stdout)

    def test_edge_cases_fix256(self):
        # Valores negativos
        neg = NatFix256(-42)
        pos = NatFix256(42)
        zero = neg + pos
        self.assertEqual(zero.to_i64(), 0)
        self.assertEqual(zero.to_f64(), 0.0)

        # Multiplicação e Divisão
        v = NatFix256(100)
        half = NatFix256(2)
        res_div = v / half
        self.assertEqual(res_div.to_i64(), 50)
        res_mul = res_div * half
        self.assertEqual(res_mul.to_i64(), 100)

        # Divisão em alta altitude (10^50 e 10^40) sem truncamento
        pos50 = NatFix256("100000000000000000000000000000000000000000000000000")
        half50 = pos50 / 2
        self.assertEqual(half50.to_str(), "50000000000000000000000000000000000000000000000000.000000")
        pos40 = NatFix256("1000000000000000000000000000000000000000")
        half40 = pos40 / 2
        self.assertEqual(half40.to_str(), "500000000000000000000000000000000000000.000000")

        # Reverse division (int / Fix256)
        r_div = 100 / NatFix256(4)
        self.assertEqual(r_div.to_i64(), 25)

        # Interval contains
        iv = NatInterval256(NatFix256(10), NatFix256(20))
        self.assertTrue(iv.contains(NatFix256(15)))
        self.assertTrue(iv.contains(NatFix256(10)))
        self.assertTrue(iv.contains(NatFix256(20)))
        self.assertFalse(iv.contains(NatFix256(5)))
        self.assertFalse(iv.contains(NatFix256(25)))

        # Operações intervalares estendidas: divisão, negação, operadores reversos
        div_scal = iv / 2
        self.assertEqual(div_scal.lo.to_f64(), 5.0)
        self.assertEqual(div_scal.hi.to_f64(), 10.0)

        div_iv = iv / NatInterval256(2, 5)
        self.assertAlmostEqual(div_iv.lo.to_f64(), 2.0, places=4)
        self.assertAlmostEqual(div_iv.hi.to_f64(), 10.0, places=4)

        neg_iv = -iv
        self.assertEqual(neg_iv.lo.to_f64(), -20.0)
        self.assertEqual(neg_iv.hi.to_f64(), -10.0)

        self.assertEqual((10 + iv).lo.to_f64(), 20.0)
        self.assertEqual((2 * iv).hi.to_f64(), 40.0)

        # Ponto médio (mid) em alta altitude (10^50)
        iv50 = NatInterval256(pos50, pos50 + NatFix256(10))
        self.assertEqual(iv50.mid().to_str(), "100000000000000000000000000000000000000000000000005.000000")


# ============================================================================
# COMMAND-LINE INTERFACE (CLI) & ENTRY POINT
# ============================================================================

def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if res.wasSuccessful() else 1


def print_nativa_help() -> None:
    print("""======================================================================
  NATIVA CLI v0.4.0 - Nativa Language (AI, Data, Spectral & Ternary Edition)
======================================================================
Usage: nativa <command> [options]

Main commands:
  nativa run <file.nat>              Execute file in Nativa interpreter
  nativa run <file.nat> --native     Compile AOT to C99/Machine Code and execute
  nativa build <file.nat>            Compile AOT generating native executable (.exe)
  nativa bundle <model.natq4> [-o]   Generate self-contained executable like llamafile
  nativa bench [options]             Run official speed and PPL benchmark
  nativa chat [model.natq4]          Start local Qwen2 chat in INT4
  nativa quantize <dir_hf> [out]     Convert Qwen2 SafeTensors to .natq4 Q4x4
  nativa test                        Run complete test suite (104 tests)
  nativa version                     Display installed version

Preloaded examples:
  nativa run @riemann                Zeros of the Riemann Zeta function
  nativa run @rede                   Multilayer Perceptron Neural Network
  nativa run @ia_tensor              Tensor operations and Autodiff
  nativa run @ternario               BitNet / Trits Ternary Computing
======================================================================""")


def print_detected_devices() -> int:
    print("====================================================================")
    print("  NATIVA - DETECTED COMPUTATION DEVICES")
    print("====================================================================")
    print(f"  [CPU Host] OS: {sys.platform} | Python {sys.version.split()[0]}")

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
                        print(f"  [{total_devs}] [{'GPU' if is_gpu else 'CPU'}] {d_name} (Platform: {p_name})")
                        total_devs += 1
            if total_devs == 0:
                print("  No active OpenCL device found.")
        else:
            print("  No active OpenCL platform found.")
    except Exception:
        print("  [OpenCL] No active OpenCL library (standard native CPU C99/OpenMP mode).")

    print("====================================================================")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    if not raw_args or (len(raw_args) == 1 and raw_args[0].lower() in ("help", "-h", "--help")):
        print_nativa_help()
        return 0

    # Modern subcommands NodeJS / Cargo / Go / Python CLI style
    subcmd = raw_args[0].lower()
    if subcmd in ("version", "--version", "-v"):
        print("Nativa v0.4.0 (AI, Data, Spectral & Ternary Edition)")
        return 0
    elif subcmd == "run":
        if len(raw_args) == 1:
            print("Usage: nativa run <file.nat|@example> [--native]")
            return 1
        clean = [a for a in raw_args[1:] if a not in ("--native", "--aot")]
        if "--native" in raw_args or "--aot" in raw_args:
            return main([clean[0], "--build", "--run"] + clean[1:])
        return main([raw_args[1], "--run"] + raw_args[2:])
    elif subcmd == "build":
        if len(raw_args) == 1:
            print("Usage: nativa build <file.nat|@example>")
            return 1
        return main([raw_args[1], "--build"] + raw_args[2:])
    elif subcmd in ("test", "--test"):
        return run_tests()
    elif subcmd == "chat":
        model_target = raw_args[1] if len(raw_args) > 1 else os.path.join(os.path.dirname(__file__), "data", "qwen2.5_3b.natq4")
        chat_script = os.path.join(os.path.dirname(__file__), "tools", "chat_natq4.py")
        if not os.path.isfile(chat_script):
            print(f"[ERROR] Chat script not found: {chat_script}", file=sys.stderr)
            return 1
        return subprocess.call([sys.executable, chat_script, model_target])
    elif subcmd == "quantize":
        quant_script = os.path.join(os.path.dirname(__file__), "tools", "nativa_quantize.py")
        if not os.path.isfile(quant_script):
            print(f"[ERROR] Quantization script not found: {quant_script}", file=sys.stderr)
            return 1
        return subprocess.call([sys.executable, quant_script] + raw_args[1:])
    elif subcmd == "bundle":
        bundle_script = os.path.join(os.path.dirname(__file__), "tools", "nativa_bundle.py")
        if not os.path.isfile(bundle_script):
            print(f"[ERROR] Bundle script not found: {bundle_script}", file=sys.stderr)
            return 1
        return subprocess.call([sys.executable, bundle_script] + raw_args[1:])
    elif subcmd in ("bench", "benchmark"):
        bench_script = os.path.join(os.path.dirname(__file__), "tools", "nativa_benchmark.py")
        if not os.path.isfile(bench_script):
            print(f"[ERROR] Benchmark script not found: {bench_script}", file=sys.stderr)
            return 1
        return subprocess.call([sys.executable, bench_script] + raw_args[1:])
    elif subcmd in ("devices", "--devices", "opencl", "--opencl"):
        return print_detected_devices()

    ap = argparse.ArgumentParser(description="Nativa v0.4 (AI, Data, Spectral & Ternary Edition)")
    ap.add_argument("arquivo", nargs="?", help="file .nat or @example | @riemann | @rede | @ia_tensor | @ternario")
    ap.add_argument("--ast", action="store_true", help="print AST")
    ap.add_argument("--tokens", action="store_true", help="print tokens")
    ap.add_argument("--run", action="store_true", help="execute program (runtime mode)")
    ap.add_argument("--emit-c", metavar="C_FILE", nargs="?", const="", help="emit corresponding C99 code")
    ap.add_argument("--build", action="store_true", help="compile AOT C99 code to native executable (.exe)")
    ap.add_argument("--test", action="store_true", help="run test suite")
    ap.add_argument("--devices", action="store_true", help="list available computation devices (CPU, OpenCL GPU/CPU)")
    ap.add_argument("--opencl", action="store_true", help="list available OpenCL devices")
    ap.add_argument("-v", "--version", action="version", version="Nativa v0.4.0 (AI, Data, Spectral & Ternary Edition)")
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
            ap.error(f"unknown example '{nome}' (options: {', '.join(EXEMPLOS)})")
        source = EXEMPLOS[nome]
    elif args.arquivo in EXEMPLOS:
        source = EXEMPLOS[args.arquivo]
    else:
        if not os.path.isfile(args.arquivo):
            print(f"[ERROR] File not found: '{args.arquivo}'", file=sys.stderr)
            return 1
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

    erros = sum(d.level in ("error", "erro") for d in diags)
    avisos = len(diags) - erros
    fns = sum(isinstance(i, Function) for i in prog.items)

    if erros:
        print(f"FAILED: {fns} function(s), {erros} error(s), {avisos} warning(s)")
        return 1

    if args.emit_c is not None:
        c_code = CTranspiler(prog).transpile()
        out_c = args.emit_c if args.emit_c else (args.arquivo.replace(".nat", "") + ".c" if not args.arquivo.startswith("@") else args.arquivo[1:] + ".c")
        with open(out_c, "w", encoding="utf-8") as f:
            f.write(c_code)
        print(f"[OK] C99 code successfully emitted: {out_c}")
        return 0

    if args.build:
        base_name = args.arquivo.replace(".nat", "") if not args.arquivo.startswith("@") else args.arquivo[1:]
        out_c = base_name + ".c"
        out_exe = base_name + (".exe" if sys.platform == "win32" else "")
        c_code = CTranspiler(prog).transpile()
        with open(out_c, "w", encoding="utf-8") as f:
            f.write(c_code)
        print(f"[*] C99 code emitted: {out_c}")
        t0 = time.perf_counter()
        ok = compile_c_to_binary(out_c, out_exe)
        t_comp = time.perf_counter() - t0
        if ok:
            print(f"[OK] Native binary successfully compiled in {t_comp:.3f}s: {out_exe}")
            if args.run:
                print(f"[*] Running {out_exe} natively:\n")
                subprocess.run([os.path.abspath(out_exe)])
            return 0
        else:
            print(f"[ERROR] Failed to compile native binary.", file=sys.stderr)
            return 1

    if args.run:
        interp = Interpreter()
        try:
            interp.run(prog)
        except NatRuntimeError as e:
            print(e.pretty(source, args.arquivo), file=sys.stderr)
            return 1
        return 0

    print(f"OK: {fns} function(s), {erros} error(s), {avisos} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

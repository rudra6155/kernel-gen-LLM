"""Candidate policy: the kernel must be written in Triton, not delegated to PyTorch.

Two layers, aimed at reward hacking and accidents rather than a hostile author:

1. check_source: static AST check, run before any candidate code executes.
2. NativeOpGuard: records every aten op the candidate triggers while it runs.
   It catches what the AST cannot see, such as `x * w + b` or `x @ w` on
   tensors, or a library call reached through an alias.

Neither layer is a security sandbox. Process isolation lives in kernelgen.sandbox.
Caching results between calls is handled by the timing run, not here: it uses new
data on every call and checks each output (kernelgen.timing.benchmark_fresh). A
candidate that behaves differently once timing starts is only caught if that
makes its output wrong.
"""

from __future__ import annotations

import ast

from torch.utils._python_dispatch import TorchDispatchMode

# Modules a candidate may import.
ALLOWED_MODULES = {"torch", "triton", "triton.language", "math", "typing", "functools", "__future__"}

# torch.<name> that host code may use: allocation, dtypes, and device queries.
TORCH_ALLOWED = {
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like",
    "full", "full_like", "Tensor", "Size", "device", "dtype", "finfo", "iinfo",
    "float16", "float32", "float64", "bfloat16", "half", "float", "double",
    "int8", "int16", "int32", "int64", "uint8", "bool", "short", "int", "long",
    "cuda",
}
CUDA_ALLOWED = {
    "current_device", "current_stream", "device", "get_device_capability", "get_device_properties",
}
TRITON_ALLOWED = {"jit", "autotune", "heuristics", "Config", "cdiv", "next_power_of_2", "language", "constexpr"}

# Method names host code may call on any non-module receiver.
ALLOWED_METHODS = {
    # tensor metadata, views, allocation
    "contiguous", "stride", "size", "dim", "numel", "element_size", "is_contiguous", "data_ptr",
    "view", "reshape", "flatten", "transpose", "permute", "t", "unsqueeze", "squeeze", "expand",
    "new_empty", "new_zeros", "new_empty_strided", "zero_", "fill_",
    # dtype casts
    "to", "float", "half", "bfloat16", "double",
    # plain containers
    "append", "extend", "insert", "pop", "get", "items", "keys", "values", "index", "count",
}

BANNED_NAMES = {
    "eval", "exec", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "open", "input", "breakpoint", "exit", "quit",
}

# aten ops that move or allocate data without computing anything.
ALLOWED_ATEN = {
    "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "new_zeros",
    "zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "zero_", "fill_",
    "view", "_unsafe_view", "alias", "_reshape_alias", "expand", "permute", "transpose", "t",
    "unsqueeze", "squeeze", "select", "slice", "as_strided", "detach",
    "clone", "copy_", "_to_copy", "set_",
}


class PolicyViolation(Exception):
    def __init__(self, ops: list[str]):
        self.ops = ops
        super().__init__("native PyTorch ops used: " + ", ".join(ops))


class NativeOpGuard(TorchDispatchMode):
    """Records, without blocking, every aten op outside ALLOWED_ATEN.

    Violations are recorded rather than raised so that a candidate cannot hide
    one behind try/except.
    """

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.name()).split(".")[0] if hasattr(func, "name") else str(func)
        namespace, _, op = name.partition("::")
        if namespace != "aten" or op not in ALLOWED_ATEN:
            if name not in self.violations:
                self.violations.append(name)
        return func(*args, **(kwargs or {}))


def _chain(node: ast.AST) -> tuple[str, list[str]] | None:
    """For `a.b.c` return ("a", ["b", "c"]); None if the base is not a plain name."""
    attrs: list[str] = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, attrs[::-1]
    return None


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.modules: dict[str, str] = {}  # local name -> dotted module
        self.origins: dict[str, str] = {}  # local name -> "module.name" for from-imports
        self.jit_kernels: set[str] = set()
        self.launched: set[str] = set()
        self.jit_depth = 0

    def flag(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    # -- imports -------------------------------------------------------------
    def collect_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name not in ALLOWED_MODULES:
                        self.flag(node, f"import of module '{a.name}' is not allowed")
                    elif a.asname:
                        self.modules[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.modules[top] = top
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level or mod not in ALLOWED_MODULES:
                    self.flag(node, f"import from module '{'.' * node.level}{mod}' is not allowed")
                    continue
                for a in node.names:
                    local = a.asname or a.name
                    if a.name == "*":
                        self.flag(node, "star import is not allowed")
                    elif mod == "torch" and (a.name not in TORCH_ALLOWED or a.name == "cuda"):
                        self.flag(node, f"'from torch import {a.name}' is not allowed")
                    elif mod == "triton" and a.name not in TRITON_ALLOWED:
                        self.flag(node, f"'from triton import {a.name}' is not allowed")
                    elif mod == "triton" and a.name == "language":
                        self.modules[local] = "triton.language"
                    else:
                        self.origins[local] = f"{mod}.{a.name}"

    def resolve(self, node: ast.AST) -> str | None:
        """Dotted origin of a name or attribute chain, following import aliases."""
        c = _chain(node)
        if c is None:
            return None
        root, attrs = c
        base = self.modules.get(root) or self.origins.get(root)
        return ".".join([base, *attrs]) if base else None

    def is_jit(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for dec in fn.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if self.resolve(target) == "triton.jit":
                return True
        return False

    # -- traversal -----------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        jit = self.is_jit(node)
        if jit:
            self.jit_kernels.add(node.name)
            self.jit_depth += 1
        self.generic_visit(node)
        if jit:
            self.jit_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self.flag(node, f"use of '{node.id}' is not allowed")
        elif node.id.startswith("__"):
            self.flag(node, f"use of '{node.id}' is not allowed")
        elif self.modules.get(node.id) in ("torch", "triton"):
            self.flag(node, f"bare reference to module '{node.id}'; use it only as {node.id}.<name>")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.flag(node, f"access to '{node.attr}' is not allowed")
        c = _chain(node)
        if c is not None and c[0] in self.modules:
            self.check_module_chain(node, self.modules[c[0]], c[1])
            return  # do not visit the base name: it is a legitimate module reference
        self.generic_visit(node)

    def check_module_chain(self, node: ast.AST, module: str, attrs: list[str]) -> None:
        if module == "torch":
            first = attrs[0] if attrs else ""
            if first not in TORCH_ALLOWED:
                self.flag(node, f"torch.{'.'.join(attrs)} is not allowed; write the kernel in Triton")
            elif first == "cuda" and (len(attrs) < 2 or attrs[1] not in CUDA_ALLOWED):
                self.flag(node, f"torch.{'.'.join(attrs)} is not allowed")
        elif module == "triton":
            first = attrs[0] if attrs else ""
            if first not in TRITON_ALLOWED:
                self.flag(node, f"triton.{'.'.join(attrs)} is not allowed")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult) and self.jit_depth == 0:
            self.flag(node, "matrix multiplication '@' outside a Triton kernel")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, ast.MatMult) and self.jit_depth == 0:
            self.flag(node, "matrix multiplication '@=' outside a Triton kernel")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.jit_depth == 0:
            func = node.func
            if isinstance(func, ast.Attribute):
                c = _chain(func)
                on_module = c is not None and c[0] in self.modules
                if not on_module and func.attr not in ALLOWED_METHODS:
                    self.flag(node, f"method call '.{func.attr}()' is not allowed in host code")
            elif isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name):
                if func.value.id in self.jit_kernels:
                    self.launched.add(func.value.id)
        self.generic_visit(node)


def check_source(source: str) -> list[str]:
    """Return policy violations in `source`, or an empty list if it passes.

    Nothing in `source` is executed.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return [f"cannot parse source: {type(exc).__name__}: {exc}"]

    checker = _Checker()
    try:
        checker.collect_imports(tree)
        checker.visit(tree)
    except RecursionError:
        return ["source is too deeply nested to check"]

    violations = checker.violations
    if not any(isinstance(n, ast.FunctionDef) and n.name == "solution" for n in tree.body):
        violations.append("no top-level function named 'solution'")
    if not checker.jit_kernels:
        violations.append("no @triton.jit kernel is defined")
    elif not checker.launched:
        violations.append("a @triton.jit kernel is defined but never launched as kernel[grid](...)")
    return violations

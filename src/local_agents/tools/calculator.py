"""A safe arithmetic calculator built on the ``ast`` module (no ``eval``)."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "log",
        "log2",
        "log10",
        "exp",
        "floor",
        "ceil",
        "fabs",
        "factorial",
        "gcd",
        "hypot",
        "degrees",
        "radians",
        "comb",
        "perm",
        "prod",
        "fsum",
        "isqrt",
        "trunc",
    )
}
_FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
MAX_POWER = 10_000
MAX_FACTORIAL = 5_000


def evaluate(expression: str) -> Any:
    """Evaluate an arithmetic expression safely."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(
            f"Invalid expression: {exc.msg}", hint=f"Expression was: {expression}"
        ) from exc
    try:
        return _eval(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc
    except OverflowError as exc:
        raise ToolError("Result is too large") from exc
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Math error: {exc}") from exc


def _eval(node: ast.AST) -> Any:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(
            f"Unknown name {node.id!r}", hint=f"Known constants: {', '.join(_CONSTANTS)}"
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(right, (int, float))
            and abs(right) > MAX_POWER
        ):
            raise ToolError(f"Exponent too large (max {MAX_POWER})")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ToolError(
                f"Unknown function {node.func.id!r}", hint=f"Known: {', '.join(sorted(_FUNCTIONS))}"
            )
        args = [_eval(a) for a in node.args]
        if node.keywords:
            raise ToolError("Keyword arguments are not supported")
        if node.func.id == "factorial" and args and args[0] > MAX_FACTORIAL:
            raise ToolError(f"factorial argument too large (max {MAX_FACTORIAL})")
        return fn(*args)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(e) for e in node.elts]
    raise ToolError(
        f"Unsupported syntax: {type(node).__name__}",
        hint="Only arithmetic and math functions are allowed.",
    )


class CalculatorTool(BaseTool):
    name = "calculate"
    description = (
        "Evaluate an arithmetic expression exactly. Supports + - * / // % **, parentheses, "
        "constants pi/e/tau, and math functions (sqrt, sin, cos, log, exp, factorial, gcd, "
        "comb, round, min, max, sum, ...). Use it for every numeric computation."
    )
    parameters = (
        ToolParameter("expression", "string", "Expression, e.g. '(17.5 * 3) ** 2 / sqrt(2)'"),
    )
    permission = Permission.READ

    async def execute(self, expression: str) -> ToolResult:
        value = evaluate(expression)
        if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
            shown: Any = int(value)
        elif isinstance(value, float):
            shown = f"{value:.12g}"
        else:
            shown = value
        return ToolResult.ok(f"{expression.strip()} = {shown}", data={"result": value})


def calculator_tools() -> list[BaseTool]:
    return [CalculatorTool()]

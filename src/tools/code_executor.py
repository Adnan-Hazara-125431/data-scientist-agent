"""Safely execute Python code and capture stdout, stderr, and status."""

from __future__ import annotations

import io
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "__import__": __import__,
    # Exception types — needed so generated code can use try/except blocks
    # (e.g. `except (ValueError, TypeError):`) without crashing with a
    # NameError because the exception class itself isn't in scope.
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "OverflowError": OverflowError,
    "FileNotFoundError": FileNotFoundError,
    "ImportError": ImportError,
    "NameError": NameError,
}


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown fences or return raw text."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def execute_code(
    code: str,
    df: pd.DataFrame | None = None,
    extra_locals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute Python code in a restricted namespace.

    IMPORTANT: We use a SINGLE namespace dict for both globals and locals.
    If you pass separate dicts to exec(), any function DEFINED inside the
    executed code gets its __globals__ bound to the globals dict only — so
    nested functions can't see anything (like `pd`, `np`, `df`) that was
    only placed in the locals dict. Using one shared dict avoids this
    scoping trap entirely.

    Returns dict with keys: success, stdout, stderr, error, result_df, figures.
    """
    code = extract_python_code(code)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    namespace: dict[str, Any] = {
        "__builtins__": ALLOWED_BUILTINS,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "df": df.copy() if df is not None else pd.DataFrame(),
    }
    if extra_locals:
        namespace.update(extra_locals)

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, namespace)

        result_df = namespace.get("df")
        if result_df is None:
            result_df = namespace.get("result_df")

        figs = [plt.figure(i) for i in plt.get_fignums()]

        return {
            "success": True,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error": None,
            "result_df": result_df if isinstance(result_df, pd.DataFrame) else None,
            "figures": figs,
            "locals": {
                k: v
                for k, v in namespace.items()
                if not k.startswith("_") and k not in ("pd", "np", "plt", "sns")
            },
        }
    except Exception:
        plt.close("all")
        return {
            "success": False,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error": traceback.format_exc(),
            "result_df": None,
            "figures": [],
            "locals": {},
        }
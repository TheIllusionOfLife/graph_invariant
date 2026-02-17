import ast
import logging
import math
import multiprocessing as mp
import os
import signal
import types
from multiprocessing.pool import Pool
from typing import Any

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows.
    resource = None

MAX_CODE_LENGTH = 100_000  # 100KB limit for source code
MAX_AST_NODES = 5_000  # Prevent complex ASTs

FORBIDDEN_CALLS = {"getattr", "setattr", "delattr", "globals", "locals", "vars"}
ALLOWED_CALLS = {
    "abs",
    "min",
    "max",
    "sum",
    "len",
    "sorted",
    "range",
    "enumerate",
    "float",
    "int",
    "list",
    "tuple",
    "dict",
    "set",
    "zip",
    "map",
    "round",
    "bool",
    "str",
    "pow",
    "any",
    "all",
    "reversed",
}
FORBIDDEN_ATTR_BASES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "nx",
    "G",
}
ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.If,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.keyword,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
)

_SAFE_NP_ATTRS: set[str] = {
    # Array creation
    "array",
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "linspace",
    "zeros_like",
    "ones_like",
    "full_like",
    "empty_like",
    "eye",
    "identity",
    "diag",
    # Element-wise math
    "abs",
    "absolute",
    "sqrt",
    "cbrt",
    "square",
    "log",
    "log2",
    "log10",
    "log1p",
    "exp",
    "exp2",
    "expm1",
    "power",
    "float_power",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "around",
    "sign",
    "clip",
    "mod",
    "remainder",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "negative",
    "positive",
    "reciprocal",
    # Trigonometric
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "degrees",
    "radians",
    "hypot",
    # Aggregation / reduction
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "median",
    "average",
    "min",
    "max",
    "argmin",
    "argmax",
    "nansum",
    "nanprod",
    "nanmean",
    "nanstd",
    "nanvar",
    "nanmedian",
    "nanmin",
    "nanmax",
    "percentile",
    "quantile",
    "count_nonzero",
    # Sorting / searching
    "sort",
    "argsort",
    "searchsorted",
    "where",
    "nonzero",
    "unique",
    # Shape manipulation
    "reshape",
    "ravel",
    "transpose",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "squeeze",
    "expand_dims",
    "tile",
    "repeat",
    "flip",
    "fliplr",
    "flipud",
    # Differences / cumulative
    "diff",
    "cumsum",
    "cumprod",
    "gradient",
    # Linear algebra basics
    "dot",
    "inner",
    "outer",
    "matmul",
    "cross",
    "trace",
    # Statistics
    "corrcoef",
    "cov",
    "histogram",
    "bincount",
    # Logic / comparison
    "isnan",
    "isinf",
    "isfinite",
    "all",
    "any",
    "allclose",
    "isclose",
    "logical_and",
    "logical_or",
    "logical_not",
    # Constants
    "pi",
    "e",
    "inf",
    "nan",
    "newaxis",
    # Types
    "float64",
    "float32",
    "int64",
    "int32",
    "uint8",
    "bool_",
    "dtype",
    # Utility
    "asarray",
    "copy",
    "convolve",
    "correlate",
    "interp",
    "polyfit",
    "polyval",
}

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "range": range,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "zip": zip,
    "map": map,
    "round": round,
    "bool": bool,
    "str": str,
    "pow": pow,
    "any": any,
    "all": all,
    "reversed": reversed,
}

_SAFE_NP_ATTRS_DICT: dict[str, Any] = {}
for name in _SAFE_NP_ATTRS:
    val = getattr(np, name, None)
    if val is not None:
        _SAFE_NP_ATTRS_DICT[name] = val

LOGGER = logging.getLogger(__name__)
_TASK_TIMEOUT_SEC = 0.0
_COMPILED_CODE_CACHE: dict[str, Any] = {}


class CandidateTimeoutError(RuntimeError):
    """Raised when a candidate evaluation exceeds per-graph timeout."""


def _timeout_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise CandidateTimeoutError("candidate evaluation timed out")


def _initialize_worker(memory_mb: int, timeout_sec: float) -> None:
    global _TASK_TIMEOUT_SEC
    global _COMPILED_CODE_CACHE
    _TASK_TIMEOUT_SEC = timeout_sec
    _COMPILED_CODE_CACHE = {}

    if resource is not None:
        memory_bytes = max(64, memory_mb) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (OSError, ValueError):
            LOGGER.debug("failed to set RLIMIT_AS for worker", exc_info=True)

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)


def _validate_ast(tree: ast.AST) -> tuple[bool, str | None]:
    # Check complexity (number of nodes)
    try:
        node_count = sum(1 for _ in ast.walk(tree))
    except (RecursionError, MemoryError) as e:
        return False, f"AST traversal failed: {type(e).__name__}"

    if node_count > MAX_AST_NODES:
        return False, f"code too complex: {node_count} AST nodes (max {MAX_AST_NODES})"

    fn_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if len(fn_defs) != 1 or fn_defs[0].name != "new_invariant":
        return False, "code must define exactly one function named `new_invariant`"

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            return False, f"disallowed syntax: {type(node).__name__}"

        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, f"forbidden name detected: {node.id}"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"forbidden attribute detected: {node.attr}"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return False, f"forbidden call detected: {node.func.id}"
                if node.func.id not in ALLOWED_CALLS:
                    return False, f"non-whitelisted call detected: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in FORBIDDEN_ATTR_BASES:
                        return False, f"forbidden call base: {node.func.value.id}"
            else:
                return False, "disallowed call expression"
    return True, None


def validate_code_static(code: str) -> tuple[bool, str | None]:
    # Best-effort defense for research use only.
    # Running fully untrusted code safely requires stronger isolation (e.g., containers/jails).

    if len(code) > MAX_CODE_LENGTH:
        return False, f"code too long: {len(code)} chars (max {MAX_CODE_LENGTH})"

    if "def new_invariant(" not in code:
        return False, "missing `new_invariant` function"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"invalid syntax: {exc.msg}"
    except RecursionError:
        return False, "code too complex: recursion limit exceeded during parsing"
    except MemoryError:
        return False, "code too complex: memory limit exceeded during parsing"
    except Exception as e:
        return False, f"parser error: {type(e).__name__}"

    return _validate_ast(tree)


def _compiled_candidate_code(code: str) -> Any:
    cached = _COMPILED_CODE_CACHE.get(code)
    if cached is not None:
        return cached
    compiled = compile(code, "<sandbox>", "exec")
    _COMPILED_CODE_CACHE[code] = compiled
    return compiled


def _safe_numpy() -> types.SimpleNamespace:
    """Return a restricted numpy namespace exposing only safe numerical functions."""
    return types.SimpleNamespace(**_SAFE_NP_ATTRS_DICT)


def _safe_globals() -> dict[str, Any]:
    return {
        "__builtins__": _SAFE_BUILTINS.copy(),
        "math": math,
        "np": _safe_numpy(),
    }


def _run_candidate(code: str, features: dict[str, Any]) -> float | None:
    return _run_candidate_detailed(code, features).get("value")


def _run_candidate_detailed(code: str, features: dict[str, Any]) -> dict[str, Any]:
    safe_globals = _safe_globals()
    safe_locals: dict[str, Any] = {}
    try:
        if _TASK_TIMEOUT_SEC > 0 and hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, _TASK_TIMEOUT_SEC)
        compiled_code = _compiled_candidate_code(code)
        exec(compiled_code, safe_globals, safe_locals)
        fn = safe_locals.get("new_invariant")
        if fn is None:
            return {"value": None, "error_type": "runtime_exception", "error_detail": "missing_fn"}
        value = fn(features)
        if value is None:
            return {
                "value": None,
                "error_type": "runtime_exception",
                "error_detail": "returned_none",
            }
        return {"value": float(value), "error_type": None, "error_detail": None}
    except CandidateTimeoutError:
        return {
            "value": None,
            "error_type": "timeout",
            "error_detail": "candidate evaluation timed out",
        }
    except Exception as exc:  # pragma: no cover - type variation depends on candidate/runtime.
        LOGGER.debug("candidate execution failed", exc_info=True)
        return {
            "value": None,
            "error_type": "runtime_exception",
            "error_detail": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)


class SandboxEvaluator:
    """Reusable evaluator that keeps a worker pool alive across calls."""

    def __init__(
        self,
        timeout_sec: float,
        memory_mb: int,
        max_workers: int | None = None,
    ):
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb
        self._pool: Pool | None = None
        cpu_workers = max(1, os.cpu_count() or 1)
        if max_workers is None:
            self._worker_count = cpu_workers
        else:
            self._worker_count = max(1, min(max_workers, cpu_workers))

    def __enter__(self) -> "SandboxEvaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        self.close()
        return False

    def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        self._pool.join()
        self._pool = None

    def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        context = mp.get_context()
        self._pool = context.Pool(
            processes=self._worker_count,
            initializer=_initialize_worker,
            initargs=(self.memory_mb, self.timeout_sec),
            maxtasksperchild=100,
        )

    def _evaluate_once(self, code: str, features_list: list[dict[str, Any]]) -> list[float | None]:
        if self._pool is None:
            raise RuntimeError("sandbox pool is not initialized")
        tasks = [(code, features) for features in features_list]
        # Keep chunksize small for fairness; tune upward if IPC becomes dominant.
        return self._pool.starmap(_run_candidate, tasks, chunksize=1)

    def _evaluate_once_detailed(
        self, code: str, features_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self._pool is None:
            raise RuntimeError("sandbox pool is not initialized")
        tasks = [(code, features) for features in features_list]
        return self._pool.starmap(_run_candidate_detailed, tasks, chunksize=1)

    def evaluate(self, code: str, features_list: list[dict[str, Any]]) -> list[float | None]:
        ok, _ = validate_code_static(code)
        if not ok:
            return [None for _ in features_list]
        if not features_list:
            return []

        self._ensure_pool()
        try:
            return self._evaluate_once(code, features_list)
        except (BrokenPipeError, EOFError, OSError):
            LOGGER.debug("sandbox pool failed; rebuilding once", exc_info=True)
            self.close()
            self._ensure_pool()
            return self._evaluate_once(code, features_list)

    def evaluate_detailed(
        self, code: str, features_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        ok, reason = validate_code_static(code)
        if not ok:
            return [
                {"value": None, "error_type": "static_invalid", "error_detail": reason}
                for _ in features_list
            ]
        if not features_list:
            return []

        self._ensure_pool()
        try:
            return self._evaluate_once_detailed(code, features_list)
        except (BrokenPipeError, EOFError, OSError):
            LOGGER.debug("sandbox pool failed; rebuilding once", exc_info=True)
            self.close()
            self._ensure_pool()
            return self._evaluate_once_detailed(code, features_list)


def evaluate_candidate_on_features(
    code: str,
    features_list: list[dict[str, Any]],
    timeout_sec: float,
    memory_mb: int,
    max_workers: int | None = None,
) -> list[float | None]:
    with SandboxEvaluator(
        timeout_sec=timeout_sec,
        memory_mb=memory_mb,
        max_workers=max_workers,
    ) as evaluator:
        return evaluator.evaluate(code, features_list)

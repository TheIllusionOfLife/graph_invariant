import ast
import logging
import math
import multiprocessing as mp
import os
import signal
from multiprocessing.pool import Pool
from typing import Any

import networkx as nx

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows.
    resource = None

FORBIDDEN_PATTERNS = [
    "import ",
    "__import__",
    "eval(",
    "exec(",
    "open(",
    "os.",
    "sys.",
    "subprocess",
    "__class__",
    "__subclasses__",
    "__globals__",
]
FORBIDDEN_CALLS = {"getattr", "setattr", "delattr", "globals", "locals", "vars"}
ALLOWED_CALLS = {"abs", "min", "max", "sum", "len", "sorted", "range", "enumerate", "float", "int"}
ALLOWED_ATTR_BASES = {"G", "math"}
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
)

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
                if not isinstance(node.func.value, ast.Name):
                    return False, "disallowed attribute call target"
                if node.func.value.id not in ALLOWED_ATTR_BASES:
                    return False, f"disallowed call base: {node.func.value.id}"
            else:
                return False, "disallowed call expression"
    return True, None


def validate_code_static(code: str) -> tuple[bool, str | None]:
    # Best-effort defense for research use only.
    # Running fully untrusted code safely requires stronger isolation (e.g., containers/jails).
    for token in FORBIDDEN_PATTERNS:
        if token in code:
            return False, f"forbidden token detected: {token}"
    if "def new_invariant(" not in code:
        return False, "missing `new_invariant` function"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"invalid syntax: {exc.msg}"
    return _validate_ast(tree)


def _compiled_candidate_code(code: str) -> Any:
    cached = _COMPILED_CODE_CACHE.get(code)
    if cached is not None:
        return cached
    compiled = compile(code, "<sandbox>", "exec")
    _COMPILED_CODE_CACHE[code] = compiled
    return compiled


def _run_candidate(code: str, graph: nx.Graph) -> float | None:
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
    }
    safe_globals: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
    }
    safe_locals: dict[str, Any] = {}
    try:
        if _TASK_TIMEOUT_SEC > 0 and hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, _TASK_TIMEOUT_SEC)
        compiled_code = _compiled_candidate_code(code)
        exec(compiled_code, safe_globals, safe_locals)
        fn = safe_locals.get("new_invariant")
        if fn is None:
            return None
        value = fn(graph)
        if value is None:
            return None
        return float(value)
    except CandidateTimeoutError:
        return None
    except Exception:
        LOGGER.debug("candidate execution failed", exc_info=True)
        return None
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)


def _run_candidate_with_queue_result(code: str, graph: nx.Graph) -> float | None:
    return _run_candidate(code, graph)


class SandboxEvaluator:
    """Reusable evaluator that keeps a worker pool alive across calls."""

    def __init__(self, timeout_sec: float, memory_mb: int, max_workers: int | None = None):
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

    def _evaluate_once(self, code: str, graphs: list[nx.Graph]) -> list[float | None]:
        if self._pool is None:
            raise RuntimeError("sandbox pool is not initialized")
        tasks = [(code, graph) for graph in graphs]
        # Keep chunksize small for fairness; tune upward if IPC becomes dominant.
        return self._pool.starmap(_run_candidate_with_queue_result, tasks, chunksize=1)

    def evaluate(self, code: str, graphs: list[nx.Graph]) -> list[float | None]:
        ok, _ = validate_code_static(code)
        if not ok:
            return [None for _ in graphs]
        if not graphs:
            return []

        self._ensure_pool()
        try:
            return self._evaluate_once(code, graphs)
        except (BrokenPipeError, EOFError, OSError):
            LOGGER.debug("sandbox pool failed; rebuilding once", exc_info=True)
            self.close()
            self._ensure_pool()
            return self._evaluate_once(code, graphs)


def evaluate_candidate_on_graphs(
    code: str,
    graphs: list[nx.Graph],
    timeout_sec: float,
    memory_mb: int,
    max_workers: int | None = None,
) -> list[float | None]:
    with SandboxEvaluator(
        timeout_sec=timeout_sec,
        memory_mb=memory_mb,
        max_workers=max_workers,
    ) as evaluator:
        return evaluator.evaluate(code, graphs)

import ast
import math
import multiprocessing as mp
from queue import Empty
from typing import Any

import networkx as nx
import numpy as np

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

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return False, f"forbidden call detected: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"forbidden attribute detected: {node.attr}"
    return True, None


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
        "np": np,
        "nx": nx,
        "math": math,
    }
    safe_locals: dict[str, Any] = {}
    try:
        exec(code, safe_globals, safe_locals)
        fn = safe_locals.get("new_invariant", safe_globals.get("new_invariant"))
        if fn is None:
            return None
        value = fn(graph)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _run_candidate_with_queue(code: str, graph: nx.Graph, queue: Any) -> None:
    queue.put(_run_candidate(code, graph))


def evaluate_candidate_on_graphs(
    code: str, graphs: list[nx.Graph], timeout_sec: float, memory_mb: int
) -> list[float | None]:
    del memory_mb  # Reserved for future process-level memory limits.
    ok, _ = validate_code_static(code)
    if not ok:
        return [None for _ in graphs]

    context = mp.get_context()
    results: list[float | None] = []
    for graph in graphs:
        queue: mp.Queue = context.Queue(maxsize=1)
        process = context.Process(target=_run_candidate_with_queue, args=(code, graph, queue))
        process.start()
        process.join(timeout=timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join()
            results.append(None)
            continue

        try:
            results.append(queue.get_nowait())
        except Empty:
            results.append(None)
        except Exception:
            results.append(None)
    return results

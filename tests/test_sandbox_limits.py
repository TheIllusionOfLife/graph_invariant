
import ast
import sys
import pytest
from graph_invariant.sandbox import validate_code_static, MAX_CODE_LENGTH, MAX_AST_NODES

# Increase recursion limit slightly to allow test setup if needed,
# though we rely on flat structures mostly
sys.setrecursionlimit(2000)

def test_large_code_check():
    """Verify that code exceeding MAX_CODE_LENGTH is rejected."""
    # Generate a large code string that is valid python but very long
    # Use a comment to inflate size cheaply without affecting parsing complexity too much
    padding = "#" * (MAX_CODE_LENGTH + 100)
    large_code = f"def new_invariant(s):\n    return 1 {padding}"

    assert len(large_code) > MAX_CODE_LENGTH

    # This should fail due to length check
    ok, reason = validate_code_static(large_code)

    assert not ok
    assert "code too long" in str(reason)

def test_recursion_error_check():
    """Verify that deep recursion (stack overflow in parsing) is caught."""
    # Trigger RecursionError in ast.parse
    # A very deep expression
    expr = "1"
    for _ in range(5000):
        expr = f"({expr} + 1)"
    code = f"def new_invariant(s):\n    return {expr}"

    # Ensure code length is within limits if we want to test recursion specifically
    # But wait, 5000 * 5 chars = 25KB < 100KB.
    assert len(code) < MAX_CODE_LENGTH

    try:
        ok, reason = validate_code_static(code)
        assert not ok
        # It might be caught as RecursionError (our new handler)
        # OR as SyntaxError (Python parser limit)
        # OR "complexity" generic message if we caught RecursionError
        assert "complexity" in str(reason) or "recursion" in str(reason) or "invalid syntax" in str(reason)
    except RecursionError:
        pytest.fail("RecursionError propagated out of validate_code_static")

def test_large_ast_check():
    """Verify that AST with too many nodes is rejected."""
    # Generate code with many nodes but low depth and short length
    # List with many elements
    # Each element in a list display is a node (Constant), plus the List node itself.
    # We want > MAX_AST_NODES (5000).
    elements = ["1"] * (MAX_AST_NODES + 1000)
    code = f"def new_invariant(s):\n    return [{', '.join(elements)}]"

    # Length: 6000 * 3 = 18000 < 100000
    assert len(code) < MAX_CODE_LENGTH

    try:
        # Verify it has enough nodes locally (if we can parse it without crashing test setup)
        # We assume ast.parse works here because it's flat structure.
        tree = ast.parse(code)
        node_count = sum(1 for _ in ast.walk(tree))
        assert node_count > MAX_AST_NODES

        ok, reason = validate_code_static(code)

        assert not ok
        assert "AST complexity exceeds limit" in str(reason)

    except RecursionError:
        pytest.skip("RecursionError during test setup prevents checking AST node limit")

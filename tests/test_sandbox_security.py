import unittest

from graph_invariant.sandbox import MAX_CODE_LENGTH, validate_code_static


class TestSandboxSecurity(unittest.TestCase):
    def test_max_code_length(self):
        # Create code slightly longer than MAX_CODE_LENGTH
        long_comment = "#" * (MAX_CODE_LENGTH + 10)
        code = f"""
def new_invariant(s):
    {long_comment}
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("code too long", reason)

    def test_max_ast_nodes(self):
        # Create code with many nodes
        # Each addition "1 + 1" adds 3 nodes (BinOp, Constant, Constant)
        # We need > MAX_AST_NODES (5000)
        # 2000 additions -> ~6000 nodes
        additions = " + ".join(["1"] * 2500)
        code = f"""
def new_invariant(s):
    return {additions}
"""
        # Ensure code length is within limit
        self.assertLess(len(code), MAX_CODE_LENGTH)

        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("code too complex", reason)
        self.assertIn("AST nodes", reason)

    def test_safe_comment_with_forbidden_word(self):
        # This was previously rejected due to "import " in comment
        code = """
def new_invariant(s):
    # This comment contains import os
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertTrue(ok, f"Reason: {reason}")

    def test_forbidden_calls_via_ast(self):
        # eval
        code = """
def new_invariant(s):
    eval("print(1)")
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("non-whitelisted call detected: eval", reason)

        # exec (as call)
        code = """
def new_invariant(s):
    exec("print(1)")
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        # exec might be parsed as Call in Py3
        # If exec is not in ALLOWED_CALLS, it fails.
        self.assertIn("non-whitelisted call detected: exec", reason)

    def test_forbidden_attributes_via_ast(self):
        # __class__
        code = """
def new_invariant(s):
    x = s.__class__
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("forbidden attribute detected: __class__", reason)

    def test_forbidden_import_ast(self):
        # import statement
        code = """
def new_invariant(s):
    import os
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("disallowed syntax: Import", reason)

    def test_forbidden_import_from_ast(self):
        # from ... import
        code = """
def new_invariant(s):
    from os import path
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("disallowed syntax: ImportFrom", reason)

    def test_bypass_attempt_with_spaces(self):
        # Previous bypass: "import  os" (two spaces)
        # This should now be caught by AST Import check
        code = """
def new_invariant(s):
    import  os
    return 1
"""
        ok, reason = validate_code_static(code)
        self.assertFalse(ok)
        self.assertIn("disallowed syntax: Import", reason)


if __name__ == "__main__":
    unittest.main()

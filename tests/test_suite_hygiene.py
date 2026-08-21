"""Guards on the test suite itself.

`unittest.main()` calls `sys.exit()`, so anything declared below a
``if __name__ == "__main__"`` guard is never defined when the file runs as a
script: it exits 0 green having executed a fraction of itself.
"""
import ast
import pathlib
import unittest

TESTS = pathlib.Path(__file__).parent


def _declarations_below_the_main_guard(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text())
    guard_line = None
    stranded = []
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("__name__"):
            guard_line = node.lineno
        elif guard_line is not None and isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            stranded.append(f"{node.name} (line {node.lineno})")
    return stranded


class TestNoTestsBelowTheMainGuard(unittest.TestCase):
    def test_every_main_guard_is_last_in_its_file(self):
        offenders = {}
        for path in sorted(TESTS.glob("test_*.py")):
            stranded = _declarations_below_the_main_guard(path)
            if stranded:
                offenders[path.name] = stranded
        self.assertEqual(
            offenders,
            {},
            "these declarations sit below `if __name__ == \"__main__\"` and are "
            "invisible to `python tests/<file>.py`. Move the guard to the bottom "
            "of the file (or delete it — pytest is the documented runner).",
        )


if __name__ == "__main__":
    unittest.main()

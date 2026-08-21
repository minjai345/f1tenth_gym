"""Keep CLAUDE.md's line anchors honest.

The brief is dense with ``[label](path#L123)`` links that rot as code moves. Two
checks: the target file exists and is long enough, and a single-symbol label lands
inside that symbol's definition. Mid-function anchors are not checkable.
"""
import ast
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIEF = ROOT / "CLAUDE.md"

LINK = re.compile(r"\[([^\]]*)\]\(([A-Za-z0-9_./-]+\.(?:py|rst|toml|yml|cfg|md))#L(\d+)(?:-L?(\d+))?\)")
LABEL_SYMBOL = re.compile(r"^`([A-Za-z_][A-Za-z0-9_]*)`$")


def _spans(path: pathlib.Path) -> dict[str, list[tuple[int, int]]]:
    tree = ast.parse(path.read_text())
    out: dict[str, list[tuple[int, int]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(node.name, []).append((node.lineno, node.end_lineno))
    return out


class TestBriefAnchors(unittest.TestCase):
    def setUp(self):
        if not BRIEF.exists():
            self.skipTest("CLAUDE.md not present")
        self.links = list(LINK.finditer(BRIEF.read_text()))
        self.assertGreater(len(self.links), 50, "anchor regex stopped matching")

    def test_every_anchor_is_in_range(self):
        bad = []
        for m in self.links:
            _, rel, lo, _ = m.groups()
            target = ROOT / rel
            if not target.exists():
                bad.append(f"{rel}#L{lo}: file does not exist")
                continue
            n = len(target.read_text().splitlines())
            if int(lo) > n:
                bad.append(f"{rel}#L{lo}: past EOF ({n} lines)")
        self.assertEqual(bad, [], "CLAUDE.md cites lines that do not exist:\n  " + "\n  ".join(bad))

    def test_symbol_labelled_anchors_point_at_their_symbol(self):
        cache: dict[str, dict] = {}
        bad = []
        for m in self.links:
            label, rel, lo, _ = m.groups()
            sym = LABEL_SYMBOL.match(label.strip())
            if not sym or not rel.endswith(".py"):
                continue
            target = ROOT / rel
            if not target.exists():
                continue
            spans = cache.setdefault(rel, _spans(target))
            ranges = spans.get(sym.group(1))
            if not ranges:
                continue
            lo = int(lo)
            # one line of slack: anchors often cite the decorator above the def
            if not any(a - 1 <= lo <= b for a, b in ranges):
                where = ", ".join(f"{a}-{b}" for a, b in ranges)
                bad.append(f"{rel}#L{lo}: `{sym.group(1)}` is at {where}")
        self.assertEqual(
            bad, [], "CLAUDE.md anchors point away from the symbol they name:\n  " + "\n  ".join(bad)
        )


if __name__ == "__main__":
    unittest.main()

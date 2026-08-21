#!/usr/bin/env python3
"""Check the docs against the house style.

Advisory by default; ``--strict`` makes findings fail the build, which is how CI
runs it. The tree is at zero findings.

    python docs/style_lint.py            # report, exit 0
    python docs/style_lint.py --strict   # report, exit 1 if anything is found

Two registries beside this file ratchet: ``_style/canonical_facts.yml`` (facts
needing exactly one full statement) and ``_style/protected.yml`` (sentences a
voice pass must not delete, keyed by content hash so they survive a move).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent
STYLE = DOCS / "_style"

MAX_LINE = 88
MAX_ADMONITIONS = 3
MAX_WARNINGS = 2
BOLD_LINES_PER_SPAN = 25
HEADING_OVERLAP = 0.60

UNDERLINE = re.compile(r"^([=\-~^\"'#*+])\1{2,}\s*$")
ADMONITION = re.compile(r"^\.\.\s+(note|warning|tip|important|caution|attention)::")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
DIRECTIVE = re.compile(r"^\s*\.\.\s+\S")

BANNED_OPENERS = [
    (re.compile(r"^\s*(This|The following)\s+(page|section|document|chapter|guide)\b", re.I),
     "opens by talking about the page instead of the simulator"),
    (re.compile(r"^\s*In this (guide|page|section|chapter|tutorial|document)\b", re.I),
     "opens with a self-referential 'In this ...'"),
    (re.compile(r"^\s*(Here|Below) we('ll| will)?\b", re.I),
     "opens with 'Here we ...'"),
]

BANNED_PHRASES = [
    "it is important to note", "please be aware", "please note", "keep in mind",
    "as mentioned above", "as mentioned earlier", "as mentioned previously",
    "may potentially", "can sometimes lead to", "for completeness",
    "everything you need", "clean interfaces", "easy to use", "easy-to-use",
    "seamless", "state-of-the-art", "comprehensive", "powerful",
    "byte-identical to before", "see below.", "included for completeness",
    # Blames the library for user error: both action columns are valid float
    # ranges, so a transposed action is a valid action. Say that instead.
    "fails silently", "fail silently",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "with", "you", "your", "this", "these",
}


class Finding:
    __slots__ = ("path", "line", "code", "message")

    def __init__(self, path: Path, line: int, code: str, message: str):
        self.path, self.line, self.code, self.message = path, line, code, message

    def __str__(self) -> str:
        return f"{self.path.relative_to(DOCS.parent)}:{self.line}: [{self.code}] {self.message}"


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS}


def normalise(text: str) -> str:
    """Collapse whitespace and markup so a moved sentence still hashes the same."""
    text = re.sub(r"``?|\*+|:[a-z:]+:`[^`]*`", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()[:16]


def sections(lines: list[str]):
    """Yield (heading, heading_lineno, body_lines) for each titled section."""
    for i in range(len(lines) - 1):
        if UNDERLINE.match(lines[i + 1]) and lines[i].strip():
            if len(lines[i + 1].strip()) < len(lines[i].strip()) - 2:
                continue  # underline too short to be a real heading
            body, j = [], i + 2
            while j < len(lines) and not (
                j + 1 < len(lines) and UNDERLINE.match(lines[j + 1]) and lines[j].strip()
            ):
                body.append(lines[j])
                j += 1
            yield lines[i].strip(), i + 1, body


def first_paragraph(body: list[str]) -> tuple[str, int]:
    para, start = [], 0
    for offset, raw in enumerate(body):
        stripped = raw.strip()
        if not stripped:
            if para:
                break
            continue
        if DIRECTIVE.match(raw) or UNDERLINE.match(raw):
            if para:
                break
            continue
        if not para:
            start = offset
        para.append(stripped)
    return " ".join(para), start


def check_page(path: Path, out: list[Finding]) -> None:
    text = path.read_text()
    lines = text.split("\n")

    for heading, hline, body in sections(lines):
        para, offset = first_paragraph(body)
        if not para:
            continue
        pline = hline + 2 + offset
        # Check every sentence of the opening paragraph, not just the first: the
        # tell usually arrives as sentence two, after a legitimate opening line
        # ("The simulator advances every agent ... This page explains the two
        # usable models, how their state vectors are laid out, ...").
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            for pattern, why in BANNED_OPENERS:
                if pattern.match(sentence):
                    out.append(Finding(path, pline, "opener", f"{why}: {sentence[:60]!r}"))
        first_sentence = re.split(r"(?<=[.!?])\s", para)[0]
        hw, sw = words(heading), words(first_sentence)
        if hw and sw and len(hw & sw) / len(hw) >= HEADING_OVERLAP:
            out.append(Finding(path, pline, "restates-heading",
                               f"first sentence restates the heading {heading!r}"))

    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        # every occurrence, not just the first: under --strict a page with five
        # instances would otherwise need five red CI runs to clear, one per push
        idx = lowered.find(phrase)
        while idx != -1:
            out.append(Finding(path, text[:idx].count("\n") + 1, "phrase",
                               f"banned phrase {phrase!r}"))
            idx = lowered.find(phrase, idx + 1)

    n_admon = n_warn = 0
    prev_admon_line = None
    for n, raw in enumerate(lines, 1):
        m = ADMONITION.match(raw)
        if m:
            n_admon += 1
            if m.group(1) == "warning":
                n_warn += 1
            if prev_admon_line is not None:
                between = [x for x in lines[prev_admon_line:n - 1] if x.strip()]
                if not any(not x.startswith(("   ", "\t")) for x in between):
                    out.append(Finding(path, n, "stacked-admonition",
                                       "admonition directly follows another with no prose between"))
            prev_admon_line = n
        if raw.rstrip().endswith("::"):
            nxt = next((x for x in lines[n:] if x.strip()), "")
            if DIRECTIVE.match(nxt):
                out.append(Finding(path, n, "literal-block",
                                   "'::' followed by a directive - drop one colon"))
        if len(raw) > MAX_LINE and not raw.lstrip().startswith(("http", "..", "|", "+--")):
            out.append(Finding(path, n, "long-line", f"source line is {len(raw)} chars (max {MAX_LINE})"))

    if n_admon > MAX_ADMONITIONS:
        out.append(Finding(path, 1, "admonition-budget",
                           f"{n_admon} admonitions (max {MAX_ADMONITIONS})"))
    if n_warn > MAX_WARNINGS:
        out.append(Finding(path, 1, "warning-budget", f"{n_warn} warnings (max {MAX_WARNINGS})"))

    # finditer, not findall: the match carries its own position, so a bold span
    # anchors where it actually is rather than at the text's first occurrence.
    bolds = list(BOLD.finditer(text))
    budget = max(1, len(lines) // BOLD_LINES_PER_SPAN)
    if len(bolds) > budget:
        out.append(Finding(path, 1, "bold-budget",
                           f"{len(bolds)} bold spans over {len(lines)} lines (budget {budget})"))
    for match in bolds:
        span = match.group(1)
        if span.rstrip().endswith(".") and len(span.split()) > 6:
            line = text[:match.start(1)].count("\n") + 1
            out.append(Finding(path, line,
                               "bold-sentence", f"whole sentence bolded: {span[:50]!r}"))

    if re.search(r"^See also\s*$", text, re.M):
        out.append(Finding(path, text[:re.search(r"^See also\s*$", text, re.M).start()].count("\n") + 1,
                           "see-also", "'See also' closer - link inline instead"))


def check_canonical(pages: list[Path], out: list[Finding]) -> None:
    registry = STYLE / "canonical_facts.yml"
    if not registry.exists():
        return
    facts = yaml.safe_load(registry.read_text()) or {}
    for key, spec in facts.items():
        pattern = re.compile(spec["pattern"], re.I)
        canonical = spec["canonical"]
        elsewhere = 0
        for path in pages:
            if path.name == canonical:
                continue
            elsewhere += len(pattern.findall(path.read_text()))
        if elsewhere > spec["max_elsewhere"]:
            out.append(Finding(registry, 1, "canonical-fact",
                               f"{key!r} stated {elsewhere}x outside {canonical} "
                               f"(ratchet allows {spec['max_elsewhere']})"))


def check_protected(pages: list[Path], out: list[Finding]) -> None:
    registry = STYLE / "protected.yml"
    if not registry.exists():
        return
    entries = yaml.safe_load(registry.read_text()) or {}
    corpus = {fingerprint(p) for path in pages
              for p in re.split(r"\n\s*\n", path.read_text())}
    for key, spec in entries.items():
        if spec["hash"] not in corpus:
            out.append(Finding(registry, 1, "protected-content",
                               f"{key!r} no longer present ({spec['note']}). "
                               f"Restore it, or waive it deliberately."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything is found")
    args = ap.parse_args()

    pages = sorted(p for p in DOCS.rglob("*.rst") if "_build" not in p.parts)
    findings: list[Finding] = []
    for path in pages:
        check_page(path, findings)
    check_canonical(pages, findings)
    check_protected(pages, findings)

    findings.sort(key=lambda f: (str(f.path), f.line))
    for f in findings:
        print(f)

    by_code: dict[str, int] = {}
    for f in findings:
        by_code[f.code] = by_code.get(f.code, 0) + 1
    print(f"\n{len(findings)} findings across {len(pages)} pages")
    for code, n in sorted(by_code.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {code}")

    if not args.strict:
        print("\nadvisory mode - not failing the build (use --strict to gate)")
        return 0
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

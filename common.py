"""
Shared helpers for the CLR-Code pipeline.

Contains: jsonl IO, code extraction from model responses, sample-test parsing
from OJBench problem statements, and a resource-limited local runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------
# jsonl
# --------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}") from e
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlAppender:
    """Append-only writer that flushes every line, so a killed run keeps its work."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def done_keys(path: str | Path, key_fields: tuple[str, ...]) -> set[tuple]:
    """Read an in-progress output file and return the set of completed keys."""
    if not Path(path).exists():
        return set()
    seen = set()
    for row in iter_jsonl(path):
        try:
            seen.add(tuple(row[k] for k in key_fields))
        except KeyError:
            continue
    return seen


# --------------------------------------------------------------------------
# Code extraction
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)

_LANG_ALIASES = {
    "cpp": {"cpp", "c++", "cc", "cxx", "c", "g++"},
    "python": {"python", "python3", "py", "py3", "pypy", "pypy3"},
}


def extract_code(content: str, language: str) -> str | None:
    """
    Pull the submission out of a model response.

    Mirrors what OJBench's own extractor does: prefer the LAST fenced block
    tagged with the target language, then the last untagged/any fenced block.
    Returns None when there is no fenced block at all.
    """
    if not content:
        return None
    blocks = _FENCE_RE.findall(content)
    if not blocks:
        return None

    wanted = _LANG_ALIASES.get(language, {language})
    tagged = [body for tag, body in blocks if tag.strip().lower() in wanted]
    if tagged:
        return tagged[-1].strip("\n")

    untagged = [body for tag, body in blocks if tag.strip() == ""]
    if untagged:
        return untagged[-1].strip("\n")

    return blocks[-1][1].strip("\n")


def code_fingerprint(code: str) -> str:
    """Whitespace-insensitive hash, for spotting literally identical candidates."""
    norm = re.sub(r"\s+", " ", code or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Sample-test parsing
# --------------------------------------------------------------------------
#
# OJBench statements are markdown converted from Luogu (NOI) and the ICPC site.
# Heading wording varies, so we classify every fenced block by the nearest
# preceding label line and then pair inputs with outputs in document order.
# ALWAYS eyeball the result with:  python step2_local_tests.py inspect-samples
# --------------------------------------------------------------------------

_IN_WORDS = re.compile(
    r"(sample\s*(input|in)|input\s*(sample|example|\d|#|:)|example\s*input"
    r"|输入样例|样例输入|输入#|input\b)",
    re.IGNORECASE,
)
_OUT_WORDS = re.compile(
    r"(sample\s*(output|out)|output\s*(sample|example|\d|#|:)|example\s*output"
    r"|输出样例|样例输出|输出#|output\b)",
    re.IGNORECASE,
)

# Blocks tagged with a real programming language are code, not sample data.
_CODE_TAGS = {
    "cpp", "c++", "c", "python", "python3", "py", "java", "pascal",
    "javascript", "js", "rust", "go",
}


@dataclass
class SampleTest:
    stdin: str
    expected: str


def _label_for(text: str, block_start: int, window: int = 220) -> str:
    """Return the text immediately before a fenced block, for classification."""
    lo = max(0, block_start - window)
    return text[lo:block_start]


def parse_samples(statement: str, max_pairs: int = 6) -> list[SampleTest]:
    """
    Best-effort extraction of (stdin, expected_stdout) pairs from a statement.

    Returns [] when nothing could be parsed confidently, which callers must
    treat as 'no sample signal available' rather than 'the code failed'.
    """
    if not statement:
        return []

    labelled: list[tuple[str, str]] = []  # (kind, body)
    for m in _FENCE_RE.finditer(statement):
        tag = m.group(1).strip().lower()
        body = m.group(2)
        if tag in _CODE_TAGS:
            continue
        label = _label_for(statement, m.start())
        # Look at the last non-empty line before the fence.
        lines = [ln.strip() for ln in label.splitlines() if ln.strip()]
        near = lines[-1] if lines else ""
        near2 = " ".join(lines[-2:]) if lines else ""

        kind = None
        for probe in (near, near2):
            if _OUT_WORDS.search(probe) and not _IN_WORDS.search(probe):
                kind = "out"
                break
            if _IN_WORDS.search(probe) and not _OUT_WORDS.search(probe):
                kind = "in"
                break
        if kind is None:
            kind = "?"
        labelled.append((kind, body))

    pairs: list[SampleTest] = []

    # Strategy 1: walk in order, pair each 'in' with the next 'out'.
    i = 0
    while i < len(labelled) - 1:
        if labelled[i][0] == "in":
            j = i + 1
            while j < len(labelled) and labelled[j][0] == "?":
                j += 1
            if j < len(labelled) and labelled[j][0] == "out":
                pairs.append(SampleTest(_norm_block(labelled[i][1]),
                                        _norm_block(labelled[j][1])))
                i = j + 1
                continue
        i += 1

    # Strategy 2: nothing labelled, but an even number of unlabelled blocks
    # that alternate -> assume in/out/in/out.
    if not pairs:
        unknown = [b for k, b in labelled if k == "?"]
        if len(unknown) >= 2 and len(unknown) % 2 == 0:
            for a, b in zip(unknown[0::2], unknown[1::2]):
                pairs.append(SampleTest(_norm_block(a), _norm_block(b)))

    # Drop degenerate pairs (empty input AND empty output).
    pairs = [p for p in pairs if p.stdin.strip() or p.expected.strip()]
    return pairs[:max_pairs]


def _norm_block(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip("\n") + "\n" if s.strip() else ""


def normalize_output(s: str) -> str:
    """Standard online-judge comparison: ignore trailing whitespace."""
    if s is None:
        return ""
    lines = s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def outputs_match(got: str, expected: str) -> bool:
    return normalize_output(got) == normalize_output(expected)


# --------------------------------------------------------------------------
# Local execution (compile + run), resource limited
# --------------------------------------------------------------------------
#
# SAFETY: this executes model-written code. It is NOT a real sandbox --
# it caps CPU, memory, processes and file size, but it does not block
# network or filesystem access. Run it on the rented GPU box or in a
# throwaway container, never on a machine you care about.
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    status: str            # ok | wrong | timeout | runtime_error | compile_error
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0


@dataclass
class Compiled:
    ok: bool
    kind: str                      # binary | script
    path: str | None = None
    error: str = ""
    workdir: str | None = None
    interpreter: list[str] = field(default_factory=list)


def _limits(mem_mb: int, cpu_s: int):
    def _apply():
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        os.setsid()
    return _apply


def compile_candidate(code: str, language: str, workdir: str,
                      compile_timeout: int = 30) -> Compiled:
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    if language == "cpp":
        src = wd / "main.cpp"
        src.write_text(code, encoding="utf-8")
        exe = wd / "main"
        try:
            p = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-pipe", "-w",
                 "-o", str(exe), str(src)],
                capture_output=True, text=True, timeout=compile_timeout,
            )
        except subprocess.TimeoutExpired:
            return Compiled(False, "binary", error="compiler timed out", workdir=str(wd))
        if p.returncode != 0:
            return Compiled(False, "binary", error=p.stderr[-4000:], workdir=str(wd))
        return Compiled(True, "binary", path=str(exe), workdir=str(wd))

    # python
    src = wd / "main.py"
    src.write_text(code, encoding="utf-8")
    interp = ["pypy3"] if _which("pypy3") else ["python3"]
    p = subprocess.run(interp + ["-c", f"compile(open({str(src)!r}).read(), 'main.py', 'exec')"],
                       capture_output=True, text=True, timeout=compile_timeout)
    if p.returncode != 0:
        return Compiled(False, "script", error=p.stderr[-4000:], workdir=str(wd))
    return Compiled(True, "script", path=str(src), workdir=str(wd), interpreter=interp)


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)


def run_candidate(comp: Compiled, stdin_data: str, timeout_s: float = 6.0,
                  mem_mb: int = 1024) -> RunResult:
    import time

    if not comp.ok:
        return RunResult("compile_error", stderr=comp.error)

    cmd = [comp.path] if comp.kind == "binary" else comp.interpreter + [comp.path]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=comp.workdir,
            preexec_fn=_limits(mem_mb, int(timeout_s) + 2),
            env={"PATH": "/usr/bin:/bin", "HOME": comp.workdir or "/tmp"},
        )
    except subprocess.TimeoutExpired:
        return RunResult("timeout", seconds=timeout_s)
    except Exception as e:  # noqa: BLE001
        return RunResult("runtime_error", stderr=str(e)[:500])
    dt = time.time() - t0
    if p.returncode != 0:
        return RunResult("runtime_error", stdout=p.stdout[:20000],
                         stderr=(p.stderr or "")[-2000:], seconds=dt)
    return RunResult("ok", stdout=p.stdout[:200000], stderr="", seconds=dt)


def temp_workdir(prefix: str = "clr_") -> str:
    return tempfile.mkdtemp(prefix=prefix)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

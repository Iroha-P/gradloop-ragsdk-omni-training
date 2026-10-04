"""Fail-closed policy gate for files staged in this repository."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 1024 * 1024

FORBIDDEN_DIRECTORIES = {
    "data",
    "models",
    "checkpoints",
    "outputs",
    "dist",
}

FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".docm",
    ".dot",
    ".dotx",
    ".dotm",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ppt",
    ".pptx",
    ".pptm",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
    ".bin",
    ".pkl",
    ".pickle",
    ".parquet",
    ".arrow",
    ".feather",
    ".npy",
    ".npz",
}

ALLOWED_SUFFIXES = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".sh",
}

ALLOWED_EXACT_NAMES = {
    ".env.example",
    ".gitignore",
    "license",
    "notice",
}

WINDOWS_DRIVE_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
)
_BACKSLASH_PATTERN = r"\\"
WINDOWS_UNC_ABSOLUTE_PATH = re.compile(
    r"""(?ix)
    (?<!<BS>)
    <BS><BS>(?:<BS><BS>)?
    (?:
        \?(?:<BS><BS>|<BS>)
        (?:
            UNC(?:<BS><BS>|<BS>)
            [^\s<BS>/"']+
            (?:<BS><BS>|<BS>)
            [^\s<BS>/"']+
            |
            [A-Za-z]:(?:<BS><BS>|<BS>)
        )
        |
        [^\s<BS>/"'?]+
        (?:<BS><BS>|<BS>)
        [^\s<BS>/"']+
    )
    """
    .replace("<BS>", _BACKSLASH_PATTERN)
)
POSIX_ABSOLUTE_PATH = re.compile(
    r"""(?x)
    (?<![A-Za-z0-9_:/.~-])
    /
    (?!/)
    [A-Za-z0-9._~+-]+
    (?:/[A-Za-z0-9._~+ -]+)*
    """
)
FILE_URI_ABSOLUTE_PATH = re.compile(
    r"""(?ix)
    \bfile://
    (?:
        /[^\s"'<>]+
        |
        [^/\s"'<>]+/[^\s"'<>]+
    )
    """
)
SECRET_ASSIGNMENT = re.compile(
    r"""(?imx)
    (?<![A-Za-z0-9_.-])
    ["']?
    [A-Za-z0-9_.-]*
    (?:
        api[_-]?key
        |
        access[_-]?key
        |
        access[_-]?token
        |
        auth[_-]?token
        |
        token
        |
        secret
        |
        password
        |
        passwd
    )
    ["']?[ \t]*[:=][ \t]*
    (?:
        ["'][^"'\r\n]+["']
        |
        [^"'${\s#,\]}][^\s#,\]}]*
    )
    """
)
BEARER_CREDENTIAL = re.compile(
    r"""(?imx)
    (?<![A-Za-z0-9_.-])
    ["']?authorization["']?[ \t]*[:=][ \t]*
    ["']?bearer[ \t]+[^\s"',}\]]+
    """
)
KNOWN_TOKEN_PREFIX = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,})\b"
)


@dataclass(frozen=True)
class ScanIssue:
    path: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    status: str
    issues: tuple[ScanIssue, ...]
    scanned_files: int


class ScanFailure(RuntimeError):
    """An internal failure that must block the commit."""


def _git(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ScanFailure(f"could not execute Git: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ScanFailure(detail or f"Git exited with status {result.returncode}")
    return result.stdout


def _staged_paths(repo_root: Path) -> list[str]:
    raw = _git(
        repo_root,
        "diff",
        "--cached",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
        "--",
    )
    try:
        return [
            item.decode("utf-8")
            for item in raw.split(b"\0")
            if item
        ]
    except UnicodeDecodeError as exc:
        raise ScanFailure("a staged filename is not valid UTF-8") from exc


def _staged_blob(repo_root: Path, path: str) -> bytes:
    return _git(repo_root, "show", f":{path}")


def _path_issue(path: str) -> str | None:
    if "\\" in path:
        return "unsafe repository path"
    normalized = path.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    original_parts = pure_path.parts
    parts = tuple(part.casefold() for part in pure_path.parts)
    basename = pure_path.name.casefold()
    is_source_test_data = (
        len(original_parts) == 3
        and original_parts[0] == "tests"
        and original_parts[1] == "data"
        and original_parts[2].startswith("test_")
        and PurePosixPath(original_parts[2]).suffix == ".py"
    )

    if pure_path.is_absolute() or ".." in parts:
        return "unsafe repository path"
    if (
        any(part in FORBIDDEN_DIRECTORIES for part in parts[:-1])
        and not is_source_test_data
    ):
        return "forbidden data, model, checkpoint, output, or distribution directory"
    if basename != ".env.example" and (
        basename == ".env" or basename.startswith(".env.")
    ):
        return "environment file variant; only .env.example is allowed"
    if basename.endswith(".sqlite") or ".sqlite" in basename:
        return "database file"
    if pure_path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        return "forbidden document, database, model, or checkpoint type"
    if (
        basename not in ALLOWED_EXACT_NAMES
        and pure_path.suffix.casefold() not in ALLOWED_SUFFIXES
    ):
        return "file type is not on the source/configuration allowlist"
    return None


def _content_issue(content: bytes) -> str | None:
    if len(content) > MAX_FILE_BYTES:
        return f"file exceeds the {MAX_FILE_BYTES}-byte limit"
    if b"\0" in content:
        return "binary content is not allowed"
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "content is not valid UTF-8 text"
    if (
        WINDOWS_DRIVE_ABSOLUTE_PATH.search(text)
        or WINDOWS_UNC_ABSOLUTE_PATH.search(text)
        or POSIX_ABSOLUTE_PATH.search(text)
        or FILE_URI_ABSOLUTE_PATH.search(text)
    ):
        return "machine-local absolute path"
    if (
        SECRET_ASSIGNMENT.search(text)
        or BEARER_CREDENTIAL.search(text)
        or KNOWN_TOKEN_PREFIX.search(text)
    ):
        return "credential or token-like content"
    return None


def _scan_paths(repo_root: Path, paths: Iterable[str]) -> ScanResult:
    issues: list[ScanIssue] = []
    scanned = 0
    for path in paths:
        scanned += 1
        reason = _path_issue(path)
        if reason is None:
            reason = _content_issue(_staged_blob(repo_root, path))
        if reason is not None:
            issues.append(ScanIssue(path=path, reason=reason))
    return ScanResult(
        status="failed" if issues else "passed",
        issues=tuple(issues),
        scanned_files=scanned,
    )


def scan_staged(repo_root: Path | str = ".") -> ScanResult:
    """Scan added, copied, modified, and renamed blobs in the Git index."""
    root = Path(repo_root).resolve()
    try:
        _git(root, "rev-parse", "--is-inside-work-tree")
        return _scan_paths(root, _staged_paths(root))
    except Exception as exc:  # noqa: BLE001 - release scanner must fail closed
        return ScanResult(
            status="failed",
            issues=(ScanIssue(path="<scanner>", reason=f"failed closed: {exc}"),),
            scanned_files=0,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged",
        action="store_true",
        help="scan blobs currently staged in the Git index",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.staged:
        print("security scan failed closed: --staged is required")
        return 2

    result = scan_staged(Path.cwd())
    if result.status == "failed":
        print("security scan blocked the commit:")
        for issue in result.issues:
            print(f"- {issue.path}: {issue.reason}")
        return 1

    print(f"security scan passed: {result.scanned_files} staged file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.security.precommit_scan import _path_issue

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCANNER = REPOSITORY_ROOT / "scripts" / "security" / "precommit_scan.py"
MAX_ALLOWED_BYTES = 1024 * 1024
FORWARD_SLASH = chr(47)
BACKSLASH = chr(92)


class GitRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def stage_text(self, relative_path: str, content: str) -> None:
        self.stage_bytes(relative_path, content.encode("utf-8"))

    def stage_bytes(self, relative_path: str, content: bytes) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.git("add", "-f", "--", relative_path)


@pytest.fixture
def repo(tmp_path: Path) -> GitRepository:
    result = subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return GitRepository(tmp_path)


def scan_staged(repo_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(SCANNER), "--staged"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


@pytest.mark.parametrize(
    "path",
    [
        "data/synthetic.jsonl",
        "data/incoming/x.jsonl",
        "data/processed/x.jsonl",
        "data/cache/x.jsonl",
        "raw.pdf",
        "private.docx",
        "sheet.xlsx",
        "slides.pptx",
        "state.db",
        "state.sqlite",
        "models/model.safetensors",
        "checkpoints/step.ckpt",
        "outputs/predictions.jsonl",
        "dist/package.txt",
    ],
)
def test_staged_forbidden_paths_are_rejected(repo: GitRepository, path: str) -> None:
    repo.stage_text(path, "canary")

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", ".env.production", "config/.env.development"],
)
def test_staged_env_variants_are_rejected(repo: GitRepository, path: str) -> None:
    repo.stage_text(path, "SAFE_PLACEHOLDER=value")

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        (
            "config/windows-drive.toml",
            'dataset_root = "'
            + "C"
            + ":"
            + BACKSLASH
            + "Users"
            + BACKSLASH
            + "student"
            + BACKSLASH
            + 'private-data"\n',
        ),
        (
            "config/windows-forward-slash.toml",
            "dataset_root = "
            + "D"
            + ":"
            + FORWARD_SLASH
            + "private/training-data\n",
        ),
        (
            "config/windows-unc.toml",
            'source = "'
            + BACKSLASH * 2
            + "server"
            + BACKSLASH
            + "share"
            + BACKSLASH
            + 'private"\n',
        ),
        (
            "config/windows-extended.toml",
            'source = "'
            + BACKSLASH * 2
            + "?"
            + BACKSLASH
            + "C"
            + ":"
            + BACKSLASH
            + "private"
            + BACKSLASH
            + 'training-data"\n',
        ),
        (
            "config/windows-extended-unc.toml",
            'source = "'
            + BACKSLASH * 2
            + "?"
            + BACKSLASH
            + "UNC"
            + BACKSLASH
            + "server"
            + BACKSLASH
            + "share"
            + BACKSLASH
            + 'private"\n',
        ),
        (
            "config/windows-escaped-unc.json",
            json.dumps(
                {
                    "source": (
                        BACKSLASH * 2
                        + "server"
                        + BACKSLASH
                        + "share"
                        + BACKSLASH
                        + "private"
                    )
                }
            ),
        ),
        (
            "config/windows-escaped-extended-unc.json",
            json.dumps(
                {
                    "source": (
                        BACKSLASH * 2
                        + "?"
                        + BACKSLASH
                        + "UNC"
                        + BACKSLASH
                        + "server"
                        + BACKSLASH
                        + "share"
                        + BACKSLASH
                        + "private"
                    )
                }
            ),
        ),
        (
            "config/posix-quoted.toml",
            'cache_location = "'
            + FORWARD_SLASH
            + 'workspace/private-cache"\n',
        ),
        (
            "scripts/train.sh",
            "cd " + FORWARD_SLASH + "home/student/private-data\n",
        ),
        (
            "config/posix-unlisted-root.yaml",
            "source: " + FORWARD_SLASH + "custom-root/private/data\n",
        ),
        (
            "config/posix-file-uri.toml",
            'source = "'
            + "file:"
            + FORWARD_SLASH * 3
            + 'workspace/private-cache"\n',
        ),
        (
            "config/windows-file-uri.toml",
            'source = "'
            + "file:"
            + FORWARD_SLASH * 3
            + "C"
            + ":"
            + FORWARD_SLASH
            + 'private/training-data"\n',
        ),
        (
            "config/unc-file-uri.toml",
            'source = "'
            + "file:"
            + FORWARD_SLASH * 2
            + 'server/share/private"\n',
        ),
    ],
)
def test_staged_machine_absolute_paths_are_rejected(
    repo: GitRepository, relative_path: str, content: str
) -> None:
    repo.stage_text(relative_path, content)

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        (
            "config/token.toml",
            "to" + 'ken = "' + "sk-" + 'secret-canary-value"\n',
        ),
        (
            "config/api-key.toml",
            "api_" + 'key = "secret-canary-value"\n',
        ),
        (
            "config/service.env.txt",
            "SERVICE_" + "TOKEN" + "=secret-canary-value\n",
        ),
        (
            "config/openai.env.txt",
            "OPENAI_" + "API_KEY" + "=secret-canary-value\n",
        ),
        (
            "config/aws.env.txt",
            "AWS_" + "SECRET_ACCESS_KEY" + "=secret-canary-value\n",
        ),
        (
            "config/client.env.txt",
            "CLIENT_" + "SECRET" + "=secret-canary-value\n",
        ),
        (
            "scripts/export-key.sh",
            "export API_" + "KEY" + "=secret-canary-value\n",
        ),
        (
            "scripts/export-prefixed-key.sh",
            "export OPENAI_" + "API_KEY" + "=secret-canary-value\n",
        ),
        (
            "config/settings.json",
            '{\n  "' + "api_key" + '": "secret-canary-value"\n}\n',
        ),
        (
            "config/nested-settings.json",
            '{"name": "canary", "' + "api_key" + '": "secret-canary-value"}\n',
        ),
        (
            "config/settings.toml",
            '"' + "access_token" + '" = "secret-canary-value"\n',
        ),
        (
            "config/settings.yaml",
            '"' + "client_secret" + '": "secret-canary-value"\n',
        ),
        (
            "config/authorization.json",
            '{"Author' + 'ization": "Bearer secret-canary-value"}\n',
        ),
        (
            "config/nested-authorization.json",
            '{"name": "canary", "Author'
            + 'ization": "Bearer secret-canary-value"}\n',
        ),
        (
            "config/authorization.toml",
            '"Author' + 'ization" = "Bearer secret-canary-value"\n',
        ),
        (
            "config/authorization.yaml",
            '"Author' + 'ization": "Bearer secret-canary-value"\n',
        ),
        (
            "scripts/curl.sh",
            'curl -H "Author'
            + 'ization: Bearer secret-canary-value" https://example.com\n',
        ),
        (
            ".env.example",
            "# API_" + "KEY" + "=secret-canary-value\n",
        ),
    ],
)
def test_staged_credentials_are_rejected(
    repo: GitRepository, relative_path: str, content: str
) -> None:
    repo.stage_text(relative_path, content)

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize(
    "content",
    [
        'homepage = "https://example.com/training/docs"\n',
        'registry = "ssh://git@example.com/team/project.git"\n',
        'database_scheme = "postgresql://db.example.com/training"\n',
    ],
)
def test_network_urls_are_not_misclassified_as_absolute_paths(
    repo: GitRepository, content: str
) -> None:
    repo.stage_text("config/network.toml", content)

    result = scan_staged(repo.root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_empty_env_example_placeholders_and_security_prose_are_allowed(
    repo: GitRepository,
) -> None:
    repo.stage_text(
        ".env.example",
        "SERVICE_" + "TOKEN" + "=\n# API_" + "KEY" + "=\n",
    )
    repo.stage_text(
        "SECURITY.md",
        "API_KEY variables must remain empty.\n"
        "Bearer authentication is prohibited in committed files.\n"
        "Tokenization settings are ordinary model configuration.\n",
    )

    result = scan_staged(repo.root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_policy_sources_are_accepted_by_their_own_scanner(
    repo: GitRepository,
) -> None:
    for relative_path in (
        "scripts/security/precommit_scan.py",
        "tests/security/test_repository_policy.py",
    ):
        repo.stage_bytes(
            relative_path,
            (REPOSITORY_ROOT / relative_path).read_bytes(),
        )

    result = scan_staged(repo.root)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "artifact.pdf",
        "artifact.doc",
        "artifact.docx",
        "artifact.docm",
        "artifact.dot",
        "artifact.dotx",
        "artifact.dotm",
        "artifact.xls",
        "artifact.xlsx",
        "artifact.xlsm",
        "artifact.xlsb",
        "artifact.ppt",
        "artifact.pptx",
        "artifact.pptm",
        "artifact.db",
        "artifact.sqlite",
        "artifact.sqlite3",
        "artifact.safetensors",
        "artifact.ckpt",
        "artifact.pt",
        "artifact.pth",
        "artifact.onnx",
        "artifact.gguf",
        "artifact.bin",
        "artifact.pkl",
        "artifact.pickle",
        "artifact.parquet",
        "artifact.arrow",
        "artifact.feather",
        "artifact.npy",
        "artifact.npz",
    ],
)
def test_artifact_suffixes_are_ignored(path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", path],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, path


def test_direct_data_child_is_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", "data/synthetic.jsonl"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0


def test_source_only_test_data_path_is_not_ignored() -> None:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.ignoreCase=true",
            "check-ignore",
            "--no-index",
            "--",
            "tests/data/test_synthetic_source.py",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 1


@pytest.mark.parametrize(
    "path",
    [
        "tests/data/private.csv",
        "tests/data/synthetic_records.jsonl",
        "tests/data/notes.txt",
        "tests/data/helper.py",
        "tests/data/nested/test_synthetic_source.py",
        "tests/data/nested/data/test_synthetic_source.py",
    ],
)
def test_non_source_test_data_paths_remain_ignored(path: str) -> None:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.ignoreCase=false",
            "check-ignore",
            "--no-index",
            "--",
            path,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, path


def test_source_only_test_data_path_is_allowed_by_scanner(
    repo: GitRepository,
) -> None:
    repo.stage_text(
        "tests/data/test_synthetic_source.py",
        "def test_synthetic_source():\n    assert True\n",
    )

    result = scan_staged(repo.root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_windows_repo_stages_source_test_normally_and_hook_passes(
    repo: GitRepository,
) -> None:
    repo.git("config", "core.ignoreCase", "true")
    repo.git("config", "user.name", "Synthetic Test")
    repo.git("config", "user.email", "synthetic@example.invalid")
    repo.stage_bytes(".gitignore", (REPOSITORY_ROOT / ".gitignore").read_bytes())

    relative_path = "tests/data/test_windows_source.py"
    path = repo.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "def test_windows_source():\n    assert True\n",
        encoding="utf-8",
    )

    add_result = repo.git("add", "--", relative_path, check=False)

    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    hook = repo.root / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!" + "/" + "bin/sh\n"
        f'"{Path(sys.executable).as_posix()}" '
        f'"{SCANNER.as_posix()}" --staged\n',
        encoding="utf-8",
    )

    commit_result = repo.git(
        "commit",
        "-m",
        "synthetic source test",
        check=False,
    )

    assert commit_result.returncode == 0, commit_result.stdout + commit_result.stderr
    assert "security scan passed" in (commit_result.stdout + commit_result.stderr)


@pytest.mark.parametrize(
    "relative_path",
    [
        "Tests/data/test_case_variant.py",
        "tests/Data/test_case_variant.py",
    ],
)
def test_windows_repo_hook_rejects_test_data_case_variants(
    repo: GitRepository,
    relative_path: str,
) -> None:
    repo.git("config", "core.ignoreCase", "true")
    repo.git("config", "user.name", "Synthetic Test")
    repo.git("config", "user.email", "synthetic@example.invalid")
    repo.stage_bytes(".gitignore", (REPOSITORY_ROOT / ".gitignore").read_bytes())

    path = repo.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "def test_case_variant():\n    assert True\n",
        encoding="utf-8",
    )

    add_result = repo.git("add", "--", relative_path, check=False)

    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    hook = repo.root / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!" + "/" + "bin/sh\n"
        f'"{Path(sys.executable).as_posix()}" '
        f'"{SCANNER.as_posix()}" --staged\n',
        encoding="utf-8",
    )

    commit_result = repo.git(
        "commit",
        "-m",
        "synthetic case variant",
        check=False,
    )

    assert commit_result.returncode != 0
    assert "blocked" in (commit_result.stdout + commit_result.stderr).lower()


@pytest.mark.parametrize(
    "path",
    [
        "tests/data/private.csv",
        "tests/data/synthetic_records.jsonl",
        "tests/data/notes.txt",
        "tests/data/helper.py",
        "tests/data/nested/test_synthetic_source.py",
        "tests/data/nested/data/test_synthetic_source.py",
        "Tests/data/test_synthetic_source.py",
        "tests/Data/test_synthetic_source.py",
    ],
)
def test_staged_non_source_test_data_paths_are_rejected(
    repo: GitRepository,
    path: str,
) -> None:
    repo.stage_text(path, "fully synthetic canary\n")

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


def test_source_test_exception_still_scans_content(
    repo: GitRepository,
) -> None:
    repo.stage_text(
        "tests/data/test_unsafe_content.py",
        "api_" + "key" + "=synthetic-canary-value\n",
    )

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "credential or token-like content" in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "data/test_synthetic_source.py",
        "src/data/test_synthetic_source.py",
        "foo/data/test_synthetic_source.py",
        "tests/data/private.csv",
        "tests/data/synthetic_records.jsonl",
        "tests/data/notes.txt",
        "tests/data/helper.py",
        "tests/data/nested/test_synthetic_source.py",
        "tests/data/nested/data/test_synthetic_source.py",
        "Tests/data/test_synthetic_source.py",
        "tests/Data/test_synthetic_source.py",
        r"tests\data\test_synthetic_source.py",
        "tests/data/../data/test_synthetic_source.py",
    ],
)
def test_test_data_exception_rejects_non_exact_or_unsafe_paths(path: str) -> None:
    assert _path_issue(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "artifact.pdf",
        "artifact.doc",
        "artifact.docx",
        "artifact.docm",
        "artifact.dot",
        "artifact.dotx",
        "artifact.dotm",
        "artifact.xls",
        "artifact.xlsx",
        "artifact.xlsm",
        "artifact.xlsb",
        "artifact.ppt",
        "artifact.pptx",
        "artifact.pptm",
        "artifact.db",
        "artifact.sqlite",
        "artifact.sqlite3",
        "artifact.safetensors",
        "artifact.ckpt",
        "artifact.pt",
        "artifact.pth",
        "artifact.onnx",
        "artifact.gguf",
        "artifact.bin",
        "artifact.pkl",
        "artifact.pickle",
        "artifact.parquet",
        "artifact.arrow",
        "artifact.feather",
        "artifact.npy",
        "artifact.npz",
    ],
)
def test_force_staged_artifact_suffixes_are_rejected(
    repo: GitRepository, path: str
) -> None:
    repo.stage_text(path, "canary")

    result = scan_staged(repo.root)

    assert result.returncode != 0
    output = (result.stdout + result.stderr).lower()
    assert "blocked" in output
    assert "file type is not on the source/configuration allowlist" not in output


def test_safe_source_docs_config_and_env_example_are_allowed(
    repo: GitRepository,
) -> None:
    repo.stage_text("src/train.py", "print('training scaffold')\n")
    repo.stage_text("README.md", "# Training scaffold\n")
    repo.stage_text("config/defaults.toml", 'dataset_name = "example"\n')
    repo.stage_text(".env.example", "SERVICE_" + "TOKEN" + "=\n")

    result = scan_staged(repo.root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_oversized_staged_file_is_rejected(repo: GitRepository) -> None:
    repo.stage_bytes("src/large.py", b"x" * (MAX_ALLOWED_BYTES + 1))

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


def test_binary_content_is_rejected_even_with_safe_extension(
    repo: GitRepository,
) -> None:
    repo.stage_bytes("src/not_text.py", b"\xff\xfe\x00\x01")

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


def test_non_whitelisted_extension_is_rejected(repo: GitRepository) -> None:
    repo.stage_text("artifact.exe", "not really executable")

    result = scan_staged(repo.root)

    assert result.returncode != 0
    assert "blocked" in (result.stdout + result.stderr).lower()


def test_scanner_fails_closed_outside_a_git_repository(tmp_path: Path) -> None:
    result = scan_staged(tmp_path)

    assert result.returncode != 0
    assert "failed closed" in (result.stdout + result.stderr).lower()

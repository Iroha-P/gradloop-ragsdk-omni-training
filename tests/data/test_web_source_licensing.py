from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.provenance import ProvenanceRecord
from gradloop_data.web_sources import (
    CollectionDisabledError,
    SourceLicense,
    SourcePolicyError,
    canonicalize_url,
    collect_source,
    dry_run_source,
    load_policy,
    source_license_from_mapping,
)

LICENSE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "PDM-1.0": "https://creativecommons.org/publicdomain/mark/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}
ALLOWED_USES = ("research", "model_training")


def policy_document() -> dict[str, object]:
    return {
        "policy_version": 1,
        "default_decision": "deny",
        "allowed_licenses": {
            license_id: {
                "canonical_license_url": license_url,
                "permitted_uses": (
                    ["research", "retrieval", "evaluation", "adaptation"]
                    if license_id == "CC-BY-SA-4.0"
                    else [
                        "research",
                        "model_training",
                        "redistribution",
                        "adaptation",
                    ]
                ),
            }
            for license_id, license_url in LICENSE_URLS.items()
        },
    }


def write_policy(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "synthetic-policy.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def assert_fixed_cli_denial(
    result: subprocess.CompletedProcess[str],
    reason: str,
    *markers: str,
) -> None:
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert result.stderr == ""
    for marker in markers:
        assert marker not in combined_output
    assert json.loads(result.stdout) == {
        "category": "unclassified",
        "decision": "denied",
        "license_id": "unknown",
        "reason": reason,
    }


def source_spec(
    *,
    license_id: str = "CC-BY-4.0",
    license_url: str | None = None,
    **overrides: object,
) -> SourceLicense:
    values: dict[str, object] = {
        "url": "https://example.org/open/corpus",
        "license_id": license_id,
        "license_url": (
            LICENSE_URLS.get(license_id, "https://licenses.example/unsupported")
            if license_url is None
            else license_url
        ),
        "attribution": "Example corpus contributors",
        "allowed_uses": ALLOWED_USES,
    }
    values.update(overrides)
    return SourceLicense(**values)


@pytest.mark.parametrize(
    ("license_id", "license_url", "allowed_uses"),
    [
        ("CC0-1.0", LICENSE_URLS["CC0-1.0"], ALLOWED_USES),
        ("PDM-1.0", LICENSE_URLS["PDM-1.0"], ALLOWED_USES),
        ("CC-BY-4.0", LICENSE_URLS["CC-BY-4.0"], ALLOWED_USES),
        (
            "CC-BY-SA-4.0",
            LICENSE_URLS["CC-BY-SA-4.0"],
            ("research", "retrieval", "evaluation"),
        ),
    ],
)
def test_only_explicitly_allowed_licenses_are_accepted(
    license_id: str, license_url: str, allowed_uses: tuple[str, ...]
) -> None:
    decision = dry_run_source(
        source_spec(
            license_id=license_id,
            license_url=license_url,
            allowed_uses=allowed_uses,
        )
    )

    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert decision.license_id == license_id


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"license_id": "", "license_url": ""}, "unknown_license"),
        (
            {
                "license_id": "MIT",
                "license_url": "https://licenses.example/mit",
            },
            "unsupported_license",
        ),
        ({"login_required": True}, "login_required"),
        ({"paywalled": True}, "paywalled"),
        ({"category": "personal_social"}, "personal_social"),
        ({"robots_allowed": False}, "robots_denied"),
        ({"terms_allowed": False}, "terms_denied"),
        ({"content_risks": ("faces",)}, "faces"),
        ({"content_risks": ("id_documents",)}, "id_documents"),
        (
            {"content_risks": ("credential_documents",)},
            "credential_documents",
        ),
        (
            {"content_risks": ("contact_information_images",)},
            "contact_information_images",
        ),
        (
            {"content_risks": ("contact_information_metadata",)},
            "contact_information_metadata",
        ),
    ],
)
def test_source_policy_denies_each_prohibited_reason(
    overrides: dict[str, object], reason: str
) -> None:
    decision = dry_run_source(source_spec(**overrides))

    assert decision.allowed is False
    assert decision.reason == reason


@pytest.mark.parametrize("category", ["official_facts", "public_web", "open_web"])
def test_category_is_not_a_substitute_for_machine_readable_license(
    category: str,
) -> None:
    decision = dry_run_source(
        source_spec(
            category=category,
            license_id="",
            license_url="",
        )
    )

    assert decision.allowed is False
    assert decision.reason == "category_only_not_a_license"


def test_explicit_allowed_license_is_required_to_collect() -> None:
    with pytest.raises(SourcePolicyError) as exc_info:
        collect_source(
            source_spec(license_id="", license_url=""),
            fetcher=lambda _: b"synthetic",
        )

    assert exc_info.value.reason == "unknown_license"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/open",
        "file:" + "/" * 3 + "tmp/open.txt",
        "https://user:password@example.org/open",
        "https://localhost/open",
        "https://127.0.0.1/open",
        "https://10.2.3.4/open",
        "https://169.254.1.2/open",
        "https://[::1]" + "/" + "open",
        "https://service.internal/open",
        "https://example.org/open?access_" + "token" + "=synthetic-secret",
        "https://example.org/open#private-fragment",
        "https://example.org/open#",
    ],
)
def test_urls_must_be_public_https_without_credentials_or_query_secrets(
    url: str,
) -> None:
    decision = dry_run_source(source_spec(url=url))

    assert decision.allowed is False
    assert decision.reason in {
        "https_required",
        "url_credentials",
        "private_network_url",
        "query_not_allowed",
        "fragment_not_allowed",
    }


@pytest.mark.parametrize(
    "host",
    [
        "127.1",
        "127.0.1",
        "0177.0.0.1",
        "0x7f.0.0.1",
        "127.0x0.0.1",
        "0x7f000001",
        "2130706433",
    ],
)
def test_noncanonical_numeric_hosts_are_denied_without_dns(host: str) -> None:
    decision = dry_run_source(source_spec(url=f"https://{host}" + "/" + "open"))

    assert decision.allowed is False
    assert decision.reason == "private_network_url"


@pytest.mark.parametrize(
    "host",
    ["224.0.0.1", "239.255.255.255", "ff02::1", "ff0e::1"],
)
def test_multicast_ip_literals_are_denied_without_dns(host: str) -> None:
    rendered = f"[{host}]" if ":" in host else host
    decision = dry_run_source(source_spec(url=f"https://{rendered}" + "/" + "open"))
    assert decision.allowed is False
    assert decision.reason == "private_network_url"


@pytest.mark.parametrize(
    "query_name",
    ["code", "key", "authToken", "jwt", "sig", "ordinary"],
)
def test_all_nonempty_query_strings_are_denied(
    query_name: str,
) -> None:
    decision = dry_run_source(
        source_spec(url=f"https://example.org/open?{query_name}=synthetic")
    )

    assert decision.allowed is False
    assert decision.reason == "query_not_allowed"


def test_canonical_url_normalizes_safe_path_and_host_forms() -> None:
    assert canonicalize_url("HTTPS://Example.ORG:443/a/../open") == canonicalize_url(
        "https://example.org/open"
    )


@pytest.mark.parametrize("control", ["\x00", "\x01", "\x1f", "\x7f"])
def test_url_rejects_ascii_controls_anywhere(control: str) -> None:
    decision = dry_run_source(
        source_spec(url=f"https://example.org/open{control}synthetic")
    )

    assert decision.allowed is False
    assert decision.reason == "invalid_url"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org:/open",
        "https://example.org:0/open",
        "https://example.org:0443/open",
        "https://example.org:65536/open",
        "https://example.org:notaport/open",
    ],
)
def test_url_rejects_empty_zero_or_malformed_explicit_ports(url: str) -> None:
    decision = dry_run_source(source_spec(url=url))

    assert decision.allowed is False
    assert decision.reason == "invalid_port"


def test_url_preserves_well_formed_nondefault_https_port() -> None:
    assert canonicalize_url("https://example.org:8443/open") == (
        "https://example.org:8443/open"
    )


def test_global_ipv4_and_ipv6_literals_are_accepted_without_dns() -> None:
    assert dry_run_source(source_spec(url="https://1.1.1.1/open")).allowed is True
    ipv6_url = "https://[2606:4700:4700::1111]" + "/" + "open"

    decision = dry_run_source(source_spec(url=ipv6_url))

    assert decision.allowed is True
    assert canonicalize_url(ipv6_url) == ipv6_url


@pytest.mark.parametrize("host", ["::1", "fc00::1", "fe80::1", "::", "2001:db8::1"])
def test_non_global_ipv6_literals_are_denied_without_dns(host: str) -> None:
    url = f"https://[{host}]" + "/" + "open"

    decision = dry_run_source(source_spec(url=url))

    assert decision.allowed is False
    assert decision.reason == "private_network_url"


def test_unicode_hostname_is_canonicalized_through_strict_idna() -> None:
    assert canonicalize_url("https://bücher.example/open") == (
        "https://xn--bcher-kva.example/open"
    )


@pytest.mark.parametrize(
    "host",
    [
        "xn--a.example",
        "example.org.",
        "example.org..",
        "example..org",
        "bad_name.example",
        "-bad.example",
        "bad-.example",
    ],
)
def test_invalid_idna_or_dns_label_hosts_are_denied(host: str) -> None:
    decision = dry_run_source(source_spec(url=f"https://{host}" + "/" + "open"))

    assert decision.allowed is False
    assert decision.reason == "private_network_url"


def test_idna_mapping_cannot_turn_unicode_host_into_numeric_alias() -> None:
    decision = dry_run_source(source_spec(url="https://１２７.０.０.１" + "/" + "open"))

    assert decision.allowed is False
    assert decision.reason == "private_network_url"


def test_collect_source_builds_complete_provenance_with_injected_clock() -> None:
    content = b"fully synthetic public-domain fixture"
    now = datetime(2026, 7, 26, 8, 30, tzinfo=timezone.utc)
    fetched_urls: list[str] = []

    def synthetic_fetcher(url: str) -> bytes:
        fetched_urls.append(url)
        return content

    record = collect_source(
        source_spec(
            url="https://EXAMPLE.org:443/open/./corpus",
            license_id="CC0-1.0",
            license_url=LICENSE_URLS["CC0-1.0"],
            attribution="Synthetic fixture authors",
            allowed_uses=("research", "redistribution"),
        ),
        fetcher=synthetic_fetcher,
        clock=lambda: now,
    )

    assert record == ProvenanceRecord(
        canonical_url="https://example.org/open/corpus",
        retrieved_at="2026-07-26T08:30:00Z",
        content_sha256=hashlib.sha256(content).hexdigest(),
        license_id="CC0-1.0",
        license_url=LICENSE_URLS["CC0-1.0"],
        attribution="Synthetic fixture authors",
        allowed_uses=("research", "redistribution"),
    )
    assert fetched_urls == ["https://example.org/open/corpus"]


def test_collect_source_normalizes_aware_non_utc_clock() -> None:
    local_time = datetime(
        2026,
        7,
        26,
        8,
        30,
        tzinfo=timezone(timedelta(hours=8)),
    )

    record = collect_source(
        source_spec(),
        fetcher=lambda _: b"synthetic",
        clock=lambda: local_time,
    )

    assert record.retrieved_at == "2026-07-26T00:30:00Z"


def test_collect_source_rejects_naive_clock_without_relabeling() -> None:
    naive_time = datetime(2026, 7, 26, 8, 30)  # noqa: DTZ001 - rejection case

    with pytest.raises(SourcePolicyError) as exc_info:
        collect_source(
            source_spec(),
            fetcher=lambda _: b"synthetic",
            clock=lambda: naive_time,
        )

    assert exc_info.value.reason == "invalid_timestamp"
    assert str(naive_time) not in str(exc_info.value)


def test_collect_source_has_no_implicit_network_fetcher() -> None:
    with pytest.raises(CollectionDisabledError):
        collect_source(source_spec())


def test_dry_run_never_calls_injected_fetcher() -> None:
    calls = 0

    def forbidden_fetcher(_: str) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError("dry-run called a fetcher")

    decision = dry_run_source(source_spec(), fetcher=forbidden_fetcher)

    assert decision.allowed is True
    assert calls == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"login_required": "false"},
        {"paywalled": 0},
        {"robots_allowed": "true"},
        {"terms_allowed": 1},
        {"allowed_uses": ["research", 7]},
        {"content_risks": ["faces", None]},
    ],
)
def test_mapping_fails_closed_on_malformed_policy_value_types(
    overrides: dict[str, object],
) -> None:
    raw: dict[str, object] = {
        "url": "https://example.org/open/corpus",
        "license_id": "CC-BY-4.0",
        "license_url": LICENSE_URLS["CC-BY-4.0"],
        "attribution": "Example corpus contributors",
        "allowed_uses": list(ALLOWED_USES),
        "category": "open_licensed",
    }
    raw.update(overrides)

    with pytest.raises(SourcePolicyError) as exc_info:
        source_license_from_mapping(raw)

    assert exc_info.value.reason == "invalid_spec"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", 7),
        ("license_id", ["CC-BY-4.0"]),
        ("license_url", None),
        ("attribution", {"name": "synthetic"}),
        ("allowed_uses", ["research"]),
        ("allowed_uses", ("research", 7)),
        ("category", 7),
        ("login_required", "false"),
        ("paywalled", 0),
        ("robots_allowed", "false"),
        ("terms_allowed", "false"),
        ("content_risks", ["faces"]),
        ("content_risks", ("faces", None)),
    ],
)
def test_direct_source_license_rejects_malformed_types_without_echo(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "url": "https://example.org/open/corpus",
        "license_id": "CC-BY-4.0",
        "license_url": LICENSE_URLS["CC-BY-4.0"],
        "attribution": "Example corpus contributors",
        "allowed_uses": ALLOWED_USES,
        "category": "open_licensed",
        "login_required": False,
        "paywalled": False,
        "robots_allowed": True,
        "terms_allowed": True,
        "content_risks": (),
    }
    values[field] = value

    with pytest.raises(SourcePolicyError) as exc_info:
        SourceLicense(**values)

    assert exc_info.value.reason == "invalid_spec"
    assert str(value) not in str(exc_info.value)


def test_runtime_decisions_follow_injected_validated_policy(tmp_path: Path) -> None:
    document = policy_document()
    document["allowed_licenses"] = {
        "CC0-1.0": {
            "canonical_license_url": LICENSE_URLS["CC0-1.0"],
            "permitted_uses": ["research"],
        }
    }
    policy_path = write_policy(tmp_path, document)

    allowed = dry_run_source(
        source_spec(
            license_id="CC0-1.0",
            license_url=LICENSE_URLS["CC0-1.0"],
            allowed_uses=("research",),
        ),
        policy_path=policy_path,
    )
    removed_license = dry_run_source(source_spec(), policy_path=policy_path)
    removed_use = dry_run_source(
        source_spec(
            license_id="CC0-1.0",
            license_url=LICENSE_URLS["CC0-1.0"],
            allowed_uses=("model_training",),
        ),
        policy_path=policy_path,
    )

    assert allowed.allowed is True
    assert removed_license.reason == "unsupported_license"
    assert removed_use.reason == "unsupported_allowed_use"


def test_replaced_policy_object_cannot_enter_public_execution_api() -> None:
    validated = load_policy()
    rule_type = type(next(iter(validated.allowed_licenses.values())))
    forged = replace(
        validated,
        allowed_licenses={
            "SYNTHETIC-1.0": rule_type(
                canonical_license_url="https://licenses.example/synthetic/",
                permitted_uses=frozenset({"research"}),
            )
        },
    )
    synthetic_spec = source_spec(
        license_id="SYNTHETIC-1.0",
        license_url="https://licenses.example/synthetic/",
        allowed_uses=("research",),
    )
    fetch_calls = 0

    def forbidden_fetcher(_: str) -> bytes:
        nonlocal fetch_calls
        fetch_calls += 1
        return b"synthetic"

    default_decision = dry_run_source(synthetic_spec)
    with pytest.raises(TypeError):
        dry_run_source(synthetic_spec, policy=forged)
    with pytest.raises(TypeError):
        collect_source(
            synthetic_spec,
            policy=forged,
            fetcher=forbidden_fetcher,
        )

    assert default_decision.allowed is False
    assert default_decision.reason == "unsupported_license"
    assert fetch_calls == 0


def test_dry_run_fails_closed_when_policy_is_missing(tmp_path: Path) -> None:
    missing_policy = tmp_path / "MISSING-POLICY-CANARY.yaml"

    decision = dry_run_source(source_spec(), policy_path=missing_policy)

    assert decision.allowed is False
    assert decision.reason == "invalid_policy"
    assert "MISSING-POLICY-CANARY" not in decision.reason


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_top_key",
        "wrong_version",
        "non_deny_default",
        "wrong_license_container",
        "unknown_license_key",
        "noncanonical_license_url",
        "wrong_use_container",
        "duplicate_use",
    ],
)
def test_policy_loader_rejects_malformed_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    document = policy_document()
    licenses = document["allowed_licenses"]
    assert isinstance(licenses, dict)
    if mutation == "unknown_top_key":
        document["extra"] = "synthetic"
    elif mutation == "wrong_version":
        document["policy_version"] = True
    elif mutation == "non_deny_default":
        document["default_decision"] = "allow"
    elif mutation == "wrong_license_container":
        document["allowed_licenses"] = []
    elif mutation == "unknown_license_key":
        entry = licenses["CC0-1.0"]
        assert isinstance(entry, dict)
        entry["extra"] = "synthetic"
    elif mutation == "noncanonical_license_url":
        entry = licenses["CC0-1.0"]
        assert isinstance(entry, dict)
        entry[
            "canonical_license_url"
        ] = "HTTPS://creativecommons.org/publicdomain/zero/1.0/"
    elif mutation == "wrong_use_container":
        entry = licenses["CC0-1.0"]
        assert isinstance(entry, dict)
        entry["permitted_uses"] = "research"
    elif mutation == "duplicate_use":
        entry = licenses["CC0-1.0"]
        assert isinstance(entry, dict)
        entry["permitted_uses"] = ["research", "research"]

    with pytest.raises(SourcePolicyError) as exc_info:
        load_policy(write_policy(tmp_path, document))

    assert exc_info.value.reason == "invalid_policy"


def test_policy_loader_rejects_duplicate_keys_and_yaml_features(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate-policy.yaml"
    duplicate_path.write_text(
        '{"policy_version":1,"policy_version":1,'
        '"default_decision":"deny","allowed_licenses":{}}',
        encoding="utf-8",
    )
    yaml_feature_path = tmp_path / "tagged-policy.yaml"
    yaml_feature_path.write_text(
        "policy_version: &version 1\n"
        "default_decision: deny\n"
        "allowed_licenses: {}\n",
        encoding="utf-8",
    )

    for path in (duplicate_path, yaml_feature_path):
        with pytest.raises(SourcePolicyError) as exc_info:
            load_policy(path)
        assert exc_info.value.reason == "invalid_policy"


def test_dry_run_cli_defaults_to_no_fetch_and_redacts_sensitive_fields() -> None:
    private_markers = [
        "BODY-CANARY",
        "PATH-CANARY",
        "HEADER-CANARY",
        "PERSON-CANARY",
        "QUERY-CANARY",
    ]
    raw_spec = {
        "url": "https://example.org/open/corpus",
        "license_id": "CC-BY-4.0",
        "license_url": LICENSE_URLS["CC-BY-4.0"],
        "attribution": "Example corpus contributors",
        "allowed_uses": list(ALLOWED_USES),
        "category": "open_licensed",
        "body": private_markers[0],
        "local_path": private_markers[1],
        "headers": {"X-Private": private_markers[2]},
        "personal_metadata": private_markers[3],
        "query_material": private_markers[4],
    }

    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py")],
        cwd=REPOSITORY_ROOT,
        input=json.dumps(raw_spec) + "\n",
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    output = result.stdout
    assert json.loads(output) == {
        "category": "unclassified",
        "decision": "denied",
        "license_id": "unknown",
        "reason": "invalid_spec",
    }
    for marker in private_markers:
        assert marker not in output
    assert "example.org" not in output
    assert "corpus" not in output


def test_dry_run_cli_does_not_leak_an_unreadable_input_path(
    tmp_path: Path,
) -> None:
    marker = "PRIVATE-PATH-CANARY"
    missing_path = tmp_path / marker / "sources.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--input",
            str(missing_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert marker not in combined_output
    assert json.loads(result.stdout) == {
        "category": "unclassified",
        "decision": "denied",
        "license_id": "unknown",
        "reason": "invalid_input",
    }


def test_cli_rejects_invalid_utf8_without_traceback_or_path(
    tmp_path: Path,
) -> None:
    marker = "UTF8-PATH-CANARY"
    input_path = tmp_path / f"{marker}.jsonl"
    input_path.write_bytes(b"\xff\xfe")

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--input",
            str(input_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert_fixed_cli_denial(result, "invalid_input", marker, "Traceback")


def test_cli_rejects_oversized_input_before_parsing(tmp_path: Path) -> None:
    marker = "OVERSIZED-CONTENT-CANARY"
    input_path = tmp_path / "oversized.jsonl"
    input_path.write_bytes(marker.encode("utf-8") + b"x" * (128 * 1024))

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--input",
            str(input_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert_fixed_cli_denial(result, "invalid_input", marker, "Traceback")


def test_cli_rejects_deeply_nested_json_without_traceback(
    tmp_path: Path,
) -> None:
    marker = "DEEP-JSON-CANARY"
    input_path = tmp_path / "deep.jsonl"
    input_path.write_text(
        "[" * 2000 + json.dumps(marker) + "]" * 2000,
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--input",
            str(input_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert_fixed_cli_denial(result, "invalid_input", marker, "Traceback")


def test_cli_rejects_unreadable_directory_input_without_path_echo(
    tmp_path: Path,
) -> None:
    marker = "DIRECTORY-PATH-CANARY"
    input_path = tmp_path / marker
    input_path.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--input",
            str(input_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert_fixed_cli_denial(result, "invalid_input", marker, "Traceback")


def test_cli_loads_same_authoritative_policy(tmp_path: Path) -> None:
    document = policy_document()
    document["allowed_licenses"] = {
        "CC0-1.0": {
            "canonical_license_url": LICENSE_URLS["CC0-1.0"],
            "permitted_uses": ["research"],
        }
    }
    policy_path = write_policy(tmp_path, document)
    raw_spec = {
        "url": "https://example.org/open/corpus",
        "license_id": "CC-BY-4.0",
        "license_url": LICENSE_URLS["CC-BY-4.0"],
        "attribution": "Synthetic attribution",
        "allowed_uses": ["research"],
    }

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--config",
            str(policy_path),
        ],
        cwd=REPOSITORY_ROOT,
        input=json.dumps(raw_spec) + "\n",
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "category": "open_licensed",
        "decision": "denied",
        "license_id": "unsupported",
        "reason": "unsupported_license",
    }


def test_cli_missing_policy_fails_closed_without_path_echo(
    tmp_path: Path,
) -> None:
    marker = "MISSING-CONFIG-CANARY"
    policy_path = tmp_path / f"{marker}.yaml"

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            "--config",
            str(policy_path),
        ],
        cwd=REPOSITORY_ROOT,
        input="",
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert_fixed_cli_denial(result, "invalid_policy", marker, "Traceback")


@pytest.mark.parametrize(
    "arguments",
    [
        ["TOKEN-CANARY-SYNTHETIC"],
        ["--unknown-option", "TOKEN-CANARY-SYNTHETIC"],
        ["--input"],
    ],
)
def test_dry_run_cli_argument_errors_never_echo_raw_values(
    arguments: list[str],
) -> None:
    marker = "TOKEN-CANARY-SYNTHETIC"

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "collect_open_sources.py"),
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert marker not in combined_output
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "category": "unclassified",
        "decision": "denied",
        "license_id": "unknown",
        "reason": "invalid_arguments",
    }


def test_allowlist_is_deny_by_default_and_contains_no_collection_targets() -> None:
    policy_text = (
        REPOSITORY_ROOT / "configs" / "sources" / "license-allowlist.yaml"
    ).read_text(encoding="utf-8")
    policy = json.loads(policy_text)

    assert policy == policy_document()
    assert policy["default_decision"] == "deny"
    assert load_policy().allowed_licenses["CC-BY-SA-4.0"].permitted_uses == frozenset(
        {"research", "retrieval", "evaluation", "adaptation"}
    )
    for license_id in LICENSE_URLS:
        assert license_id in policy["allowed_licenses"]
    assert "sources:" not in policy_text
    assert "collection_targets:" not in policy_text

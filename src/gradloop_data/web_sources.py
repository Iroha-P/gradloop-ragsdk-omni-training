"""Deny-by-default policy gate for explicitly licensed public web sources."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

from .provenance import ProvenanceRecord

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "sources"
    / "license-allowlist.yaml"
)
MAX_POLICY_BYTES = 64 * 1024
KNOWN_CATEGORIES = frozenset(
    {
        "open_licensed",
        "official_facts",
        "public_web",
        "open_web",
        "personal_social",
    }
)
CATEGORY_ONLY_LABELS = frozenset({"official_facts", "public_web", "open_web"})
PROHIBITED_RISKS = frozenset(
    {
        "faces",
        "id_documents",
        "credential_documents",
        "contact_information_images",
        "contact_information_metadata",
    }
)
_PRIVATE_HOST_SUFFIXES = (".internal", ".local", ".localhost", ".home", ".lan")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_LEGACY_NUMERIC_LABEL = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)", re.IGNORECASE)
_LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,63}")
_USE_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_POLICY_ROOT_KEYS = frozenset(
    {"policy_version", "default_decision", "allowed_licenses"}
)
_LICENSE_RULE_KEYS = frozenset({"canonical_license_url", "permitted_uses"})
_SOURCE_SPEC_FIELDS = frozenset(
    {
        "url",
        "license_id",
        "license_url",
        "attribution",
        "allowed_uses",
        "category",
        "login_required",
        "paywalled",
        "robots_allowed",
        "terms_allowed",
        "content_risks",
    }
)


class SourcePolicyError(ValueError):
    """A stable, non-sensitive reason why a source was denied."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CollectionDisabledError(RuntimeError):
    """Raised when collection is requested without an explicit fetcher."""


@dataclass(frozen=True)
class LicenseRule:
    """Validated rule loaded from the authoritative policy document."""

    canonical_license_url: str
    permitted_uses: frozenset[str]


@dataclass(frozen=True)
class LicensePolicy:
    """Immutable, validated runtime license policy."""

    allowed_licenses: Mapping[str, LicenseRule]


@dataclass(frozen=True)
class SourceLicense:
    """License declaration plus policy facts needed before collection."""

    url: str
    license_id: str
    license_url: str
    attribution: str
    allowed_uses: tuple[str, ...]
    category: str = "open_licensed"
    login_required: bool = False
    paywalled: bool = False
    robots_allowed: bool = True
    terms_allowed: bool = True
    content_risks: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        text_values = (
            self.url,
            self.license_id,
            self.license_url,
            self.attribution,
            self.category,
        )
        boolean_values = (
            self.login_required,
            self.paywalled,
            self.robots_allowed,
            self.terms_allowed,
        )
        if any(type(value) is not str for value in text_values):
            raise SourcePolicyError("invalid_spec")
        if any(type(value) is not bool for value in boolean_values):
            raise SourcePolicyError("invalid_spec")
        if type(self.allowed_uses) is not tuple or any(
            type(value) is not str for value in self.allowed_uses
        ):
            raise SourcePolicyError("invalid_spec")
        if type(self.content_risks) is not tuple or any(
            type(value) is not str for value in self.content_risks
        ):
            raise SourcePolicyError("invalid_spec")


@dataclass(frozen=True)
class SourceDecision:
    """Safe dry-run result containing policy labels, never source content."""

    allowed: bool
    reason: str
    category: str
    license_id: str


Fetcher = Callable[[str], bytes]
Clock = Callable[[], datetime]


def _safe_category(category: object) -> str:
    if isinstance(category, str) and category in KNOWN_CATEGORIES:
        return category
    return "unclassified"


def _safe_license_id(
    license_id: object,
    policy: LicensePolicy | None,
) -> str:
    if (
        policy is not None
        and isinstance(license_id, str)
        and license_id in policy.allowed_licenses
    ):
        return license_id
    if not license_id:
        return "unknown"
    return "unsupported"


def _canonical_public_host(hostname: str) -> str | None:
    if not hostname or hostname.endswith("."):
        return None
    lowered = hostname.casefold()
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        labels = lowered.split(".")
        if all(_LEGACY_NUMERIC_LABEL.fullmatch(label) for label in labels):
            return None
        try:
            ascii_host = hostname.encode("idna").decode("ascii").casefold()
        except UnicodeError:
            return None
        ascii_labels = ascii_host.split(".")
        if (
            len(ascii_host) > 253
            or len(ascii_labels) < 2
            or any(not label for label in ascii_labels)
            or all(_LEGACY_NUMERIC_LABEL.fullmatch(label) for label in ascii_labels)
            or ascii_host == "localhost"
            or ascii_host.endswith(_PRIVATE_HOST_SUFFIXES)
        ):
            return None
        for label in ascii_labels:
            if _HOST_LABEL.fullmatch(label) is None:
                return None
            if label.startswith("xn--"):
                try:
                    decoded = label.encode("ascii").decode("idna")
                    round_trip = decoded.encode("idna").decode("ascii").casefold()
                except UnicodeError:
                    return None
                if round_trip != label:
                    return None
        return ascii_host
    canonical = address.compressed.casefold()
    if address.is_multicast or not address.is_global or canonical != lowered:
        return None
    return canonical


def _explicit_port_text(netloc: str) -> str | None:
    authority = netloc.rsplit("@", 1)[-1]
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            raise SourcePolicyError("invalid_url")
        suffix = authority[closing + 1 :]
        if not suffix:
            return None
        if not suffix.startswith(":") or ":" in suffix[1:]:
            raise SourcePolicyError("invalid_port")
        return suffix[1:]
    if ":" not in authority:
        return None
    if authority.count(":") != 1:
        raise SourcePolicyError("invalid_url")
    return authority.rsplit(":", 1)[1]


def canonicalize_url(url: str) -> str:
    """Validate and canonicalize a public HTTPS URL without network access."""

    if (
        not isinstance(url, str)
        or not url
        or any(char.isspace() for char in url)
        or any(ord(char) < 32 or ord(char) == 127 for char in url)
    ):
        raise SourcePolicyError("invalid_url")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise SourcePolicyError("invalid_url") from exc
    if parsed.scheme.casefold() != "https":
        raise SourcePolicyError("https_required")
    if parsed.username is not None or parsed.password is not None:
        raise SourcePolicyError("url_credentials")
    if parsed.hostname is None:
        raise SourcePolicyError("invalid_url")
    canonical_host = _canonical_public_host(parsed.hostname)
    if canonical_host is None:
        raise SourcePolicyError("private_network_url")
    port_text = _explicit_port_text(parsed.netloc)
    if port_text is None:
        port = None
    elif re.fullmatch(r"[0-9]+", port_text) is None:
        raise SourcePolicyError("invalid_port")
    else:
        port = int(port_text)
        if port < 1 or port > 65535 or port_text != str(port):
            raise SourcePolicyError("invalid_port")

    if parsed.query:
        raise SourcePolicyError("query_not_allowed")
    if "#" in url:
        raise SourcePolicyError("fragment_not_allowed")

    rendered_host = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    netloc = rendered_host if port in (None, 443) else f"{rendered_host}:{port}"
    path = posixpath.normpath(parsed.path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    if parsed.path.endswith("/") and not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit(("https", netloc, path, "", ""))


def _reject_duplicate_policy_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate policy key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> object:
    raise ValueError("non-standard JSON constant")


def _validated_policy(document: object) -> LicensePolicy:
    if type(document) is not dict or set(document) != _POLICY_ROOT_KEYS:
        raise ValueError("invalid policy root")
    if type(document["policy_version"]) is not int:
        raise ValueError("invalid policy version")
    if document["policy_version"] != 1:
        raise ValueError("unsupported policy version")
    if type(document["default_decision"]) is not str:
        raise ValueError("invalid default decision")
    if document["default_decision"] != "deny":
        raise ValueError("policy must deny by default")
    licenses = document["allowed_licenses"]
    if type(licenses) is not dict:
        raise ValueError("invalid license container")

    validated: dict[str, LicenseRule] = {}
    for license_id, raw_rule in licenses.items():
        if (
            type(license_id) is not str
            or _LICENSE_ID.fullmatch(license_id) is None
            or type(raw_rule) is not dict
            or set(raw_rule) != _LICENSE_RULE_KEYS
        ):
            raise ValueError("invalid license rule")
        license_url = raw_rule["canonical_license_url"]
        raw_uses = raw_rule["permitted_uses"]
        if type(license_url) is not str or type(raw_uses) is not list:
            raise ValueError("invalid license rule types")
        if (
            not raw_uses
            or any(
                type(use) is not str or _USE_ID.fullmatch(use) is None
                for use in raw_uses
            )
            or len(set(raw_uses)) != len(raw_uses)
        ):
            raise ValueError("invalid permitted uses")
        if canonicalize_url(license_url) != license_url:
            raise ValueError("license URL is not canonical")
        validated[license_id] = LicenseRule(
            canonical_license_url=license_url,
            permitted_uses=frozenset(raw_uses),
        )
    return LicensePolicy(
        allowed_licenses=MappingProxyType(validated),
    )


def load_policy(path: Path | str | None = None) -> LicensePolicy:
    """Load the authoritative JSON-compatible YAML policy fail closed."""

    try:
        policy_path = DEFAULT_POLICY_PATH if path is None else Path(path)
        with policy_path.open("rb") as stream:
            raw = stream.read(MAX_POLICY_BYTES + 1)
        if len(raw) > MAX_POLICY_BYTES:
            raise ValueError("policy is oversized")
        text = raw.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_policy_keys,
            parse_constant=_reject_json_constant,
        )
        return _validated_policy(document)
    except Exception as exc:
        raise SourcePolicyError("invalid_policy") from exc


def _validate_source(spec: SourceLicense, policy: LicensePolicy) -> str:
    canonical_url = canonicalize_url(spec.url)

    if spec.category not in KNOWN_CATEGORIES:
        raise SourcePolicyError("unknown_category")
    if spec.category == "personal_social":
        raise SourcePolicyError("personal_social")
    if spec.login_required:
        raise SourcePolicyError("login_required")
    if spec.paywalled:
        raise SourcePolicyError("paywalled")
    if not spec.robots_allowed:
        raise SourcePolicyError("robots_denied")
    if not spec.terms_allowed:
        raise SourcePolicyError("terms_denied")

    for risk in spec.content_risks:
        if risk in PROHIBITED_RISKS:
            raise SourcePolicyError(risk)
        raise SourcePolicyError("unsupported_content_risk")

    if not spec.license_id:
        if spec.category in CATEGORY_ONLY_LABELS:
            raise SourcePolicyError("category_only_not_a_license")
        raise SourcePolicyError("unknown_license")
    rule = policy.allowed_licenses.get(spec.license_id)
    if rule is None:
        raise SourcePolicyError("unsupported_license")
    expected_license_url = rule.canonical_license_url
    try:
        actual_license_url = canonicalize_url(spec.license_url)
        canonical_license_url = canonicalize_url(expected_license_url)
    except SourcePolicyError as exc:
        raise SourcePolicyError("invalid_license_url") from exc
    if actual_license_url != canonical_license_url:
        raise SourcePolicyError("license_url_mismatch")
    if not isinstance(spec.attribution, str) or not spec.attribution.strip():
        raise SourcePolicyError("missing_attribution")
    if not spec.allowed_uses:
        raise SourcePolicyError("missing_allowed_uses")
    if any(use not in rule.permitted_uses for use in spec.allowed_uses):
        raise SourcePolicyError("unsupported_allowed_use")
    return canonical_url


def dry_run_source(
    spec: SourceLicense,
    *,
    fetcher: Fetcher | None = None,
    policy_path: Path | str | None = None,
) -> SourceDecision:
    """Evaluate policy only; ``fetcher`` is intentionally never called."""

    del fetcher
    resolved_policy: LicensePolicy | None = None
    try:
        resolved_policy = load_policy(policy_path)
        _validate_source(spec, resolved_policy)
    except SourcePolicyError as exc:
        return SourceDecision(
            allowed=False,
            reason=exc.reason,
            category=_safe_category(spec.category),
            license_id=_safe_license_id(spec.license_id, resolved_policy),
        )
    return SourceDecision(
        allowed=True,
        reason="allowed",
        category=_safe_category(spec.category),
        license_id=_safe_license_id(spec.license_id, resolved_policy),
    )


def _utc_timestamp(clock: Clock) -> str:
    try:
        value = clock()
        if not isinstance(value, datetime):
            raise TypeError
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError
        value = value.astimezone(timezone.utc)
        return value.isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception as exc:
        raise SourcePolicyError("invalid_timestamp") from exc


def collect_source(
    spec: SourceLicense,
    *,
    fetcher: Fetcher | None = None,
    clock: Clock | None = None,
    policy_path: Path | str | None = None,
) -> ProvenanceRecord:
    """Collect approved content only through an explicitly injected fetcher."""

    resolved_policy = load_policy(policy_path)
    canonical_url = _validate_source(spec, resolved_policy)
    if fetcher is None:
        raise CollectionDisabledError(
            "collection requires an explicitly injected fetcher"
        )
    timestamp = _utc_timestamp(clock or (lambda: datetime.now(timezone.utc)))
    content = fetcher(canonical_url)
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("fetcher must return bytes")
    body = bytes(content)
    rule = resolved_policy.allowed_licenses[spec.license_id]
    return ProvenanceRecord(
        canonical_url=canonical_url,
        retrieved_at=timestamp,
        content_sha256=hashlib.sha256(body).hexdigest(),
        license_id=spec.license_id,
        license_url=rule.canonical_license_url,
        attribution=spec.attribution.strip(),
        allowed_uses=spec.allowed_uses,
    )


def source_license_from_mapping(values: Mapping[str, object]) -> SourceLicense:
    """Build a source declaration, failing closed on unrecognized fields."""

    if any(key not in _SOURCE_SPEC_FIELDS for key in values):
        raise SourcePolicyError("invalid_spec")
    text_fields = ("url", "license_id", "license_url", "attribution", "category")
    for name in text_fields:
        if name in values and not isinstance(values[name], str):
            raise SourcePolicyError("invalid_spec")
    boolean_fields = (
        "login_required",
        "paywalled",
        "robots_allowed",
        "terms_allowed",
    )
    for name in boolean_fields:
        if name in values and type(values[name]) is not bool:
            raise SourcePolicyError("invalid_spec")
    allowed_uses = values.get("allowed_uses", ())
    content_risks = values.get("content_risks", ())
    if isinstance(allowed_uses, str) or not isinstance(allowed_uses, Sequence):
        raise SourcePolicyError("invalid_spec")
    if isinstance(content_risks, str) or not isinstance(content_risks, Sequence):
        raise SourcePolicyError("invalid_spec")
    if any(not isinstance(item, str) for item in allowed_uses):
        raise SourcePolicyError("invalid_spec")
    if any(not isinstance(item, str) for item in content_risks):
        raise SourcePolicyError("invalid_spec")
    try:
        return SourceLicense(
            url=values.get("url", ""),
            license_id=values.get("license_id", ""),
            license_url=values.get("license_url", ""),
            attribution=values.get("attribution", ""),
            allowed_uses=tuple(allowed_uses),
            category=values.get("category", "open_licensed"),
            login_required=values.get("login_required", False),
            paywalled=values.get("paywalled", False),
            robots_allowed=values.get("robots_allowed", True),
            terms_allowed=values.get("terms_allowed", True),
            content_risks=tuple(content_risks),
        )
    except (TypeError, ValueError) as exc:
        raise SourcePolicyError("invalid_spec") from exc

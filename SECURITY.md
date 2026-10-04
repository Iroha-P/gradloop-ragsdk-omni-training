# Security policy

## Public-safe repository rule

Assume every tracked byte can become public. Do not add private repository
source, real user or experiment data, Office or PDF documents, databases,
credentials, machine-local absolute paths, model artifacts, checkpoints, or
generated outputs.

The staged-file gate applies three controls:

1. a filename and extension allowlist plus explicit forbidden directories;
2. a one-mebibyte limit and UTF-8 text-only requirement for every staged blob;
3. content checks for credentials and machine-local absolute paths.

The scanner is fail-closed: Git failures, unreadable index blobs, decoding
errors, and unsupported file types block the commit. `.env.example` is the only
allowed environment-file variant, and it may contain empty placeholders only.

## Stage A public-release boundary

Private corpora, private workspaces, user materials, credentials, and
machine-local source material are excluded from this repository and from Stage
A bundle inputs. A candidate source must have an explicit, verified license
record that permits its intended use; category labels and unknown licenses do
not qualify. `CC-BY-SA-4.0` is not eligible for public model training.

Before publication, the release gate scans sample text, every referenced
evidence value, OCR text, transcripts, and metadata for personal-data patterns,
credentials, and machine-local paths. It also rejects missing or unreferenced
evidence, invalid source/provenance records, duplicates, and near-copy answers
across every bounded scan chunk. Findings contain only safe identifiers and
codes, never raw matched content or local paths.

Bundle publication is fail-closed. A failed schema, split, source-license,
provenance, evidence, source-leakage, or release check prevents the final
bundle directory from being published; an existing destination is never
merged with or overwritten. Deleting data from a cloud location cannot untrain
a model, so privacy rejection occurs before any training or publication step.

Source-manifest attribution text passes the same privacy, credential, and
machine-path policy before it can be written. Public retrieval times must use
the canonical UTC form `YYYY-MM-DDTHH:MM:SSZ` and represent a real calendar
time. Source IDs own unique provenance IDs, and every sample provenance set
must exactly equal the provenance identities of its source set.

If a staged file is rejected, remove it from the index, move it to an ignored
local directory when appropriate, and run the scanner again. Do not bypass the
hook to force a commit.

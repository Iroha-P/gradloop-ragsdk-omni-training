# GradLoop Stage A Data Card

## Purpose

Stage A is a local, public-safe data and evaluation foundation for GradLoop
RAGSDK Omni. It validates the data path, source policy, source-connected
splits, release checks, and aggregate evaluation interfaces before any training
work. This repository contains no real dataset. Generated bundles are local
artifacts under ignored data and output roots and are never release inputs by
default.

## Supported Scenarios

Stage A supports exactly these scenarios:

- `course_learning`
- `technical_interview`
- `research_defense`

The approved design allocates 30% to course learning, 30% to technical
interview, and 20% to research defense. The remaining 20% is reserved in that
design for other task categories; it does not add another Stage A scenario.
Current deterministic fixtures deliberately use equal counts for the three
supported scenarios. They validate pipeline behavior only and do not claim
production training quality or represent the target distribution.

## Allowed Licenses

The deny-by-default policy accepts explicit source records with `CC0-1.0`,
`PDM-1.0`, or `CC-BY-4.0` for public model training. `CC-BY-4.0` use requires
the applicable attribution obligations to be preserved. `CC-BY-SA-4.0` is
restricted to retrieval, evaluation, and adaptation in Stage A; it is not
eligible for public model training.

## Excluded Sources

Do not use private corpora, private workspaces, user materials, login-required
or paywalled content, machine-local sources, unknown-license content,
non-commercial or no-derivatives licenses, or sources lacking an explicit
training permission. A source category is not a substitute for a verified
license record.

## Modalities

The canonical schema supports `text`, `document`, `image`, `audio`, and
`video`. The deterministic Stage A fixtures exercise text together with
document, image, and audio sidecars. They contain no binary media: document and
image fixtures use synthetic OCR text, audio fixtures use a synthetic
transcript, and the asset manifest explicitly marks every fixture as
sidecar-only. Candidate sidecar text passes the same release gate as the sample
and its referenced evidence.

## Sample Schema

Each immutable training sample has these fields:

```text
sample_id
scenario
task_type
modalities
source_ids
evidence_ids
prompt
reference_answer
follow_ups
rubric
refusal_reason
provenance_ids
generation_method
quality_status
privacy_status
asset_sidecars
```

`asset_sidecars` records an asset identifier, modality, OCR text, transcript,
and metadata text. The schema rejects incomplete or invalid records before a
bundle can be built.

## Source-Level Split Policy

Samples are first connected by shared source identifiers. Each connected
component is deterministically assigned to exactly one of `train`,
`validation`, or `test`; only after that assignment are bundle records written.
The bundle checks that no source identifier occurs in more than one split.
`evaluation_cases.jsonl` is derived only from `test` assignments, and the
publisher verifies that evaluation sources are absent from both `train` and
`validation`.

## Provenance and Publication Manifest

Publication consumes validated source/provenance records rather than a caller
supplied source-ID allowlist. Public-content records are rechecked against the
training-license policy and joined to retrieval timestamp and content digest
metadata. Deterministic fixtures use a distinct `synthetic_recipe` provenance
kind with stable recipe name, version, digest, and provenance identity.
`source_manifest.json` contains only those privacy-safe decisions and
obligations; it omits raw content and source URLs. Attribution obligations pass
the same privacy, credential, and machine-path checks as candidate text.
Retrieval timestamps must be real UTC calendar values in exact
`YYYY-MM-DDTHH:MM:SSZ` form. Provenance ownership is unique, and each sample's
provenance IDs must exactly equal those owned by its source IDs; missing,
duplicate, or orphan provenance blocks the bundle.

## Privacy and Near-Copy Gate

Every candidate is checked before release for personal-data patterns,
credentials, machine-local absolute paths, unknown sources, missing evidence,
duplicate identifiers, and near-copy reference answers. The gate scans prompts,
answers, follow-ups, rubrics, refusal reasons, every referenced evidence value,
OCR text, transcripts, and metadata text across bounded chunks. Unreferenced
evidence is rejected before writing. Findings retain only a sample identifier,
field, and stable code; they never retain matched content or an input path. Any
finding blocks the entire bundle.

## Frozen Evaluation Coverage

Every publishable fixture set contains deterministic answerable and
evidence-insufficient/refusal fixtures. The evaluator reports recall, citation,
refusal, structure validity, and mean latency for each scenario as well as
overall. Recall, citation, and refusal positive/negative support counts are
reported explicitly. A set with zero refusal-positive support reports refusal
accuracy as `0.0`, not a perfect score.

Before publication, the frozen subset must be non-empty, contain cases for all
three supported scenarios, and contain at least one refusal-positive and one
refusal-negative case. These are behavioral postconditions, not a count
shortcut. With the current deterministic recipe and split salt, counts from 1
through 16 fixtures per scenario do not satisfy the complete contract and
return `bundle_blocked`; 17 is the first satisfying count. The documented
20-per-scenario build retains a larger five-case frozen subset.

## Known Limitations

The deterministic synthetic fixtures are small, template-derived validation
inputs with sidecar metadata rather than real document, image, or audio
binaries. They establish interface and release-gate coverage only; they do not
establish representativeness, real-media quality, model quality, safety
performance, or production readiness. Stage A does not perform model training,
network collection, or cloud publication.

## Redistribution Rules

Only content that has passed the source-license and release gates may enter a
candidate bundle. Preserve attribution and redistribution obligations from the
verified source record. Do not redistribute raw source content, generated
bundles, model artifacts, or private material through this repository.

## Reproduction Evidence

Run the local builder with 20 deterministic fixtures per supported scenario.
The expected summary is 60 samples total, 20 per scenario, 30 refusal samples,
60 sidecar assets, 5 test-only evaluation cases (3 refusal-positive and 2
refusal-negative), zero findings, and zero source leakage. The generated bundle
remains ignored, while its summary, source and asset manifests, split manifest,
release report, and evaluation cases provide local evidence of the pipeline
run.

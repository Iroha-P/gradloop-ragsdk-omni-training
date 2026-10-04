# Stage A Local Runbook

Run these commands from an environment using the repository's Anaconda Python
interpreter. They make no network request and write generated artifacts only
to ignored local roots.

## Verify and Build

```text
python -m pytest -q
python scripts/build_stage_a_bundle.py --synthetic-per-scenario 20 --output dist/stage-a-public
python scripts/security/precommit_scan.py --staged
```

The first command runs the complete local test suite. The builder creates a
fresh `dist/stage-a-public` directory only after schema validation,
source-connected splitting, source-leakage verification, and release-gate
approval. It refuses an existing destination rather than merging with or
overwriting it.

Publication also checks the builder/evaluator boundary: the frozen subset must
be non-empty, cover all three scenarios, and include refusal-positive and
refusal-negative support. These postconditions determine usability. For the
current deterministic recipe and split salt, 17 fixtures per scenario is the
first passing count; counts from 1 through 16 return fixed `bundle_blocked` and
leave no directory. The recommended count remains 20.

## Generated Bundle Contents

- `samples.jsonl` contains canonical, deterministic sample records.
- `evidence.json` maps evidence identifiers to the synthetic evidence used by
  the fixtures; every value is referenced and release-scanned.
- `source_manifest.json` records validated license decisions or explicit
  synthetic-recipe provenance, attribution obligations, and stable provenance
  identity without raw source text or URLs. Attribution text is release-scanned,
  public retrieval times use strict canonical UTC values, and source/sample
  provenance joins are exact.
- `asset_manifest.json` records document, image, and audio sidecar assets and
  marks that the Stage A fixtures contain no binary media.
- `split_manifest.json` records the source-connected split assignment.
- `release_report.json` records per-sample allow decisions and privacy-safe
  findings, if any.
- `evaluation_cases.jsonl` contains frozen evaluation case records derived only
  from `test` assignments.
- `summary.json` records release status, sample/asset/evaluation/refusal counts,
  modality and scenario counts, refusal-positive/refusal-negative evaluation
  support, finding count, and source-leakage count.

For this command, `summary.json` must report 60 samples, 20 each for
`course_learning`, `technical_interview`, and `research_defense`, 30 refusal
samples, 60 sidecar assets, 5 test-only evaluation cases with 3 positive and 2
negative refusal supports, zero findings, and zero source leakage.

## Frozen Evaluation

With public-safe prediction records prepared in an ignored output root, compute
aggregate-only metrics with:

```text
python scripts/evaluate_frozen_set.py --cases dist/stage-a-public/evaluation_cases.jsonl --predictions dist/stage-a-public/predictions.jsonl --output dist/stage-a-metrics.json
```

The evaluator accepts strict JSON Lines inputs, rejecting duplicate object keys
and non-standard constants, and writes one aggregate metrics JSON file. It
reports recall, citation precision, refusal accuracy, structure validity,
latency, and applicable support counts for all three scenarios separately,
including a zero-case scenario when applicable. Refusal accuracy is `0.0` when
the evaluated set has no refusal-positive support.

## Stable Failures and Release Boundary

The bundle builder emits only fixed reason codes: `invalid_arguments` (exit
status 2) for unsafe, malformed, or existing output destinations, and
`bundle_blocked` (exit status 1) when release findings, provenance checks, or
frozen-set postconditions block publication. The
frozen evaluator emits `invalid_arguments` (exit status 2) for unsafe arguments
or output destinations and `invalid_input` (exit status 1) for invalid cases or
predictions. These responses intentionally omit raw input values and content.

The staged-file scanner must pass before a commit. Never stage any file under
ignored data or output roots, including `data/`, `outputs/`, and `dist/`.
Generated bundles, predictions, metrics, model artifacts, and private material
remain local and untracked.

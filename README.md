# GradLoop RAGSDK Omni Training

GradLoop RAGSDK Omni is an evidence-constrained multimodal learning and
interview-training agent. This repository is its independent, public-safe
training foundation: it contains source, configuration, tests, and
documentation only, never a real dataset or model artifact.

## 项目导航

本仓库负责 GradLoop 的数据、训练、模型评测和模型服务；[应用仓库](https://github.com/Iroha-P/gradloop-ragsdk-omni)负责学习闭环、API 和 UI。项目分工、维护与归档方式见 [项目指南](./PROJECT_GUIDE.md)，协作规则见 [AGENTS](./AGENTS.md)。

公开版本只包含源码、测试、文档与配置，不包含真实数据、模型权重、adapter、训练输出或本地历史资料。

## Stage A: public data and evaluation foundation

Stage A validates deterministic synthetic text plus document/image/audio
sidecar fixtures, refusal behavior, source-license and recipe provenance,
source-connected splits, privacy and near-copy release checks, and frozen
test-only evaluation. It supports exactly:

- `course_learning`
- `technical_interview`
- `research_defense`

Build a local public-safe bundle and evaluate public-safe predictions with:

```text
python scripts/build_stage_a_bundle.py --synthetic-per-scenario 20 --output dist/stage-a-public
python scripts/evaluate_frozen_set.py --cases dist/stage-a-public/evaluation_cases.jsonl --predictions dist/stage-a-public/predictions.jsonl --output dist/stage-a-metrics.json
```

These generated paths are ignored and must remain untracked. The deterministic
fixtures validate the Stage A pipeline; they do not claim production training
quality. Publication additionally requires a non-empty frozen set spanning all
three scenarios with both refusal-positive and refusal-negative support. Under
the current deterministic recipe and split salt, 17 fixtures per scenario is
the first count satisfying that contract; smaller counts fail closed. Stage B
Ascend execution results are summarized below.

## Stage B: NPU baseline preparation

Stage B preparation now includes a privacy-safe Ascend environment probe,
an offline MiniCPM-o cache/data preflight, deterministic ten-case text/image/
audio smoke inputs, four-route Base execution, frozen evaluation, a one-update
SWIFT LoRA feasibility gate, adapter reload, and Base/LoRA comparison. These
paths retain only hashes, aggregate metrics, timings, and memory evidence.
The personal computer never stores MiniCPM-o weights. On 2026-07-29, the
pinned MiniCPM-o 4.5 revision completed Base and reloaded-LoRA smoke on an
Ascend 910C HiDevLab TorchNPU environment: both passed 10/10 deterministic
text, image, audio, and image-audio cases. A one-update SWIFT LoRA run saved
adapter-only artifacts and reloaded successfully. On the five-case frozen
synthetic set, Base and LoRA both scored 0.1333 on the predeclared composite,
so the promotion decision is `retain_base`; feasibility is not presented as a
quality improvement. The final release scan covered nine aggregate reports
with zero findings.

See `docs/stage-b-npu-results-2026-07-29.md` for measured results and
limitations, and `docs/stage-b-npu-runbook.md` before using any NPU resource.

## Cloud demo service (Base only)

`scripts/npu/serve_minicpmo.py` is the narrow service boundary used by the
application demo. It loads the already-cached Base model only and accepts
request-scoped `public_or_synthetic` text, image, and WAV-audio inputs. It
does not train, write request content to disk, accept paths or filenames, or
claim video support. The service must be installed and started inside the
Ascend environment; it has not yet completed a new cloud end-to-end run in
this repository revision.

```text
pip install -e ".[service]"
python scripts/npu/serve_minicpmo.py --model-cache MODEL_CACHE_DIRECTORY
```

## Repository boundary

- Build training code here without reading or copying private repository code.
- Keep real data under the ignored `data/incoming`, `data/processed`, or
  `data/cache` directories.
- Keep weights, checkpoints, outputs, and packages in their ignored directories.
- Commit only small UTF-8 source, documentation, and configuration files.
- Use `.env.example` only for empty placeholders. Put real values in a local
  `.env` variant.

## Security check

Run the staged-file gate before every commit:

```text
python scripts/security/precommit_scan.py --staged
```

The local pre-commit hook in this checkout invokes the same command and exits
nonzero whenever the scanner reports a violation or cannot complete its checks.

Run the policy tests with:

```text
python -m pytest tests/security/test_repository_policy.py -q
```

## Source licensing

No source-code license has been declared yet. Public visibility is not a grant of redistribution rights. Model and dataset licenses must be checked independently before use.

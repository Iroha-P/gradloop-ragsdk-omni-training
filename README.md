# GradLoop RAGSDK Omni Training

> GradLoop 的数据、模型实验、评测与推理服务工程

**🏆 所属 GradLoop 项目：星火杯 · 全球总决赛 20 强**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ascend](https://img.shields.io/badge/NPU-Huawei%20Ascend-D71920)](docs/stage-b-npu-runbook.md)
[![MiniCPM-o](https://img.shields.io/badge/Model-MiniCPM--o%204.5-1C3C3C)](docs/stage-b-npu-results-2026-07-29.md)
[![LoRA](https://img.shields.io/badge/Experiment-Base%20%2F%20LoRA-6F42C1)](docs/stage-b-npu-runbook.md)
[![Evaluation](https://img.shields.io/badge/Evaluation-Frozen%20Protocol-18A058)](docs/stage-a-runbook.md)

**[应用与在线体验](https://github.com/Iroha-P/gradloop-ragsdk-omni)** · **[功能亮点](#功能亮点)** · **[运行手册](docs/stage-a-runbook.md)** · **[NPU 实测记录](docs/stage-b-npu-results-2026-07-29.md)** · **[安全政策](SECURITY.md)**

这是 GradLoop 的模型工程底座：把**数据来源、可复验实验、冻结评测和服务接口**连接起来，为课程学习、技术面试和科研答辩提供有证据的模型能力。它不是第二套应用，也不要求用户为了体验网页而先训练模型。

GradLoop 项目参加星火杯并取得**全球总决赛 20 强**成绩。本仓库承担其中的数据、模型实验与服务部分；比赛成绩属于项目整体，不是本仓库单独获奖，也不替代模型效果评测。

GradLoop RAGSDK Omni is an evidence-constrained multimodal learning and
interview-training agent. This repository is its independent, public-safe
training foundation: it contains source, configuration, tests, and
documentation only, never a real dataset or model artifact.

## 项目导航

本仓库负责数据、模型实验、评测与推理服务；[应用仓库](https://github.com/Iroha-P/gradloop-ragsdk-omni)负责产品 UI、API 与学习闭环。维护说明见 [项目指南](PROJECT_GUIDE.md)，协作规则见 [AGENTS](AGENTS.md)。

公开仓库包含源码、测试、配置和文档，不发布真实数据、模型权重或训练产物。

## 功能亮点

- **可追溯数据管线**：合成文本与文档/图像/音频 sidecar，记录来源许可、recipe 与 provenance，避免来源不明的数据进入实验。
- **泄漏与发布门禁**：来源关联划分、隐私及近似复制检查，真实数据和模型产物与公开源码分离。
- **冻结评测协议**：预声明指标与评测集，保留版本、哈希和聚合结果，不在冻结 test 上继续调参。
- **Ascend Base / LoRA 实验**：环境预检、四路 Base smoke、单步 SWIFT LoRA 可行性、adapter 重载与对照决策。
- **窄推理服务边界**：独立 Base 服务向应用提供公开/合成文本、图像和 WAV 音频推理，不把训练流程混入业务 API。
- **如实保留失败与限制**：区分“能运行”和“质量改善”；现有 Base/LoRA 对照选择 `retain_base`，不包装成微调效果提升。

## 与应用仓库怎样协作

```mermaid
flowchart LR
    S[公开或合成来源] --> D[数据管线与 provenance]
    D --> P[来源关联划分与冻结协议]
    P --> B[Ascend Base / LoRA 实验]
    B --> E[聚合评测与发布门禁]
    E --> M[独立 Base 推理服务]
    M --> A[GradLoop 应用 API / UI]
```

| 工程 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| [gradloop-ragsdk-omni](https://github.com/Iroha-P/gradloop-ragsdk-omni) | RAG、Agent、学习闭环、网页、PDF/Word 文档模式 | 不在公开网页训练或持久化上传文件 |
| 本仓库 | 数据、实验、模型评测、Base 推理服务 | 不重复实现产品 UI，不公开真实数据与权重 |

## 已有证据与限制

| 验证项 | 历史结果 | 含义 |
| --- | --- | --- |
| Ascend 910C / MiniCPM-o 4.5 Base smoke | 10/10 合成案例通过 | 有限条件下文本、图像、音频、图音路径可运行 |
| 单步 SWIFT LoRA 与重载 smoke | adapter 保存与重载，10/10 案例通过 | 证明可行性，不证明质量提升 |
| 5 案例冻结合成对照 | Base / LoRA composite 均为 `0.1333` | 决策为 `retain_base`，保留基线 |
| 模型服务 | 公开安全接口与自动测试 | 当前服务版本尚未完成新一轮云端 E2E，不能宣称已部署 |

上述历史证据以 [2026-07-29 报告](docs/stage-b-npu-results-2026-07-29.md)为准；不是本次文档修改重跑的实验。

## 目录导航

```text
src/gradloop_data/    来源、合成数据、provenance、划分与发布检查
src/gradloop_eval/    冻结案例 schema 与指标
src/gradloop_npu/     Ascend 预检、Base/LoRA、重载与服务边界
scripts/             数据构建、实验运行器与提交安全门禁
configs/             可版本化的空配置和公开示例
docs/                Stage A/B 协议、运行手册与实测聚合报告
tests/               数据、评测、NPU 流程和服务契约回归
```

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

## 源码许可

本仓库尚未声明源码许可证；公开可见不等于授予再分发权利。第三方模型、数据和工具仍须遵守各自的许可证。没有把应用仓库的 Apache-2.0 自动套用到本仓库。

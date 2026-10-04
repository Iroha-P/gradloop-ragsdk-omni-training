# Stage B NPU Base 运行手册

## 当前状态

本手册已覆盖准备、环境探针、Base 四路线 smoke、冻结评测、LoRA 单步
可行性门、适配器重载 smoke、Base/LoRA 对比和发布扫描。2026-07-29，
固定 revision 的 MiniCPM-o 4.5 已在云端完成上述闭环；Base 与 LoRA
均通过 10/10 多模态 smoke，但冻结合成集复合分数均为 0.1333，因此
晋级结论为 `retain_base`。这证明训练与适配器重载链路可行，不证明
单步 LoRA 带来质量提升。

2026-07-29 已在 HiDevLab Atlas 800T A3 环境完成最小 TorchNPU 验证：
Python 3.11.15、PyTorch 2.10.0、torch-npu 2.10.0，NPU 可见且矩阵乘法
成功。该结果只证明基础张量运行时可用，不证明 MiniCPM-o 模型图、音频、
视觉或训练框架已经兼容。

同日已在独立的 `--system-site-packages` 虚拟环境完成依赖层与模型验证：
torchvision 0.25.0、Transformers 4.51.3、Accelerate 1.14.0、
torchaudio 2.10.0、timm 1.0.28、SoundFile 0.14.0、
minicpmo-utils 1.0.6。Linux ARM64 没有上游 decord 0.6.0 wheel，
因此使用提供同名 `decord` 导入接口的 decord2 3.4.0；探针会先查
`decord`，缺失时再查 `decord2`，不会跳过视频依赖检查。

LoRA 使用 ms-swift 4.3.1。当前 Transformers 4.51.3 组合下，
`enable_npu_model_patch` 必须为 `false`：ms-swift 的通用 NPU 补丁会
额外导入与本模型无关、但当前 Transformers 不支持的 Qwen3-VL-MoE。
这里保留 TorchNPU 原生执行，不修改 MiniCPM-o 模型代码。

## 1. 无负载环境探针

在仓库根目录运行：

```text
python scripts/npu/probe_environment.py
```

该命令只导入运行库并查询 NPU 可见性，不加载模型、不读取数据、不输出
主机名、用户名、环境变量、工作目录或设备序列号。返回 `ready` 才能进入
下一步；返回 `blocked` 时根据稳定阻断码处理。

HiDevLab 当前环境的可复现安装顺序如下。先锁定 Torch/TorchNPU 兼容栈，
再单独安装 MiniCPM-o 工具包，避免依赖解析器替换系统 Torch：

```text
python -m venv --system-site-packages .venv-torch210
. .venv-torch210/bin/activate
python -m pip install --index-url https://pypi.org/simple \
  torch==2.10.0 torchvision==0.25.0 transformers==4.51.3 \
  accelerate timm soundfile modelscope sentencepiece decord2 \
  "pillow<=10.4" librosa==0.9.0
python -m pip install --index-url https://pypi.org/simple --no-deps \
  minicpmo-utils==1.0.6 torchaudio==2.10.0
python -m pip install --index-url https://pypi.org/simple moviepy==2.1.2
```

如果任何安装步骤试图升级或替换 Torch，应立即停止并重新建立隔离环境。
本次实测中，PyPI 连接不稳定时使用了华为云 PyPI 镜像；所有模型权重仍由
HiDevLab 环境直接从 ModelScope 获取，未经过个人电脑。

## 2. 生成十条合成 smoke 输入

```text
python scripts/npu/build_smoke_assets.py --output dist/npu-smoke-inputs
```

输出包含：

- 一张确定性合成 PNG；
- 一段 16 kHz 单声道确定性 WAV；
- 十条覆盖纯文本、视觉、音频、图音混合四条路线的案例；
- 只含相对文件名和 SHA-256 的 manifest。

目标目录必须不存在，命令不会覆盖已有内容。该数据不来自用户材料或
网络内容。

## 3. 模型缓存与发布包 preflight

模型必须由云端直接获得并固定精确 revision。模型缓存完成后运行：

```text
python scripts/npu/prepare_baseline.py --bundle "$PUBLIC_BUNDLE" --model-cache "$MODEL_CACHE" --model-revision "$MODEL_REVISION" --output "$PREFLIGHT_REPORT"
```

preflight 只检查：

- revision 是否为精确提交；
- 模型配置、索引和权重分片是否存在；
- Stage A 包是否通过发布、安全、来源隔离和拒答支持门；
- 八个 Stage A 工件的 SHA-256。

它不会加载权重，不会联网，不会在报告中写入原始路径。

## 4. Base 四路线 smoke

当前资源页标识为昇腾 910C、TorchNPU 环境；`npu-smi` 对运行时设备只返回
通用 `Ascend910` 名称，因此报告同时保留“资源页型号”和“运行时通用名”，
不据此推断额外物理卡。

允许开始 Base 的必要条件：

1. 当前云端环境空闲；
2. 环境探针为 `ready`；
3. TorchNPU、Transformers 和 MiniCPM-o 依赖版本满足门禁；
4. 模型直接缓存在云端并固定 revision；
5. preflight 为 `ready`；
6. 先完成 Base smoke，再评审是否占用更多 NPU 进行 LoRA。

满足上述门禁后：

```text
python scripts/npu/run_base_smoke.py \
  --smoke-root "$SMOKE_ROOT" \
  --model-cache "$MODEL_CACHE" \
  --model-revision "$MODEL_REVISION" \
  --code-commit "$CODE_COMMIT" \
  --config configs/stage-b/minicpmo-baseline.yaml \
  --output "$BASE_SMOKE_REPORT"
```

脚本按纯文本、视觉、音频、图音混合顺序分别加载模块，显式关闭 TTS 和
全双工模式。报告只保存输出 SHA-256、字符数、首响应片段时延、端到端时延、
峰值 NPU 内存和状态；不保存提示词、原始回答或路径。

## 5. 冻结 Base 评测

Base smoke 全部通过后：

```text
python scripts/npu/run_frozen_eval.py \
  --bundle "$PUBLIC_BUNDLE" \
  --model-cache "$MODEL_CACHE" \
  --model-revision "$MODEL_REVISION" \
  --code-commit "$CODE_COMMIT" \
  --config configs/stage-b/minicpmo-baseline.yaml \
  --output "$BASE_EVAL_REPORT"
```

冻结评测只使用 Stage A 五条确定性合成案例，计算引用精度、拒答准确率、
结构有效率和时延。`recall_at_k` 对应冻结输入证据，不属于模型自行检索
成绩，报告中必须保留这一说明。

## 6. LoRA 单步可行性训练

先从 smoke 包生成十条 SWIFT 标准格式的确定性合成训练样本：

```text
python scripts/npu/build_lora_assets.py \
  --smoke-root "$SMOKE_ROOT" \
  --output "$LORA_DATASET"
```

随后执行单步训练：

```text
python scripts/npu/run_lora_feasibility.py \
  --swift-executable "$SWIFT_EXECUTABLE" \
  --config configs/stage-b/minicpmo-lora-feasibility.yaml \
  --model-cache "$MODEL_CACHE" \
  --model-revision "$MODEL_REVISION" \
  --dataset-root "$LORA_DATASET" \
  --output-root "$LORA_OUTPUT" \
  --log "$LORA_PRIVATE_LOG" \
  --report "$LORA_REPORT" \
  --adapter-manifest "$ADAPTER_MANIFEST" \
  --code-commit "$CODE_COMMIT"
```

只有 `trainer_state.json` 证明至少一次优化更新、LoRA 适配器与配置均存在，
且输出目录不含完整模型权重时，脚本才返回 `trained`。完整训练日志仅留在
云端忽略目录，不进入公开报告或 Git。

## 7. 适配器重载、冻结复评和晋级门

使用 `run_base_smoke.py` 和 `run_frozen_eval.py` 的 `--adapter`、
`--adapter-manifest` 参数重载适配器，重新执行十条多模态 smoke 与同一
五条冻结评测。随后：

```text
python scripts/npu/compare_base_lora.py \
  --base-report "$BASE_EVAL_REPORT" \
  --lora-report "$LORA_EVAL_REPORT" \
  --output "$COMPARISON_REPORT"
```

复合分数是引用精度、拒答准确率和结构有效率的等权平均。LoRA 只有在同一
模型 revision、同一冻结案例下绝对提升至少 0.05 时才允许晋级；否则保留
Base。LoRA 可行性成功不等同于效果提升。

## 8. 断开环境前验收

只有以下证据全部存在并通过，才可断开 HiDevLab：

1. 环境探针和离线 preflight 为 `ready`；
2. Base 四路线 smoke 全部通过；
3. Base 冻结评测完成；
4. LoRA 至少一次更新、适配器保存、重载和十条 smoke 全部通过；
5. LoRA 冻结复评和晋级决策完成；
6. 发布扫描确认报告不含原始提示词、回答、私有路径、凭证或个人信息；
7. 源码提交不含数据、模型、检查点、日志或实验输出。

2026-07-29 的十份脱敏聚合证据已下载到本地忽略目录
`dist/stage-b-npu-1e86eaf/reports`。压缩包 SHA-256 为
`d84deb187eaf66289e5698ff62876bfa5270e6ef769ac1eb0ceef47ae644b639`；
包内不含模型、适配器、数据集、日志、原始提示词、原始回答或路径。
详细指标见 `docs/stage-b-npu-results-2026-07-29.md`。

## 9. 明确禁止

- 不在个人电脑下载模型；
- 不上传私有语料、本地索引、本地 Qwen 或个人资料；
- 不把模型、数据、检查点、报告提交进源码仓库；
- 不在其他项目运行时安装依赖、加载模型或占用 NPU；
- 不在 Base 未通过时启动 LoRA；
- 不把基础张量验证描述为 MiniCPM-o 已适配或训练已完成。

## 10. 官方参考

- MiniCPM-o 4.5 官方模型卡：
  https://huggingface.co/openbmb/MiniCPM-o-4_5
- MiniCPM-o 官方仓库：
  https://github.com/OpenBMB/MiniCPM-o
- ms-swift 官方支持矩阵：
  https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md
- MiniCPM 与昇腾挑战赛：
  https://ascend.openbmb.cn/

# Stage B Ascend NPU 实测结果（2026-07-29）

## 结论

MiniCPM-o 4.5 Base 已在 HiDevLab Ascend 910C / TorchNPU 环境完成
文本、图像、音频和图音混合四路线验证。单步 LoRA 完成一次优化更新，
保存了纯适配器工件，并成功重载完成相同 smoke 与冻结复评。

这次实验验证的是 Ascend 上的全模态推理、LoRA 训练、保存、重载和
评测闭环。LoRA 在五条冻结合成案例上没有提升预声明的复合分数，因此
保留 Base，不把“训练成功”表述为“效果提升”。

## 运行身份

- 云环境：HiDevLab，资源页标识 Ascend 910C；
- Python：3.11.15；
- PyTorch / torch-npu：2.10.0 / 2.10.0；
- Transformers：4.51.3；
- ms-swift：4.3.1；
- 模型：`openbmb/MiniCPM-o-4_5`；
- 固定 revision：`438cf37032d6a94b57d7d7a6cb6eda040c4cc922`；
- 实验代码基线：`1e86eaf`；
- 数据：10 条确定性合成 LoRA/smoke 案例和 5 条冻结合成评测案例；
- 模型权重由云端直接缓存，个人电脑未保存模型权重。

## Base 四路线 smoke

| 指标 | 结果 |
| --- | ---: |
| 通过案例 | 10 / 10 |
| 路线分布 | text 2、vision 3、audio 3、omni 2 |
| 成功率 | 1.0000 |
| 平均首块时延 | 668.8 ms |
| 平均端到端时延 | 2826.0 ms |
| 峰值 NPU 内存 | 31.16 GiB |

报告只保留输出哈希、字符数、时延、峰值内存和状态，不保留提示词或回答。

## Base 冻结评测

| 指标 | 结果 |
| --- | ---: |
| 案例数 | 5 |
| citation precision | 0.0000 |
| refusal accuracy | 0.4000 |
| structure valid rate | 0.0000 |
| recall@k | 1.0000 |
| 平均时延 | 3264.8 ms |
| 复合分数 | 0.1333 |

`recall@k` 只衡量冻结案例中预置证据是否可用，不代表系统已完成独立检索，
也不应写成 RAG 检索质量。

## LoRA 可行性训练与重载

- 框架：ms-swift；
- 调参方式：LoRA；
- 合成样本：10；
- 优化更新：1；
- 保存内容：`adapter_model.safetensors` 与 `adapter_config.json`；
- 完整模型权重：未写入输出目录；
- Hub push：关闭；
- 适配器重载 smoke：10 / 10 通过；
- 重载后平均首块时延：516.1 ms；
- 重载后平均端到端时延：4024.4 ms；
- 重载后峰值 NPU 内存：31.32 GiB。

ms-swift 4.3.1 与 Transformers 4.51.3 组合使用 TorchNPU 原生执行。
通用 `enable_npu_model_patch` 会导入与 MiniCPM-o 无关且当前版本不支持的
Qwen3-VL-MoE，因此配置显式关闭该通用补丁；模型源码和固定 revision
均未修改。

## LoRA 冻结复评与晋级门

| 指标 | Base | LoRA |
| --- | ---: | ---: |
| citation precision | 0.0000 | 0.0000 |
| refusal accuracy | 0.4000 | 0.4000 |
| structure valid rate | 0.0000 | 0.0000 |
| 复合分数 | 0.1333 | 0.1333 |
| 平均时延 | 3264.8 ms | 5897.8 ms |

- 绝对增益：0.0000；
- 晋级阈值：至少 +0.0500；
- 决策：`retain_base`。

五条合成案例和一次优化更新只能验证工程闭环，不能用于证明泛化、业务效果
或生产可用性。下一阶段若要评估效果，必须扩大公开可授权训练数据和独立
冻结集，并继续保持来源隔离与人工审查。

## 隐私与发布证据

最终扫描覆盖 9 份聚合报告，结果为：

- `status = passed`；
- `scanned_report_count = 9`；
- `finding_count = 0`；
- 不保留命中内容和本地路径。

本地证据目录为忽略路径 `dist/stage-b-npu-1e86eaf`。证据压缩包：

- 文件：`gradloop-stage-b-safe-evidence-1e86eaf.zip`；
- 大小：9006 字节；
- SHA-256：
  `d84deb187eaf66289e5698ff62876bfa5270e6ef769ac1eb0ceef47ae644b639`；
- 内容：10 份聚合 JSON 报告和 1 份哈希清单；
- 不包含：模型、LoRA 适配器、数据集、日志、原始提示词、原始输出、
  私有路径、凭证或个人信息。

## 可复现边界

仓库提交公开的是代码、配置、测试和本结果说明。模型权重、训练数据生成物、
适配器、运行日志和证据包都留在忽略目录或云端环境，不进入 Git。复现实验
需要重新在受控 NPU 环境中由官方模型源获取相同 revision，并重新生成
确定性合成输入。

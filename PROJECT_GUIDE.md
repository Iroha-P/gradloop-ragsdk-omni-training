# GradLoop Training 项目指南

本工程是 GradLoop 的独立模型实验与服务工程，负责数据准备、来源许可与 provenance、数据划分、冻结评测、NPU Base/LoRA 运行和应用模型服务。

## 与应用工程的分工

| 仓库 | 负责什么 |
| --- | --- |
| [Training](https://github.com/Iroha-P/gradloop-ragsdk-omni-training) | 数据、训练、模型评测、adapter 重载与推理服务 |
| [应用工程](https://github.com/Iroha-P/gradloop-ragsdk-omni) | RAG/Agent、学习闭环、API、UI 与公开演示 |


## 目录和阶段

- `src/gradloop_data`：公开合成数据、来源许可、provenance、划分与发布门禁。
- `src/gradloop_eval`：冻结案例 schema 和评测指标。
- `src/gradloop_npu`：预检、Base/LoRA、重载、比较与模型服务边界。
- `scripts`：数据构建、实验操作与安全检查入口。
- `tests`：数据、评测、服务、NPU 流程及仓库政策测试。

Stage A 验证合成数据与评测流程。Stage B 的 [历史结果](./docs/stage-b-npu-results-2026-07-29.md)证明有限条件下的可运行性，不宣称 LoRA 改善质量。使用云端或 NPU 资源前阅读 [操作手册](./docs/stage-b-npu-runbook.md)，并取得相应任务授权。

## 开发、发布与归档

模型实验和服务在本仓库开发，应用产品功能在应用仓库开发。两者通过明确的服务协议连接，各自保留独立的评测和版本证据。

本公开版本只包含选定源码、测试、配置和文档。真实数据、权重、checkpoint、adapter、数据库、凭据及训练输出保留在批准的本地或计算环境，不上传。

完成的实验协议和结果保持冻结，新实验使用新版本。清理前保留唯一实验材料与运行证据，只处理经确认可重建的缓存。发布前遵循 [SECURITY](./SECURITY.md)，对源码和独立发布候选分别扫描。

## 许可

本项目尚未声明源码开源许可证；公开可见不等于获得再分发授权。第三方模型和数据仍须遵循各自的许可证和使用条款。

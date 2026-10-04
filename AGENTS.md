# GradLoop Training 公开仓库协作指导

- 项目用途与应用仓库的分工见 [PROJECT_GUIDE](./PROJECT_GUIDE.md)。
- 功能修改前阅读 README、[SECURITY](./SECURITY.md)和对应阶段手册。
- 只提交小型 UTF-8 源码、测试、文档和空配置示例。
- 不导入私人代码、语料、学习记录、真实配置、凭据、数据库、模型和训练产物。
- 模型加载、训练、云端或 NPU 任务须有具体授权；不得因文档存在而启动。
- 保留冻结实验与未提交工作。新实验使用新版本，分别报告可运行性和质量。
- 每次提交前执行 `python scripts/security/precommit_scan.py --staged`。

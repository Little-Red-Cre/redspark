# MiniCPM5-2B 最终后训练版本（上游参考）

![RedSpark 赤霄初耀](../../../../assets/redspark-logo.png)

## 目录定位

本目录保留上游最终后训练模型 `openbmb/MiniCPM5-2B` 的原始权重，作为 RedSpark 的**参考基线和回退路径**，不参与训练。

RedSpark 当前阶段的训练起点是 `openbmb/MiniCPM5-2B-Base`，位于 `model/redspark/model-weights/base/`。当 Base 权重尚未就绪时，`model/redspark/loading.py` 会自动回退到本目录，以保证推理与训练链路可用。

## 当前状态

- 模型类型：2.5B 参数量级的 causal language model
- 上游基模：`openbmb/MiniCPM5-2B`（最终后训练版本）
- 上下文长度：131,072 tokens
- 运行框架：Hugging Face Transformers `>=5.6,<6`
- 推理设备：支持 CPU，也支持 NVIDIA CUDA
- 用途：参考基线与权重回退，不作为 RedSpark 的训练起点

## RedSpark 工程入口

从仓库根目录运行：

```powershell
py -3 inference/chat.py "解释什么是 RAG"
py -3 inference/benchmark.py --device cuda --max-new-tokens 64
py -3 inference/benchmark.py --device cpu --max-new-tokens 64
py -3 data/validate.py data/processed/sft.jsonl
py -3 training/sft.py --data data/processed/sft.jsonl --output artifacts/sft
```

完整的工程架构、依赖安装和数据格式见仓库根目录 [README.md](../../../../README.md)。

## 能力边界

本目录权重用于工程验证和领域微调的参考对照，不代表 RedSpark 已经完成大规模训练或生产级评测。模型输出需要经过事实性、代码正确性、安全性和许可证合规检查；推理速度会随硬件、精度、上下文长度和生成参数变化。

## 上游归属与许可证

本目录权重来自 openbmb/MiniCPM5-2B。模型配置、Tokenizer、权重及本目录中的许可证信息保留上游归属；RedSpark 自有模型源码位于 `model/redspark/`，训练、推理和数据飞轮代码位于仓库的 `training/`、`inference/` 和 `data/` 目录。

MiniCPM5-2B 模型文件使用 Apache-2.0，详见本目录的 [LICENSE](LICENSE)。项目整体的归属边界见仓库根目录 [NOTICE](../../../../NOTICE)。

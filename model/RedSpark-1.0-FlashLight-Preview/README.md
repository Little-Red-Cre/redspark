# RedSpark-1.0-FlashLight-Preview

![RedSpark 赤霄初耀](../../assets/redspark-logo.png)

## 模型定位

`RedSpark-1.0-FlashLight-Preview` 是 RedSpark 项目的第一版预览模型，面向开源软件知识和代码问答场景。

本模型基于上游指令模型 `Qwen/Qwen3-1.7B`，并不是从零训练的新基础模型。当前目录中的权重是 Qwen3-1.7B 初始权重，后续通过 RedSpark 的领域数据飞轮和 SFT/LoRA 训练形成项目版本。

## 当前状态

- 模型类型：1.7B 参数量级的 causal language model
- 上游基模：`Qwen/Qwen3-1.7B`
- 上下文长度：32,768 tokens
- 运行框架：Hugging Face Transformers `>=4.51.0`
- 推理设备：支持 CPU，也支持 NVIDIA CUDA
- 产品阶段：Preview，当前重点是验证数据、训练和推理闭环

## RedSpark 工程入口

从仓库根目录运行：

```powershell
py -3 inference/chat.py "解释什么是 RAG" --no-thinking
py -3 inference/benchmark.py --max-new-tokens 64
py -3 data/validate.py data/processed/sft.jsonl
py -3 training/sft.py --data data/processed/sft.jsonl --output artifacts/sft
```

默认模型路径为当前目录。完整的工程架构、依赖安装和数据格式见仓库根目录 [README.md](../../README.md)。

## 能力边界

本版本用于工程验证和领域微调，不代表 RedSpark 已经完成大规模训练或生产级评测。模型输出需要经过事实性、代码正确性、安全性和许可证合规检查；推理速度会随硬件、精度、上下文长度和生成参数变化。

## 上游归属与许可证

本模型基于 Qwen/Qwen3-1.7B。模型配置、Tokenizer、权重及本目录中的许可证信息保留上游归属；RedSpark 自有训练、推理和数据飞轮代码位于仓库的 `training/`、`inference/` 和 `data/` 目录。

Qwen3-1.7B 模型文件使用 Apache-2.0，详见本目录的 [LICENSE](LICENSE)。项目整体的归属边界见仓库根目录 [NOTICE](../../NOTICE)。

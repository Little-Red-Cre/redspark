<p align="center">
  <img src="assets/redspark-logo.png" alt="RedSpark 赤霄初耀" width="260">
</p>

# RedSpark 赤霄初耀

RedSpark 是面向开源软件知识与代码问答的 LLM 工程，当前模型产品名为 `RedSpark-1.0-FlashLight-Preview`。项目以 `openbmb/MiniCPM5-2B` 最终后训练版本为指令模型起点，通过数据飞轮持续沉淀高质量领域数据，再进行 SFT/LoRA 训练和本地推理验证。

## 工程架构

```text
数据采集 → 清洗/去重/质检 → SFT 数据集
                              ↓
              RedSpark-1.0-FlashLight-Preview
                              ↓
                    SFT/LoRA 训练产物
                              ↓
                 推理评测 → 错误样本回流数据集
```

仓库按职责拆成四个核心域，所有代码均直接属于 RedSpark，不包含嵌套 Git 子仓库：

```text
redspark/
├── model/                 模型快照、Tokenizer、配置和模型资产说明
│   ├── README.md          RedSpark 模型产品说明
│   └── RedSpark-1.0-FlashLight-Preview/
├── training/              SFT/LoRA 训练入口和训练超参数
│   ├── sft.py
│   └── sft_config.yaml
├── inference/             本地推理与性能基准
│   ├── chat.py
│   └── benchmark.py
├── data/                  数据飞轮：输入、处理、契约和质量报告
│   ├── raw/
│   ├── processed/
│   ├── schemas/
│   └── validate.py
├── requirements.txt       通用 Python 依赖
├── requirements-cuda128.txt  CUDA 12.8 PyTorch 依赖
└── NOTICE                  上游模型与项目代码的归属边界
```

## 推理

Python 3.10–3.14 均在 PyTorch Windows 支持范围内。GPU 环境先安装 CUDA 依赖，再安装通用依赖：

```powershell
py -3 -m pip install -r requirements-cuda128.txt
py -3 inference/chat.py "解释什么是 RAG" --no-thinking
```

默认模型目录为 `model/RedSpark-1.0-FlashLight-Preview`。如果只拉取了代码和配置，可以使用以下命令补齐模型文件：

```powershell
hf download openbmb/MiniCPM5-2B --local-dir model/RedSpark-1.0-FlashLight-Preview
```

性能基准：

```powershell
py -3 inference/benchmark.py --device cuda --max-new-tokens 64
py -3 inference/benchmark.py --device cpu --max-new-tokens 64
```

基准脚本会用固定提示、预热和重复生成报告 CPU/GPU 的生成速度、加载时间和内存。GPU 使用 BF16，CPU 使用 FP32；结果只代表当次设备、精度和生成长度。

## 训练

SFT 数据使用 JSONL，每行一个 `messages` 数组，格式约束见 `data/schemas/sft_record.schema.json`。原始数据放入 `data/raw/`，清洗、去重、切分后的数据放入 `data/processed/`。

校验数据：

```powershell
py -3 data/validate.py data/processed/sft.jsonl
```

启动 SFT：

```powershell
py -3 training/sft.py --data data/processed/sft.jsonl --output artifacts/sft
```

## 技术边界

- MiniCPM5-2B 是标准 `LlamaForCausalLM`，由 Hugging Face Transformers 直接加载，要求 `transformers>=5.6,<6`。
- `RedSpark-1.0-FlashLight-Preview` 基于已经完成预训练和后训练的 `openbmb/MiniCPM5-2B` 最终版本，本项目当前定位是领域 SFT，不是从零预训练。
- MiniCPM5-2B 仅支持 Think 模式。为兼容旧命令，`--no-thinking` 仍可传入，但会被忽略并给出提示。
- `artifacts/`、数据处理产物和 safetensors 权重默认不进入 Git；模型权重可通过 Hugging Face 重新下载。

## 许可证与归属

`RedSpark-1.0-FlashLight-Preview` 使用的 MiniCPM5-2B 模型文件使用 Apache-2.0。RedSpark 自有训练、推理和数据飞轮代码的归属边界，以及模型上游文件的保留说明见 `NOTICE`。

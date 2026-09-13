<p align="center">
  <img src="assets/redspark-logo.png" alt="RedSpark 赤霄初耀" width="260">
</p>

# RedSpark 赤霄初耀

RedSpark 是面向开源软件知识与代码问答的 LLM 工程。当前阶段以 `openbmb/MiniCPM5-2B-Base` 为权重起点，在保持 LLaMA 兼容基线可复现的前提下，逐步演进 RedSpark 自有 Transformer 实现，并通过官方开源代码数据和 RedSpark 自有数据增强 Coding 能力。

## 工程架构

```text
数据采集 → 清洗/去重/质检 → SFT 数据集
                              ↓
          RedSpark Transformer source + Base weights
                              ↓
           继续预训练 → 全参数 SFT → 可选 LoRA 实验
                              ↓
                 推理评测 → 错误样本回流数据集
```

仓库按职责拆成四个核心域，所有代码均直接属于 RedSpark，不包含嵌套 Git 子仓库：

```text
redspark/
├── model/
│   ├── README.md          模型资产说明
│   └── redspark/          RedSpark 模型源码
│       └── model-weights/ Base、参考和训练产物权重
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
├── tools/                 权重管理与回归验证脚本
│   ├── model_weights.py   权重状态 / 拉取 / 卸载（含 Git LFS 副本检查）
│   └── verify_refactor.py
├── third_party/           上游参考源码（只读，Apache-2.0）
│   └── MiniCPM/           OpenBMB/MiniCPM @ e944abd
├── requirements.txt       通用 Python 依赖
├── requirements-cuda128.txt  CUDA 12.8 PyTorch 依赖
└── NOTICE                  上游模型与项目代码的归属边界
```

## 推理

Python 3.10–3.14 均在 PyTorch Windows 支持范围内。GPU 环境先安装 CUDA 依赖，再安装通用依赖：

```powershell
py -3 -m pip install -r requirements-cuda128.txt
py -3 inference/chat.py "解释什么是 RAG"
py -3 inference/chat.py "解释什么是 RAG" --no-thinking
```

默认开启 Think 模式，prompt 以 `<think>\n` 结尾，模型先输出推理再输出答案。`--no-thinking` 改为让模板直接开合一个空的推理块（`<think>\n\n</think>\n\n`），prompt 不再要求模型先续写推理。

默认模型目录为 `model/redspark/model-weights/base`。仓库默认不带权重：权重文件约 5 GB，已被 `.gitignore` 排除，不进入版本库。补齐和卸载都走同一个入口：

```powershell
py -3 tools/model_weights.py status                       # 权重目录、体积与磁盘占用
py -3 tools/model_weights.py download --target base       # 拉取 Base 并校验 sha256
py -3 tools/model_weights.py download --target all        # base + final-reference
py -3 tools/model_weights.py clean --target all --dry-run # 预览卸载
```

`download` 拉取后按固定 sha256 校验，不匹配即以非零退出码结束。`clean` 只删除 `*.safetensors` 等权重文件，保留 `config.json`、`tokenizer.json`、`chat_template.jinja` 等溯源文件。

**卸载时注意**：Git LFS 会在 `.git/lfs/objects/` 保留一份副本，只删工作区文件不会释放空间。`clean` 会报告这些副本，加 `--lfs-orphans` 才会删除其中无引用的部分。

也可以直接使用 Hugging Face CLI 拉取同一份权重：

```powershell
hf download openbmb/MiniCPM5-2B-Base --local-dir model/redspark/model-weights/base
```

当前工作区的 Base 权重来自 ModelScope 的 `OpenBMB/MiniCPM5-2B-Base`，两种来源任选其一即可。

性能基准：

```powershell
py -3 inference/benchmark.py --device cuda --max-new-tokens 64
py -3 inference/benchmark.py --device cpu --max-new-tokens 64
```

基准脚本会用固定提示、预热和重复生成报告 CPU/GPU 的生成速度、加载时间和内存。GPU 使用 BF16，CPU 使用 FP32；结果只代表当次设备、精度和生成长度。

## 训练

SFT 数据使用 JSONL，每行一个 `messages` 数组，契约声明见 `data/schemas/sft_record.schema.json`（供编辑器与 CI 使用，仓库内没有代码执行它，因为 jsonschema 不是本项目的依赖），可执行的门禁是 `data/validate.py`，它是该 schema 的严格超集。原始数据放入 `data/raw/`，清洗、去重、切分后的数据放入 `data/processed/`。

assistant 轮采用 thinking 格式：`content` 自带 `<think>...</think>` 推理块，后接答案。推理内容直接写在 `content` 内，不额外引入字段。

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>\n推理过程\n</think>\n\n答案"}]}
```

训练与推理共用 `model/redspark/chat_template.jinja` 这一份模板，模板会把 `<think>` 块切分出来再重新拼接，因此数据必须满足：

- 每个 assistant 轮有且只有一对 `<think>` / `</think>`，`<think>` 在最前，`</think>` 之后有非空答案；
- 内容中不得出现字面 `</think>`。模板按**第一个和最后一个** `</think>` 切分，落在两者之间的内容会被静默丢弃。

校验数据（`valid_records` 为通过条数，违反上述规则的记录以非零退出码报错）：

```powershell
py -3 data/validate.py data/processed/smoke_sft.jsonl
```

两类不阻断的 warning 也值得看：推理块为空（等于教模型「推理永远为空」，正是 thinking 格式要避免的），以及内容里出现 `<|im_start|>` / `<|im_end|>`（它们各编码成单个控制 token，会注入回合边界或截断生成）。

启动 SFT：

```powershell
py -3 training/sft.py --data data/processed/smoke_sft.jsonl --output artifacts/sft
```

## 回归验证

`tools/verify_refactor.py` 是重构门禁，检查参数字典键、属性键、建层次数、模板渲染与 assistant 掩码跨度，任一检查失败即以非零退出码结束。

```powershell
py -3 tools/verify_refactor.py --data data/processed/smoke_sft.jsonl
```

`--data` 会把真实 SFT 数据逐条渲染一遍，先套用 `data/validate.py` 的数据契约，再断言每条记录的推理与答案都完整落在 assistant 掩码跨度内。改动模型类或模板后，可以留一份基线再比对：

```powershell
py -3 tools/verify_refactor.py --out before.json
py -3 tools/verify_refactor.py --out after.json
py -3 tools/verify_refactor.py --compare before.json after.json
```

## 技术边界

- MiniCPM5-2B 是标准 `LlamaForCausalLM`。`model/redspark/` 通过 RedSpark 自有类保持同形状基线，并提供后续结构改造边界。
- 训练起点是 `openbmb/MiniCPM5-2B-Base`，当前阶段包含继续预训练、全参数 SFT 和 Coding 能力评测；LoRA 只用于低显存实验或冒烟验证。
- 单张 8 GB GPU 可运行 BF16 推理和 LoRA 冒烟训练，但不能承载该 2.5B 模型的 AdamW 全参数训练；全参训练需要 ZeRO/FSDP 或 CPU/NVMe offload。
- `artifacts/`、数据处理产物和 safetensors 权重默认不进入 Git；模型权重用 `tools/model_weights.py download` 重新拉取（带固定 sha256 校验），用 `clean` 卸载。

## 许可证与归属

RedSpark 使用的 MiniCPM5-2B 模型文件使用 Apache-2.0。RedSpark 自有模型源码、训练、推理和数据飞轮代码的归属边界，以及模型上游文件的保留说明见 `NOTICE`。

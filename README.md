# Task23 API 实验代码

本目录是 `Task23/API_Exp` 的**纯代码整理版**，用于保存和共享量表测量、下游任务生成及二者相关分析的正式实现。整理日期为 2026-07-16。

本目录不包含题库、模型清单、API 密钥、原始回答、计分结果、图片、Slurm 日志或历史归档。代码运行时需要把下文列出的外部输入放到相同的相对路径中。

## 目录结构

```text
Code_Only/
├── README.md
├── requirements.txt
├── Submit.sh
├── run_no_memory_batch.py
├── Tools/
│   ├── README.md
│   ├── diagnose_scale_responses.py
│   ├── example_single_turn_test.py
│   └── read_results_workbook.py
└── 下游任务/
    ├── README.md
    ├── Submit_Downstream.sh
    ├── Run_Part2_Analysis.sh
    └── scripts/
        ├── build_downstream_tasks.py
        ├── run_downstream.py
        ├── parse_downstream.py
        ├── aggregate_downstream.py
        ├── prepare_scale_scores.py
        └── analyze_correlations.py
```

## 两条正式流程

### 1. 量表测量

正式设计为 9 个模型、7 个量表、中英文、PromptA/PromptB 分开测试，每个条件 5 次重复。7 个量表共 157 道题，因此每个 prompt condition 的计划调用数为：

```text
9 models × 157 items × 2 languages × 5 repeats = 14,130
```

量表请求行为：

- 每题均为独立单轮请求，无上下文记忆。
- payload 传入 `model`、`messages` 和 `max_tokens=32`。
- 不传 `temperature`、`top_p`、`seed` 或 reasoning 参数。
- 只接受合法选项、明确答案标记或整条回答为 `N/A`。
- 默认最多尝试 5 次；格式错误后会在原题后追加一条**仅约束输出格式**的提醒。
- 每道题完成后写 checkpoint；中断后可以从未完成题继续。
- 正式工作簿保存解析后的答案，不保存每一次有效请求的完整原始 JSON。
- 如需审计原始响应、token usage 和 finish/status 信息，使用 `Tools/diagnose_scale_responses.py`。

主要入口：

```bash
sbatch Submit.sh
```

PromptA 和 PromptB 必须写入不同结果目录。例如：

```bash
PROMPT_VARIANT=A RESULTS_ROOT=Outputs/Results_9API_7Scale_PromptA sbatch Submit.sh
PROMPT_VARIANT=B RESULTS_ROOT=Outputs/Results_9API_7Scale_PromptB sbatch Submit.sh
```

建议每次提交一轮，通过 `REPEAT_START=1..5` 依次运行；`--skip-existing` 防止覆盖已完成工作簿。

量表侧详细说明见 `Tools/README.md` 和代码中的模块注释。

### 2. 下游任务与相关分析

下游设计为 9 个模型、24 个任务、中英文、5 次重复，共 2160 个正式请求。任务包括 18 个 MFV 情境题，以及 3 个 OUS Instrumental Harm 题和 3 个 OUS Impartial Beneficence 分配题。

下游请求行为：

- 每题均为独立单轮请求，无上下文记忆。
- payload 只传 `model` 和 `messages`，不显式传生成参数或 `max_tokens`。
- 格式错误、空回答或请求错误时，最多原样重发 5 次，不追加修复提示词。
- 只有整条回答精确为 `N/A` 才记作合法 N/A；合法 N/A 不计零分，也不重试。
- 每条原始记录保存完整提示词、提示词 SHA-256、原始回答、尝试次数、耗时和错误。

生成与解析入口：

```bash
cd 下游任务
sbatch Submit_Downstream.sh
```

聚合和相关分析入口：

```bash
cd 下游任务
./Run_Part2_Analysis.sh
```

相关分析将 PromptA、PromptB 的正式量表分数分别作为主分析输入，并额外计算等权 `PromptMean=(PromptA+PromptB)/2` 作为敏感性分析。五轮重复先在模型内聚合，相关分析单位始终是模型，不把重复轮次当作独立样本。

下游侧的逐脚本输入输出和 N/A 聚合规则见 `下游任务/README.md`。

## 外部输入

代码运行时至少需要以下文件。它们因属于题库、配置或数据而未收录在本目录中：

```text
Test_File/Scale_16_Q_F.xlsx
Test_File/_AI_Protocol_C/_AI_Protocol.xlsx
Test_File/_AI_Protocol_E/_AI_Protocol.xlsx
Test_File/API_Selected_Final9.txt
Test_File/Sheet_Selected_Final7.txt
下游任务/data/model_list.json
统计/PromptA/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
统计/PromptB/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
```

API 凭据优先从环境变量读取：

```bash
export N1N_BASE_URL="https://example.com/v1"
export N1N_API_KEY="..."
```

也兼容 `OPENAI_BASE_URL`/`OPENAI_API_KEY`、`API_BASE_URL`/`API_KEY`，或本地 `Test_File/api_config.json`：

```json
{
  "base_url": "https://example.com/v1",
  "api_key": "..."
}
```

不要把真实密钥放入代码包或版本控制。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

主要依赖为 `requests`、`pandas`、`openpyxl`、`numpy`、`scipy` 和 `matplotlib`。

## 无 API 检查

量表 dry-run：

```bash
python run_no_memory_batch.py --dry-run --prompt A --repeats 1
```

下游 dry-run：

```bash
cd 下游任务
DOWNSTREAM_DRY_RUN=1 python scripts/run_downstream.py
```

编译和 Shell 语法检查：

```bash
python -m compileall -q .
bash -n Submit.sh
bash -n 下游任务/Submit_Downstream.sh
bash -n 下游任务/Run_Part2_Analysis.sh
```

## 复现边界

- `build_downstream_tasks.py` 是下游题目和冻结提示词的代码来源，运行后生成 `下游任务/prompts/downstream_tasks.json`。
- 模型名称和量表选择由外部 allowlist 决定，不硬编码在分析脚本中。
- 量表最终计分表属于统计数据产品，不在纯代码目录中；下游分析只读取官方维度 CSV。
- `Outputs/`、`统计/` 和 `Archive/` 中的结果、图和历史脚本均不属于本代码包。
- 如需修改正式提示词，应新建独立结果批次，不能与已有 prompt hash 或工作簿混合。

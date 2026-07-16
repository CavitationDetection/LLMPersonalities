# 下游任务代码

本目录包含下游情境任务的构建、API 调用、严格解析、五轮聚合，以及与 PromptA/PromptB 量表分数的相关分析代码。这里只收录代码，任务 JSON、模型清单、原始回答和分析结果均作为外部运行文件生成或提供。

## 脚本执行顺序

```text
scripts/build_downstream_tasks.py
        ↓
scripts/run_downstream.py
        ↓
scripts/parse_downstream.py
        ↓
scripts/aggregate_downstream.py
        ↓
scripts/prepare_scale_scores.py
        ↓
scripts/analyze_correlations.py
```

`Submit_Downstream.sh` 执行前三步并输出质量摘要；`Run_Part2_Analysis.sh` 执行后三步。

## 各脚本职责

### `build_downstream_tasks.py`

- 代码内定义 18 个 MFV 情境题和 6 个 OUS 下游题。
- 为每题生成冻结的中英文 system/user prompt。
- 写入 `prompts/downstream_tasks.json`。
- 任务 JSON 是生成产物；修改题目或提示词必须修改本脚本并开启新的正式结果批次。

### `run_downstream.py`

- 从 `data/model_list.json` 读取启用模型。
- 按模型、任务、语言和 repeat 生成唯一请求键。
- payload 只包含 `model` 和 `messages`，不传 `temperature`、`top_p`、`seed` 或 `max_tokens`。
- 失败或格式无效时原样重发，最多 5 次；不会追加“修复提示词”。
- 已有记录会按唯一键跳过，除非显式设置 overwrite。
- 对已写入记录校验 `prompt_sha256`，防止不同提示词版本混合。
- 输出 `outputs/downstream_raw_responses.jsonl`。

### `parse_downstream.py`

- 再次按任务类型独立解析原始回答，不直接信任调用阶段的格式判断。
- MFV 接受 1 到 5；OUS IH 接受 1 到 7；OUS IB 只接受整数 `A=... B=... C=...` 且总和为 100。
- 只把整条精确 `N/A` 视为合法 N/A。
- 输出 `downstream_parsed_responses.csv` 和 `downstream_scores_long.csv`。

### `aggregate_downstream.py`

- 每个模型、题目、语言先在 5 次重复中对可用分数取均值。
- N/A 不计零分，也不插补；只从该均值的分母中排除，并保留 `n_na`/`na_rate`。
- 某题五轮全为 N/A 时，该题均值为空。
- 维度层面再对有分数的题目等权平均，同时保存实际计分题数。
- 输出题目聚合、维度长/宽表和 repeat 稳定性表。

### `prepare_scale_scores.py`

- 读取 PromptA 和 PromptB 的官方量表维度 CSV。
- 每个 prompt condition 先在模型、语言、量表和维度内聚合 5 轮。
- `PromptMean` 只在 A/B 两侧均有有效分数时计算，公式为 `(PromptA + PromptB) / 2`。
- 只保留下游分析需要的 MFQ-30 与 OUS-9 官方维度。

### `analyze_correlations.py`

- 以模型为统计单位计算 Spearman 相关，不把 5 次重复当作独立样本。
- MFQ-30 与 MFV 输出完整 5×5 矩阵及理论匹配关系。
- OUS 输出 Instrumental Harm 和 Impartial Beneficence 的匹配相关。
- 缺失值采用 pairwise complete cases；N/A 率只作质量指标，不作为 0 分或相关权重。
- 计算 BH-FDR、leave-one-model-out 范围、语言一致性、prompt condition 对照和图形。

## N/A 的完整数据路径

合法 N/A 在解析表中记录为：

```text
response_valid = true
is_na = true
score_available = false
score_value = missing
```

示例：同一道题五轮有 4 个数值和 1 个 N/A，则题目均值只使用 4 个数值，`na_rate=0.2`。若一个三题维度中有一题五轮全部 N/A，则维度均值使用剩余两题，并记录 `n_tasks_scored=2`。只有整个下游维度没有任何可用分数时，该模型才会在对应相关中被成对排除。

## 运行生成与解析

```bash
sbatch Submit_Downstream.sh
```

主要环境变量：

```text
DOWNSTREAM_REPEAT_START       默认 1
DOWNSTREAM_N_REPEATS          默认 5
DOWNSTREAM_LANGUAGES          默认 "zh en"
DOWNSTREAM_TIMEOUT            默认 90
DOWNSTREAM_MAX_RETRIES        默认 5
DOWNSTREAM_RETRY_BACKOFF_BASE 默认 2
DOWNSTREAM_RETRY_BACKOFF_MAX  默认 20
DOWNSTREAM_OVERWRITE          默认 0
DOWNSTREAM_DRY_RUN            默认 0
```

dry-run 不发送 API 请求：

```bash
DOWNSTREAM_DRY_RUN=1 python scripts/run_downstream.py
```

## 运行聚合与相关分析

```bash
./Run_Part2_Analysis.sh
```

量表输入路径：

```text
../统计/PromptA/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
../统计/PromptB/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
```

主要输出：

```text
outputs/downstream_task_aggregates.csv
outputs/downstream_dimension_aggregates_long.csv
outputs/repeat_stability.csv
outputs/analysis_summary.csv
outputs/main_matched_correlations.csv
outputs/mfq_mfv_correlation_long.csv
outputs/ous_correlation_summary.csv
outputs/correlation_model_pairs.csv
outputs/ANALYSIS_SUMMARY.md
outputs/figures/
```

当前仅有 9 个模型，所有显著性结果均应按探索性分析解释，并与 N/A 率、repeat 稳定性、语言方向和 leave-one-model-out 结果共同报告。

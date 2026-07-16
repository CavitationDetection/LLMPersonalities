# 量表辅助工具

本目录中的脚本用于预览、诊断和读取量表结果，不属于正式批量调用的主路径。正式入口是上一级的 `Submit.sh` 和 `run_no_memory_batch.py`。

## `example_single_turn_test.py`

用途：选择一道量表题，打印最终 system/user message；只有显式加 `--run` 才会调用 API。

预览示例：

```bash
python Tools/example_single_turn_test.py \
  --model gpt-5.4-2026-03-05 \
  --sheet "IPIP BFFM-50" \
  --number 1 \
  --language zh \
  --prompt A
```

实际发送并显示完整返回 JSON：

```bash
python Tools/example_single_turn_test.py \
  --model gpt-5.4-2026-03-05 \
  --sheet "IPIP BFFM-50" \
  --number 1 \
  --language en \
  --prompt B \
  --run \
  --show-json
```

注意：该工具是人工抽查工具，其实现保留了与正式批量脚本相同的消息构造规则，但它不会写入正式 repeat 目录。

## `diagnose_scale_responses.py`

用途：按随机种子从正式 7 个量表中抽取一道题，用相同题目调用全部最终模型，并保存完整请求与原始响应。该工具专门用于检查 `max_tokens=32`、reasoning token、finish reason、空 content 和解析状态。

```bash
python Tools/diagnose_scale_responses.py \
  --prompt A \
  --max-tokens 32 \
  --timeout 90 \
  --seed 20260716
```

默认输出到：

```text
Outputs/Diagnostics/scale_max_tokens32_9api_<timestamp>/
```

主要文件：

- `raw_responses.jsonl`：逐模型完整请求、HTTP 状态、响应 JSON、可见文本、usage、reasoning/output token 和解析结果。
- `summary.csv`：便于筛查的扁平汇总。
- `SUMMARY.md`：人类可读摘要。
- `test_metadata.json`：抽题种子、题目、模型和调用参数。

诊断脚本默认不重试，因此不会把后续尝试混入第一次原始响应。

## `read_results_workbook.py`

用途：读取正式 `*_results.xlsx`，并保证字面量 `N/A` 不被 pandas 自动转换为 `NaN`。

```bash
python Tools/read_results_workbook.py \
  Outputs/Results_9API_7Scale_PromptA/repeat_01/<model>_results.xlsx
```

该工具只读文件，不修改工作簿。

## 凭据与数据

三个工具与正式 runner 共用：

- `Test_File/api_config.json` 或 API 环境变量。
- `Test_File/Scale_16_Q_F.xlsx`。
- 中英文 `_AI_Protocol.xlsx`。
- `API_Selected_Final9.txt` 和 `Sheet_Selected_Final7.txt`。

任何诊断输出都不得包含可复用的 API key；响应 JSON 可以保留模型返回和 token usage，但不能写入请求 Authorization header。

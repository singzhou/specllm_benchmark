# AICC Agent Benchmark

以用户提供的原版源码为蓝本（与仓库 `04eb5b5` 的 Agent 一致）。`ResultWriter` 和整个 `BenchmarkFramework` 保持原样；运行前检查依赖与 Excel，运行结束后自动写 Excel。

## 安装

```bash
python -m pip install -r AICC_llm_benchmark/requirements.txt
```

Python 3.9 或更新版本。启动时检查 `requests`、`tqdm`、`openpyxl`；首个请求前实际读取、保存报表，提前发现依赖缺失、坏表、占用和读写权限问题。`tokenizer_path` 保留为可选兼容参数，无需 transformers。

## 两份 
| 文件 | 格式 |
| --- | --- |
| `testsets/common_200_openai.jsonl` | OpenAI `messages` 与空 `tools` 列表 |

两份各 200 题，题目、ID、顺序、类别及来源信息逐项相同，每题一个请求。沿用原 Agent 的 `turns` 转换和 `messages/tools` 路径，不改变题目内容。

在项目根目录运行；测试另一种格式时仅替换 `--dataset` 路径：

```bash
python AICC_llm_benchmark/agent_benchmark.py \
  --dataset AICC_llm_benchmark/testsets/common_200.jsonl \
  --output_file /dpc/hot/w00958190/101_workspace/dspark/test_output/1020_domain/common_200.jsonl \
  --ip 127.0.0.1 --port 45008 --model qwen3.8-27b \
  --concurrency 8 --spec_step_num 7 --eval_target final_answer --enable_thinking \
  --max_tokens 8192 --temperature 0.7 --top_p 0.95 --request_timeout 600
```

输出沿用原版时间戳命名。`--eval_target` 支持原版的 `final_answer` 和 `tool_call`；采样、请求发送、SSE 读取、线程并发、超时和请求失败处理均沿用用户原版。

## 自动 Excel

每次测完，自动向结果目录的 `目录名_analyzer.xlsx` 保存结果。例如上面的目录对应 `1020_domain_analyzer.xlsx`；`Overall` 每场一行，`Categories` 保存分类统计。保留以前场次，重新分析同一结果时更新对应行。

Excel 取原版 analyzer 同一次计算的结果；`Requests` 沿用原版有效请求数。整场没有有效数据时也记录 `no_valid_data`，不生成有效性能数据。测试期间请关闭目标 Excel。

离线重新导出：

```bash
python AICC_llm_benchmark/agent_benchmark.py --analyze_only --output_file /path/to/result.jsonl
```

`--spec_step_num` 仅用于统计，应与服务端投机配置相同。原版使用整场 `/metrics` 差值补充投机计数时，须独占该服务；分类计数缺失时显示 `N/A`。

## 本地基本验证

```bash
python -m pip install pytest
python -m pytest -q AICC_llm_benchmark/tests
```

测试使用本地 HTTP/SSE 模拟服务，覆盖两份 common200、自动 Excel 和依赖预检。

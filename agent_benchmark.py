"""
Agent-Aware Concurrent OpenAI-compatible Benchmark Framework.
支持两种测试集模式（通过 --eval_target 控制）：
  1. tool_call: 评测模型输出工具调用的能力（输入包含对话历史及工具定义，期望模型输出 tool_calls）
  2. final_answer: 评测模型根据工具执行结果进行最终总结的能力（输入包含 tool 返回结果，期望模型输出最终回复）
"""
import json
import time
import argparse
import threading
try:
    import requests
    from tqdm import tqdm
    from openpyxl import Workbook, load_workbook
except ImportError as exc:
    raise SystemExit(
        f'Missing/broken benchmark dependency: {exc}. '
        f'Install with: python -m pip install -r {__file__.rsplit("/", 1)[0]}/requirements.txt'
    ) from exc
import os
import math
import re
import shutil
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from req_metadata import ReqMetadata
from benchmark_args import BenchmarkArgs
from datetime import datetime
from pathlib import Path
from excel_report import preflight_excel, append_result

def parse_spec_metrics(text):
    """Read vLLM Prometheus counters without an extra parser dependency."""
    names = {
        "vllm:spec_decode_num_drafts": "num_spec_steps",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_per_pos": "positions",
    }
    families = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(.*)$", line)
        if not match:
            continue
        name, rest = match.groups()
        base = name[:-6] if name.endswith("_total") else name
        if base not in names:
            continue
        labels = ""
        rest = rest.strip()
        if rest.startswith("{"):
            end = rest.rfind("}")
            if end < 0:
                continue
            labels, rest = rest[1:end], rest[end + 1:].strip()
        try:
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(value) or value < 0:
            continue
        families.setdefault(name, []).append((labels, value))
    result = {}
    for base, key in names.items():
        samples = families.get(base + "_total", families.get(base))
        if samples is None:
            continue
        if key == "positions":
            positions = {}
            for labels, value in samples:
                match = re.search(r'(?:^|,)\s*position="(\d+)"', labels)
                if match:
                    index = int(match.group(1))
                    positions[index] = positions.get(index, 0) + value
            result[key] = positions
        else:
            result[key] = sum(value for _, value in samples)
    required = ("num_spec_steps", "num_draft_tokens", "num_accepted_draft_tokens")
    return result if all(key in result for key in required) else None


def spec_metrics_delta(before, after, k):
    if before is None or after is None:
        return None
    keys = ("num_spec_steps", "num_draft_tokens", "num_accepted_draft_tokens")
    delta = {key: after[key] - before[key] for key in keys}
    if any(value < 0 for value in delta.values()):
        return None
    if delta["num_accepted_draft_tokens"] > delta["num_draft_tokens"]:
        return None
    old_pos, new_pos = before.get("positions", {}), after.get("positions", {})
    if k > 0 and all(index in new_pos for index in range(k)):
        positions = [new_pos[index] - old_pos.get(index, 0) for index in range(k)]
        if all(0 <= value <= delta["num_spec_steps"] for value in positions):
            delta["num_accepted_tokens_per_pos"] = positions
    return delta


class ResultWriter:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.file = open(path, 'a', encoding='utf-8')
    def write(self, metadata: ReqMetadata):
        line = metadata.dump_json()
        with self.lock:
            self.file.write(line + '\n')
            self.file.flush()
    def close(self):
        self.file.close()

class BenchmarkFramework:
    def __init__(self, args):
        self.args = args
        # Agent 测试统一走 /v1/chat/completions 接口
        self.url = f'http://{args.ip}:{args.port}/v1/chat/completions'
        self.writer = ResultWriter(args.output_file)

    def read_server_spec_metrics(self):
        host = "127.0.0.1" if self.args.ip == "0.0.0.0" else self.args.ip
        url = getattr(self.args, "metrics_url", None) or f"http://{host}:{self.args.port}/metrics"
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    snapshot = parse_spec_metrics(response.text)
            if snapshot is None:
                print("[Warning] /metrics 未暴露所需的投机计数；无法用服务端累计计数补全指标。")
            return snapshot
        except Exception as exc:
            print(f"[Warning] 读取 /metrics 失败: {exc}")
            return None

    def single_infer(self, task, *, warmup=False):
        """
        根据新版测试集结构和评测目标（eval_target）组装 messages 与 tools。
        """
        category = task.get('category', 'default')
        base_id = task.get('id', 'unknown')
        session_id = task.get('session_id', base_id)

        messages = task.get('messages', [])
        if not messages and 'turns' in task:
            messages = [
                {"role": "user", "content": t} if isinstance(t, str) else t
                for t in task['turns']
            ]
        tools = task.get('tools', [])

        if not messages:
            raise ValueError(f"Task {session_id!r}: 'messages' must not be empty")

        # 根据评测目标过滤或截断 messages
        eval_target = getattr(self.args, 'eval_target', 'final_answer')

        if eval_target == 'tool_call':
            # 模式 A：寻找最后一次 Assistant 发起 tool_call 的位置，截断在其之前，
            # 让模型重新预测或执行这一步的 tool call。
            # 如果整段历史里没有 tool_call，则退化为直接使用全部 messages。
            target_idx = -1
            for idx in range(len(messages) - 1, -1, -1):
                if messages[idx].get('role') == 'assistant' and messages[idx].get('tool_calls'):
                    target_idx = idx
                    break

            if target_idx != -1:
                # 截断到 tool_call 发起的前一步，让模型输出当前的 tool_call
                truncated_messages = messages[:target_idx]
                req_id = f"{session_id}_toolcall"
            else:
                truncated_messages = messages
                req_id = f"{session_id}_chat"

        elif eval_target == 'final_answer':
            # 模式 B：评测最终总结。输入包含完整的对话历史、tool_calls 以及 tool 的执行结果，
            # 期望模型生成最终的 assistant 回复（reference_answer）。
            truncated_messages = messages
            req_id = f"{session_id}_final"

        else:
            raise ValueError(f"Unsupported eval_target: {eval_target}")

        return self._execute_request(req_id, category, truncated_messages, tools, warmup=warmup)

    def _execute_request(self, req_id, category, messages, tools, *, warmup=False):
        """构造 OpenAI Chat Completion 格式的请求并发送流式评测"""
        req_meta = ReqMetadata(
            spec_step_num=self.args.spec_step_num,
            req_id=req_id,
            category=category,
            concurrency=self.args.concurrency
        )

        # 动态新增字段，兼容原版 req_metadata.py，无需修改该文件。
        req_meta.completion_tokens = None
        req_meta.speculative_metrics = None
        req_meta.tool_call_deltas = []

        js = {
            "model": self.args.model,
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # 如果测试集中带有 tools 定义，则带上 tools
        if tools:
            js["tools"] = tools

        # 兼容 thinking 模型开关
        if not self.args.is_magistral:
            js["chat_template_kwargs"] = {
                "enable_thinking": self.args.enable_thinking
            }

        if self.args.force_output_tokens is not None:
            js.update(
                max_tokens=self.args.force_output_tokens,
                min_tokens=self.args.force_output_tokens,
                ignore_eos=True,
            )
            js.pop('stop', None)

        if warmup:
            tokens = self.args.warmup_max_tokens
            js.update(max_tokens=tokens, min_tokens=tokens, ignore_eos=True)

        req_meta.put_req_js(js)

        response = None
        try:
            start_request_time = time.perf_counter()
            response = requests.post(
                self.url,
                json=js,
                stream=True,
                timeout=self.args.request_timeout,
            )
            first_token_received = False
            stream_done = False
            stream_error = False
            prev_token_time = 0.0

            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith('data:'):
                        continue
                    data_payload = line[len('data:'):].strip()
                    if data_payload == "[DONE]":
                        stream_done = True
                        break
                    try:
                        data_json = json.loads(data_payload)
                        if data_json.get('error'):
                            stream_error = True
                        # usage/metrics 通常位于 choices 为空的尾包，必须先读取。
                        usage = data_json.get("usage")
                        if isinstance(usage, dict):
                            n = usage.get("completion_tokens")
                            if isinstance(n, int) and n >= 0:
                                req_meta.completion_tokens = n
                        metrics = data_json.get("metrics")
                        if isinstance(metrics, dict):
                            spec = metrics.get("speculative_decoding")
                            if isinstance(spec, dict):
                                req_meta.speculative_metrics = spec
                        choices = data_json.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        r_text = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        c_text = delta.get("content") or ""
                        tool_deltas = delta.get("tool_calls") or []
                        if tool_deltas:
                            req_meta.tool_call_deltas.append(tool_deltas)
                        tool_text = ""
                        for tc in tool_deltas:
                            fn = tc.get("function") or {}
                            tool_text += fn.get("name") or ""
                            tool_text += fn.get("arguments") or ""
                        # 仅用于输出留存和计时，不用此文本推算 token 数或验证轮数。
                        # 同一 SSE 中的 reasoning/content/tool_calls 只记录一个到达事件。
                        content = r_text + c_text + tool_text
                        if not content:
                            continue
                        cur_time = time.perf_counter()
                        if not first_token_received:
                            req_meta.put_prefill(content, cur_time - start_request_time)
                            first_token_received = True
                        else:
                            req_meta.put_new_decode(content, cur_time - prev_token_time)
                        prev_token_time = cur_time
                    except (KeyError, TypeError, json.JSONDecodeError) as exc:
                        print(f"\n[Warning] {req_id} ignored malformed SSE data: {exc}")
                if warmup:
                    return (stream_done and not stream_error and req_meta.prefill_latency > 0
                            and req_meta.completion_tokens is not None
                            and req_meta.completion_tokens >= self.args.warmup_max_tokens)
                self.writer.write(req_meta)
            else:
                error_text = response.text[:1000].replace('\n', ' ')
                print(f"\n[Error] {req_id} returned status {response.status_code}: {error_text}")
        except Exception as e:
            print(f"\n[Exception] {req_id} failed: {e}")
        finally:
            if response is not None:
                response.close()
        return False

    def run_warmup(self):
        count = self.args.warmup_requests
        if count is None:
            count = 2 * self.args.concurrency
        report = {'requests': count, 'concurrency': self.args.concurrency,
                  'max_tokens': self.args.warmup_max_tokens,
                  'status': 'disabled' if count == 0 else 'running', 'prompt_source': 'synthetic'}
        if count == 0:
            return report
        print(f"Warmup: {count} requests, concurrency={self.args.concurrency}, "
              f"tokens={self.args.warmup_max_tokens}; excluded from measured results.", flush=True)
        nonce = os.urandom(8).hex()
        started = time.perf_counter()
        passed = 0
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as executor:
            futures = [executor.submit(self.single_infer, {
                'id': f'warmup_{nonce}_{i}', 'category': '[WARMUP]',
                'messages': [{'role': 'user', 'content':
                    f'{nonce}-{i}\nExplain step by step how to write and debug a Python program. '
                    'Include detailed examples.'}],
            }, warmup=True) for i in range(count)]
            for future in tqdm(as_completed(futures), total=count, desc='Warmup'):
                passed += bool(future.result())
        if passed != count:
            raise RuntimeError(f'Warmup failed ({passed}/{count} valid requests); measured run not started.')
        # Exclude asynchronously published warmup counters from the measured baseline.
        time.sleep(self.args.metrics_settle_seconds)
        report.update(status='passed', valid_requests=passed, elapsed_seconds=time.perf_counter() - started)
        print('Warmup passed; starting measured run.', flush=True)
        return report

    def post_sort_results(self):
        print("Sorting results by Category...")
        results = []
        header = None
        with open(self.args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data = json.loads(line)
                    if header is None and raw_data.get("__type__", "") == "meta":
                        header = raw_data
                        continue
                    results.append(raw_data)
        if header is not None:
            header["__payload__"]["server_spec_metrics_delta"] = getattr(self, "server_spec_delta", None)
            header["__payload__"]["server_spec_metrics_by_category"] = getattr(self, "category_spec_deltas", {})
        results.sort(key=lambda x: x.get('category', ''))
        if header is not None:
            results.insert(0, header)
        with open(self.args.output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def run(self, dataset_path):
        tasks = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        print(f"Total tasks: {len(tasks)}. Concurrency: {self.args.concurrency}, Eval Target: {self.args.eval_target}")
        tasks_by_category = defaultdict(list)
        for task in tasks:
            tasks_by_category[task.get('category', 'default')].append(task)
        metrics_by_category = getattr(self.args, 'metrics_by_category', False)
        try:
            warmup = self.run_warmup() if tasks else {'requests': 0, 'status': 'disabled'}
        except BaseException:
            self.writer.close()
            raise

        if os.path.getsize(self.writer.path) == 0:
            header = {
                "__type__": "meta",
                "__payload__": {
                    "url": f"{self.args.ip}:{self.args.port}",
                    "dataset": self.args.dataset,
                    "model": self.args.model,
                    "tokenizer": self.args.tokenizer_path,
                    "concurrency": self.args.concurrency,
                    "spec_step_num": self.args.spec_step_num,
                    "eval_target": self.args.eval_target,
                    "task_count": len(tasks),
                    "metrics_scope_categories": sorted({t.get("category", "default") for t in tasks}),
                    "category_task_counts": {c: len(batch) for c, batch in tasks_by_category.items()},
                    "metrics_by_category": metrics_by_category,
                    "warmup": warmup,
                    "max_tokens": (
                        self.args.force_output_tokens
                        if self.args.force_output_tokens is not None
                        else self.args.max_tokens
                    ),
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                }
            }
            self.writer.file.write(json.dumps(header, ensure_ascii=False) + '\n')

        if self.args.spec_step_num > 0:
            print("投机统计：优先逐请求计数，否则使用 /metrics 前后差值；测试期间请独占该服务。")
        if metrics_by_category:
            print(f"分类统计：类别依次运行，每类内部并发 {self.args.concurrency}；分别采集 /metrics 差值。")
        batches = tasks_by_category if metrics_by_category else {None: tasks}
        self.server_spec_delta = None
        self.category_spec_deltas = {}
        deltas = []
        try:
            with ThreadPoolExecutor(max_workers=self.args.concurrency) as executor:
                for category, batch in batches.items():
                    before = self.read_server_spec_metrics() if self.args.spec_step_num > 0 else None
                    futures = [executor.submit(self.single_infer, task) for task in batch]
                    for future in tqdm(as_completed(futures), total=len(batch),
                                       desc=category if metrics_by_category else "Benchmarking Agent"):
                        future.result()
                    delta = None
                    if before is not None:
                        # Let asynchronous frontend counters settle before the next category.
                        time.sleep(getattr(self.args, "metrics_settle_seconds", 1.0))
                        after = self.read_server_spec_metrics()
                        delta = spec_metrics_delta(before, after, self.args.spec_step_num)
                    if self.args.spec_step_num > 0 and delta is None:
                        print(f"[Warning] {category if metrics_by_category else 'OVERALL'}: "
                              "/metrics 差值不可用，或计数在测试期间被重置。")
                    deltas.append(delta)
                    if metrics_by_category:
                        self.category_spec_deltas[category] = delta
        finally:
            self.writer.close()
        if deltas and all(d is not None for d in deltas):
            keys = ('num_spec_steps', 'num_draft_tokens', 'num_accepted_draft_tokens')
            self.server_spec_delta = {key: sum(d[key] for d in deltas) for key in keys}
            if all('num_accepted_tokens_per_pos' in d for d in deltas):
                self.server_spec_delta['num_accepted_tokens_per_pos'] = [
                    sum(d['num_accepted_tokens_per_pos'][p] for d in deltas)
                    for p in range(self.args.spec_step_num)]
        self.post_sort_results()

# 使用服务端 usage 和投机计数；不再通过 SSE 文本片段推算 token 数。
def _display_width(text):
    """Terminal cell width for Chinese/English text; no extra dependency."""
    return sum(0 if unicodedata.combining(c) else
               2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
               for c in str(text))


def _print_wrapped(text, columns, prefix=''):
    # Sanitize tabs/newlines so category names cannot break the report layout.
    text = ' '.join(str(text).split())
    line, used = prefix, _display_width(prefix)
    for char in text:
        width = _display_width(char)
        if used + width > columns and used > _display_width(prefix):
            print(line)
            line, used = ' ' * _display_width(prefix), _display_width(prefix)
        line += char
        used += width
    print(line)


def _print_table(headers, rows, columns):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(_display_width(row[i]) for row in [headers] + rows)
              for i in range(len(headers))]
    def render(row):
        cells = []
        for i, (value, width) in enumerate(zip(row, widths)):
            padding = ' ' * (width - _display_width(value))
            cells.append(value + padding if i == 0 else padding + value)
        return ' | '.join(cells)
    if sum(widths) + 3 * (len(widths) - 1) > columns:
        # Very narrow terminals: keep each metric on its own line.
        for row in rows:
            print(row[0])
            for key, value in zip(headers[1:], row[1:]):
                _print_wrapped(f'{key}: {value}', columns, '  ')
        return
    print(render(headers))
    rule = '-+-'.join('-' * width for width in widths)
    print(rule)
    for index, row in enumerate(rows):
        if row[0] == 'OVERALL' and index > 0:
            print(rule)
        print(render(row))


def _print_metrics_report(rows, columns):
    print()
    _print_wrapped('[Performance summary]', columns)
    _print_table(['ID', 'N', 'TTFT(ms)', 'TPOT(ms)', 'Acc(%)', 'SpecLen', 'OutLen'],
                 [r['cells'] for r in rows], columns)
    _print_wrapped('Acc = SpecAcc; SpecLen = AvgSpecLen; OutLen = AvgDecodeLen.', columns)
    print()
    _print_wrapped('[Category names]', columns)
    for row in rows:
        if row['id'] != 'OVERALL':
            _print_wrapped(row['category'], columns, row['id'] + '  ')
    print()
    _print_wrapped('[Per-position acceptance (%)]', columns)
    available = [r for r in rows if r['positions'] is not None]
    if not available:
        print('N/A')
    else:
        max_positions = max(len(r['positions']) for r in available)
        # Print positions in groups that fit the terminal.
        block_size = max(1, (columns - 10) // 9)
        for start in range(0, max_positions, block_size):
            end = min(start + block_size, max_positions)
            headers = ['ID'] + [f'P{p + 1}' for p in range(start, end)]
            values = [[r['id']] + [f"{r['positions'][p] * 100:.1f}"
                      if p < len(r['positions']) else 'N/A'
                      for p in range(start, end)] for r in available]
            _print_table(headers, values, columns)
    print()
    _print_wrapped('[Spec metric sources]', columns)
    labels = {'request': '逐请求服务端计数',
              'server': '/metrics 整场差值（须无其他流量）',
              'server_category': '/metrics 分类差值（类间串行，类内并发；须无其他流量）',
              'missing': '缺少可归属的服务端计数，显示 N/A'}
    for source, label in labels.items():
        ids = ', '.join(r['id'] for r in rows if r['source'] == source)
        if ids:
            _print_wrapped(f'{ids}: {label}', columns)
    if any(r['source'] == 'missing' and r['id'] != 'OVERALL' for r in rows):
        _print_wrapped('服务端无逐请求投机计数时，可加 --metrics-by-category 重跑以采集分类计数；'
                       '整场累计计数无法事后拆分给各类。', columns)


class MetricsAnalyzer:
    def __init__(self, result_file, tokenizer_path):
        self.result_file = result_file
        self.tokenizer_path = tokenizer_path  # 保留构造参数，不再通过重新分词计算指标。
    def analyze(self):
        results, header = [], None
        with open(self.result_file, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get('__type__') == 'meta':
                    header = data
                    continue
                results.append(ReqMetadata.from_dict(data))
        valid = [r for r in results
                 if isinstance(getattr(r, "completion_tokens", None), int)
                 and getattr(r, "completion_tokens", None) > 0 and r.prefill_latency > 0]
        print(f'有效请求: {len(valid)}/{len(results)}；缺少 usage 或可见输出的请求不参与统计')
        if not valid:
            print('没有有效数据，请确认服务端返回 usage，并重新评测。')
            payload = header.get('__payload__', {}) if header else {}
            return payload, [{'Category': '[OVERALL]', 'Requests': 0,
                              'Expected Requests': payload.get('task_count'),
                              'Run Status': 'no_valid_data'}]
        groups = defaultdict(list)
        for r in valid:
            groups[r.category].append(r)
        groups['[OVERALL]'] = valid
        def fmt(value):
            return 'N/A' if value is None else f'{value:.2f}'
        columns = max(20, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)
        _print_wrapped('AvgDecodeLen 含首 token；TPOT 按 sum(末次-首次可见输出时间)/sum(N-1) 计算。', columns)
        payload = header.get("__payload__", {}) if header else {}
        server_delta = payload.get("server_spec_metrics_delta")
        category_deltas = payload.get("server_spec_metrics_by_category", {})
        report_rows = []
        category_index = 0
        for category, items in groups.items():
            output_tokens = sum(getattr(r, "completion_tokens", None) for r in items)
            decode_ms = sum(sum(r.decode_raw_latency) * 1000 for r in items)
            tpot_tokens = sum(getattr(r, "completion_tokens", None) - 1 for r in items)
            ttft = sum(r.prefill_latency * 1000 for r in items) / len(items)
            tpot = decode_ms / tpot_tokens if tpot_tokens else None
            acc, spec_len, positions = None, None, None
            accepted, drafted, steps = None, None, None
            metrics = [getattr(r, "speculative_metrics", None) for r in items]
            keys = ('num_accepted_draft_tokens', 'num_draft_tokens', 'num_spec_steps')
            complete_request_metrics = all(
                isinstance(m, dict) and all(k in m for k in keys) for m in metrics
            )
            single_category_complete = (
                payload.get("metrics_scope_categories") == [category]
                and payload.get("task_count") == len(items)
                and len(items) == len(results)
            )
            source = 'request' if complete_request_metrics else 'missing'
            if not complete_request_metrics:
                fallback, fallback_source = None, 'missing'
                if category == '[OVERALL]' or single_category_complete:
                    fallback, fallback_source = server_delta, 'server'
                elif payload.get('category_task_counts', {}).get(category) == len(items):
                    # A category delta covers the whole batch, not a subset of valid requests.
                    fallback, fallback_source = category_deltas.get(category), 'server_category'
                if isinstance(fallback, dict) and all(k in fallback for k in keys):
                    metrics = [fallback]
                    source = fallback_source
                    complete_request_metrics = True
            if complete_request_metrics:
                accepted = sum(m['num_accepted_draft_tokens'] for m in metrics)
                drafted = sum(m['num_draft_tokens'] for m in metrics)
                steps = sum(m['num_spec_steps'] for m in metrics)
                acc = 100 * accepted / drafted if drafted else None
                spec_len = 1 + accepted / steps if steps else None
                histograms = [m.get('acceptance_histogram') for m in metrics]
                if (steps and all(isinstance(h, list) and h for h in histograms)
                        and len({len(h) for h in histograms}) == 1
                        and all(sum(h) == m['num_spec_steps'] for h, m in zip(histograms, metrics))):
                    positions = [sum(sum(h[p + 1:]) for h in histograms) / steps
                                 for p in range(len(histograms[0]) - 1)]
                counter_positions = [m.get('num_accepted_tokens_per_pos') for m in metrics]
                if (positions is None and steps
                        and all(isinstance(v, list) and v for v in counter_positions)
                        and len({len(v) for v in counter_positions}) == 1):
                    positions = [sum(v[p] for v in counter_positions) / steps
                                 for p in range(len(counter_positions[0]))]
            if category == '[OVERALL]':
                category_id = 'OVERALL'
            else:
                category_index += 1
                category_id = f'C{category_index:02d}'
            report_rows.append({
                'id': category_id, 'category': category, 'positions': positions,
                'source': source,
                'excel': {
                    'Category': category, 'Requests': len(items),
                    'Run Status': 'completed',
                    'SpecAcc': 'N/A' if acc is None else f'{acc:.2f}%',
                    'SpecLenAvg': round(spec_len, 2) if spec_len is not None else 'N/A',
                    'Accepted Draft Tokens': accepted, 'Draft Tokens': drafted, 'Spec Steps': steps,
                    'Spec Metric Source': source, 'Per-pos Acc': positions,
                    'Avg TTFT (ms)': round(ttft, 2),
                    'Avg TPOT (ms)': round(tpot, 2) if tpot is not None else 'N/A',
                    'Avg DecodeLen': round(output_tokens / len(items), 1),
                    'Total DecodeLen': output_tokens,
                },
                'cells': [category_id, str(len(items)), f'{ttft:.2f}', fmt(tpot),
                          fmt(acc), fmt(spec_len), f'{output_tokens / len(items):.1f}'],
            })
        _print_metrics_report(report_rows, columns)
        concurrency = header['__payload__']['concurrency'] if header else valid[0].concurrency
        total_ttft = sum(r.prefill_latency * 1000 for r in valid)
        total_decode = sum(sum(r.decode_raw_latency) * 1000 for r in valid)
        total_tokens = sum(getattr(r, "completion_tokens", None) for r in valid)
        denominator = sum(getattr(r, "completion_tokens", None) - 1 for r in valid)
        print(f'Total Output Token Number: {total_tokens}')
        print(f'TPOT Total Avg: {fmt(total_decode / denominator if denominator else None)}ms')
        if concurrency > 0:
            print('以下 batch avg 为请求时间之和/concurrency，不是实测整场耗时。')
            print(f'Total Prefill Time({concurrency} batch avg): {total_ttft / concurrency:.3f}ms')
            print(f'Total Decode Time({concurrency} batch avg): {total_decode / concurrency:.3f}ms')
            print(f'Total E2E time({concurrency} batch avg): {(total_ttft + total_decode) / concurrency:.3f}ms')
        if header:
            print('[Benchmark metadata]')
            for key, value in header['__payload__'].items():
                _print_wrapped(f'{key}: {value}', columns)
        return payload, [row['excel'] for row in report_rows]

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    BenchmarkArgs.add_args(parser)

    # 新增参数：控制评测目标是 tool_call 阶段还是 final_answer 阶段
    parser.add_argument(
        '--eval_target',
        choices=['tool_call', 'final_answer'],
        default='final_answer',
        help=(
            "tool_call: 评测模型输出结构化 tool_calls 的能力；"
            "final_answer: 评测模型根据 tool 返回结果输出最终总结（参考答案）的能力。"
        ),
    )
    parser.add_argument(
        '--request_timeout',
        type=float,
        default=120,
        help='HTTP request timeout in seconds',
    )
    parser.add_argument(
        '--force-output-tokens',
        type=int,
        default=None,
        metavar='N',
        help='Force exactly N generated tokens.',
    )
    parser.add_argument('--metrics-url', default=None,
                        help='Override Prometheus URL; default http://IP:PORT/metrics')
    parser.add_argument('--metrics-settle-seconds', type=float, default=1.0,
                        help='Wait before final metrics scrape (default: 1 second)')
    parser.add_argument('--metrics-by-category', action='store_true',
                        help='Run categories sequentially (concurrent within each category) '
                             'and snapshot /metrics around each category')
    parser.add_argument('--warmup-requests', type=int, default=None,
                        help='Warmup request count (default: 2 * concurrency; 0 disables warmup)')
    parser.add_argument('--warmup-max-tokens', type=int, default=256,
                        help='Force this many output tokens per warmup request (default: 256)')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    if (args.warmup_requests is not None and args.warmup_requests < 0) or args.warmup_max_tokens < 1:
        parser.error('--warmup-requests must be nonnegative; --warmup-max-tokens must be positive')
    if not 0 <= args.metrics_settle_seconds <= 30:
        parser.error('--metrics-settle-seconds must be between 0 and 30')

    result_directory = Path(args.output_file).resolve().parent
    excel_path = result_directory / f'{result_directory.name}_analyzer.xlsx'
    try:
        preflight_excel(excel_path)
        result_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        parser.error(f'Excel preflight failed before inference: {exc}')

    base, ext = os.path.splitext(args.output_file)
    if not args.analyze_only:
        args.output_file = f"{base}_{timestamp}{ext}"
        framework = BenchmarkFramework(args)
        framework.run(args.dataset)

    analyzer = MetricsAnalyzer(args.output_file, args.tokenizer_path)
    metadata, summary_rows = analyzer.analyze()
    try:
        append_result(excel_path, args.output_file, metadata, summary_rows)
    except Exception:
        print(f'Results retained at {args.output_file}. Re-export with: '
              f'python agent_benchmark.py --analyze_only --output_file {args.output_file}', flush=True)
        raise
    print()

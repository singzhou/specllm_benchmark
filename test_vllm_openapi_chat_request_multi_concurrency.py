
# encoding: utf-8

import json
import threading
import time
import argparse
import requests

from rich.live import Live
from rich.panel import Panel
from rich.console import Group, Console
from rich.table import Table
from transformers import AutoTokenizer


stream_outputs = {}
metrics_data = {}
console = Console()
output_lock = threading.Lock()


def extract_text(value):
    """
    兼容字符串及部分接口返回的结构化文本。
    """
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    result.append(item["text"])
                elif isinstance(item.get("content"), str):
                    result.append(item["content"])
        return "".join(result)

    if isinstance(value, dict):
        for key in ("text", "content"):
            if isinstance(value.get(key), str):
                return value[key]

    return ""


def extract_reasoning_and_content(data):
    """
    支持如下两类响应：

    流式：
        choices[0].delta.reasoning_content
        choices[0].delta.content

    非流式：
        choices[0].message.reasoning_content
        choices[0].message.content

    同时兼容部分服务的 reasoning / thinking 字段。
    """
    if not isinstance(data, dict):
        return "", ""

    reasoning = ""

    for key in ("reasoning_content", "reasoning", "thinking"):
        value = extract_text(data.get(key))
        if value:
            reasoning = value
            break

    content = extract_text(data.get("content"))

    return reasoning, content


def render_ui(questions_len):
    panels = []

    with output_lock:
        outputs = [
            stream_outputs.get(i, "等待响应中...")
            for i in range(questions_len)
        ]

    for i, output in enumerate(outputs):
        panels.append(
            Panel(
                output,
                title=f"[bold cyan]Question {i + 1}[/bold cyan]"
            )
        )

    return Group(*panels)


def infer(question, num, url, is_stream, tokenizer, enable_thinking=True):

    headers = {
        "content-type": "application/json",
        "Inference-Type": "stream"
    }

    payload = {
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_completion_tokens": 300,
        "temperature": 0,
        "stream": is_stream,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking
        }
    }

    start_time = time.perf_counter()
    first_token_time = None
    end_time = None

    reasoning_buffer = ""
    content_buffer = ""

    current_section = None
    succeeded = False

    with output_lock:
        stream_outputs[num] = ""

    def append_output(text, section):
        nonlocal current_section

        if not text:
            return

        with output_lock:
            if section != current_section:
                if section == "reasoning":
                    stream_outputs[num] += "\n【思考过程】\n"
                else:
                    stream_outputs[num] += "\n【最终回答】\n"

                current_section = section

            stream_outputs[num] += text

    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            stream=is_stream,
            verify=False,
            timeout=600
        ) as response:

            response.raise_for_status()

            if is_stream:
                for raw_line in response.iter_lines():

                    if not raw_line:
                        continue

                    line = raw_line.decode("utf-8").strip()

                    if not line.startswith("data:"):
                        continue

                    # 只去掉 data: 前缀，不能用 lstrip("data:")
                    line = line[len("data:"):].strip()

                    if line == "[DONE]":
                        break

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in event:
                        raise RuntimeError(str(event["error"]))

                    choices = event.get("choices") or []

                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}

                    reasoning, content = extract_reasoning_and_content(delta)

                    if reasoning or content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                    if reasoning:
                        reasoning_buffer += reasoning
                        append_output(reasoning, "reasoning")

                    if content:
                        content_buffer += content
                        append_output(content, "content")

            else:
                response_data = response.json()

                choices = response_data.get("choices") or []

                if not choices:
                    raise ValueError("响应中不存在 choices")

                message = choices[0].get("message") or {}

                reasoning, content = extract_reasoning_and_content(message)

                if reasoning or content:
                    first_token_time = time.perf_counter()

                reasoning_buffer = reasoning
                content_buffer = content

                append_output(reasoning, "reasoning")
                append_output(content, "content")

            succeeded = True

    except Exception as e:
        with output_lock:
            stream_outputs[num] += f"\n[Request Error]: {str(e)}"

    finally:
        end_time = time.perf_counter()

    total_time = end_time - start_time
    raw_text_buffer = reasoning_buffer + content_buffer

    metrics_text = (
        f"\n\n[bold green]✓ Done in {total_time:.2f}s[/bold green]"
        if succeeded
        else f"\n\n[bold red]✗ Failed in {total_time:.2f}s[/bold red]"
    )

    if first_token_time is not None:
        ttft = first_token_time - start_time

        metrics_text += (
            f"\n[bold yellow]• 首字时延 (TTFT):[/bold yellow] "
            f"{ttft * 1000:.2f} ms"
        )

        if tokenizer and raw_text_buffer:

            # 思维链与最终回答均参与 Token 统计
            reasoning_tokens = len(
                tokenizer.encode(
                    reasoning_buffer,
                    add_special_tokens=False
                )
            )

            content_tokens = len(
                tokenizer.encode(
                    content_buffer,
                    add_special_tokens=False
                )
            )

            token_count = reasoning_tokens + content_tokens

            incremental_time = end_time - first_token_time

            # 首 Token 包含在 TTFT 中，因此增量 Token 数减一
            incremental_tokens = max(token_count - 1, 0)

            tpot = (
                incremental_time * 1000 / incremental_tokens
                if incremental_tokens > 0
                else 0.0
            )

            throughput = (
                token_count / total_time
                if total_time > 0
                else 0.0
            )

            metrics_text += (
                f"\n[bold yellow]• 增量时延 (TPOT):[/bold yellow] "
                f"{tpot:.2f} ms/token"
            )

            metrics_text += (
                f"\n[bold yellow]• 吞吐量:[/bold yellow] "
                f"{throughput:.2f} tokens/s"
                f" (共 {token_count} tokens)"
            )

            metrics_text += (
                f"\n• Thinking Tokens: {reasoning_tokens}"
                f"\n• Content Tokens: {content_tokens}"
            )

            if succeeded:
                with output_lock:
                    metrics_data[num] = {
                        "ttft": ttft,
                        "tpot": tpot,
                        "tokens": token_count,
                        "reasoning_tokens": reasoning_tokens,
                        "content_tokens": content_tokens,
                        "incremental_time": incremental_time,
                        "incremental_tokens": incremental_tokens
                    }

    with output_lock:
        stream_outputs[num] += metrics_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", default="43118")

    parser.add_argument(
        "--model_path",
        default="/dpc/hot/model/z00927893/Qwen3.8-27B/",
        help="模型路径，用于加载 Tokenizer"
    )

    parser.add_argument(
        "--q_num",
        type=int,
        default=1,
        help="并发请求数"
    )

    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启流式响应"
    )

    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启 Thinking"
    )

    args = parser.parse_args()

    q_num = args.q_num

    url = f"http://{args.ip}:{args.port}/v1/chat/completions"

    tokenizer = None

    if args.model_path:
        console.print(
            f"[cyan]Loading Tokenizer from {args.model_path}...[/cyan]"
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=True
            )

            console.print(
                "[green]Tokenizer loaded successfully![/green]\n"
            )

        except Exception as e:
            console.print(
                f"[red]Failed to load tokenizer: {e}[/red]\n"
            )

    # 保留原来的测试问题
    question1 = (
        "Write a Python function to compute the nth "
        "Fibonacci number iteratively."
    )

    question2 = """
Can you give me a C function that can achieve the following functionality?

Chandler is participating in a series of car races.
He starts with a certain amount of fuel, and each race requires
a certain amount of fuel to participate.

After each race, Chandler earns money which he uses to buy more fuel.

However, Chandler can only participate in a race if he has
more fuel than the race requires.

Each race can only be participated in once.

Your task is to determine the maximum number of races
Chandler can participate in.

You need to write a program that takes the following inputs:

- The number of test cases
- For each test case:
  - The initial amount of fuel Chandler has
  - The number of races
  - For each race, the amount of fuel required to participate
    and the amount of fuel Chandler can buy after the race

The program should output the maximum number of races
Chandler can participate in for each test case.
"""

    question3 = "给我写个故事吧，400字左右"

    question4 = (
        "VMware能通过Intel处理器虚拟出其他架构的虚拟机么，如ARM"
    )

    question_bank = [
        question1,
        question2,
        question3,
        question4
    ]

    # 支持任意 q_num，而不是最多16个问题
    questions = [
        question_bank[i % len(question_bank)]
        for i in range(q_num)
    ]

    threads = []

    overall_start_time = time.perf_counter()

    with Live(
        render_ui(len(questions)),
        refresh_per_second=15
    ) as live:

        for j, question in enumerate(questions):

            thread = threading.Thread(
                target=infer,
                args=(
                    question,
                    j,
                    url,
                    args.stream,
                    tokenizer,
                    args.thinking
                )
            )

            threads.append(thread)
            thread.start()

        # 统一刷新，避免某个慢请求阻塞其他请求的展示
        while any(thread.is_alive() for thread in threads):
            live.update(render_ui(len(questions)))
            time.sleep(0.1)

        for thread in threads:
            thread.join()

        live.update(render_ui(len(questions)))

    overall_end_time = time.perf_counter()

    overall_wall_time = overall_end_time - overall_start_time

    console.print("\n所有并发请求已执行完毕！\n")

    if metrics_data and tokenizer:

        valid_requests = len(metrics_data)

        avg_ttft = (
            sum(m["ttft"] for m in metrics_data.values())
            / valid_requests
        )

        total_tokens = sum(
            m["tokens"] for m in metrics_data.values()
        )

        total_reasoning_tokens = sum(
            m["reasoning_tokens"] for m in metrics_data.values()
        )

        total_content_tokens = sum(
            m["content_tokens"] for m in metrics_data.values()
        )

        # 全部请求的增量时间总和
        total_incremental_time = sum(
            m["incremental_time"]
            for m in metrics_data.values()
        )

        # 全部请求的增量 Token 总数
        total_incremental_tokens = sum(
            m["incremental_tokens"]
            for m in metrics_data.values()
        )

        # 按所有增量 Token 加权统计，而非请求 TPOT 算术平均
        avg_tpot = (
            total_incremental_time * 1000
            / total_incremental_tokens
            if total_incremental_tokens > 0
            else 0.0
        )

        system_throughput = (
            total_tokens / overall_wall_time
            if overall_wall_time > 0
            else 0.0
        )

        table = Table(
            title="[bold magenta]整体并发测试统计报告[/bold magenta]",
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("性能指标", justify="left")
        table.add_column("数值结果", justify="right")

        table.add_row("成功请求数", f"{valid_requests} 个")

        table.add_row(
            "总耗时 (Wall-clock Time)",
            f"{overall_wall_time:.2f} s"
        )

        table.add_row(
            "平均首字时延 (Avg TTFT)",
            f"{avg_ttft * 1000:.2f} ms"
        )

        table.add_row(
            "平均增量时延 (Avg TPOT)",
            f"{avg_tpot:.2f} ms/token"
        )

        table.add_row(
            "生成总 Token 数",
            f"{total_tokens} tokens"
        )

        table.add_row(
            "Thinking Token 数",
            f"{total_reasoning_tokens} tokens"
        )

        table.add_row(
            "Content Token 数",
            f"{total_content_tokens} tokens"
        )

        table.add_row(
            "系统总吞吐量 (System Throughput)",
            f"{system_throughput:.2f} tokens/s"
        )

        console.print(table)

    else:
        console.print(
            "[yellow]没有可用的性能统计数据。[/yellow]"
        )

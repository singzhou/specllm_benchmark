import argparse
class BenchmarkArgs:

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
    # 基础连接
        parser.add_argument("--ip", default="127.0.0.1")
        parser.add_argument("--port", default="18080")
        parser.add_argument("--tokenizer_path", default="", help="Compatibility only; no tokenizer is needed")
        parser.add_argument("--dataset", default="")
        parser.add_argument("--output_file", default=f"benchmark_results.jsonl")
        parser.add_argument("--model", default="Magistral")

        # 并发与投机设置
        parser.add_argument("--concurrency", type=int, default=4)
        parser.add_argument("--spec_step_num", type=int, default=0)

        # 推理超参数
        parser.add_argument("--max_tokens", type=int, default=1024)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_p", type=float, default=0.95)

        # 系统 Prompt 设置
        parser.add_argument("--with_system_prompt", action='store_true')
        parser.add_argument("--system_prompt", default='')

        # 功能开关
        parser.add_argument("--analyze_only", action="store_true")

        parser.add_argument("--message", default='')

        parser.add_argument("--enable_thinking", action="store_true")

        parser.add_argument("--is_magistral", action="store_true")

        parser.add_argument("--multi_turn", action="store_true", help="开启多轮拼接压测")
        parser.add_argument("--turn_separator", default="\n\n", help="多轮问题拼接时的分隔符，默认双换行")

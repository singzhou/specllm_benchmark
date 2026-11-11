"""Real HTTP/SSE checks for original Agent scheduling and preflight/Excel additions."""
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

BENCH = Path(os.environ.get('AICC_BENCHMARK_DIR', Path(__file__).resolve().parents[1]))
ENTRY = BENCH / 'agent_benchmark.py'
COMMON = BENCH / 'testsets' / 'common_200.jsonl'


@pytest.fixture
def server():
    state = {'requests': [], 'metrics_reads': [], 'active': 0, 'peak': 0, 'gate': None,
             'lock': threading.Lock()}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == '/metrics'
            with state['lock']:
                count = len(state['requests'])
                state['metrics_reads'].append(count)
            data = (f'vllm:spec_decode_num_drafts_total {count}\n'
                    f'vllm:spec_decode_num_draft_tokens_total {7 * count}\n'
                    f'vllm:spec_decode_num_accepted_tokens_total {5 * count}\n').encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            assert self.path == '/v1/chat/completions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['stream'] and body['stream_options']['include_usage']
            with state['lock']:
                state['requests'].append(body)
                position = len(state['requests'])
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
            try:
                # Hold the first batch until all eight workers have reached HTTP.
                # This also makes the initial dataset-order assertion deterministic.
                if state['gate'] is not None and position <= 8:
                    state['gate'].wait(timeout=8)
                if body['model'] == 'broken':
                    self.send_response(500)
                    self.send_header('Content-Length', '7')
                    self.end_headers()
                    self.wfile.write(b'failure')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                events = [
                    {'choices': [{'index': 0, 'delta': {'content': 'Hello'}}]},
                    {'choices': [{'index': 0, 'delta': {'content': ' world'}, 'finish_reason': 'stop'}]},
                    {'choices': [], 'usage': {'completion_tokens': 2, 'prompt_tokens': 25},
                     'metrics': {'speculative_decoding': {'num_spec_steps': 1, 'num_draft_tokens': 7,
                                                          'num_accepted_draft_tokens': 5}}},
                ]
                if body['model'] == 'sse-error':
                    events.append({'error': {'message': 'backend failed'}})
                for event in [json.dumps(item) for item in events] + ['[DONE]']:
                    data = f'data: {event}\n\n'.encode()
                    self.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')
                    self.wfile.flush()
                self.wfile.write(b'0\r\n\r\n')
            finally:
                with state['lock']:
                    state['active'] -= 1
                self.close_connection = True

    class HTTPServer(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

    httpd = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_port, state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def run_cli(server, dataset, output, extra=(), env=None):
    command = [sys.executable, str(ENTRY), '--ip', '127.0.0.1', '--port', str(server[0]),
               '--model', 'fixture', '--dataset', str(dataset), '--output_file', str(output),
               '--spec_step_num', '7', '--concurrency', '8',
               '--max_tokens', '32', '--request_timeout', '10', '--metrics-settle-seconds', '0',
               *extra]
    return subprocess.run(command, capture_output=True, text=True, timeout=40, env=env)


def result_path(path):
    if path.exists():
        return path
    matches = list(path.parent.glob(f'{path.stem}_*{path.suffix}'))
    assert len(matches) == 1, matches
    return matches[0]


def read_results(path):
    rows = [json.loads(line) for line in result_path(path).read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get('__type__') != 'meta']


def write_dataset(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


@pytest.mark.parametrize('dataset_name', ['common_200.jsonl', 'common_200_openai.jsonl'])
def test_common_200_c8_uses_original_mixed_schedule_and_exports_excel(server, tmp_path, dataset_name):
    from openpyxl import load_workbook
    dataset = COMMON.with_name(dataset_name)
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert len(rows) == 200
    if dataset_name == 'common_200_openai.jsonl':
        assert all('turns' not in row and row['tools'] == [] for row in rows)
        assert all(len(row['messages']) == 1 and row['messages'][0]['role'] == 'user' for row in rows)
    messages = [row.get('messages') or [{'role': 'user', 'content': row['turns'][0]}] for row in rows]
    server[1]['gate'] = threading.Barrier(8)
    output = tmp_path / 'common.jsonl'
    result = run_cli(server, dataset, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(read_results(output)) == len(server[1]['requests']) == 200
    assert server[1]['peak'] >= 8
    assert server[1]['metrics_reads'] == [0, 200]
    expected = Counter(items[0]['content'] for items in messages[:8])
    actual = Counter(request['messages'][0]['content'] for request in server[1]['requests'][:8])
    assert actual == expected  # No grouping by category or implicit warmup.
    assert Counter(json.dumps(items, sort_keys=True) for items in messages) == Counter(
        json.dumps(request['messages'], sort_keys=True) for request in server[1]['requests'])
    book = load_workbook(tmp_path / f'{tmp_path.name}_analyzer.xlsx')
    try:
        assert book['Overall'].max_row == 2
        assert book['Categories'].max_row == 5
        sheet = book['Overall']
        report = dict(zip((cell.value for cell in sheet[1]), next(sheet.iter_rows(min_row=2, values_only=True))))
        assert report['Requests'] == 200
        assert report['SpecAcc'] == '71.43%' and '71.43' in result.stdout
        assert report['Accepted Draft Tokens'] == 1000
        assert report['Draft Tokens'] == 1400
        assert report['Total DecodeLen'] == 400
    finally:
        book.close()


def test_two_runs_append_excel_and_preserve_agent_inputs(server, tmp_path):
    from openpyxl import load_workbook
    messages = [{'role': 'user', 'content': 'look up the value'},
                {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'call_1', 'type': 'function',
                     'function': {'name': 'lookup', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call_1', 'content': '42'}]
    tools = [{'type': 'function', 'function': {'name': 'lookup',
              'parameters': {'type': 'object', 'properties': {}}}}]
    dataset = write_dataset(tmp_path / 'input.jsonl', [
        {'id': 'common', 'category': 'common', 'turns': ['first turn', 'do not send this turn']},
        {'id': 'agent', 'category': 'agent', 'messages': messages, 'tools': tools},
    ])
    for name in ('first', 'second'):
        result = run_cli(server, dataset, tmp_path / f'{name}.jsonl')
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(read_results(tmp_path / f'{name}.jsonl')) == 2
    assert len(server[1]['requests']) == 4
    reexport = subprocess.run([sys.executable, str(ENTRY), '--analyze_only',
                               '--output_file', str(result_path(tmp_path / 'first.jsonl'))],
                              capture_output=True, text=True, timeout=15)
    assert reexport.returncode == 0, reexport.stdout + reexport.stderr
    for request in server[1]['requests']:
        if request['messages'][0]['content'] == 'first turn':
            assert request['messages'] == [{'role': 'user', 'content': 'first turn'},
                                            {'role': 'user', 'content': 'do not send this turn'}]
        else:
            assert request['messages'] == messages
            assert request['tools'] == tools
    book = load_workbook(tmp_path / f'{tmp_path.name}_analyzer.xlsx')
    try:
        assert book['Overall'].max_row == 3
        assert book['Categories'].max_row == 5
        headers = [cell.value for cell in book['Overall'][1]]
        index = headers.index('File Name')
        assert {row[index].split('_', 1)[0] for row in book['Overall'].iter_rows(min_row=2, values_only=True)} == {
            'first', 'second'}
    finally:
        book.close()


@pytest.mark.parametrize('dependency', ['requests', 'tqdm', 'openpyxl'])
def test_missing_dependency_fails_before_first_post(server, tmp_path, dependency):
    blocked = tmp_path / 'blocked'
    blocked.mkdir()
    (blocked / f'{dependency}.py').write_text(f'raise ImportError("missing fixture {dependency}")\n')
    env = {**os.environ, 'PYTHONPATH': str(blocked)}
    result = run_cli(server, COMMON, tmp_path / 'never.jsonl', env=env)
    assert result.returncode != 0
    assert dependency in result.stdout + result.stderr
    assert not server[1]['requests']


def test_corrupt_excel_fails_before_first_post_and_is_preserved(server, tmp_path):
    workbook = tmp_path / f'{tmp_path.name}_analyzer.xlsx'
    original = b'not a valid workbook; preserve me'
    workbook.write_bytes(original)
    result = run_cli(server, COMMON, tmp_path / 'never.jsonl')
    assert result.returncode != 0
    assert not server[1]['requests']
    assert workbook.read_bytes() == original


def test_empty_result_still_exports_excel(server, tmp_path):
    from openpyxl import load_workbook
    dataset = write_dataset(tmp_path / 'one.jsonl', [
        {'id': 'one', 'category': 'failed', 'turns': ['test failure']}])
    output = tmp_path / 'failed.jsonl'
    result = run_cli(server, dataset, output, extra=['--model', 'broken'])
    assert result.returncode == 0, result.stdout + result.stderr  # Original request handling.
    assert read_results(output) == []
    book = load_workbook(tmp_path / f'{tmp_path.name}_analyzer.xlsx')
    try:
        sheet = book['Overall']
        report = dict(zip((cell.value for cell in sheet[1]), next(sheet.iter_rows(min_row=2, values_only=True))))
        assert report['Requests'] == 0
        assert report['Expected Requests'] == 1
        assert report['Run Status'] == 'no_valid_data'
    finally:
        book.close()


def test_excel_owner_lock_fails_before_first_post(server, tmp_path):
    workbook = tmp_path / f'{tmp_path.name}_analyzer.xlsx'
    workbook.with_name('~$' + workbook.name).write_bytes(b'open in Excel')
    result = run_cli(server, COMMON, tmp_path / 'never.jsonl')
    assert result.returncode != 0
    assert not server[1]['requests']
    assert not (tmp_path / 'never.jsonl').exists()

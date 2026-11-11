#!/usr/bin/env python3
"""Estimate draft-weighted acceptance from saved Agent SSE text fragments."""
import argparse
import json
from pathlib import Path

from req_metadata import ReqMetadata


COUNTERS = ('requests', 'used_requests', 'skipped_failed_requests', 'no_decode_requests',
            'zero_token_chunks', 'clipped_chunks', 'rounds', 'accepted', 'drafted')


def load_results(path):
    metadata, records = {}, []
    seen_header = False
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('expected a JSON object')
                if row.get('__type__') == 'meta':
                    if seen_header or records or not isinstance(row.get('__payload__'), dict):
                        raise ValueError('expected one metadata header before request records; do not merge runs')
                    metadata, seen_header = row['__payload__'], True
                else:
                    chunks = row.get('decode_raw_texts')
                    if not isinstance(chunks, list) or not all(isinstance(s, str) for s in chunks):
                        raise ValueError('expected decode_raw_texts as a list of SSE text fragments; '
                                         'full output text alone is insufficient')
                    if not isinstance(row.get('category', 'default'), str):
                        raise ValueError('category must be a string')
                    records.append(row)
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    if not records:
        raise ValueError(f'{path}: no request records')
    return metadata, records


def new_counts(k):
    return {**dict.fromkeys(COUNTERS, 0), 'positions': [0] * k}


def summarize(category, counts):
    rounds, drafted = counts['rounds'], counts['drafted']
    return {
        'category': category,
        **{key: counts[key] for key in COUNTERS[:6]},
        'estimated_rounds': rounds,
        'estimated_accepted_tokens': counts['accepted'],
        'estimated_draft_tokens': drafted,
        'estimated_spec_acc_pct': 100 * counts['accepted'] / drafted if drafted else None,
        'estimated_spec_len': 1 + counts['accepted'] / rounds if rounds else None,
        'estimated_per_position_acc_pct': [100 * n / rounds for n in counts['positions']] if rounds else None,
    }


def estimate(records, tokenizer, k):
    groups, overall = {}, new_counts(k)
    for row in records:
        counts = new_counts(k)
        counts['requests'] = 1
        if row.get('request_error') or row.get('stream_done') is False:
            counts['skipped_failed_requests'] = 1
        else:
            request = ReqMetadata.from_dict(row)
            request.spec_step_num = k
            # Reuse the old per-fragment tokenization; prefill_token is excluded.
            lengths = request.get_decode_token_num_list(tokenizer)
            counts['zero_token_chunks'] = len(row['decode_raw_texts']) - len(lengths)
            if not lengths:
                counts['no_decode_requests'] = 1
            else:
                # ponytail: assumes fixed K drafts and one verification per SSE fragment;
                # use server counters when exact acceptance is required.
                accepted = [min(length - 1, k) for length in lengths]
                counts.update(used_requests=1, rounds=len(lengths), accepted=sum(accepted),
                              drafted=k * len(lengths), clipped_chunks=sum(n > k + 1 for n in lengths))
                counts['positions'] = [sum(n >= p + 1 for n in accepted) for p in range(k)]
        category = row.get('category', 'default')
        if category not in groups:
            groups[category] = new_counts(k)
        for target in (groups[category], overall):
            for key in COUNTERS:
                target[key] += counts[key]
            for p in range(k):
                target['positions'][p] += counts['positions'][p]
    return ([summarize(category, groups[category]) for category in sorted(groups)],
            summarize('[OVERALL]', overall))


def print_report(rows):
    headers = ['Category', 'N', 'Used', 'EstAcc(%)', 'EstLen', 'EstAccept', 'EstDraft', 'Rounds', 'Clipped']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.2f}'
    cells = [[r['category'], str(r['requests']), str(r['used_requests']), fmt(r['estimated_spec_acc_pct']),
              fmt(r['estimated_spec_len']), str(r['estimated_accepted_tokens']),
              str(r['estimated_draft_tokens']), str(r['estimated_rounds']), str(r['clipped_chunks'])]
             for r in rows]
    widths = [max(len(row[i]) for row in [headers] + cells) for i in range(len(headers))]
    for row in [headers] + cells:
        print(' | '.join(value.ljust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One Agent benchmark result JSONL')
    parser.add_argument('--tokenizer-path', '--tokenizer_path',
                        help='Same local tokenizer as the served model; defaults to metadata.tokenizer')
    parser.add_argument('--spec-step-num', '--spec_step_num', type=int,
                        help='Fixed draft length K; defaults to the saved per-request/header setting')
    parser.add_argument('--output', type=Path, help='New JSON report; defaults to INPUT.sse_estimate.json')
    parser.add_argument('--trust-remote-code', action='store_true', help='Allow local custom tokenizer code')
    args = parser.parse_args()
    try:
        metadata, records = load_results(args.input)
        k = args.spec_step_num
        if k is None:
            steps = [row.get('spec_step_num', metadata.get('spec_step_num')) for row in records]
            if any(type(n) is not int or n <= 0 for n in steps):
                raise ValueError('a positive --spec-step-num is required when saved K is missing or zero')
            if len(set(steps)) != 1:
                raise ValueError('inconsistent spec_step_num; analyze each configuration separately')
            k = steps[0]
        if type(k) is not int or k <= 0:
            raise ValueError('a positive --spec-step-num is required when saved K is missing or zero')
        tokenizer_path = args.tokenizer_path or metadata.get('tokenizer')
        if not tokenizer_path:
            raise ValueError('--tokenizer-path is required when metadata.tokenizer is empty')
        output = args.output or args.input.with_suffix('.sse_estimate.json')
        if output.resolve() == args.input.resolve() or output.exists():
            raise ValueError(f'refusing to overwrite {output}; choose a new --output path')
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ValueError('install the offline analyzer dependency: python -m pip install transformers') from exc
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=args.trust_remote_code)
        categories, overall = estimate(records, tokenizer, k)
        report = {
            'method': 'sse_text_estimate_draft_weighted_v1',
            'input_file': str(args.input.resolve()), 'tokenizer': str(tokenizer_path), 'spec_step_num': k,
            'spec_step_override': args.spec_step_num is not None, 'prefill_chunk_excluded': True,
            'assumptions': 'One nonempty retokenized decode fragment = one round of K drafts; '
                           'accepted per round = min(fragment_tokens - 1, K). Not server counters.',
            'categories': categories, 'overall': overall,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write('\n')
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print('SSE text ESTIMATES only; accepted/drafted are estimates, not server counters.')
    print_report(categories + [overall])
    if overall['clipped_chunks']:
        print(f"Warning: {overall['clipped_chunks']} fragments exceeded K+1 tokens; "
              'accepted estimates were capped at K. Fragment/round boundaries are unreliable.')
    print('EstAcc = 100 * sum(estimated accepted) / sum(estimated drafts); no request averaging.')
    print(f'JSON report: {output.resolve()}')


if __name__ == '__main__':
    main()

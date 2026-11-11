"""Append one completed Agent benchmark to its result directory's workbook."""
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _report(path):
    from openpyxl import Workbook, load_workbook

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_name('~$' + path.name).exists():
        raise RuntimeError(f'Close the Excel workbook before benchmarking: {path}')
    lock = path.with_name(path.name + '.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Excel is being written: {lock}; if a previous process crashed, '
                           'verify it has stopped before removing this lock') from exc
    book = None
    try:
        os.write(fd, str(os.getpid()).encode())
        if path.exists():
            book = load_workbook(path)
            for name in ('Overall', 'Categories'):
                if name not in book.sheetnames:
                    raise ValueError(f'{path}: expected sheets Overall and Categories')
                headers = [cell.value for cell in book[name][1]]
                if headers != [None] and (not all(isinstance(h, str) for h in headers)
                                         or len(set(headers)) != len(headers)
                                         or not {'Result File', 'Category'} <= set(headers)):
                    raise ValueError(f'{path}: incompatible {name} headers; choose a new Excel filename')
        else:
            book = Workbook()
            book.active.title = 'Overall'
            book.create_sheet('Categories')
        yield book
    finally:
        if book is not None:
            book.close()
        os.close(fd)
        lock.unlink()


def _save(book, path):
    """Replace only after serialization succeeds; preserve the last good report."""
    path = Path(path).resolve()
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.stem}-', suffix='.xlsx', dir=path.parent)
    os.close(fd)
    try:
        book.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def preflight_excel(path):
    """Check dependency, existing format, locks and actual read/write before inference."""
    with _report(path) as book:
        _save(book, path)
    print(f'Excel preflight passed: {path}', flush=True)


def append_result(path, result_file, metadata, summary_rows):
    """Keep earlier runs; reanalyzing the same result replaces only that run's rows."""
    result_file = Path(result_file).resolve()
    common = {'File Name': result_file.name, 'Result File': str(result_file),
              'Model': metadata.get('model', ''), 'Dataset': metadata.get('dataset', ''),
              'Concurrency': metadata.get('concurrency'), 'Max Tokens': metadata.get('max_tokens'),
              'Step': metadata.get('spec_step_num')}
    rows = [{**common, **summary} for summary in summary_rows]
    with _report(path) as book:
        for name in ('Overall', 'Categories'):
            sheet = book[name]
            selected = [row for row in rows if (row['Category'] == '[OVERALL]') == (name == 'Overall')]
            headers = [cell.value for cell in sheet[1]]
            if headers == [None]:
                headers = []
            for row in selected:
                headers += [key for key in row if key not in headers]
            for col, header in enumerate(headers, 1):
                cell = sheet.cell(1, col, header)
                cell.data_type = 's'
            if 'Result File' in headers:
                file_column = headers.index('Result File') + 1
                for index in range(sheet.max_row, 1, -1):
                    if sheet.cell(index, file_column).value == str(result_file):
                        sheet.delete_rows(index)
            for row in selected:
                index = sheet.max_row + 1
                for col, header in enumerate(headers, 1):
                    value = row.get(header)
                    if isinstance(value, (dict, list, tuple)):
                        value = json.dumps(value, ensure_ascii=False)
                    cell = sheet.cell(index, col, value)
                    if isinstance(value, str):
                        cell.data_type = 's'
            if headers:
                sheet.freeze_panes = 'C2'
                sheet.auto_filter.ref = sheet.dimensions
                for cells in sheet.columns:
                    sheet.column_dimensions[cells[0].column_letter].width = min(45, max(14, len(str(cells[0].value)) + 2))
        _save(book, path)
    print(f'Excel saved: {path}', flush=True)

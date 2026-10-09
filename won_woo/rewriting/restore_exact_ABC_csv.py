"""Copy the eight linked source CSVs without parsing or rewriting their bytes."""

import csv
import json
import shutil
from pathlib import Path

from analyze_val_domain import write_csv
from build_object_action_regions import sha256


def main():
    workspace = Path(__file__).resolve().parent.parent
    base = workspace / 'rewriting/val_domain_analysis'
    source_dir = base / 'diagram_synonym_pairs'
    out = workspace / 'tvr_ABC_csv'
    out.mkdir(exist_ok=True)
    audit = json.loads((base / 'diagram_region_audit/context_review/final_audit/summary.json').read_text())
    with (base / 'diagram_region_audit/context_review/final_region_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        summary = {r['csv_file']: r for r in csv.DictReader(handle)}
    reports = []
    for aspect in ('object', 'action'):
        for region, split in [('A', 'train'), ('B', 'train'), ('B', 'val'), ('C', 'val')]:
            name = f'{aspect}_{region}_{split}.csv'
            source, target = source_dir / name, out / name
            temporary = target.with_suffix('.csv.copying')
            shutil.copy2(source, temporary)
            original_hash, copied_hash = sha256(source), sha256(temporary)
            assert original_hash == audit['checked_csv_sha256'][str(source)], f'Source changed: {name}'
            assert copied_hash == original_hash and source.stat().st_size == temporary.stat().st_size
            with source.open(encoding='utf-8-sig', newline='') as original, temporary.open(encoding='utf-8-sig', newline='') as copied:
                original_header, copied_header = next(csv.reader(original)), next(csv.reader(copied))
                assert original_header == copied_header and len(copied_header) == 32
            temporary.replace(target)
            reports.append({'file': name, 'source': str(source), 'copy': str(target),
                            'rows': int(summary[name]['rows']), 'columns': len(copied_header),
                            'unique_source_lemmas': int(summary[name]['unique_source_lemmas']),
                            'bytes': target.stat().st_size, 'source_sha256': original_hash,
                            'copy_sha256': copied_hash, 'byte_identical': True})
            print(f'{name}: original and copy SHA-256 match; all {summary[name]["rows"]} rows preserved', flush=True)
    write_csv(out / 'exact_copy_verification.csv', list(reports[0]), reports)
    (out / 'exact_copy_verification.json').write_text(json.dumps({'copies': reports,
        'method': 'Binary copy; no CSV parsing/re-serialization, filtering, deduplication, column changes or compression',
        'reference': 'The eight source files linked in the user-provided original table',
        'checks': 'All source hashes match the existing complete audit; all copy hashes match source hashes',
        'script_sha256': sha256(Path(__file__))}, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 원본과 동일한 A/B/C CSV', '',
             '아래 8개 파일은 처음 표에 연결했던 원본 CSV의 바이트 단위 복사본입니다. 원본과 복사본의 SHA-256이 모두 일치합니다.',
             '모든 행, 32개 열, 열 순서, WordNet 후보, 문맥 검토 열, 인코딩과 줄바꿈을 그대로 보존했습니다. 압축하지 않았습니다.', '',
             '숫자는 고유 기본형 수이며, CSV에는 해당 쿼리와 유의어 쌍의 전체 행이 들어 있습니다.', '',
             '| 구분 | A · Train | B · Train | B · Val | C · Val |', '| --- | --- | --- | --- | --- |']
    for aspect in ('object', 'action'):
        items = [r for r in reports if r['file'].startswith(aspect+'_')]
        lines.append('| '+aspect.title()+' | '+' | '.join(f"[{r['unique_source_lemmas']:,}]({r['file']})" for r in items)+' |')
    lines += ['', '[원본 동일성 검증 CSV](exact_copy_verification.csv) · [검증 상세](exact_copy_verification.json)', '',
              '이전 `_words.csv` 파일은 단어별 요약, `_preview.csv`는 일부 행 미리보기입니다. 위 표의 원본 복사본과 구분해야 합니다.', '',
              '재복사: `python rewriting/restore_exact_ABC_csv.py`']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'files_copied': len(reports), 'all_byte_identical': True,
                      'total_rows': sum(r['rows'] for r in reports), 'output': str(out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

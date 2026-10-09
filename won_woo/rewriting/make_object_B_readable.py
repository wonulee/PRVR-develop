"""Make small, uncompressed Object B CSV views and lossless row partitions."""

import csv
import hashlib
import json
from pathlib import Path

from analyze_val_domain import write_csv
from build_object_action_regions import sha256


def row_bytes(row):
    return json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'


def main():
    base = Path(__file__).resolve().parent / 'val_domain_analysis'
    source_dir = base / 'diagram_synonym_pairs'
    out = source_dir / 'object_B_readable'
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((base / 'diagram_region_audit/context_review/final_audit/summary.json').read_text())
    manifest, source_reports = [], []
    for split, source_name in [('train', 'object_B_train.csv'), ('val', 'object_B_val.csv')]:
        source = source_dir / source_name
        source_hash = sha256(source)
        assert source_hash == audit['checked_csv_sha256'][str(source)]
        part_dir = out / f'{split}_parts'
        part_dir.mkdir(exist_ok=True)
        preview_path = out / f'object_B_{split}_preview.csv'
        source_digest = hashlib.sha256()
        count, part_count, paths = 0, 0, []
        with source.open(encoding='utf-8-sig', newline='') as handle, preview_path.open('w', encoding='utf-8-sig', newline='') as preview:
            reader = csv.reader(handle)
            header = next(reader)
            preview_writer = csv.writer(preview)
            preview_writer.writerow(header)
            region_idx, aspect_idx = header.index('term_region'), header.index('term_type')
            part = None
            try:
                for row in reader:
                    assert len(row) == len(header) and row[region_idx] == 'B' and row[aspect_idx] == 'object'
                    if count % 5000 == 0:
                        if part is not None:
                            part.close()
                        part_count += 1
                        path = part_dir / f'object_B_{split}_part_{part_count:03d}.csv'
                        paths.append(path)
                        part = path.open('w', encoding='utf-8-sig', newline='')
                        part_writer = csv.writer(part)
                        part_writer.writerow(header)
                    part_writer.writerow(row)
                    if count < 500:
                        preview_writer.writerow(row)
                    source_digest.update(row_bytes(row))
                    count += 1
            finally:
                if part is not None:
                    part.close()
        restored_digest = hashlib.sha256()
        restored_count = 0
        for path in paths:
            start = restored_count + 1
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.reader(handle)
                assert next(reader) == header
                for row in reader:
                    restored_digest.update(row_bytes(row))
                    restored_count += 1
            manifest.append({'split': split, 'file': str(path.relative_to(out)),
                             'source_row_start': start, 'source_row_end': restored_count,
                             'rows': restored_count - start + 1, 'bytes': path.stat().st_size})
        assert restored_count == count and source_digest.digest() == restored_digest.digest()
        with source.open(encoding='utf-8-sig', newline='') as handle, preview_path.open(encoding='utf-8-sig', newline='') as preview:
            source_reader, preview_reader = csv.reader(handle), csv.reader(preview)
            assert next(source_reader) == next(preview_reader)
            for row in preview_reader:
                assert row == next(source_reader)
        assert sha256(source) == source_hash
        source_reports.append({'split': split, 'source': str(source), 'source_sha256': source_hash,
                               'rows': count, 'columns': len(header), 'parts': len(paths),
                               'preview_rows': min(500, count), 'partition_records_sha256': source_digest.hexdigest(),
                               'verified_all_fields_and_row_order_preserved': True})
        print(f'{split}: {count:,} rows, {len(paths)} parts; all fields/order verified', flush=True)
    term_path = base / 'diagram_region_audit/context_review/final_term_regions.csv'
    with term_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        terms = [r for r in reader if r['aspect'] == 'object' and r['term_region'] == 'B']
    assert sum(r['split'] == 'train' for r in terms) == 2991
    assert sum(r['split'] == 'validation' for r in terms) == 2209
    terms.sort(key=lambda r: (r['split'] != 'train', r['lemma']))
    for split, label in [('train', 'train'), ('validation', 'val')]:
        write_csv(out / f'object_B_{label}_words.csv', fields, [r for r in terms if r['split'] == split])
    write_csv(out / 'parts_manifest.csv', list(manifest[0]), manifest)
    metadata = {'source_files': source_reports, 'term_source_sha256': sha256(term_path),
                'preview_and_parts_unit': 'Synonym-pair rows; all original columns preserved',
                'word_summary_unit': 'Distinct Object B lemma per split; not synonym-pair rows',
                'encoding': 'UTF-8 BOM', 'compression': 'none', 'script_sha256': sha256(Path(__file__))}
    (out / 'verification.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Object B 바로 보기', '', '최종 문맥 검토를 반영한 Object B CSV입니다. 모든 파일은 압축하지 않았습니다.', '',
             '| 내용 | Train | Val |', '| --- | --- | --- |',
             '| 처음 500행, 원래 32개 열 유지 | [미리보기](object_B_train_preview.csv) | [미리보기](object_B_val_preview.csv) |',
             '| 기본형별 WordNet 후보 요약 | [2,991개 단어](object_B_train_words.csv) | [2,209개 단어](object_B_val_words.csv) |', '',
             '전체 분할본은 파일마다 최대 5,000행입니다. 헤더·모든 필드·행 순서의 보존을 다시 읽어 검증했습니다.',
             '단어별 요약은 기본형마다 한 행이며, 미리보기와 분할본은 기존 tvrtrain_synonym_pairs 형식의 후보 쌍 행입니다.',
             'WordNet 전체 후보와 반대 split 후보를 구분해서 볼 수 있습니다. 사전 후보가 문맥에 맞는 대체 표현임을 보장하지는 않습니다.', '',
             '## 전체 분할본', '', '| 출처 | 파일 | 원본 행 범위 | 크기(MB) |', '| --- | --- | --- | ---: |']
    for r in manifest:
        lines.append(f"| {r['split']} | [{Path(r['file']).name}]({r['file']}) | {r['source_row_start']:,}–{r['source_row_end']:,} | {r['bytes']/1_000_000:.2f} |")
    lines += ['', '[분할 목록 CSV](parts_manifest.csv) · [검증 결과](verification.json)', '',
              '재생성: `python rewriting/make_object_B_readable.py`']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(out), 'parts': len(manifest), 'word_summary_rows': len(terms)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Check all eight final region CSVs and export small, uncompressed local views."""

import csv
import hashlib
import json
from pathlib import Path

from analyze_val_domain import write_csv
from build_object_action_regions import sha256
from make_object_B_readable import row_bytes


VIEWER = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Object / Action A·B·C 보기</title>
<style>body{font:15px system-ui;margin:24px;color:#17212b}h1{font-size:24px}select,input,button{font:inherit;padding:8px;margin:4px}input{width:300px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ddd;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eef3f8}td:first-child{font-weight:600;min-width:100px}.muted{color:#52616b}summary{cursor:pointer}a{color:#0764ad}</style>
<h1>Object / Action A·B·C</h1>
<p>기본형마다 한 행인 단어별 요약입니다. WordNet 전체 후보와 반대 데이터셋 후보를 함께 볼 수 있습니다.</p>
<label>영역 <select id="group"></select></label>
<label>검색 <input id="search" placeholder="단어 또는 후보 검색"></label>
<button id="download">검색 결과 CSV 저장</button>
<p id="count" class="muted"></p><p id="links"></p>
<button id="prev">이전</button><span id="page"></span><button id="next">다음</button>
<table><thead><tr><th>기본형</th><th>반대 split 후보</th><th>WordNet 전체 후보</th><th>선택 의미의 후보</th><th>선택 의미 ID</th></tr></thead><tbody id="rows"></tbody></table>
<details><summary>전체 유의어 쌍 분할본</summary><p>원래 32개 열을 유지했습니다. 파일마다 최대 5,000행입니다.</p><div id="parts"></div></details>
<p class="muted">A/C도 WordNet 후보가 있을 수 있습니다. 미검토 사전 후보는 문맥에 적합한 대체 표현인지 확인이 필요합니다.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),byId=id=>document.getElementById(id);
const groups=data.groups,select=byId('group'),fields=data.fields,pageSize=50;let page=0;
for(const g of groups){const o=document.createElement('option');o.value=g.stem;o.textContent=g.label+' · '+g.words.length.toLocaleString()+'개';select.appendChild(o)}select.value='object_B_train';
function current(){return groups.find(g=>g.stem===select.value)}
function filtered(){const q=byId('search').value.toLocaleLowerCase().trim();return current().words.filter(r=>!q||fields.map(k=>r[k]).join(' ').toLocaleLowerCase().includes(q))}
function link(label,path){const a=document.createElement('a');a.textContent=label;a.href=path;a.download=path.split('/').pop();return a}
function render(){const g=current(),items=filtered(),pages=Math.max(1,Math.ceil(items.length/pageSize));page=Math.min(page,pages-1);byId('rows').replaceChildren();
 for(const r of items.slice(page*pageSize,(page+1)*pageSize)){const tr=document.createElement('tr');for(const k of ['lemma','other_split_candidates','wordnet_candidates','context_wordnet_candidates','selected_wordnet_synsets']){const td=document.createElement('td');td.textContent=r[k]||'—';tr.appendChild(td)}byId('rows').appendChild(tr)}
 byId('count').textContent=g.label+' · 고유 기본형 '+g.words.length.toLocaleString()+'개 / 검색 결과 '+items.length.toLocaleString()+'개 / 전체 유의어 쌍 파일 '+g.source_rows.toLocaleString()+'행';
 byId('page').textContent=' '+(page+1)+' / '+pages+' ';byId('prev').disabled=page===0;byId('next').disabled=page+1>=pages;
 byId('links').replaceChildren(link('단어별 전체 CSV',g.stem+'_words.csv'),document.createTextNode(' · '),link('원래 형식 100행 미리보기',g.stem+'_preview.csv'));
 const list=document.createElement('ul');for(const p of g.parts){const li=document.createElement('li');li.appendChild(link(p.file.split('/').pop(),p.file));li.appendChild(document.createTextNode(' · '+p.rows.toLocaleString()+'행 · '+(p.bytes/1e6).toFixed(2)+'MB'));list.appendChild(li)}byId('parts').replaceChildren(list)}
select.onchange=()=>{page=0;render()};byId('search').oninput=()=>{page=0;render()};byId('prev').onclick=()=>{page--;render()};byId('next').onclick=()=>{page++;render()};
byId('download').onclick=()=>{const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const records=[fields,...filtered().map(r=>fields.map(k=>r[k]))];const csv='\\ufeff'+records.map(r=>r.map(quote).join(',')).join('\\r\\n')+'\\r\\n';const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=link('CSV',url);a.download=current().stem+'_filtered_words.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();
</script></html>'''


def main():
    workspace = Path(__file__).resolve().parent.parent
    base = workspace / 'rewriting/val_domain_analysis'
    source_dir = base / 'diagram_synonym_pairs'
    out = workspace / 'tvr_ABC_csv'
    out.mkdir(exist_ok=True)
    audit = json.loads((base / 'diagram_region_audit/context_review/final_audit/summary.json').read_text())
    summary_path = base / 'diagram_region_audit/context_review/final_region_summary.csv'
    with summary_path.open(encoding='utf-8-sig', newline='') as handle:
        expected = {(r['aspect'], r['split'], r['region']): r for r in csv.DictReader(handle)
                    if r['region'] in {'A', 'B', 'C'}}
    term_path = base / 'diagram_region_audit/context_review/final_term_regions.csv'
    with term_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        word_fields, terms = reader.fieldnames, list(reader)
    reports, manifest, groups = [], [], []
    for aspect in ('object', 'action'):
        for region, split in [('A', 'train'), ('B', 'train'), ('B', 'val'), ('C', 'val')]:
            stem = f'{aspect}_{region}_{split}'
            source = source_dir / f'{stem}.csv'
            source_hash = sha256(source)
            assert source_hash == audit['checked_csv_sha256'][str(source)], f'Input changed: {source}'
            summary = expected[(aspect, 'validation' if split == 'val' else split, region)]
            words = sorted([r for r in terms if r['aspect'] == aspect and r['term_region'] == region
                            and r['split'] == ('validation' if split == 'val' else split)], key=lambda r: r['lemma'])
            assert len(words) == int(summary['unique_source_lemmas'])
            write_csv(out / f'{stem}_words.csv', word_fields, words)
            part_dir = out / 'parts' / stem
            part_dir.mkdir(parents=True, exist_ok=True)
            digest, count, paths, source_lemmas = hashlib.sha256(), 0, [], set()
            with source.open(encoding='utf-8-sig', newline='') as handle, (out / f'{stem}_preview.csv').open('w', encoding='utf-8-sig', newline='') as preview:
                reader = csv.reader(handle, strict=True)
                header = next(reader)
                assert len(header) == len(set(header)) == 32
                indices = {key: header.index(key) for key in ('term_type', 'term_region', 'source_lemma')}
                preview_writer = csv.writer(preview)
                preview_writer.writerow(header)
                part = None
                try:
                    for row in reader:
                        assert len(row) == len(header), f'{stem}: invalid field count at record {count+1}'
                        assert row[indices['term_type']] == aspect and row[indices['term_region']] == region
                        source_lemmas.add(row[indices['source_lemma']])
                        if count % 5000 == 0:
                            if part is not None:
                                part.close()
                            path = part_dir / f'{stem}_{len(paths)+1:03d}.csv'
                            paths.append(path)
                            part = path.open('w', encoding='utf-8-sig', newline='')
                            part_writer = csv.writer(part)
                            part_writer.writerow(header)
                        part_writer.writerow(row)
                        if count < 100:
                            preview_writer.writerow(row)
                        digest.update(row_bytes(row))
                        count += 1
                        if count % 250000 == 0:
                            print(f'{stem}: parsed {count:,} rows', flush=True)
                finally:
                    if part is not None:
                        part.close()
            assert count == int(summary['rows']) and source_lemmas == {r['lemma'] for r in words}
            restored_digest, restored_count, group_parts = hashlib.sha256(), 0, []
            for path in paths:
                start = restored_count + 1
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    reader = csv.reader(handle, strict=True)
                    assert next(reader) == header
                    for row in reader:
                        restored_digest.update(row_bytes(row))
                        restored_count += 1
                item = {'dataset': stem, 'file': str(path.relative_to(out)),
                        'source_row_start': start, 'source_row_end': restored_count,
                        'rows': restored_count-start+1, 'bytes': path.stat().st_size}
                manifest.append(item)
                group_parts.append(item)
            assert restored_count == count and restored_digest.digest() == digest.digest()
            with source.open(encoding='utf-8-sig', newline='') as handle, (out / f'{stem}_preview.csv').open(encoding='utf-8-sig', newline='') as preview:
                original, small = csv.reader(handle), csv.reader(preview)
                assert next(original) == next(small)
                assert all(row == next(original) for row in small)
            assert sha256(source) == source_hash
            reports.append({'file': str(source), 'source_sha256': source_hash, 'bytes': source.stat().st_size,
                            'rows': count, 'columns': len(header), 'unique_lemmas': len(words),
                            'csv_parse_errors': 0, 'source_hash_matches_audit': True,
                            'parts': len(paths), 'all_fields_and_row_order_verified': True,
                            'records_sha256': digest.hexdigest()})
            label = f"{aspect.title()} · {region} · {'Train' if split == 'train' else 'Val'}"
            groups.append({'stem': stem, 'label': label, 'words': words, 'source_rows': count, 'parts': group_parts})
            print(f'{stem}: {count:,} rows / {len(words):,} words / {len(paths)} parts verified', flush=True)
    write_csv(out / 'parts_manifest.csv', list(manifest[0]), manifest)
    write_csv(out / 'file_status.csv', list(reports[0]), reports)
    embedded = json.dumps({'groups': groups, 'fields': word_fields}, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    (out / 'index.html').write_text(VIEWER.replace('__DATA__', embedded), encoding='utf-8')
    metadata = {'files_checked': reports, 'csv_errors': 0, 'source_inputs_unchanged': True,
                'term_source_sha256': sha256(term_path), 'region_summary_sha256': sha256(summary_path),
                'encoding': 'UTF-8 BOM', 'compression': 'none',
                'word_summary_unit': 'One valid source lemma per aspect/split/region, with all eleven term-catalog columns',
                'preview': 'First 100 synonym-pair rows, all 32 columns preserved',
                'full_data': 'Maximum 5000 records per part; all records and order verified',
                'ui_error_status': 'Local CSV reads verified; no client-side error message received or reproduced',
                'script_sha256': sha256(Path(__file__))}
    (out / 'verification.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    lines = ['# A/B/C CSV 새 보기', '',
             '[검색 가능한 HTML 보기](index.html) — 브라우저로 열면 모든 영역의 단어와 WordNet 후보를 검색하고 검색 결과를 CSV로 저장할 수 있습니다.', '',
             '아래 숫자는 고유 기본형 수이며, 링크는 단어별 WordNet 후보 요약입니다.', '',
             '| 유형 | A · Train | B · Train | B · Val | C · Val |', '| --- | --- | --- | --- | --- |']
    for aspect in ('object', 'action'):
        items = [g for g in groups if g['stem'].startswith(aspect+'_')]
        lines.append('| '+aspect.title()+' | '+' | '.join(f"[{len(g['words']):,}]({g['stem']}_words.csv)" for g in items)+' |')
    lines += ['', '## 원래 형식 미리보기', '', '| 유형 | A · Train | B · Train | B · Val | C · Val |', '| --- | --- | --- | --- | --- |']
    for aspect in ('object', 'action'):
        items = [g for g in groups if g['stem'].startswith(aspect+'_')]
        lines.append('| '+aspect.title()+' | '+' | '.join(f"[{min(100,g['source_rows'])}행]({g['stem']}_preview.csv)" for g in items)+' |')
    lines += ['', '전체 자료는 HTML의 “전체 유의어 쌍 분할본” 또는 아래 목록에서 볼 수 있습니다. 압축하지 않았습니다.', '',
              '[전체 분할 목록 CSV](parts_manifest.csv) · [원본 파일 검사 결과](file_status.csv) · [검증 상세](verification.json)', '',
              '로컬 CSV의 존재·인코딩·행과 열·영역·고유 단어 수·감사 해시를 확인했습니다. 실제 화면의 링크 열기 오류는 오류 문구가 없어 재현하지 못했습니다.', '',
              '재생성: `python rewriting/make_ABC_csv_readable.py`']
    readable_index = '\n'.join(lines)+'\n'
    (out / 'READABLE_VIEWS.md').write_text(readable_index, encoding='utf-8')
    if not (out / 'README.md').exists():
        (out / 'README.md').write_text(readable_index, encoding='utf-8')
    print(json.dumps({'output': str(out), 'files_checked': len(reports), 'total_records': sum(r['rows'] for r in reports),
                      'parts': len(manifest), 'csv_errors': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

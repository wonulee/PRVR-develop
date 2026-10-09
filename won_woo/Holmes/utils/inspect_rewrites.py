"""Show actual rewrite candidates and progress without restarting the experiment."""
import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.holmes_ar import PROTECTED

LABELS = {
    "accepted": "채택",
    "unchanged": "원문 유지: 모델 출력이 원문과 같음",
    "temporal_or_negation_change": "제외: 시간·부정 단어가 달라짐",
    "token_budget": "제외: 토큰 제한 초과",
    "invalid_json": "제외: JSON 형식 오류",
    "invalid_text": "제외: 문장 형식 오류",
}


def read_new(stream):
    while True:
        offset = stream.tell()
        raw = stream.readline()
        if not raw:
            return
        if not raw.endswith(b"\n"):
            stream.seek(offset)
            return
        yield json.loads(raw)


def status(row):
    return "accepted" if row["accepted"] else row["reason"]


def show(counts, examples, total):
    done = sum(counts.values())
    accepted = counts["accepted"]
    unchanged = counts["unchanged"]
    excluded = done - accepted - unchanged
    progress = f"{done:,}/{total:,} ({100 * done / total:.1f}%)" if total else f"{done:,}"
    print(f"처리 {progress} | 채택 {accepted:,} | 원문 유지 {unchanged:,} | 제외 {excluded:,}")
    print(f"채택률 {100 * accepted / done if done else 0:.1f}% (처리 완료 쿼리 기준)")
    for key, count in counts.items():
        if key not in ("accepted", "unchanged"):
            print(f"  {LABELS.get(key, key)}: {count:,}")
    for row in examples:
        print(f"\n{row['id']} — {LABELS.get(status(row), status(row))}")
        print(f"  원문: {row['original']}")
        print(f"  생성: {row['candidate']}")
        if row["reason"] == "temporal_or_negation_change":
            before = PROTECTED.findall(row["original"].lower())
            after = PROTECTED.findall(row["candidate"].lower())
            print(f"  검사된 시간·부정 표현: {before} → {after}")
        print(f"  RoBERTa 토큰: {row['tokens']}")
    if not examples:
        print("표시할 생성문이 아직 없습니다.")
    print("\n채택은 자동 검사 통과를 뜻하며, 의미 보존·검색 성능 개선을 보장하지 않습니다.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=ROOT.parent / "experiments/mca_train_test/context/rewrites/audit.jsonl")
    p.add_argument("--watch", action="store_true", help="Refresh as the running job appends results")
    p.add_argument("--last", type=int, default=3, help="Number of latest accepted examples")
    p.add_argument("--id", help="Inspect one caption ID, including rejected/unchanged results")
    p.add_argument("--csv", type=Path, help="Write all completed records to a new CSV file")
    args = p.parse_args()
    if args.last < 1:
        p.error("--last must be positive")
    if args.csv and args.watch:
        p.error("Use --csv for a snapshot or --watch for live viewing")
    if not args.audit.is_file():
        p.error(f"Audit file is not available yet: {args.audit}")
    originals = args.audit.parent / "original.caption.txt"
    total = None
    if originals.exists():
        with originals.open(encoding="utf-8") as reader:
            total = sum(bool(line.strip()) for line in reader)
    counts, examples = Counter(), deque(maxlen=args.last)
    csv_stream = None
    try:
        if args.csv:
            csv_stream = args.csv.open("x", newline="", encoding="utf-8-sig")
            writer = csv.DictWriter(csv_stream, fieldnames=("id", "status", "original", "candidate", "reason", "tokens"))
            writer.writeheader()
        with args.audit.open("rb") as stream:
            first = True
            while True:
                updated = False
                for row in read_new(stream):
                    key = status(row)
                    counts[key] += 1
                    matches = row["id"] == args.id if args.id else row["accepted"]
                    if matches:
                        examples.append(row)
                    if csv_stream:
                        writer.writerow(dict(id=row["id"], status=LABELS.get(key, key),
                                             original=row["original"], candidate=row["candidate"],
                                             reason=row["reason"], tokens=row["tokens"]))
                    updated = True
                if updated or first:
                    if args.watch and sys.stdout.isatty():
                        print("\033[2J\033[H", end="")
                    show(counts, examples, total)
                    first = False
                if not args.watch:
                    break
                time.sleep(3)
    except KeyboardInterrupt:
        print("\n조회 종료 (학습/재작성 프로세스는 계속 실행됩니다).")
    finally:
        if csv_stream:
            csv_stream.close()
    if args.csv:
        print(f"비교표 저장: {args.csv}")


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    openpyxl = None

# Add root and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Validations.validations import validations, get_gt, cal_perf
from Utils.utils import set_seed, gpu

# [수정된 부분] utils.py를 수정하지 않아도 되도록 스크립트 내부에서 체크포인트를 불러오는 함수 정의
def custom_load_ckpt(ckpt_file):
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    return ckpt['config'], ckpt['state_dict'], ckpt['optimizer'], ckpt['epoch'], ckpt['model_val']


def build_excel_report(df, output_path, model_name="Holmes TVR Test Model"):
    if openpyxl is None:
        print("[Warning] openpyxl is not installed. Saving as CSV instead...")
        csv_path = output_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved to: {csv_path}")
        return

    print(f"Generating Excel report from {len(df)} records -> {output_path}...")

    for col in ['R@1_correct', 'R@5_correct', 'R@10_correct', 'R@100_correct']:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    if 'error_category' not in df.columns:
        def get_category(row):
            if row.get('R@1_correct', False):
                return '정답 (Rank 1)'
            elif row.get('R@5_correct', False):
                return '근접 오답 (Rank 2~5)'
            elif row.get('R@10_correct', False):
                return '상위 오답 (Rank 6~10)'
            elif row.get('R@100_correct', False):
                return '중위 오답 (Rank 11~100)'
            else:
                return '완전 실패 (Rank > 100)'
        df['error_category'] = df.apply(get_category, axis=1)

    df['R@1_결과'] = df['R@1_correct'].apply(lambda x: '정답 (Success)' if x else '오답 (Failure)')
    df['R@5_결과'] = df['R@5_correct'].apply(lambda x: 'O' if x else 'X')
    df['R@10_결과'] = df['R@10_결과'].apply(lambda x: 'O' if x else 'X') if 'R@10_결과' in df.columns else df['R@10_correct'].apply(lambda x: 'O' if x else 'X')
    df['R@100_결과'] = df['R@100_correct'].apply(lambda x: 'O' if x else 'X')

    total_q = len(df)
    r1_count = df['R@1_correct'].sum()
    r5_count = df['R@5_correct'].sum()
    r10_count = df['R@10_correct'].sum()
    r100_count = df['R@100_correct'].sum()

    r1_pct = (r1_count / total_q) * 100
    r5_pct = (r5_count / total_q) * 100
    r10_pct = (r10_count / total_q) * 100
    r100_pct = (r100_count / total_q) * 100
    rsum = r1_pct + r5_pct + r10_pct + r100_pct

    cat_counts = df['error_category'].value_counts()

    wb = openpyxl.Workbook()
    ws_summary = wb.active
    ws_summary.title = "1. 대시보드 요약"

    navy_header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    navy_sub_fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    gray_header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    white_bold_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=16, bold=True, color="1F4E79")
    subtitle_font = Font(name="Calibri", size=11, italic=True, color="595959")
    regular_font = Font(name="Calibri", size=10)
    bold_regular_font = Font(name="Calibri", size=10, bold=True)

    success_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    failure_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    near_miss_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    ws_summary.views.sheetView[0].showGridLines = True
    ws_summary["A1"] = "Partially-Relevant Video Retrieval (PRVR) 평가 결과 대시보드"
    ws_summary["A1"].font = title_font
    ws_summary["A2"] = f"모델/체크포인트: {model_name} | 총 테스트 쿼리 수: {total_q:,}개"
    ws_summary["A2"].font = subtitle_font

    ws_summary["A4"] = "주요 성능 지표 (Retrieval Performance)"
    ws_summary["A4"].font = Font(name="Calibri", size=12, bold=True, color="1F4E79")

    metrics_headers = ["Metric", "정답 수 (Correct Count)", "비율 (Recall %)", "평가 의미"]
    for col_num, h in enumerate(metrics_headers, 1):
        cell = ws_summary.cell(row=5, column=col_num)
        cell.value = h
        cell.fill = navy_header_fill
        cell.font = white_bold_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    metrics_rows = [
        ("R@1", f"{r1_count:,} / {total_q:,}", f"{r1_pct:.2f}%", "1위로 정답 비디오를 정확히 찾아낸 비율 (Strict Top-1)"),
        ("R@5", f"{r5_count:,} / {total_q:,}", f"{r5_pct:.2f}%", "Top-5 내에 정답 비디오가 포함된 비율"),
        ("R@10", f"{r10_count:,} / {total_q:,}", f"{r10_pct:.2f}%", "Top-10 내에 정답 비디오가 포함된 비율"),
        ("R@100", f"{r100_count:,} / {total_q:,}", f"{r100_pct:.2f}%", "Top-100 내에 정답 비디오가 포함된 비율"),
        ("Rsum", "-", f"{rsum:.2f}", "R@1 + R@5 + R@10 + R@100 총합 (핵심 비교 지표)")
    ]

    for r_idx, row_data in enumerate(metrics_rows, 6):
        for c_idx, val in enumerate(row_data, 1):
            cell = ws_summary.cell(row=r_idx, column=c_idx)
            cell.value = val
            cell.border = thin_border
            cell.font = bold_regular_font if (r_idx == 6 or r_idx == 10) else regular_font
            cell.alignment = Alignment(horizontal="center" if c_idx in [1, 3] else ("right" if c_idx == 2 else "left"), vertical="center")
            if r_idx == 6:
                cell.fill = success_fill
            elif r_idx == 10:
                cell.fill = gray_header_fill

    ws_summary["A12"] = "결과 상세 구간 분포 (Category Breakdown)"
    ws_summary["A12"].font = Font(name="Calibri", size=12, bold=True, color="1F4E79")

    cat_headers = ["구분 (Category)", "쿼리 수", "비율 (%)", "연구적 가치 및 분석 방향 (교수님 조언 연계)"]
    for col_num, h in enumerate(cat_headers, 1):
        cell = ws_summary.cell(row=13, column=col_num)
        cell.value = h
        cell.fill = navy_sub_fill
        cell.font = white_bold_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    tier_order = [
        ("정답 (Rank 1)", cat_counts.get("정답 (Rank 1)", 0), "현재 모델이 확실하게 맞춘 쿼리. 쿼리 길이 및 키워드 특성 파악용."),
        ("근접 오답 (Rank 2~5)", cat_counts.get("근접 오답 (Rank 2~5)", 0), "★ 1순위 타겟! 마진이 근소하여 디테일 보강(Selective Rewriting) 시 즉시 정답 전환 가능성 높음."),
        ("상위 오답 (Rank 6~10)", cat_counts.get("상위 오답 (Rank 6~10)", 0), "주요 동사/객체는 매칭되었으나 시간적 순서(Temporal Order) 또는 배경 정보 차이로 밀린 샘플."),
        ("중위 오답 (Rank 11~100)", cat_counts.get("중위 오답 (Rank 11~100)", 0), "모호한 서술로 다른 유사 비디오들과 혼동된 쿼리. 시간 부사(before/after) 보강 필요."),
        ("완전 실패 (Rank > 100)", cat_counts.get("완전 실패 (Rank > 100)", 0), "도메인 불일치 또는 심각한 텍스트-비디오 간 괴리. 리라이팅 시 환각 발생 여부 중점 점검 대상.")
    ]

    for r_idx, (tier_name, count, desc) in enumerate(tier_order, 14):
        pct = (count / total_q) * 100
        row_vals = [tier_name, f"{count:,}", f"{pct:.2f}%", desc]
        for c_idx, val in enumerate(row_vals, 1):
            cell = ws_summary.cell(row=r_idx, column=c_idx)
            cell.value = val
            cell.border = thin_border
            cell.font = bold_regular_font if c_idx == 1 else regular_font
            cell.alignment = Alignment(horizontal="center" if c_idx in [1, 3] else ("right" if c_idx == 2 else "left"), vertical="center")
            if "정답" in tier_name:
                cell.fill = success_fill
            elif "근접 오답" in tier_name:
                cell.fill = near_miss_fill

    ws_summary["A20"] = "💡 교수님 지도 방향 기반 오답 분석 팁 (Qualitative Analysis Guide):"
    ws_summary["A20"].font = Font(name="Calibri", size=11, bold=True, color="C00000")
    guidance = [
        "1. [5. 근접 오답 (Rank 2~5)] 시트를 먼저 확인하여, 쿼리의 어떤 부분이 아쉬워서 1위를 놓쳤는지 비디오와 대조해 보세요.",
        "2. [3. 오답 목록] 시트에서 리라이팅 적용 전/후의 순위 변동을 비교하고, 왜 쿼리 리라이팅 후 순위가 하락했는지 확인하세요.",
        "3. 모든 쿼리를 일괄 변환하지 말고, 모호하거나 짧은 쿼리만 골라내는 '선별 규칙(Adaptive Rule)'을 수립할 때 이 데이터가 기준이 됩니다."
    ]
    for idx, line in enumerate(guidance, 21):
        ws_summary[f"A{idx}"] = line
        ws_summary[f"A{idx}"].font = regular_font

    ws_summary.column_dimensions['A'].width = 28
    ws_summary.column_dimensions['B'].width = 25
    ws_summary.column_dimensions['C'].width = 18
    ws_summary.column_dimensions['D'].width = 85

    display_columns = [
        ('caption_id', 'Caption ID', 35),
        ('query', 'Query (자연어 검색 쿼리)', 70),
        ('gt_video_ids', '정답 비디오 ID (Ground Truth)', 30),
        ('predicted_video_id', '예측 비디오 ID (Top-1 Pred)', 30),
        ('gt_rank', '정답 순위 (Rank)', 16),
        ('R@1_결과', 'R@1 결과', 16),
        ('R@5_결과', 'R@5', 10),
        ('R@10_결과', 'R@10', 10),
        ('R@100_결과', 'R@100', 10),
        ('error_category', '오답 상세 분류', 22),
        ('top5_predictions', 'Top-5 예측 비디오', 50)
    ]

    avail_cols = [col_info for col_info in display_columns if col_info[0] in df.columns]

    def create_data_sheet(sheet_title, filtered_df):
        ws = wb.create_sheet(title=sheet_title)
        ws.views.sheetView[0].showGridLines = True

        ws.cell(row=1, column=1, value="No.").fill = navy_header_fill
        ws.cell(row=1, column=1).font = white_bold_font
        ws.cell(row=1, column=1).alignment = Alignment(horizontal="center", vertical="center")

        for col_idx, (col_id, col_name, width) in enumerate(avail_cols, 2):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.fill = navy_header_fill
            cell.font = white_bold_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for r_idx, (_, row) in enumerate(filtered_df.iterrows(), 2):
            no_cell = ws.cell(row=r_idx, column=1, value=r_idx - 1)
            no_cell.alignment = Alignment(horizontal="center", vertical="center")
            no_cell.font = regular_font
            no_cell.border = thin_border

            is_correct = row.get('R@1_correct', False)
            row_fill = success_fill if is_correct else failure_fill

            for c_idx, (col_id, _, _) in enumerate(avail_cols, 2):
                val = row[col_id]
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.font = regular_font
                cell.border = thin_border

                if col_id in ['R@1_결과', 'R@5_결과', 'R@10_결과', 'R@100_결과', 'gt_rank', 'error_category']:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif col_id == 'query':
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

                if col_id == 'R@1_결과':
                    cell.fill = row_fill
                    cell.font = bold_regular_font
                elif col_id == 'error_category' and '근접 오답' in str(val):
                    cell.fill = near_miss_fill

        max_col_letter = get_column_letter(len(avail_cols) + 1)
        ws.auto_filter.ref = f"A1:{max_col_letter}{len(filtered_df) + 1}"

        ws.column_dimensions['A'].width = 8
        for c_idx, (_, _, width) in enumerate(avail_cols, 2):
            col_letter = get_column_letter(c_idx)
            ws.column_dimensions[col_letter].width = width

    print("Writing Sheet 2: 전체 결과...")
    create_data_sheet("2. 전체 결과", df)

    print("Writing Sheet 3: 오답 목록...")
    df_failed = df[df['R@1_correct'] == False]
    if 'gt_rank' in df_failed.columns:
        df_failed = df_failed.sort_values(by='gt_rank', ascending=True)
    create_data_sheet("3. 오답 목록 (R@1 실패)", df_failed)

    print("Writing Sheet 4: 정답 목록...")
    df_success = df[df['R@1_correct'] == True]
    create_data_sheet("4. 정답 목록 (R@1 성공)", df_success)

    print("Writing Sheet 5: 근접 오답...")
    df_near = df[df['error_category'] == '근접 오답 (Rank 2~5)']
    create_data_sheet("5. 근접 오답 (Rank 2~5)", df_near)

    wb.save(output_path)
    print(f"Successfully saved Excel report to {output_path}!")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and export per-query results to Excel")
    parser.add_argument('-d', '--dataset_name', default='tvr', type=str, choices=['tvr', 'act', 'cha'], help='Dataset name')
    parser.add_argument('--resume', required=True, type=str, help='Path to checkpoint file (.ckpt)')
    parser.add_argument('--split', default='test', choices=['test', 'val'], help='Split to evaluate')
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')
    parser.add_argument('--output_excel', default=None, type=str, help='Output path for Excel (.xlsx)')
    parser.add_argument('--output_csv', default=None, type=str, help='Output path for CSV (.csv)')
    return parser.parse_args()

@torch.no_grad()
def evaluate_and_extract_records(model, context_loader, query_loader, cfg):
    model.eval()
    val_module = validations(cfg)

    print("\n[1/3] Computing video context representations...")
    context_info = val_module.compute_context_info(model, context_loader)

    print("\n[2/3] Computing query embeddings and similarity scores...")
    score_sum, query_metas = val_module.compute_query2ctx_info(model, query_loader, context_info)

    video_metas = context_info['video_metas']
    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

    print("\n[3/3] Calculating retrieval performance and ranking...")
    r1, r5, r10, r100 = cal_perf(-1 * score_sum, t2v_gt)
    rsum = r1 + r5 + r10 + r100
    print(f"\n=======================================================")
    print(f"Overall Retrieval Metrics ({cfg.get('dataset_name', 'dataset')}):")
    print(f"R@1: {r1:.2f}% | R@5: {r5:.2f}% | R@10: {r10:.2f}% | R@100: {r100:.2f}% | Rsum: {rsum:.2f}")
    print(f"=======================================================\n")

    n_q, n_m = score_sum.shape
    records = []

    caption_map = getattr(query_loader.dataset, 'captions', {})

    for i in tqdm(range(n_q), desc="Processing detailed query results"):
        cap_id = query_metas[i]
        query_text = caption_map.get(cap_id, "")

        s = score_sum[i]
        sorted_idxs = torch.argsort(s, descending=True)

        gt_indices = t2v_gt.get(i, [])
        gt_vids = [video_metas[k] for k in gt_indices]

        rank = n_m + 1
        for k in gt_indices:
            pos_matches = (sorted_idxs == k).nonzero(as_tuple=True)[0]
            if len(pos_matches) > 0:
                cur_rank = pos_matches[0].item() + 1
                if cur_rank < rank:
                    rank = cur_rank

        top1_idx = sorted_idxs[0].item()
        pred_top1_vid = video_metas[top1_idx]
        top5_vids = [video_metas[sorted_idxs[j].item()] for j in range(min(5, n_m))]

        is_r1 = bool(rank <= 1)
        is_r5 = bool(rank <= 5)
        is_r10 = bool(rank <= 10)
        is_r100 = bool(rank <= 100)

        if is_r1:
            category = "정답 (Rank 1)"
        elif is_r5:
            category = "근접 오답 (Rank 2~5)"
        elif is_r10:
            category = "상위 오답 (Rank 6~10)"
        elif is_r100:
            category = "중위 오답 (Rank 11~100)"
        else:
            category = "완전 실패 (Rank > 100)"

        records.append({
            'caption_id': cap_id,
            'query': query_text,
            'gt_video_ids': ", ".join(gt_vids),
            'predicted_video_id': pred_top1_vid,
            'gt_rank': rank,
            'R@1_correct': is_r1,
            'R@5_correct': is_r5,
            'R@10_correct': is_r10,
            'R@100_correct': is_r100,
            'error_category': category,
            'top5_predictions': ", ".join(top5_vids),
            'video_id_from_caption': cap_id.split('#')[0]
        })

    return records, (r1, r5, r10, r100, rsum)


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = get_configs(args.dataset_name)
    set_seed(cfg.get('seed', 0))

    ckpt_name = os.path.splitext(os.path.basename(args.resume))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.output_excel and not args.output_csv:
        args.output_excel = f"./{args.dataset_name}_{args.split}_{ckpt_name}_{timestamp}.xlsx"

    print(f"==> Loading dataset ({args.dataset_name}, split: {args.split})...")
    cfg, train_loader, val_ctx_loader, val_q_loader, test_ctx_loader, test_q_loader = get_datasets(cfg)

    if args.split == 'test':
        ctx_loader = test_ctx_loader
        q_loader = test_q_loader
    else:
        ctx_loader = val_ctx_loader
        q_loader = val_q_loader

    print(f"==> Loading model and checkpoint: {args.resume}")
    model = get_models(cfg)
    
    # [수정된 부분] utils.py의 load_ckpt 대신 새로 정의한 custom_load_ckpt 사용
    cfg_ckpt, model_state, _, epoch, best_val = custom_load_ckpt(args.resume)
    
    model.load_state_dict(model_state)

    if torch.cuda.is_available():
        model = model.cuda()

    records, metrics = evaluate_and_extract_records(model, ctx_loader, q_loader, cfg)
    df = pd.DataFrame(records)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        print(f"==> Saved raw CSV to: {args.output_csv}")

    if args.output_excel:
        model_title = f"{args.dataset_name.upper()} - {ckpt_name} (Epoch {epoch})"
        build_excel_report(df, args.output_excel, model_name=model_title)
        print(f"==> Saved formatted Excel to: {args.output_excel}")

    print("\n[Done] All tasks completed successfully!")


if __name__ == '__main__':
    main()
import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add multisrc and parent paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
multisrc_dir = os.path.join(parent_dir, 'multisrc')
sys.path.insert(0, multisrc_dir)
sys.path.insert(0, parent_dir)

from Configs.builder import get_configs
from Models.builder import get_models
from Validations.validations import validations, get_gt, cal_perf
from Datasets.data_provider import VisDataSet4PRVR, TxtDataSet4PRVR, collate_frame_val, collate_text_val, read_video_ids
from Utils.basic_utils import read_dict
from Utils.utils import set_seed, gpu

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Holmes on tvrval_rewritten_3.txt")
    parser.add_argument("--ckpt", default="/home/viglab2/won_woo/Holmes_ICML/tvr-multi/tvr/20260918-171248(rewriting3)/best.ckpt", help="Rewriting3 checkpoint path")
    parser.add_argument("--baseline-ckpt", default="/home/viglab2/won_woo/Holmes_ICML/tvr-multi/tvr/20260914-230815(org)/best.ckpt", help="Baseline checkpoint path")
    parser.add_argument("--caption-file", default="/home/viglab2/won_woo/tvr/TextData/tvrval_rewritten_3.txt", help="Validation caption file with rewritten queries")
    parser.add_argument("--feat-file", default="/home/viglab2/won_woo/tvr/TextData/roberta_tvr_val_feat_rewriting3.hdf5", help="Validation RoBERTa feature file")
    parser.add_argument("--device", default="cuda:1", help="Device to run evaluation on")
    parser.add_argument("--query-bsz", type=int, default=64, help="Query batch size")
    parser.add_argument("--context-bsz", type=int, default=100, help="Context batch size")
    parser.add_argument("--eval-baseline", action="store_true", default=True, help="Also evaluate baseline checkpoint for comparison")
    return parser.parse_args()

def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt['config'], ckpt['state_dict'], ckpt.get('epoch', -1), ckpt.get('model_val', None)

def evaluate_model_on_subsets(model, val_criterion, context_info, query_loader, cfg):
    """
    Computes scores for all queries, then partitions into:
    1) All queries (Total)
    2) Original queries (#enc#0 ~ #enc#4)
    3) Rewritten queries (#enc#5 ~ #enc#N)
    """
    model.eval()
    with torch.no_grad():
        score_sum, query_metas = val_criterion.compute_query2ctx_info(model, query_loader, context_info)
        video_metas = context_info['video_metas']

        # Determine subsets
        orig_indices = []
        rewr_indices = []
        for idx, q_id in enumerate(query_metas):
            try:
                enc_idx = int(q_id.split('#enc#')[1])
                if enc_idx < 5:
                    orig_indices.append(idx)
                else:
                    rewr_indices.append(idx)
            except Exception:
                orig_indices.append(idx)

        # 1. All queries
        v2t_gt_all, t2v_gt_all = get_gt(video_metas, query_metas)
        r1_all, r5_all, r10_all, r100_all = cal_perf(-1 * score_sum, t2v_gt_all)
        rsum_all = r1_all + r5_all + r10_all + r100_all

        # 2. Original queries only
        score_orig = score_sum[orig_indices]
        q_metas_orig = [query_metas[i] for i in orig_indices]
        _, t2v_gt_orig = get_gt(video_metas, q_metas_orig)
        r1_orig, r5_orig, r10_orig, r100_orig = cal_perf(-1 * score_orig, t2v_gt_orig)
        rsum_orig = r1_orig + r5_orig + r10_orig + r100_orig

        # 3. Rewritten queries only
        score_rewr = score_sum[rewr_indices]
        q_metas_rewr = [query_metas[i] for i in rewr_indices]
        _, t2v_gt_rewr = get_gt(video_metas, q_metas_rewr)
        r1_rewr, r5_rewr, r10_rewr, r100_rewr = cal_perf(-1 * score_rewr, t2v_gt_rewr)
        rsum_rewr = r1_rewr + r5_rewr + r10_rewr + r100_rewr

    return {
        "all": {"n": len(query_metas), "R@1": r1_all, "R@5": r5_all, "R@10": r10_all, "R@100": r100_all, "Rsum": rsum_all},
        "orig": {"n": len(orig_indices), "R@1": r1_orig, "R@5": r5_orig, "R@10": r10_orig, "R@100": r100_orig, "Rsum": rsum_orig},
        "rewr": {"n": len(rewr_indices), "R@1": r1_rewr, "R@5": r5_rewr, "R@10": r10_rewr, "R@100": r100_rewr, "Rsum": rsum_rewr},
    }

def main():
    args = parse_args()

    # Use CUDA device as set by CUDA_VISIBLE_DEVICES or default
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    cfg = get_configs("tvr")
    set_seed(cfg.get('seed', 0))

    cfg['eval_query_bsz'] = args.query_bsz
    cfg['eval_context_bsz'] = args.context_bsz

    print(f"Loading visual features for validation videos...")
    rootpath = cfg['data_root']
    collection = cfg['collection']
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'feature.hdf5')
    video2frames = read_dict(os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))
    
    # 2,179 val video IDs
    val_video_ids = read_video_ids(os.path.join(rootpath, collection, 'TextData', 'tvrval.caption.txt'))
    print(f"Total validation videos: {len(val_video_ids)}")

    val_video_dataset = VisDataSet4PRVR(visual_feat_path, video2frames, cfg, video_ids=val_video_ids)
    context_dataloader = DataLoader(
        val_video_dataset,
        collate_fn=collate_frame_val,
        batch_size=cfg['eval_context_bsz'],
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    print(f"Loading queries from {args.caption_file} and features from {args.feat_file}...")
    val_text_dataset = TxtDataSet4PRVR(args.caption_file, args.feat_file, cfg)
    print(f"Total queries to evaluate: {len(val_text_dataset)}")

    query_eval_loader = DataLoader(
        val_text_dataset,
        collate_fn=collate_text_val,
        batch_size=cfg['eval_query_bsz'],
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    val_criterion = validations(cfg)

    # Models to evaluate
    models_to_eval = [("Rewriting3 Model", args.ckpt)]
    if args.eval_baseline and os.path.exists(args.baseline_ckpt):
        models_to_eval.append(("Baseline Model (org)", args.baseline_ckpt))

    results = {}

    for model_name, ckpt_path in models_to_eval:
        print(f"\n=======================================================")
        print(f"Evaluating {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"=======================================================")
        
        cfg_ckpt, state_dict, epoch, best_val = load_checkpoint(ckpt_path)
        print(f"Loaded checkpoint from Epoch {epoch}")

        model = get_models(cfg)
        model.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()

        with torch.no_grad():
            print("Encoding video contexts...")
            context_info = val_criterion.compute_context_info(model, context_dataloader)

            print("Computing query embeddings & evaluating retrieval scores...")
            res = evaluate_model_on_subsets(model, val_criterion, context_info, query_eval_loader, cfg)
            results[model_name] = res

        print(f"\n--- {model_name} Results ---")
        for subset, metrics in res.items():
            print(f"[{subset.upper()}] (N={metrics['n']}) R@1: {metrics['R@1']:.2f}% | R@5: {metrics['R@5']:.2f}% | R@10: {metrics['R@10']:.2f}% | R@100: {metrics['R@100']:.2f}% | Rsum: {metrics['Rsum']:.2f}")

        # Clean up GPU memory
        del model
        del context_info
        torch.cuda.empty_cache()

    # Print Final Summary Comparison Table
    print("\n\n" + "="*80)
    print("FINAL RETRIEVAL PERFORMANCE ON tvrval_rewritten_3.txt")
    print("="*80)
    header = f"{'Model':<25} | {'Subset':<10} | {'Count':<7} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'R@100':<7} | {'Rsum':<7}"
    print(header)
    print("-" * len(header))
    for m_name, res in results.items():
        for s_name in ['all', 'orig', 'rewr']:
            m = res[s_name]
            label = "Total (All)" if s_name == 'all' else ("Original" if s_name == 'orig' else "Rewritten")
            print(f"{m_name:<25} | {label:<10} | {m['n']:<7} | {m['R@1']:<7.2f} | {m['R@5']:<7.2f} | {m['R@10']:<7.2f} | {m['R@100']:<7.2f} | {m['Rsum']:<7.2f}")
        print("-" * len(header))

    # Save summary json
    output_json = "/home/viglab2/won_woo/Holmes_ICML/tvr-multi/tvr/val_rewriting3_evaluation_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {output_json}")

if __name__ == '__main__':
    main()

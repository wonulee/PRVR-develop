"""Use the native Holmes epoch, validation, loss and optimizer functions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

from utils.holmes_ar import positive
from .evaluate import ROOT


def load_checkpoint(path):
    import torch
    from Opts.optimization import WarmupLinearSchedule
    with torch.serialization.safe_globals([WarmupLinearSchedule]):
        return torch.load(path, map_location="cpu", weights_only=True)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def rng_state():
    import numpy as np
    import torch
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def native_entry(backbone, gpu):
    path = ROOT / backbone / "main.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("holmes_native_training", path)
    entry = importlib.util.module_from_spec(spec)
    previous = sys.argv
    try:
        sys.argv = [str(path), "-d", "tvr", "--gpu", gpu]
        spec.loader.exec_module(entry)
    finally:
        sys.argv = previous
    return entry


def run(args):
    # Native Holmes uses .cuda(); expose exactly one requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    entry = native_entry(args.backbone, args.gpu)
    from .evaluate import check_features
    from utils.holmes_ar import read_captions
    if not torch.cuda.is_available():
        raise RuntimeError("Holmes training requires CUDA")
    saved = None
    last = args.output / "last.ckpt"
    best = args.output / "best.ckpt"
    if args.mode == "test":
        saved = load_checkpoint(best)
    elif last.exists():
        saved = load_checkpoint(last)
    cfg = dict(saved["config"] if saved else entry.get_configs("tvr"))
    cfg.update(data_root=str(args.data_root.resolve()), model_root=str(args.output.resolve()),
               root=str(args.output.resolve()), seed=args.seed, batchsize=args.batch_size,
               num_workers=0, pin_memory=True, eval_query_bsz=args.query_batch,
               eval_context_bsz=args.context_batch)
    if args.epochs is not None:
        cfg["n_epoch"] = args.epochs
    if saved and args.mode == "train":
        previous = saved["config"]
        for key in ("n_epoch", "batchsize", "seed", "data_root"):
            if cfg[key] != previous[key]:
                raise ValueError(f"Resume configuration changed: {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    text = args.data_root / "tvr/TextData"
    rows = {split: read_captions(text / f"tvr{split}.caption.txt") for split in ("train", "val", "test")}
    videos = {split: {cid.split("#")[0] for cid in captions} for split, captions in rows.items()}
    if videos["train"] & (videos["val"] | videos["test"]):
        raise ValueError("Training videos overlap with val/test")
    if len(videos["train"]) < 2 or args.batch_size < 2 or len(videos["train"]) % args.batch_size == 1:
        raise ValueError("Contrastive training needs at least two videos in every batch; adjust --batch-size")
    all_ids = set().union(*(captions.keys() for captions in rows.values()))
    check_features(text / "roberta_tvr_query_feat.hdf5", all_ids, cfg["q_feat_size"])
    split_info = dict(val_test_identical=rows["val"] == rows["test"],
                      val_test_shared_videos=len(videos["val"] & videos["test"]),
                      independent_test=not bool(videos["val"] & videos["test"]))
    entry.set_seed(args.seed)
    logger = entry.set_log(str(args.output), "log.txt")
    logger.info("Holmes %s | backbone=%s | GPU=%s | seed=%d",
                args.mode, args.backbone, args.gpu, args.seed)
    logger.info("Epoch budget: %d | batch size: %d videos", cfg["n_epoch"], cfg["batchsize"])
    logger.info("Checkpoint directory: %s", args.output)
    if saved:
        logger.info("%s: %s (epoch %d)", "Testing from" if args.mode == "test" else "Resume from",
                    best if args.mode == "test" else last, saved["epoch"])
    logger.info("Split information: %s", split_info)
    logger.info("Loading Data ......")
    cfg, train, val_context, val_query, test_context, test_query = entry.get_datasets(cfg)
    logger.info("Loading Model ......")
    model = entry.get_models(cfg).cuda()
    logger.info("Total Parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("Trainable Parameters: %.2fM",
                sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    if saved:
        model.load_state_dict({k.removeprefix("module."): v for k, v in saved["state_dict"].items()})
    validation = entry.get_validations(cfg)
    if args.mode == "test":
        with torch.inference_mode():
            metrics = validation(model, test_context, test_query)
        result = dict(checkpoint=str(best), best_epoch=saved["epoch"],
                      best_validation=saved["model_val"][:5], test=metrics[:5], **split_info)
        atomic_json(args.output / "test_metrics.json", result)
        logger.info("Test results | best epoch: %d", saved["epoch"])
        for name, value in zip(("R@1", "R@5", "R@10", "R@100", "Rsum"), metrics[:5]):
            logger.info("%s: %.2f", name, value)
        logger.info("Independent test: %s", split_info["independent_test"])
        logger.info("Saved test metrics: %s", args.output / "test_metrics.json")
        return

    criterion = entry.get_losses(cfg)
    optimizer = entry.get_opts(cfg, model, train)
    epoch = saved["epoch"] if saved else -1
    best_val = saved["model_val"] if saved else [0., 0., 0., 0., -1., None]
    es_count = saved.get("es_count", 0) if saved else 0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        if "rng" in saved:
            restore_rng(saved["rng"])
    stopped = cfg["max_es_cnt"] != -1 and es_count > cfg["max_es_cnt"]
    for epoch in range(epoch + 1, cfg["n_epoch"]) if not stopped else []:
        average_loss = entry.train_one_epoch(epoch, train, model, criterion, cfg, optimizer)
        with torch.no_grad():
            metrics, best_val, failed = entry.val_one_epoch(
                epoch, val_context, val_query, model, validation, cfg, optimizer,
                best_val, average_loss, logger,
            )
        es_count = es_count + 1 if failed else 0
        state = dict(config=cfg, epoch=epoch, model_val=best_val, state_dict=model.state_dict(),
                     optimizer=optimizer.state_dict(), es_count=es_count, rng=rng_state())
        temporary = last.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(last)
        atomic_json(args.output / "progress.json", dict(epoch=epoch, best_validation=best_val[:5],
                                                        loss=average_loss, es_count=es_count))
        stopped = cfg["max_es_cnt"] != -1 and es_count > cfg["max_es_cnt"]
        if stopped:
            logger.info("Early Stop !!!")
            break
    if not best.exists():
        raise RuntimeError("Training finished without a validation-selected best.ckpt")
    atomic_json(args.output / "train_complete.json", dict(
        last_epoch=epoch, best_checkpoint=str(best), early_stopped=stopped,
        train_queries=len(rows["train"]), train_videos=len(videos["train"]), **split_info,
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train", "test"), default="train")
    p.add_argument("--backbone", choices=("src", "multisrc"), default="multisrc")
    p.add_argument("--data-root", type=Path, required=True, help="Prepared dataset parent")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", default="1")
    p.add_argument("--epochs", type=positive, help="Defaults to native Holmes: src=100, multisrc=150")
    p.add_argument("--batch-size", type=positive, default=128)
    p.add_argument("--query-batch", type=positive, default=16)
    p.add_argument("--context-batch", type=positive, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.gpu.isdigit():
        p.error("--gpu must be one physical GPU index")
    run(args)


if __name__ == "__main__":
    main()

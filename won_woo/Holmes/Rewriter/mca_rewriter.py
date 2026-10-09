"""Run from Holmes: python -m Rewriter.mca_rewriter --help."""

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from utils.holmes_ar import positive, read_captions, write_captions
from .context_manager import build_video_context_db, get_sibling_queries, split_caption_id
from .linguistic_rules import detect_linguistic_flags, rejection
from .prompt_templates import build_messages


class RewriteRejected(ValueError):
    def __init__(self, reason, candidate):
        super().__init__(reason)
        self.reason = reason
        self.candidate = candidate


def validate_length(rewritten_q, tokenizer, max_tokens=30):
    if max_tokens < 3:
        raise ValueError("max_tokens must be at least 3, including special tokens")
    return isinstance(rewritten_q, str) and bool(rewritten_q.strip()) and len(
        tokenizer.encode(rewritten_q, add_special_tokens=True)
    ) <= max_tokens


def rewrite_with_context(target_q, sibling_qs, llm_pipe, tokenizer, max_tokens=30):
    """Return an accepted string or raise RewriteRejected; never truncate text.

    tokenizer is the feature extractor's RoBERTa tokenizer, not the LLM's.
    An over-budget result receives one compression attempt.
    Generation/runtime errors propagate, rather than masquerading as rejections.
    """
    if not isinstance(target_q, str) or not target_q.strip():
        raise ValueError("target_q must be nonempty text")
    sibling_qs = list(sibling_qs)
    if any(not isinstance(q, str) or not q.strip() for q in sibling_qs):
        raise ValueError("sibling_qs must contain nonempty text")
    validate_length(target_q, tokenizer, max_tokens)
    messages = build_messages(target_q, sibling_qs, max_tokens)
    for attempt in range(2):
        prompt = llm_pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        response = llm_pipe(
            prompt, max_new_tokens=128, do_sample=False, return_full_text=False,
        )[0]["generated_text"]
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RewriteRejected("invalid_json", response) from exc
        candidate = payload.get("query") if isinstance(payload, dict) else None
        if isinstance(candidate, str) and any(c in candidate for c in "\r\u0085\u2028\u2029"):
            raise RewriteRejected("invalid_text", candidate)
        reason = rejection(target_q, candidate, tokenizer, max_tokens)
        if reason is None:
            return candidate.strip()
        if reason != "token_budget" or attempt == 1:
            raise RewriteRejected(reason, candidate)
        messages.extend([
            {"role": "assistant", "content": response},
            {"role": "user", "content": (
                f"Compress to at most {max_tokens} RoBERTa tokens including special tokens. "
                "Preserve meaning and all temporal and negation words. Return JSON only."
            )},
        ])


def rewrite_batch_with_context(queries, llm_pipe, tokenizer, max_tokens=30, batch_size=1):
    """Generate a batch; retry only over-budget queries and retain input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    messages = []
    for target_q, sibling_qs in queries:
        if not isinstance(target_q, str) or not target_q.strip():
            raise ValueError("target_q must be nonempty text")
        sibling_qs = list(sibling_qs)
        if any(not isinstance(q, str) or not q.strip() for q in sibling_qs):
            raise ValueError("sibling_qs must contain nonempty text")
        validate_length(target_q, tokenizer, max_tokens)
        messages.append(build_messages(target_q, sibling_qs, max_tokens))

    results = [None] * len(messages)
    active = list(range(len(messages)))
    for attempt in range(2):
        if not active:
            break
        prompts = [llm_pipe.tokenizer.apply_chat_template(
            messages[i], tokenize=False, add_generation_prompt=True,
        ) for i in active]
        responses = llm_pipe(
            prompts, batch_size=batch_size, max_new_tokens=128,
            do_sample=False, return_full_text=False,
        )
        if len(responses) != len(active):
            raise ValueError("LLM returned the wrong number of batched responses")
        retry = []
        for i, generated in zip(active, responses):
            response = generated[0]["generated_text"]
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                results[i] = RewriteRejected("invalid_json", response)
                continue
            candidate = payload.get("query") if isinstance(payload, dict) else None
            if isinstance(candidate, str) and any(c in candidate for c in "\r\u0085\u2028\u2029"):
                results[i] = RewriteRejected("invalid_text", candidate)
                continue
            target_q = queries[i][0]
            reason = rejection(target_q, candidate, tokenizer, max_tokens)
            if reason is None:
                results[i] = candidate.strip()
            elif reason == "token_budget" and attempt == 0:
                messages[i].extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": (
                        f"Compress to at most {max_tokens} RoBERTa tokens including special tokens. "
                        "Preserve meaning and all temporal and negation words. Return JSON only."
                    )},
                ])
                retry.append(i)
            else:
                results[i] = RewriteRejected(reason, candidate)
        active = retry
    return results


def rewrite_file(caption_path, output, llm_pipe, tokenizer, max_tokens=30,
                 limit=None, use_siblings=True, resume=False, run_settings=None,
                 batch_size=1):
    """Write paired originals, accepted rewrites, and a rejection audit.

    Rewritten IDs are unchanged for pairing with original scores/features.
    Rejected queries are absent from rewritten.caption.txt, never silently clipped.
    """
    if max_tokens < 3 or (limit is not None and limit < 1) or batch_size < 1:
        raise ValueError("max_tokens must be >= 3, limit and batch_size must be positive")
    rows = read_captions(caption_path)
    contexts = build_video_context_db(caption_path)
    if limit is not None:
        rows = dict(list(rows.items())[:limit])
    output = Path(output)
    accepted, done = {}, set()
    if output.exists() and resume:
        if read_captions(output / "original.caption.txt") != rows:
            raise ValueError("Resume input captions differ from the original run")
        audit_path = output / "audit.jsonl"
        if audit_path.exists():
            valid_bytes = 0
            with audit_path.open("rb") as reader:
                for raw in reader:
                    if not raw.endswith(b"\n"):
                        break  # A killed process may leave one incomplete final record.
                    record = json.loads(raw)
                    cid = record["id"]
                    if cid not in rows or cid in done or record["original"] != rows[cid]:
                        raise ValueError("Invalid resume audit ID or original")
                    vid, idx = split_caption_id(cid)
                    expected = get_sibling_queries(contexts, vid, idx) if use_siblings else []
                    if record["siblings"] != expected:
                        raise ValueError("Resume sibling context changed")
                    if record["accepted"]:
                        candidate = record["candidate"]
                        if rejection(rows[cid], candidate, tokenizer, max_tokens) is not None:
                            raise ValueError("Resume candidate no longer passes validation")
                        accepted[cid] = candidate
                    done.add(cid)
                    valid_bytes += len(raw)
            with audit_path.open("r+b") as writer:
                writer.truncate(valid_bytes)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_captions(output / "original.caption.txt", rows)
    if run_settings is not None:
        (output / "settings.json").write_text(json.dumps(run_settings, indent=2), encoding="utf-8")
    with (output / "audit.jsonl").open("a", encoding="utf-8") as audit:
        pending = [item for item in rows.items() if item[0] not in done]
        with tqdm(total=len(rows), initial=len(done), desc="Rewriting queries",
                  unit="query") as progress:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                siblings_batch = []
                for cid, _ in batch:
                    video_id, index = split_caption_id(cid)
                    siblings_batch.append(
                        get_sibling_queries(contexts, video_id, index) if use_siblings else []
                    )
                if batch_size == 1:
                    try:
                        outcomes = [rewrite_with_context(
                            batch[0][1], siblings_batch[0], llm_pipe, tokenizer, max_tokens,
                        )]
                    except RewriteRejected as exc:
                        outcomes = [exc]
                else:
                    outcomes = rewrite_batch_with_context(
                        [(query, siblings) for (_, query), siblings in zip(batch, siblings_batch)],
                        llm_pipe, tokenizer, max_tokens, batch_size,
                    )
                for (cid, query), siblings, outcome in zip(batch, siblings_batch, outcomes):
                    candidate = outcome.candidate if isinstance(outcome, RewriteRejected) else outcome
                    reason = outcome.reason if isinstance(outcome, RewriteRejected) else None
                    if reason is None:
                        accepted[cid] = candidate
                    audit.write(json.dumps(dict(
                        id=cid, original=query, siblings=siblings, candidate=candidate,
                        accepted=reason is None, reason=reason,
                        linguistic_flags=detect_linguistic_flags(query),
                        tokens=len(tokenizer.encode(candidate, add_special_tokens=True))
                        if isinstance(candidate, str) else None,
                    ), ensure_ascii=False) + "\n")
                    audit.flush()
                    progress.update()
    temporary = output / "rewritten.caption.txt.tmp"
    with temporary.open("w", encoding="utf-8") as writer:
        for cid in rows:
            if cid in accepted:
                writer.write(f"{cid} {accepted[cid]}\n")
    temporary.replace(output / "rewritten.caption.txt")
    return dict(total=len(rows), accepted=len(accepted), rejected=len(rows) - len(accepted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--roberta", default="roberta-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=positive, default=30)
    parser.add_argument("--batch-size", type=positive, default=1,
                        help="LLM generation batch size; reduce if GPU memory is exhausted")
    parser.add_argument("--limit", type=positive, help="Number of targets; use all their siblings")
    parser.add_argument("--resume", action="store_true", help="Continue a matching interrupted run")
    parser.add_argument("--no-siblings", action="store_true", help="Target-only ablation")
    args = parser.parse_args()
    if args.max_tokens < 3:
        parser.error("--max-tokens must be at least 3")
    if args.output.exists() and not args.resume:
        parser.error("--output must be a new directory")
    build_video_context_db(args.caption_file)
    from .prompt_templates import SYSTEM_PROMPT
    metadata = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ("resume", "device")}
    metadata["caption_sha256"] = hashlib.sha256(args.caption_file.read_bytes()).hexdigest()
    metadata["prompt_sha256"] = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    settings = args.output / "settings.json"
    if args.output.exists():
        if not settings.is_file() or json.loads(settings.read_text()) != metadata:
            parser.error("Resume settings/input do not match; use a new output directory")

    import torch
    from transformers import AutoTokenizer, RobertaTokenizer, pipeline

    with tqdm(total=3, desc="Loading models", unit="stage") as loading:
        loading.set_postfix_str("RoBERTa tokenizer")
        tokenizer = RobertaTokenizer.from_pretrained(args.roberta, do_lower_case=False)
        loading.update()
        loading.set_postfix_str("LLM tokenizer")
        llm_tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if llm_tokenizer.pad_token_id is None:
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        loading.update()
        loading.set_postfix_str("LLM model weights")
        llm_pipe = pipeline(
            "text-generation", model=args.model, tokenizer=llm_tokenizer,
            device=args.device,
            torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        )
        loading.update()
    summary = rewrite_file(args.caption_file, args.output, llm_pipe, tokenizer,
                           args.max_tokens, args.limit, not args.no_siblings, args.resume,
                           metadata, args.batch_size)
    (args.output / "run.json").write_text(
        json.dumps(dict(**metadata, **summary), indent=2), encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()

"""Extract MS-SL/Holmes-compatible token-level RoBERTa features for TVR.

The input caption file must contain one query per line in this format:

    <caption_id> <raw query text>

The output HDF5 maps each caption ID to a float32 array with shape
``[number_of_BPE_tokens, hidden_size]``. By default, the penultimate RoBERTa
layer is saved and the leading ``<s>`` and trailing ``</s>`` tokens are
removed, matching the upstream TVRetrieval extraction procedure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caption-file",
        type=Path,
        action="append",
        required=True,
        help=(
            "Caption file (caption_id followed by query text). Repeat this "
            "option to combine train/val/test in one HDF5."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 path, typically roberta_tvr_query_feat.hdf5.",
    )
    parser.add_argument(
        "--model",
        default="roberta-base",
        help=(
            "Hugging Face model name or local roberta-base_tuned_model path. "
            "Use the original TVR-tuned checkpoint for the closest MS-SL match."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Tokenizer length including <s> and </s>, as in TVRetrieval.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-2,
        help="Hidden-state index to save; -2 is the penultimate layer.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=768,
        help="Fail if the model hidden size differs; TVR uses 768.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N, or mps.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "L2-normalize each token before saving. Leave disabled for the "
            "original MS-SL/Holmes format, whose data loader normalizes later."
        ),
    )
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="Keep <s> and </s>; disabled by default to match TVRetrieval.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after extraction succeeds.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_captions(path: Path) -> list[tuple[str, str]]:
    captions: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as reader:
        for line_number, raw_line in enumerate(reader, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                caption_id, caption = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: expected '<caption_id> <query text>'"
                ) from exc
            if caption_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate caption ID {caption_id!r}"
                )
            if not caption.strip():
                raise ValueError(f"{path}:{line_number}: query text is empty")
            seen_ids.add(caption_id)
            captions.append((caption_id, caption))

    if not captions:
        raise ValueError(f"No captions found in {path}")
    return captions


def batches(
    items: Sequence[tuple[str, str]], batch_size: int
) -> Iterator[Sequence[tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def l2_normalize(array: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    denominator = np.linalg.norm(array, axis=-1, keepdims=True) + epsilon
    return array / denominator


def extract(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length < 3:
        raise ValueError("--max-length must be at least 3")
    for caption_file in args.caption_file:
        if not caption_file.is_file():
            raise FileNotFoundError(caption_file)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it"
        )

    captions = []
    seen_ids = set()
    for caption_file in args.caption_file:
        for caption_id, caption in read_captions(caption_file):
            if caption_id in seen_ids:
                raise ValueError(
                    f"Duplicate caption ID {caption_id!r} across input files. "
                    "Every transformed query needs a unique HDF5 key."
                )
            seen_ids.add(caption_id)
            captions.append((caption_id, caption))
    device = resolve_device(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = RobertaForMaskedLM.from_pretrained(args.model)
    model.config.output_hidden_states = True
    model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    if hidden_size != args.expected_dim:
        raise ValueError(
            f"Model hidden size is {hidden_size}, expected {args.expected_dim}. "
            "TVR requires RoBERTa-base (768 dimensions)."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    print(
        f"Extracting {len(captions)} queries with {args.model!r} on {device}; "
        f"layer={args.layer}, max_length={args.max_length}"
    )

    try:
        with h5py.File(temporary_output, "w") as output_file:
            output_file.attrs["model"] = str(args.model)
            output_file.attrs["hidden_state_layer"] = args.layer
            output_file.attrs["max_length_including_special_tokens"] = args.max_length
            output_file.attrs["special_tokens_removed"] = not args.keep_special_tokens
            output_file.attrs["l2_normalized"] = args.normalize

            completed = 0
            for batch in batches(captions, args.batch_size):
                caption_ids = [item[0] for item in batch]
                texts = [item[1] for item in batch]
                encoded = tokenizer(
                    texts,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.inference_mode():
                    outputs = model.roberta(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    feature_batch = outputs.hidden_states[args.layer]

                valid_lengths = attention_mask.sum(dim=1).cpu().tolist()
                feature_batch = feature_batch.float().cpu().numpy()

                for caption_id, feature, valid_length in zip(
                    caption_ids, feature_batch, valid_lengths
                ):
                    start = 0 if args.keep_special_tokens else 1
                    end = int(valid_length) if args.keep_special_tokens else int(valid_length) - 1
                    token_feature = feature[start:end].astype(np.float32, copy=False)
                    if args.normalize:
                        token_feature = l2_normalize(token_feature).astype(
                            np.float32, copy=False
                        )
                    output_file.create_dataset(caption_id, data=token_feature)

                completed += len(batch)
                print(f"\rSaved {completed}/{len(captions)} queries", end="", flush=True)

        os.replace(temporary_output, args.output)
    except BaseException:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(f"\nDone: {args.output}")


def main() -> None:
    extract(parse_args())


if __name__ == "__main__":
    main()



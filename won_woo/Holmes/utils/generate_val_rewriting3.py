import os
import re
import argparse
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Multi-Caption Weaving for TVR validation captions")
    parser.add_argument("--caption-file", default="/home/viglab2/won_woo/tvr/TextData/tvrval.caption.txt", help="Input validation caption file")
    parser.add_argument("--grouped-output", default="/home/viglab2/won_woo/tvr/TextData/tvrval_rewritten_grouped.txt", help="Output grouped comparison text file")
    parser.add_argument("--final-output", default="/home/viglab2/won_woo/tvr/TextData/tvrval_rewritten_3.txt", help="Output combined tvrval_rewritten_3.txt file")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct", help="LLM model identifier")
    parser.add_argument("--device", default="cuda:1", help="Device to run LLM on")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (number of clips per batch)")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Max new tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Reading original captions from {args.caption_file}...")

    clips = defaultdict(list)
    clip_order = []

    with open(args.caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap_id, query = line.split(' ', 1)
            clip_id = cap_id.split('#enc#')[0]
            if clip_id not in clips:
                clip_order.append(clip_id)
            clips[clip_id].append({'cap_id': cap_id, 'query': query})

    print(f"Total clips: {len(clip_order)}, Total original captions: {sum(len(v) for v in clips.values())}")

    print(f"Loading HuggingFace LLM ({args.model_id}) on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine device index or str
    if args.device.startswith("cuda:"):
        device_idx = int(args.device.split(":")[1])
    elif args.device == "cuda":
        device_idx = 0
    else:
        device_idx = -1

    llm_pipe = pipeline(
        "text-generation",
        model=args.model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device_idx
    )

    system_prompt = (
        "You are an expert AI for enhancing video retrieval queries. You will be provided with multiple captions describing the same video clip.\n"
        "Your task is to weave these sentences together into a natural, highly visual narrative or a set of refined sentences based on these 4 strict rules:\n"
        "1. Affirmative Conversion: Remove any negative words ('not', 'without', 'never'). Convert the sentence into a positive visual state.\n"
        "2. Visual Salience Amplification: Remove invisible human emotions, mental states, or abstract concepts. Focus ONLY on tangible objects and active verbs that can be seen as pixels.\n"
        "3. Subject-Object Grounding: Replace all ambiguous pronouns ('He', 'She', 'It', 'They') with concrete visible nouns (e.g., 'The man', 'The woman').\n"
        "4. Token Economy & Front-loading: Eliminate non-visual filler words (adverbs, conjunctions). Place the most unique, deterministic visual action/object at the VERY FRONT of the sentence.\n\n"
        "Instructions:\n"
        "- Remove redundancies and merge overlapping actions logically.\n"
        "- You may output one or multiple sentences. Number them if there are multiple distinct actions (e.g., '1. ...\n2. ...').\n"
        "- Output ONLY the rewritten text, with no conversational fillers, thoughts, or explanations."
    )

    print(f"Applying Multi-Caption Weaving + LLM Rewriting (Batch Size: {args.batch_size})...")
    grouped_blocks = []
    final_combined_lines = []

    for i in tqdm(range(0, len(clip_order), args.batch_size)):
        chunk_clip_ids = clip_order[i:i+args.batch_size]
        llm_inputs = []

        for clip_id in chunk_clip_ids:
            clip_captions = clips[clip_id]
            user_content = "Captions for the video clip:\n"
            for idx, cap in enumerate(clip_captions):
                user_content += f"{idx}: {cap['query']}\n"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content.strip()}
            ]
            llm_inputs.append(messages)

        llm_outputs = []
        try:
            llm_outputs = llm_pipe(
                llm_inputs,
                batch_size=len(llm_inputs),
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        except Exception as e:
            print(f"Batch generation failed at index {i}: {e}")
            llm_outputs = [None] * len(llm_inputs)

        for idx, clip_id in enumerate(chunk_clip_ids):
            clip_captions = clips[clip_id]
            final_rewritten = ""

            if llm_outputs and llm_outputs[idx] is not None:
                try:
                    final_rewritten = llm_outputs[idx][0]["generated_text"][-1]["content"].strip()
                except Exception:
                    pass

            if not final_rewritten:
                final_rewritten = "\n".join([f"{j}. {cap['query']}" for j, cap in enumerate(clip_captions)])
            else:
                final_rewritten = re.sub(r'^(Here is.*?:\s*|Rewritten.*:\s*)', '', final_rewritten, flags=re.IGNORECASE).strip('"\' \n')

            # Build human-readable block
            block = f"==================================================\n"
            block += f"Clip ID: {clip_id}\n"
            block += f"[Original]\n"
            for cap in clip_captions:
                enc_idx = cap['cap_id'].split('#enc#')[-1]
                block += f"{enc_idx}: {cap['query']}\n"
            block += f"\n[Rewritten]\n"
            block += f"{final_rewritten}\n"
            block += f"==================================================\n\n"
            grouped_blocks.append(block)

            # Build final format: original captions followed by rewritten captions as #enc#5, #enc#6, ...
            for enc_num, cap in enumerate(clip_captions):
                final_combined_lines.append(f"{clip_id}#enc#{enc_num} {cap['query']}\n")

            # Parse individual rewritten sentences
            raw_rewr_lines = [l.strip() for l in final_rewritten.split('\n') if l.strip()]
            rewr_sentences = []
            for r_line in raw_rewr_lines:
                clean_sentence = re.sub(r'^\d+[\.\)]\s*', '', r_line).strip()
                if clean_sentence:
                    rewr_sentences.append(clean_sentence)

            start_rewr_enc = len(clip_captions)
            for r_idx, sent in enumerate(rewr_sentences):
                final_combined_lines.append(f"{clip_id}#enc#{start_rewr_enc + r_idx} {sent}\n")

    print(f"Saving grouped file to {args.grouped_output}...")
    with open(args.grouped_output, 'w', encoding='utf-8') as f:
        f.writelines(grouped_blocks)

    print(f"Saving final combined file to {args.final_output} (Total lines: {len(final_combined_lines)})...")
    with open(args.final_output, 'w', encoding='utf-8') as f:
        f.writelines(final_combined_lines)

    print("Multi-Caption Weaving for Val Completed Successfully!")

if __name__ == '__main__':
    main()

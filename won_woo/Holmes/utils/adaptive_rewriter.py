import os
import re
from collections import defaultdict
from tqdm import tqdm

# =======================================================
# Configuration
# =======================================================
CAPTION_FILE = '/home/viglab2/won_woo/tvr/TextData/tvrtrain.caption.txt'
OUTPUT_FILE  = '/home/viglab2/won_woo/tvr/TextData/tvrtrain_rewritten_grouped.txt'
BATCH_SIZE = 8  # ✨ 클립 단위 배치 처리이므로 OOM 방지를 위해 사이즈 축소

USE_LLM_FOR_IRM = True  # API/로컬 LLM 사용 시 True 변경

if USE_LLM_FOR_IRM:
    import torch
    from transformers import pipeline, AutoTokenizer
    print("Loading HuggingFace LLM...")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    # ✨ 왼쪽 패딩 적용 및 pad_token 설정 (배치 추론 시 필수)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    llm_pipe = pipeline("text-generation", model=model_id, tokenizer=tokenizer,
                        model_kwargs={"torch_dtype": torch.bfloat16}, device=0)
else:
    llm_pipe = None


def main():
    print(f"Reading original captions from {CAPTION_FILE}...")
    
    # 그룹핑 로직: clip_id -> list of (cap_id, query)
    clips = defaultdict(list)
    clip_order = []  # 순서 보장을 위해
    
    with open(CAPTION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cap_id, query = line.split(' ', 1)
            # cap_id e.g., friends_s09e23-24_seg02_clip_33#enc#0
            clip_id = cap_id.split('#enc#')[0]
            if clip_id not in clips:
                clip_order.append(clip_id)
            clips[clip_id].append({'cap_id': cap_id, 'query': query})
            
    print(f"Applying Multi-Caption Weaving + LLM Rewriting (Batch Size: {BATCH_SIZE})...")
    rewritten_lines = []
    
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
    
    for i in tqdm(range(0, len(clip_order), BATCH_SIZE)):
        chunk_clip_ids = clip_order[i:i+BATCH_SIZE]
        
        llm_inputs = []
        
        for clip_id in chunk_clip_ids:
            clip_captions = clips[clip_id]
            user_content = "Captions for the video clip:\n"
            for idx, cap in enumerate(clip_captions):
                user_content += f"{idx}: {cap['query']}\n"
                
            if USE_LLM_FOR_IRM:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content.strip()}
                ]
                llm_inputs.append(messages)

        # Batch Inference
        llm_outputs = []
        if llm_inputs and llm_pipe is not None:
            try:
                # max_new_tokens 늘림 (여러 문장이 합쳐져서 길어질 수 있으므로)
                llm_outputs = llm_pipe(llm_inputs, batch_size=len(llm_inputs), max_new_tokens=150, temperature=0.1, do_sample=False)
            except Exception as e:
                print(f"Batch generation failed: {e}")
                llm_outputs = [None] * len(llm_inputs)

        # LLM 결과를 정리하여 저장
        for idx, clip_id in enumerate(chunk_clip_ids):
            clip_captions = clips[clip_id]
            final_rewritten = ""
            
            if USE_LLM_FOR_IRM and llm_outputs[idx] is not None:
                try:
                    final_rewritten = llm_outputs[idx][0]["generated_text"][-1]["content"].strip()
                except Exception:
                    pass
            
            # fallback 처리
            if not final_rewritten:
                final_rewritten = "\n".join([f"{j}. {cap['query']}" for j, cap in enumerate(clip_captions)])
            else:
                # 불필요한 LLM 서두 제거
                final_rewritten = re.sub(r'^(Here is.*?:\s*|Rewritten.*:\s*)', '', final_rewritten, flags=re.IGNORECASE).strip('"\' \n')
            
            # 텍스트 포맷으로 Append
            block = f"==================================================\n"
            block += f"Clip ID: {clip_id}\n"
            block += f"[Original]\n"
            for cap in clip_captions:
                # cap_id의 마지막 번호를 사용
                enc_idx = cap['cap_id'].split('#enc#')[-1]
                block += f"{enc_idx}: {cap['query']}\n"
            
            block += f"\n[Rewritten]\n"
            block += f"{final_rewritten}\n"
            block += f"==================================================\n\n"
            
            rewritten_lines.append(block)
        
    print(f"Saving rewritten grouped captions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(rewritten_lines)
        
    print("✨ Adaptive Rewriting (Multi-Caption Weaving) 완료!")

if __name__ == '__main__':
    main()

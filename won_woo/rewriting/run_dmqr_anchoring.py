import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 1. 모델 로드 (Qwen2.5-3B-Instruct)
model_name = "Qwen/Qwen2.5-3B-Instruct"
print("⏳ Qwen2.5-3B 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16
).to("cuda")

# 2. 데이터 로드
input_file = "context_anchored_samples.json" 
output_file = "rewritten_dmqr_samples.json"

with open(input_file, 'r', encoding='utf-8') as f:
    samples = json.load(f)

# 3. 🌟 DMQR 맞춤형 프롬프트 설계 🌟
system_prompt = (
    "You are an expert in partially relevant video retrieval (PRVR). "
    "Your task is to apply Diverse Multi-Query Rewriting (DMQR) to a Target Event. "
    "Generate 3 distinct queries based on the provided [Target Event], [Previous Event], and [Next Event].\n\n"
    "1. 'Action_Focused': Describe ONLY the core actions, movements, and verbs of the Target Event.\n"
    "2. 'Entity_Focused': Describe the Target Event by focusing heavily on the specific nouns, objects, and subjects involved.\n"
    "3. 'Temporal_Anchored': Rewrite the Target Event by explicitly bounding it chronologically using the Previous and Next events (e.g., 'After [Prev], [Target] happens, followed by [Next]').\n\n"
    "IMPORTANT: Do NOT hallucinate any visual details not present in the text. "
    "You MUST return ONLY a valid JSON object in the exact format below, without any other conversational text:\n"
    "{\n"
    '  "Action_Focused": "...",\n'
    '  "Entity_Focused": "...",\n'
    '  "Temporal_Anchored": "..."\n'
    "}"
)

print("🚀 DMQR 쿼리 생성을 시작합니다...")
for sample in tqdm(samples):
    user_prompt = (
        f"[Previous Event]: {sample['Prev_Query']}\n"
        f"[Target Event]: {sample['Target_Query']}\n"
        f"[Next Event]: {sample['Next_Query']}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # max_new_tokens를 늘림 (3개의 문장을 생성해야 하므로)
    generated_ids = model.generate(**model_inputs, max_new_tokens=200, temperature=0.2)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 4. JSON 파싱 (안전 장치 추가)
    try:
        # LLM이 종종 붙이는 마크다운 블록(```json ... ```) 찌꺼기 제거
        clean_response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.IGNORECASE|re.MULTILINE).strip()
        dmqr_output = json.loads(clean_response)
    except json.JSONDecodeError:
        print(f"\n⚠️ JSON 파싱 에러 발생 (Target_Idx: {sample.get('Target_Idx', 'Unknown')}). Raw 텍스트로 저장합니다.")
        dmqr_output = {"Raw_Output": response}
    
    # 생성된 3개의 쿼리를 기존 샘플 데이터에 병합
    sample.update(dmqr_output)

# 5. 결과 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)

print(f"✅ DMQR 생성 완료! 결과가 {output_file}에 저장되었습니다.")
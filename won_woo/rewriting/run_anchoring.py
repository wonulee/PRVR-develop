import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 1. 모델 로드 (Qwen2.5-3B-Instruct)
model_name = "Qwen/Qwen2.5-3B-Instruct"
print("⏳ Qwen2.5-3B 모델을 로드하는 중입니다... (A100 GPU 권장)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16
).to("cuda")  # device_map="auto" 대신 수동으로 GPU 할당


# 2. 1단계에서 추출한 데이터 로드
# (오타로 저장하셨을 수 있으니 실제 파일명에 맞게 수정해주세요)
input_file = "context_anchored_samples.json" 
output_file = "rewritten_anchored_samples.json"

with open(input_file, 'r', encoding='utf-8') as f:
    samples = json.load(f)

# 3. 프롬프트 템플릿 설계 (논문의 TCP/IRM 철학 완벽 반영)
system_prompt = (
    "You are an expert in video retrieval. Your task is to rewrite the Target Event by embedding it "
    "into a strict chronological sequence using the Previous and Next events. "
    "Rules: \n"
    "1. Focus ONLY on the temporal order (e.g., 'After [Prev], [Target], and then [Next]').\n"
    "2. Do NOT add unnecessary adjectives, adverbs or hallucinate visual details.\n"
    "3. Keep the core action of the Target Event perfectly intact."
)

print("🚀 앵커링 쿼리 생성을 시작합니다...")
# 4. 추론 루프 (100개 샘플)
for sample in tqdm(samples):
    user_prompt = f"""
    [Previous Event]: {sample['Prev_Query']}
    [Target Event]: {sample['Target_Query']}
    [Next Event]: {sample['Next_Query']}
    
    Rewrite the [Target Event] into a single concise sentence anchored chronologically.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Qwen 포맷에 맞게 프롬프트 적용
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Temperature를 낮춰서 창의성(환각)을 억제하고 규칙을 엄격히 따르도록 설정
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, temperature=0.2)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 생성된 텍스트를 JSON에 추가
    sample["Anchored_Query"] = response

# 5. 결과 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)

print(f"✅ 생성 완료! 결과가 {output_file}에 저장되었습니다.")
# zh_translator.py
import re
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ======== Cấu hình model ========
# Zh -> En (direct)
MODEL_ZH_EN = "Helsinki-NLP/opus-mt-zh-en"
# En -> Vi (dùng để pivot nếu không có zh->vi tốt)
MODEL_EN_VI = "Helsinki-NLP/opus-mt-en-vi"

DEVICE = 0 if torch.cuda.is_available() else -1  # 0 means cuda:0, -1 means cpu

# ======== Loader helper ========
def load_model_and_tokenizer(model_name: str):
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

# Load models lazily (only when needed)
tokenizer_zh_en, model_zh_en = load_model_and_tokenizer(MODEL_ZH_EN)
# Try loading en->vi (optional). If not present, you can skip pivot.
try:
    tokenizer_en_vi, model_en_vi = load_model_and_tokenizer(MODEL_EN_VI)
    PIVOT_AVAILABLE = True
except Exception as e:
    print("Không thể nạp model En->Vi:", e)
    tokenizer_en_vi, model_en_vi = None, None
    PIVOT_AVAILABLE = False

# ======== Utilities ========
def split_into_sentences(text: str, max_chars: int = 300) -> List[str]:
    """
    Chia text lớn thành các chunk nhỏ theo dấu câu, giữ mỗi chunk <= max_chars.
    Điều này giúp tránh tràn max_length khi generate.
    """
    # split by punctuation (including Chinese punctuation)
    sentences = re.split(r'(?<=[。！？!?\.])\s*', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if not s:
            continue
        if len(current) + len(s) <= max_chars:
            current += s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    # fallback: if still too long, split by characters
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            # split evenly
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars])
    return final

def generate_translation(tokenizer, model, texts: List[str], max_length=512, num_beams=4):
    outputs = []
    for t in texts:
        inputs = tokenizer.prepare_seq2seq_batch([t], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model = model.to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        outputs.append(out)
    return " ".join(outputs)

# ======== API chính ========
def translate_zh_to_en(text: str) -> str:
    chunks = split_into_sentences(text, max_chars=300)
    return generate_translation(tokenizer_zh_en, model_zh_en, chunks, max_length=512)

def translate_en_to_vi(text: str) -> str:
    if not PIVOT_AVAILABLE:
        raise RuntimeError("Model En->Vi không sẵn sàng trên máy này.")
    chunks = split_into_sentences(text, max_chars=300)
    return generate_translation(tokenizer_en_vi, model_en_vi, chunks, max_length=512)

def translate_zh_to_vi(text: str) -> str:
    """
    Nếu không có model trực tiếp, đi qua bước pivot: Zh -> En -> Vi.
    Ưu/nhược: đơn giản nhưng có thể giảm chất lượng.
    """
    en = translate_zh_to_en(text)
    vi = translate_en_to_vi(en)
    return vi

# ======== CLI demo ========
if __name__ == "__main__":
    print("Trình dịch tiếng Trung (text-only) — nhập 'exit' để thoát.")
    while True:
        txt = input("Chinese > ").strip()
        if not txt or txt.lower() == "exit":
            break
        # Thử Zh -> En và Zh -> Vi (pivot) nếu model en->vi có sẵn
        try:
            en = translate_zh_to_en(txt)
            print("-> English:", en)
            if PIVOT_AVAILABLE:
                vi = translate_en_to_vi(en)
                print("-> Vietnamese (pivot):", vi)
            else:
                print("-> Vietnamese (pivot): Không có model en->vi, hãy cài model tương ứng nếu cần.")
        except Exception as e:
            print("Lỗi khi dịch:", e)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLM_MODEL, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE

class QwenLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map = "auto",
            torch_dtype = torch.bfloat16,
            trust_remote_code=True
        )

    def invoke(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            temperature=LLM_TEMPERATURE,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=True
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
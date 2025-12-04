import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import EMBEDDING_MODEL, EMBED_BATCH_SIZE

class EmbeddingEncoder:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        self.device = next(self.model.parameters()).device
        self.batch_size = EMBED_BATCH_SIZE

    def embed_text(self, texts: str):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._encode(batch)
            embeddings.extend(vectors)

    def _encode(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().list()
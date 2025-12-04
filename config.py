# config the LLM model and Embedding model
from transformers import AutoModelForCausalLM

LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen-base"
EMBED_BATCH_SIZE = 512
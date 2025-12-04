import argparse
from llm.qwen_llm import QwenLLM
from memory.milvus_client import milvus_connect, create_memory_collection
from memory.memory_store import AgentMemory
from embeddings.encoder import EmbeddingEncoder
from graph.nodes import build_graph


def main():
    # connect + setup
    milvus_connect()
    create_memory_collection()
    memory = AgentMemory()
    encoder = EmbeddingEncoder()
    llm = QwenLLM()

    graph = build_graph()

    state = {
        "input": "Summarize the Lakers game",
        "memory": memory,
        "encoder": encoder,
        "llm": llm,
        "tools": {
            "search_api": lambda x: {"result": "Lakers won 120-115"}
        }
    }

    result = graph.run(state)
    print(result["final_response"])

if __name__ == "__main__":
    main()

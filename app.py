import argparse

from memory.milvus_client import milvus_connect, create_memory_collection
from memory.memory_store import AgentMemory
from embeddings.encoder import EmbeddingEncoder
from graph.nodes import build_graph

from openai import OpenAI

def main():
    # connect + setup
    milvus_connect()
    create_memory_collection()
    memory = AgentMemory()
    encoder = EmbeddingEncoder()
    llm = OpenAI().chat.completions

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

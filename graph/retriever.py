from langgraph.graph import add_node

@add_node
def rag_retriever(state):
    memory = state['memory']
    encoder = state['encoder']
    query = state['plan']
    emb = encoder.embed_text(query)
    results = memory.search(emb, top_k=5)
    retrieved = [hit.entity.get("text") for hit in results[0]]

    state["context"] = "\n".join(retrieved)
    return state
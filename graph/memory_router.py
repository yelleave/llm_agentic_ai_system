from langgraph.graph import add_node

@add_node
def memory_router(state):
    encoder = state['encoder']
    memory = state['memory']
    llm = state['llm']
    text = state['final_response']
    should_store = llm.invoke(
        f"Should this be stored as memory? {text}. Answer YES/NO"
    )
    if "YES" in should_store:
        emb = encoder.embed_text(text)
        memory.insert_memory(emb, text, mem_type="episodic")

    return state
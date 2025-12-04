from langgraph.graph import add_node

@add_node
def planning_agent(state):
    llm = state['llm']
    user_query = state['input']
    plan = llm.invoke(
        f"Decompose this task into steps: {user_query}"
    )
    state["plan"] = plan
    return state
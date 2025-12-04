from langgraph.graph import add_node

@add_node
def tool_agent(state):
    tools = state['tools']
    step = state['current_step']
    if "search" in step:
        result = tools["search_api"](step)
    else:
        result = None

    state["tool_result"] = result
    return state
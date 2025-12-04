from langgraph.graph import StateGraph
from .planner import planning_agent
from .retriever import rag_retriever
from .tool_agent import tool_agent
from .memory_router import memory_router

def build_graph():
    g = StateGraph()
    g.add_node("planner", planning_agent)
    g.add_node("retriever", rag_retriever)
    g.add_node("tool", tool_agent)
    g.add_node("memory_router", memory_router)

    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "tool")
    g.add_edge("tool", "memory_router")
    g.set_entry_point("planner")

    return g.compile()
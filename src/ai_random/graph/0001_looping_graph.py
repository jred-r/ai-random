import random
from pathlib import Path
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int


def greeting_node(state: AgentState) -> AgentState:
    state["name"] = f"Hi there, {state['name']}"
    state["counter"] = 0
    return state


def random_node(state: AgentState) -> AgentState:
    state["number"].append(random.randint(1, 10))
    state["counter"] += 1
    return state


def should_continue(state: AgentState) -> str:
    if state["counter"] < 5:
        return "loop"
    return "exit"


graph = StateGraph(AgentState)

graph.add_node("greet", greeting_node)
graph.add_node("random", random_node)

graph.add_edge(START, "greet")
graph.add_edge("greet", "random")
graph.add_edge("random", END)

graph.add_conditional_edges("random", should_continue, {"loop": "random", "exit": END})

app = graph.compile()
Path("001.png").write_bytes(app.get_graph().draw_mermaid_png())
data: AgentState = {"name": "J", "counter": 0, "number": []}
result = app.invoke(data)
print(result["number"])

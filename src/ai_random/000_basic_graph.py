from math import prod
from typing import TypedDict

from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    values: list[int]
    operation: str
    result: str


def perform_operation(state: AgentState) -> AgentState:
    """Perform addition or multiplication on the state's values."""
    operation = state["operation"]
    values = state["values"]
    name = state["name"]

    match operation:
        case "+":
            answer = sum(values)
        case "*":
            answer = prod(values)
        case _:
            result = f"hi {name}, Invalid operation"
            return {**state, "result": result}

    result = f"hi {name}, your answer is {answer}"
    return {**state, "result": result}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("process", perform_operation)
    graph.set_entry_point("process")
    graph.set_finish_point("process")
    return graph.compile()


def main() -> None:
    app = build_graph()
    examples: list[AgentState] = [
        {"name": "J", "operation": "+", "values": [1, 2, 3, 4], "result": ""},
        {"name": "J", "operation": "*", "values": [1, 2, 3, 4], "result": ""},
    ]

    for data in examples:
        print(app.invoke(data)["result"])


if __name__ == "__main__":
    main()

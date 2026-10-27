import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = "You are a math tutor. Always use tools for calculations."
EXIT_COMMANDS = {"exit", "quit"}


@tool(description="This tool is used to add two numbers, arguments are n1 and n2")
def sum_two_numbers(n1: int, n2: int) -> int:
    print(f"sum_two_numbers called with input {n1}, {n2}")
    return n1 + n2


@tool(description="This tool is used to subtract two numbers, arguments are n1 and n2")
def subtract_two_numbers(n1: int, n2: int) -> int:
    print(f"subtract_two_numbers Tool called with input {n1}, {n2}")
    return n2 - n1


@tool(description="This tool is used to divide two numbers, arguments are n1 and n2")
def divide_two_numbers(n1: int, n2: int) -> int:
    print(f"divide_two_numbers called with input {n1}, {n2}")
    if n2 == 0:
        raise ValueError("Cannot divide by zero.")
    return n2 // n1


@tool(description="This tool is used to multiply two numbers, arguments are n1 and n2")
def multiply_two_numbers(n1: int, n2: int) -> int:
    print(f"multiply_two_numbers Tool called with input {n1}, {n2}")
    return n1 * n2


TOOLS = [
    sum_two_numbers,
    subtract_two_numbers,
    multiply_two_numbers,
    divide_two_numbers,
]


def build_graph(model_name: str):
    llm = ChatOpenRouter(model=model_name, max_retries=3).bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    

    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    load_dotenv()
    model_name = str(os.getenv("OPEN_ROUTER_MODEL"))

    graph = build_graph(model_name)
    diagram_path = Path(__file__).with_name("05_langgraph_math_tutor.png")
    diagram_path.write_bytes(graph.get_graph().draw_mermaid_png())
    config: RunnableConfig = {"configurable": {"thread_id": "math-tutor"}}

    print("Math tutor ready. Enter a problem, or type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() in EXIT_COMMANDS:
            break
        if not question:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        print(f"Tutor: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()

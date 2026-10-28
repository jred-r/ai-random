import os
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

load_dotenv()
model = str(os.getenv("OPEN_ROUTER_MODEL"))


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOpenRouter(model=model, max_retries=3)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

# Path("0002.png").write_bytes(app.get_graph().draw_mermaid_png())

user_input = input("Enter Message: ")

while user_input != "exit":
    data: AgentState = {"messages": [HumanMessage(user_input)]}
    app.invoke(data)
    user_input = input("Enter Message: ")

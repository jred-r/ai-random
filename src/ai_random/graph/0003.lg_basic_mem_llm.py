import os
from pathlib import Path
from typing import List, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

load_dotenv()

model = str(os.getenv("OPEN_ROUTER_MODEL"))
llm = ChatOpenRouter(model=model, max_retries=3)


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    message = response.content
    print(f"AI : {message}")
    state["messages"].append(AIMessage(message))
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(user_input))
    response = app.invoke({"messages": conversation_history})
    conversation_history = response["messages"]
    user_input = input("Enter: ")
    
print("Thank you for using the bot!")

import os

from dotenv import load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = str(os.getenv("OPEN_ROUTER_MODEL"))


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


available_tools = [
    sum_two_numbers,
    subtract_two_numbers,
    divide_two_numbers,
    multiply_two_numbers,
]

llm_model = ChatOpenRouter(model=model, max_retries=3).bind_tools(
    available_tools
)  # bind the tools


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a math tutor. Always use tools for calculations."
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

print("You are interacting with a math tutor, enter your problem -> ex: 5 + 5 * 10")
question = input()

messages = prompt.format_messages(question=question)
while True:
    response = llm_model.invoke(messages)

    messages.append(response)

    if not response.tool_calls:
        print(response.content)
        break

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = {t.name: t for t in available_tools}[tool_name]
        result = selected_tool.invoke(tool_call["args"])

        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

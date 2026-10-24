import os

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = os.environ.get("OPEN_ROUTER_MODEL")

open_llm = ChatOpenRouter(
    model=str(model),
    max_retries=3,
)

messages = [
    (
        SystemMessage(
            "Hey, You are a Helpful AI Agent, and you are very advanced for your time!"
        )
    ),
    (HumanMessage("How do i calculate the sum of all irrational numbers?")),
]

ai_messge = open_llm.invoke(messages)
print(ai_messge.content)

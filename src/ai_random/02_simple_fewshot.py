import os

from dotenv import load_dotenv
from langchain_core.prompts import (ChatPromptTemplate,
                                    FewShotChatMessagePromptTemplate)
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = str(os.getenv("OPEN_ROUTER_MODEL"))

llm_model = ChatOpenRouter(model=model, max_retries=3)

few_shot_examples = [
    {"question": "I was charged twice!", "category": "Billing"},
    {
        "question": "My PC is stuck in a boot loop, I am so frustrated!",
        "category": "Tech Support",
    },
    {"question": "Reset my password!", "category": "Account"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{category}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a support ticket classifier. Reply only with the category.",
        ),
        few_shot_prompt,
        ("human", "{question}"),
    ]
)

print("Enter your concern!")
question = input("> ")

messages = final_prompt.format_messages(question=question)

response = llm_model.invoke(messages)

print(response.content)

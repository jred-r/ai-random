import os

from dotenv import load_dotenv
from langchain_core.prompts import (ChatPromptTemplate,
                                    FewShotChatMessagePromptTemplate)
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = str(os.getenv("OPEN_ROUTER_MODEL"))

llm_model = ChatOpenRouter(model=model, max_retries=3)

# create the example
few_shot_examples = [
    {"question": "I was charged twice!", "category": "Billing"},
    {
        "question": "My PC is stuck in a boot loop, I am so frustrated!",
        "category": "Tech Support",
    },
    {"question": "Reset my password!", "category": "Account"},
]

# make a chat prompt template
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{category}"),
    ]
)

# convert it to few shot by adding examples and template created above
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
)

# add the system instructions, the prompt and the final question
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

# create the chain and link the llm
chain = final_prompt | llm_model

question = input("Enter your concern: ")

response = chain.invoke({"question": question})

print(response.content)

from init_model import model
from langchain_core.messages import (
    SystemMessage,
    trim_messages,
    AIMessage,
    HumanMessage,
    BaseMessage,
)
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful assistant. Answer questions to the best of your ability in {language}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


workflow = StateGraph(state_schema=State)


trimmer = trim_messages(
    max_tokens=65,
    strategy="last",  # "first" or "last"
    token_counter=model,  # model will be used to count tokens
    include_system=True,
    allow_partial=False,  # Do not allow partial messages according to the strategy
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Trimming messages
trimmed_messages = trimmer.invoke(messages)

print(trimmed_messages)

print("Before trimming:")
for message in messages:
    print(f"{message.content}")

print("\nAfter trimming:")
for message in trimmed_messages:
    print(f"{message.content}")
print(f"*" * 40)


def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "language": state["language"],
        }
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
app = workflow.compile(checkpointer=MemorySaver())

# first query, you will get negative response as the human message is trimmed
config = {"configurable": {"thread_id": "8240a568-10e9-4ee4-a2c5-9266f7e1260b"}}
query = "What is my name?"
language = "English"
input_messages = messages + [HumanMessage(content=query)]
output = app.invoke({"messages": input_messages, "language": language}, config=config)
print(output["messages"][-1].content)
config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

# second query, you will get positive response as the human message is not trimmed
input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
print(output["messages"][-1].content)

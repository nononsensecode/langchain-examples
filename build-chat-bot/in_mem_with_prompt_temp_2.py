from init_model import model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Sequence

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful assistant. Answer questions to the best of your ability in {language}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "8240a568-10e9-4ee4-a2c5-9266f7e1260b"}}
query = "Hi, I'm Jim"
language = "spanish"
input_messages = [HumanMessage(content=query)]
output = app.invoke({"messages": input_messages, "language": language}, config=config)
print(output["messages"][-1].content)

# The above code stores the state in memory. So from now on, we can omit the language parameter,
# and the model will remember it.

query = "What is my name?"
input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
print(output["messages"][-1].content)

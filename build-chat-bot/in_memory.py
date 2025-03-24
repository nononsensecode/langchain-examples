from init_model import model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import asyncio

messages = [
    HumanMessage(content="My name is Bob."),
    AIMessage(content="Hello Bob! How can I assist you today?"),
    HumanMessage(content="What is my name?"),
]

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

query = "What is my name?"
config = {"configurable": {"thread_id": "8240a568-10e9-4ee4-a2c5-9266f7e1260b"}}
response = app.invoke(
    {
        "messages": messages,
    },
    config=config,
)
print(response["messages"][-1].content)

# The above code is synchronous. Below is the asynchronous version.

async_workflow = StateGraph(state_schema=MessagesState)


async def call_model_async(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}


async_workflow.add_node("model", call_model_async)
async_workflow.add_edge(START, "model")
async_app = async_workflow.compile(checkpointer=MemorySaver())


async def print_response():
    response = await async_app.ainvoke(
        {
            "messages": messages,
        },
        config=config,
    )
    print(response["messages"][-1].content)


asyncio.run(print_response())

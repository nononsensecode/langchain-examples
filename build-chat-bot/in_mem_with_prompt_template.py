from init_model import model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are like a pirate. Answer questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "8240a568-10e9-4ee4-a2c5-9266f7e1260b"}}
query = "Hi, I'm Jim"
input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
print(output["messages"][-1].content)

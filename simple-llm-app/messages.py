from langchain_core.messages import HumanMessage, SystemMessage
from init_model import model


messages = [
    SystemMessage("Translate the following from English to Hindi"),
    HumanMessage("Hi! How are you?"),
]

response = model.invoke(messages)
print(response)
print("===" * 20)

# Below are the openai format. This is also supported

messages = [
    ("system", "Translate the following from English to Malayalam"),
    ("user", "How old are you ?"),
]
response = model.invoke(messages)
print(response)
print("===" * 20)

# We can stream too

messages = [
    ("system", "Translate the following from English to French"),
    ("user", "What is your name ?"),
]

for token in model.stream(messages):
    print(token.content)

print("===" * 20)

from langchain_core.prompts import ChatPromptTemplate
from init_model import model

system_template = "Translate the following from English to {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)
prompt = prompt_template.invoke({"language": "Hindi", "text": "Hi! How are you?"})
print(f"Prompt: {prompt}")
messages = prompt.to_messages()
print(f"Messages: {messages}")
response = model.invoke(prompt)
print(response)

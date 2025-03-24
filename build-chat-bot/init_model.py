from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="gemini-1.5-flash", temperature=0, model_provider="google_genai"
)

import os
from dotenv import load_dotenv

load_dotenv()

# Securely load from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model from your Groq Playground
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

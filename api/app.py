from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"
)

# Initialize OpenAI Chat model
openai_model = ChatOpenAI()

# Initialize Ollama model
ollama_model = Ollama(model="gemma")

# Prompts for different tasks
prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic} with 100 words")

# Add routes for OpenAI model
add_routes(
    app,
    openai_model,
    path="/openai"
)

# Add routes for essay generation using OpenAI model
add_routes(
    app,
    prompt1 | openai_model,
    path="/essay"
)

# Add routes for poem generation using Ollama model
add_routes(
    app,
    prompt2 | ollama_model,
    path="/poem"
)

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

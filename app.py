from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.llms.base import LLM
import streamlit as st
import os
import requests
from dotenv import load_dotenv
from typing import Optional, Any, List, Mapping
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class GroqLLM(LLM, BaseModel):
    api_key: str = Field(default=groq_api_key)
    model_name: str = Field(default="mixtral-8x7b-32768")
    temperature: float = Field(default=0.7)

    @property
    def _llm_type(self) -> str:
        return "groq_llm"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        if stop:
            payload["stop"] = stop

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",  # Updated to correct endpoint
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Groq API error: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer as concisely as possible."),
    ("user", "{question}")
])

# Streamlit UI Setup
st.title("Langchain Demo With Groq API")
input_text = st.text_input("Search the topic you want")

# Initialize LLM
if not groq_api_key:
    st.error("Groq API key not found. Please check your .env file.")
else:
    llm = GroqLLM(temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Handle user input
    if input_text:
        try:
            response = chain.invoke({"question": input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
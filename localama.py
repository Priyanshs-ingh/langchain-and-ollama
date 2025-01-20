from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit UI Setup
st.title('Langchain Demo With Gemma')
model_version = st.selectbox('Select Gemma Model Version', ['gemma:2b', 'gemma:7b'])

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer as concisely as possible."),
    ("user", "Question: {question}"),
])

# Input field
input_text = st.text_input('Search the topic you want')

try:
    # Initialize Ollama with Gemma
    llm = Ollama(
        model=model_version,
        temperature=0.7,
        base_url="http://localhost:11434",  # Default Ollama API endpoint
    )
    
    # Create the chain
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Process input and display results
    if input_text:
        with st.spinner('Generating response...'):
            try:
                response = chain.invoke({'question': input_text})
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                
except Exception as e:
    st.error(f"Error initializing Ollama: {str(e)}")
    st.info("Please make sure Ollama is running and Gemma model is installed.")
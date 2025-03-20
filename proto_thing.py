import openai
import pandas as pd
import streamlit as st
from llama_index import (Document, ServiceContext, SimpleDirectoryReader,
                         VectorStoreIndex)
from llama_index.llms import OpenAI

# Configure the Streamlit page
st.set_page_config(page_title="Spotify Review Q&A Chatbot", layout="wide")

# Set up your OpenAI API key (make sure it's in your Streamlit secrets or environment)
openai.api_key = st.secrets.get("openai_key", "YOUR_OPENAI_API_KEY")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me any question about our music streaming app based on user reviews."}
    ]


@st.cache_resource(show_spinner=True)
def load_index():
    with st.spinner("Loading and indexing review data..."):
        # Load documents from the review dataset directory (adjust path as needed)
        csv_path = "./data/reviews.csv"
        df = pd.read_csv(csv_path)

        docs = [Document(text=row["review_text"])
                for index, row in df.iterrows()]

        # Create a service context using an OpenAI LLM
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-4o-mini",
                temperature=0.5,
                system_prompt="You are an expert on our music streaming application. Answer queries based solely on user review insights."
            )
        )
        # Build a vector index from the documents for efficient retrieval
        index = VectorStoreIndex.from_documents(
            docs, service_context=service_context)
        return index


# Load (or cache) the review index
index = load_index()

# Initialize the chat engine if it hasn't been already set
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True)

# Get user input using Streamlit's chat input widget
prompt = st.chat_input("Your question:")

if prompt:
    # Append the user's question to the conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            # Pass the prompt to the chat engine to generate a response
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            # Save the assistant's response in the session state
            st.session_state.messages.append(
                {"role": "assistant", "content": response.response})

# Display the conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

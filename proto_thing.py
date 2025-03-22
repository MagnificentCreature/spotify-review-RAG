import os
import pickle

import openai
import streamlit as st
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PandasCSVReader

openai.api_key = st.secrets["secrets"]["openai_key"]
st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Spotify reviews!"}
    ]


system_prompt = """You are a summary agent to answer users questions on Spotify reviews.

Follow these rules in order to answer the user's question:
1) Your answer should be short (maximum 3 short coherent sentences).
2) Use the contextual information on Google Store reviews for Spotify to extract actionable insights.
3) Your answer should be a coherent question answering the question with the given context.
4) Keep your answers technical and based on facts do not hallucinate features.
5) If the question is not clear, ask for clarification.
6) If the question is out of scope, politely decline.

"""


cache_file = "open_ai_embeddings_cache.pkl"


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs! This should take 3-4 minutes."):
        parser = PandasCSVReader(concat_rows=False, pandas_config={
            "usecols": ["review_text"], "nrows": 10000})
        docs = parser.load_data("data/spotify_reviews_dedup.csv")
        print("Loaded Data")
        llm = OpenAI(model="gpt-4o-mini", temperature=0.5,
                     system_prompt=system_prompt)
        Settings.llm = llm
        print("Readied LLM")
        index = VectorStoreIndex.from_documents(docs)
        print("Generated index")
        return index


index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    print(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            # Add response to message history
            st.session_state.messages.append(message)
            st.session_state.messages.append(message)

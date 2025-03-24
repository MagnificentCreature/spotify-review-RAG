import os
import pickle
import time

import faiss
import openai
import streamlit as st
from llama_index.core import (Document, Settings, StorageContext,
                              VectorStoreIndex, load_index_from_storage)
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PandasCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore

openai.api_key = st.secrets["secrets"]["openai_key"]
st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Spotify reviews!"}
    ]


system_prompt = """You are a summary agent to answer users questions on Spotify reviews.

Follow these rules in order to answer the user's question:
1) Your answer should be short (maximum 3 short coherent sentences).
2) Process your thoughts with the contextual information on Google Store reviews for Spotify to extract actionable insights.
3) Your answer should be a coherent question answering the question with the given context.
4) Keep your answers technical and based on facts, do not hallucinate features.
5) If the question is not clear, ask for clarification.
6) If the question is out of scope, politely decline.

"""

FAISS_STORAGE_PATH = os.path.join("storage")


@st.cache_resource(show_spinner=False)
def load_vector_data():
    with st.spinner(text="Loading and indexing the docs! This should take 1-2 minutes."):
        if os.path.exists(FAISS_STORAGE_PATH):
            t0 = time.time()

            print("Loading FAISS index from disk")

            faiss_index = faiss.read_index(os.path.join(
                FAISS_STORAGE_PATH, "faiss_index.bin"))
            vector_store = FaissVectorStore(faiss_index)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=FAISS_STORAGE_PATH)
            index = load_index_from_storage(storage_context)
            print("FAISS index loaded successfully")
            t1 = time.time()
            print(t1-t0)
            return index
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


index = load_vector_data()

chat_engine = index.as_query_engine(
    chat_mode="condense_question", verbose=True, similarity_top_k=5)

# UI Section

curated_prompts = [
    "What do users like about our application?",
    "In comparison to our application, which music streaming platform are users most likely to compare ours with?",
    "What are the primary reasons users express dissatisfaction with Spotify?",
    "Can you identify emerging trends or patterns in recent user reviews that may impact our product strategy?"
]

if "has_user_input" not in st.session_state:
    st.session_state.has_user_input = False

# Prompt for user input and save to chat history
if user_input := st.chat_input("Your question"):
    st.session_state.has_user_input = True
    st.session_state.messages.append({"role": "user", "content": user_input})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display curated prompts as buttons
if st.session_state.has_user_input == False:
    st.write("Select a curated prompt:")
    for prompt in curated_prompts:
        if st.button(prompt):
            st.session_state.has_user_input = True
            user_input = prompt
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            break

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.query(prompt)
            print(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            # Add response to message history
            st.session_state.messages.append(message)
            print(f"Message: {message}")

            # Display retrieved documents
            if hasattr(response, "source_nodes") and response.source_nodes:
                with st.expander("Retrieved Documents"):
                    for i, node in enumerate(response.source_nodes):
                        print(node.text)

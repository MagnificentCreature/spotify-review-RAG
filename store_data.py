import os

import faiss
import openai
import streamlit as st
from llama_index.core import (Settings, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PandasCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore

openai.api_key = st.secrets["secrets"]["openai_key"]


FAISS_STORAGE_PATH = "./storage"
DOCS_PATH = "./storage/default__vector_store.json"


@st.cache_resource(show_spinner=False)
def load_data():
    # Create FAISS index

    if os.path.exists(FAISS_STORAGE_PATH) and os.path.exists(DOCS_PATH):
        print("Loading Index from cache")
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=FAISS_STORAGE_PATH)
        index = load_index_from_storage(storage_context=storage_context)
        return index

    with st.spinner(text="Loading and indexing the docs! This should take 3-4 minutes."):
        print("Loading Data")
        parser = PandasCSVReader(concat_rows=False, pandas_config={
                                 "usecols": ["review_text"], "nrows": 1000})
        docs = parser.load_data("data/spotify_reviews_dedup.csv")

        print("Generating index")
        # Adjust based on embedding dimensions
        faiss_index = faiss.IndexFlatL2(1536)
        vector_store = FaissVectorStore(faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, show_progress=True)

        print("Saving index")
        # save index to disk
        index.storage_context.persist()

        return index


index = load_data()

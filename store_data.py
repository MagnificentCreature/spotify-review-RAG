import os

import faiss
import openai
import streamlit as st
from llama_index.core import (Settings, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PandasCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore


def get_embedding_dim(embed_model):
    return len(embed_model.get_text_embedding("test"))


openai.api_key = st.secrets["secrets"]["openai_key"]


embed_model = OpenAIEmbedding()

FAISS_STORAGE_PATH = "./storage"
DATA_PATH = os.path.join(FAISS_STORAGE_PATH, "SAMPLE_SPOTIFY_REVIEWS_DEDUP")


def store_data():
    print("Loading Data")
    parser = PandasCSVReader(concat_rows=False, pandas_config={
        "usecols": ["review_text"]})
    docs = parser.load_data(DATA_PATH)
    print("Generating index")

    # Adjust based on embedding dimensions
    co = faiss.GpuMultipleClonerOptions()
    co.shard = False
    res = faiss.StandardGpuResources()  # Create a single GPU resource
    faiss_index = faiss.IndexFlatL2(1536)
    gpu_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # res = [faiss.StandardGpuResources() for _ in range(num_gpus)]  # Multi-GPU resources
    # gpu_index = faiss.index_cpu_to_gpu_multiple(res, list(range(num_gpus)), faiss_index)

    vector_store = FaissVectorStore(gpu_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context)

    print("Saving index")
    # save index to disk
    # index.storage_context.persist()

    if not os.path.exists(FAISS_STORAGE_PATH):
        os.makedirs(FAISS_STORAGE_PATH)
        print(f"Created directory: {FAISS_STORAGE_PATH}")

    # Convert back to CPU before saving
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, f"{FAISS_STORAGE_PATH}/faiss_index.bin")
    # faiss.write_index(cpu_index, FAISS_INDEX_PATH)  # Save FAISS index separately
    storage_context.persist(persist_dir=FAISS_STORAGE_PATH)

    return index


index = store_data()

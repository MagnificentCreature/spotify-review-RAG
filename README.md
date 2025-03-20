## Q&A RAG for Music Streaming Application Review
This is the implementation of an AI-powered Q&A chatbot designed to extract actionable insights from a dataset of Google Store reviews for a music streaming application (e.g., Spotify).

### Overview
The project aims to address the "management's" need for a more efficient solution to analyze and interpret unstructured user reviews. The chatbot utilises natural language processing techniques to provide insightful responses to various management queries about user preferences, comparisons with competitors, reasons for dissatisfaction, and emerging trends etc.

### Project Architecture

- **Data Ingestion:**  
  The application reads a CSV file containing reviews. Repeated and irrelevant data (e.g., reviews with too few tokens, or reviews that are too similar) are removed during preprocessing.
  
- **Text Preprocessing & Splitting:**  
  Raw reviews are preprocessed by lowercasing, removing punctuation, and filtering out stopwords. 

- **Embedding & Indexing:**  
  Reviews are vectorized using TF-IDF and optionally enhanced with GPU-accelerated embedding models (e.g., using RAPIDS or cuML on CUDA-enabled hardware). Similarity is computed (using cosine similarity) to build a retrieval index over the dataset.

- **Retrieval-Augmented Generation (RAG):**  
  When the user submits a query, the index is searched to retrieve the most relevant review excerpts. These are then aggregated and fed into a generative model (or a Q&A system) to produce a coherent, actionable answer.

- **Deployment:**  
  The solution is containerized using Docker and enhanced as a VS Code dev container to simplify dependency management. RAPIDS is used for GPU acceleration, and additional tools (e.g., pre-commit, symspell, etc.) further optimize the development workflow.

### Data

- **Source File:**  
  The primary data source is the CSV file located at `Data/SPOTIFY_REVIEWS.csv` containing user reviews including text, rating, and likes.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

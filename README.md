## Q&A RAG for Music Streaming Application Review
This is the implementation of an AI-powered Q&A chatbot designed to extract actionable insights from a dataset of Google Store reviews for a music streaming application (e.g., Spotify).

### Overview
The project aims to address the "management's" need for a more efficient solution to analyze and interpret unstructured user reviews. The chatbot utilises natural language processing techniques to provide insightful responses to various management queries about user preferences, comparisons with competitors, reasons for dissatisfaction, and emerging trends etc.

![Example query on Streamlit UI](/Readme_images/UI%20example.png)
![Example query on Streamlit UI](/Readme_images/UI%20example%202.png)

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

### UI

Simple and intuitive chat UI with sample queries, made using Streamlit.

### How to use

1)Run 
```
pip install -r requirements.txt
```
Alternatively, run the app in docker

2)Get an openAI API key, and put it under [./.streamlit/secrets.toml](./.streamlit/secrets.toml)
The file should have these two lines only
```
[secrets]
openai_key = "<your api key>"
```

3)Download the dataset.<br />
Option 1:
Download the [dataset](https://www.kaggle.com/datasets/bwandowando/3-4-million-spotify-google-store-reviews?resource=download)
Put the unzipped csv as ./data/SPOTIFY_REVIEWS.csv

Run 
```
python preprocess_data.py
```
This might take a couple of hours depending on your GPU

When it is done, run
```
python store_data.py
```

Option 2:
Download the reduced [processed dataset](https://drive.google.com/file/d/1xR-dVr6785sw0uXMRDrCxjDm2ODM2XvY/view?usp=sharing) and put it in ./storage
(while this step is simpler, it might produce less optimal results, if you have the time and GPU I recommend doing option 1)

4)Run the app using 
```
streamlit run app.py
```
This should open your browser to use the app, otherwise it is likely on [localhost:8501](http://localhost:8501/) but may vary based off your streamlist default port settings

Note before running:
Make sure (minimally) that your directory looks like this
```
.streamlit
|-- secrets.toml
storage
|-- docstore.json
|-- faiss_index.bin
|-- graph_store.json
|-- index_store.json
app.py
scoring.py
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

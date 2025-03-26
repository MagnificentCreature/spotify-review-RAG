import cudf
import cupy as cp  # GPU-accelerated NumPy
import cupyx.scipy.sparse as cpx  # Sparse CuPy operations
import pandas as pd
import torch
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
from cuml.metrics.pairwise_distances import pairwise_distances
from cuml.neighbors import NearestNeighbors  # RAPIDS GPU-optimized ANN
from tqdm import tqdm


def remove_short_lines(df, length=8):
    return df.loc[df["review_text"].str.count(" ") >= length]


def preprocess_text(df):
    df.drop_duplicates(subset=["review_text"], inplace=True)
    df.dropna(subset=["review_text"], inplace=True)
    df["review_text"] = (
        df["review_text"]
        # replace non-alphanumeric with space
        .str.replace(r'\W+', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)  # collapse whitespace
        .str.lower()
    )
    return df


def sort_by_recent(df):
    df.sort_values(by=['review_timestamp'], ascending=False)
    return df


def vectorized_deduplicate_dataframe(df, text_column="review_text", threshold=0.8):
    """
    Efficiently deduplicates a dataframe based on text similarity using TF-IDF and cosine similarity on GPU using RAPIDS cuML.

    Args:
    - df (pd.DataFrame): Input DataFrame containing text data.
    - text_column (str): Column name of text data.
    - threshold (float): Similarity threshold (0 to 1), where higher means stricter deduplication.

    Returns:
    - pd.DataFrame: Deduplicated DataFrame with all original columns preserved.
    """

    # Compute TF-IDF embeddings on GPU
    vectorizer = cuTfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df[text_column])  # Sparse matrix

    print("Fitted and normalised matrix")

    # tfidf_matrix_gpu = cp.asarray(tfidf_matrix.toarray())  # Move data to GPU

    # # Compute cosine similarity matrix on GPU
    # similarity_matrix_gpu = pairwise_distances(tfidf_matrix_gpu, tfidf_matrix_gpu, metric="cosine")
    # print("Computed similarity matrix on GPU")

    # Convert to sparse CuPy matrix (Keep it sparse!)
    tfidf_matrix_gpu = cpx.csr_matrix(tfidf_matrix)

    # Use RAPIDS NearestNeighbors (FAISS alternative)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine",
                          algorithm="brute", output_type="numpy")
    nn.fit(tfidf_matrix_gpu)

    print("Computed Nearest Neighbors on GPU")

    # Identify duplicates
    unique_indexes = []
    seen = set()

    distances, indices = nn.kneighbors(tfidf_matrix_gpu, n_neighbors=5)

    for i in tqdm(range(len(df))):
        if i in seen:
            continue
        # Find similar reviews
        # similar_indexes = pairwise_distances(similarity_matrix_gpu[i], similarity_matrix_gpu[i], metric="cosine").toarray()
        # similar_indexes = [idx for idx, val in enumerate(similar_indexes) if val > threshold]

        similar_indexes = indices[i][distances[i] < threshold]
        seen.update(similar_indexes)  # Mark them as seen
        unique_indexes.append(i)  # Keep only the first occurrence

    # Return deduplicated DataFrame
    return df.iloc[unique_indexes].reset_index(drop=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Load documents from the review dataset directory (adjust path as needed)
# csv_path = "Data/SPOTIFY_REVIEWS.csv"
csv_path = "data/SPOTIFY_REVIEWS.csv"

df = pd.read_csv(csv_path, usecols=[
                 "review_text", "review_rating", "review_likes", "review_timestamp"])

# remove rows with review text less than 8 tokens
df = remove_short_lines(df)
df = preprocess_text(df)


# Assuming df is your DataFrame
df = vectorized_deduplicate_dataframe(df)
df = df.dropna(subset=["review_text"])
# df = sort_by_recent(df)
sample_size = 500000  # Number of rows to randomly pick

# Randomly sample X rows
df_sampled = df.sample(n=sample_size, random_state=42)
df.to_csv("../Data/SPOTIFY_REVIEWS_DEDUP.csv", index=False)

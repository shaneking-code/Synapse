import polars as pl
import faiss
import numpy as np
from math import floor, sqrt
import os

def verify_normalized(vectors, tolerance=1e-6):
    norms = np.linalg.norm(vectors, axis=1)
    return np.all(np.abs(norms - 1.0) < tolerance)

def build_and_save_index(vectors_path, index_path, dim=384):
    # Load embeddings
    print("Loading parquet file...")
    wiki_embedded = pl.read_parquet(vectors_path)
    
    # Debug info
    print(f"Columns in parquet: {wiki_embedded.columns}")
    raw_embeddings = wiki_embedded["sentence_embedding"].to_numpy()
    print(f"Raw embeddings shape: {raw_embeddings.shape}")
    
    # Stack the embeddings into a 2D array
    print("Converting to 2D array...")
    embeddings = np.stack(raw_embeddings)
    print(f"Processed embeddings shape: {embeddings.shape}")
    
    # Verify dimensionality
    if embeddings.shape[1] != dim:
        raise ValueError(f"Expected dimension {dim}, got {embeddings.shape[1]}")
    
    # Verify normalization
    if not verify_normalized(embeddings):
        print("Warning: Embeddings are not normalized, normalizing now...")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # Calculate optimal number of clusters (4*sqrt(n))
    n_clusters = floor(4 * sqrt(len(embeddings)))
    
    # Build index
    print(f"Building index with {n_clusters} clusters...")
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])  # Using inner product
    index = faiss.IndexIVFFlat(
        quantizer, 
        embeddings.shape[1], 
        n_clusters,
        faiss.METRIC_INNER_PRODUCT
    )
    
    # Train and add vectors
    print("Training index...")
    index.train(embeddings)
    print("Adding vectors...")
    index.add(embeddings)
    
    # Set better default nprobe
    index.nprobe = floor(sqrt(n_clusters))
    
    # Save index
    print("Saving index...")
    faiss.write_index(index, index_path)
    
    print(f"FAISS index built with {index.ntotal} vectors and {n_clusters} clusters")
    print(f"Index saved to {index_path}")
    return index

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.read_index(index_path)

if __name__ == "__main__":
    vectors_path = "./data/wikisimple_embedded_all.parquet"
    index_path = "./data/wiki_index.faiss"
    
    try:
        # Using 384 dimensions for all-MiniLM-L6-v2 embeddings
        index = build_and_save_index(vectors_path, index_path, dim=384)
    except Exception as e:
        print(f"Error building index: {e}")


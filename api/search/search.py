import functools
import time
from typing import List, Dict, Any, Tuple
import logging

import faiss
import polars as pl
import torch
import sentence_transformers
from dataclasses import dataclass

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f"Function '{func.__name__}' executed in {(end - start):.6f} seconds")
        return result
    return wrapper

@dataclass
class SearchResult:
    score: float
    distance: float
    title: str
    url: str
    text: str
    sentence: str

class SearchEngine:
    def __init__(self, 
                 index_path: str = "./data/index.faiss",
                 metadata_path: str = "./data/meta.parquet",
                 sentences_path: str = "./data/sentences.parquet"):
        """Initialize the search engine with required models and data."""
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.sentences_path = sentences_path
        self.initialize()

    def initialize(self) -> None:
        """Load all required models and data."""
        try:
            # Initialize device
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Load models
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            self.encoder = sentence_transformers.SentenceTransformer(
                "all-MiniLM-L6-v2", 
                device=self.device
            ).eval()
            self.ranker = sentence_transformers.CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                device=self.device
            )

            # Load data
            self.metadata = pl.read_parquet(self.metadata_path)
            self.sentences = pl.read_parquet(self.sentences_path)
            
        except Exception as e:
            logging.error(f"Failed to initialize search engine: {str(e)}")
            raise RuntimeError("Search engine initialization failed") from e

    @timeit
    def semantic_search(self, query: str, k: int = 5) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Perform initial semantic search."""
        query_embedding = self.encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k=k)
        return distances, indices

    @timeit
    def rerank_results(self, query: str, sentences: pl.DataFrame, 
                      distances: pl.DataFrame, indices: pl.DataFrame) -> List[Tuple]:
        """Rerank results using cross-encoder."""
        selected_rows = [sentences.row(i) for i in indices[0]]
        pairs = [(query, row[1]) for row in selected_rows]
        scores = self.ranker.predict(pairs)
        
        return sorted(
            zip(scores, distances[0], indices[0], selected_rows),
            key=lambda x: -x[0]
        )

    def _get_article_metadata(self, index: int) -> Dict[str, Any]:
        """Retrieve article metadata for a given index."""
        article_id, _ = self.sentences.row(index)
        return self.metadata.filter(
            pl.col("article_id") == article_id
        ).to_dicts()[0]

    @timeit
    def format_search_results(self, reranked_results: List[Tuple]) -> List[SearchResult]:
        """Format the final search results."""
        results = []
        
        for score, distance, idx, row in reranked_results:
            article_data = self._get_article_metadata(idx)
            # Convert numpy.float32 to native float
            results.append(SearchResult(
                score=float(score),
                distance=float(distance),
                title=article_data["title"],
                url=article_data["url"],
                text=article_data["text"],
                sentence=row[1]
            ))
            
        return results

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Main search method that combines semantic search, reranking, and formatting.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[SearchResult]: Ranked list of search results
        """
        try:
            distances, indices = self.semantic_search(query, k)
            reranked = self.rerank_results(query, self.sentences, distances, indices)
            return self.format_search_results(reranked)
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            raise RuntimeError("Search operation failed") from e

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = SearchEngine()
    results = engine.search("Who invented artificial intelligence?")
    
    for result in results:
        print(f"Score: {result.score:.4f}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Title: {result.title}")
        print(f"Sentence: {result.sentence}")
        print(f"URL: {result.url}")
        print("*" * 100)

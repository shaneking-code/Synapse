from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from dataclasses import asdict
from search.search import SearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Search Engine API",
    description="An API for semantic search using FAISS and Sentence-Transformers",
    version="1.0.0"
)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the search engine once during startup
search_engine = SearchEngine()

@app.get("/search")
async def search(
    query: str = Query(
        ...,  # ... means required
        min_length=1,
        max_length=100,
        regex="^[\x20-\x7E]+$",  # printable ASCII characters only
        description="Search query string (ASCII characters only)",
        example="sample search query"
    ),
    k: int = Query(
        default=5,
        ge=1,
        le=9,
        description="Number of results to return (max 9)",
        example=5
    )
):
    try:
        results = search_engine.search(query, k)
        # Convert each SearchResult data class instance into a dictionary
        return [asdict(result) for result in results]
    except Exception as e:
        logging.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import { useState } from 'react'
import './App.css'

interface SearchResult {
  score: number;
  distance: number;
  title: string;
  url: string;
  text: string;
  sentence: string;
}

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(query)}&k=5`);
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to perform search. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const highlightSentence = (text: string, sentence: string) => {
    // Escape special regex characters in the sentence
    const escapedSentence = sentence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escapedSentence})`, 'gi');
    const parts = text.split(regex);

    return (
      <>
        {parts.map((part, i) => 
          regex.test(part) ? (
            <mark key={i} className="highlight">{part}</mark>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </>
    );
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Search Engine</h1>
      </header>

      <main className="main">
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="search-input"
          />
          <button type="submit" className="search-button" disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        <div className="results">
          {results.map((result, index) => (
            <article key={index} className="result-card">
              <h2>
                <a href={result.url} target="_blank" rel="noopener noreferrer">
                  {result.title}
                </a>
              </h2>
              <p>{highlightSentence(result.text, result.sentence)}</p>
              <div className="score">
                Relevance Score: {result.score.toFixed(4)}
              </div>
            </article>
          ))}
        </div>
      </main>
    </div>
  )
}

export default App

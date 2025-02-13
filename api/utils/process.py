import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

wiki   = pd.read_parquet("./data/wiki.parquet")
corpus = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SentenceTransformer('all-MiniLM-L6-v2').to(device)
model.eval()  # Set model to evaluation mode for inference

with torch.no_grad():  # Disable grad computations
    for row in tqdm(wiki.itertuples(index=False), total=len(wiki)):
        
        article_id    = row.id
        article_url   = row.url
        article_title = row.title
        article_text  = row.text

        # Use NLTK's sent_tokenize for fast sentence splitting
        article_sentences = sent_tokenize(article_text)
    
        # Increase batch size to 256 and encode all sentences at once
        encoded_tensor = model.encode(
            article_sentences,
            batch_size=256,
            device=device,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
    
        # Transfer the entire batch from GPU to CPU in one call
        encoded_sentences = encoded_tensor.cpu().numpy()

        assert len(article_sentences) == len(encoded_sentences)
        article_metadata = [
            {
                'article_id': article_id,
                'article_title': article_title,
                'sentence_id': sentence_id,
                'sentence_text': sentence,
                'sentence_embeddings': sentence_embedding
            }
            for sentence_id, (sentence, sentence_embedding) in enumerate(zip(article_sentences, encoded_sentences))
        ]
    
        corpus.extend(article_metadata)

assert len(corpus) == len(wiki)
corpus = pd.DataFrame(corpus)
corpus.to_parquet("./data/corpus.parquet")
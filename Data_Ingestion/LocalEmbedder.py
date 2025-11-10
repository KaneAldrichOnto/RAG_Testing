import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding model
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
        - all-mpnet-base-v2: Higher quality (768 dimensions)
        - all-MiniLM-L12-v2: Balance of speed/quality (384 dimensions)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Check if GPU is available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print("Using GPU for embeddings")
        else:
            print("Using CPU for embeddings")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed single text or list of texts
        Returns numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
     
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            # Fall back: compute embedding for a single dummy string and infer dimension
            emb = self.embed_text(" ")
            # emb shape is (1, dim) -> take second axis
            dim = int(emb.shape[1])
        return int(dim)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self.embed_text([text1, text2])
        
        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        return dot_product / (norm1 * norm2)

# Test the embedder
if __name__ == "__main__":
    # Initialize embedder
    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    
    # Test single text
    test_text = "This is a test document about machine learning."
    embedding = embedder.embed_text(test_text)
    print(f"Single text embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    
    # Test similarity
    text_a = "Machine learning is a field of artificial intelligence."
    text_b = "Artificial intelligence includes machine learning techniques."
    sim = embedder.similarity(text_a, text_b)
    print(f"Similarity between text A and B: {sim:.4f}")
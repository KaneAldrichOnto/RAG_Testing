import os
import dotenv
from openai import AzureOpenAI
import numpy as np
from typing import List, Union
import time

class LocalEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize Azure OpenAI embedding model
        
        Supported Azure OpenAI embedding models:
        - text-embedding-3-small: 1536 dimensions, fast and efficient
        - text-embedding-3-large: 3072 dimensions, highest quality
        - text-embedding-ada-002: 1536 dimensions, legacy model
        """
        # Load environment variables
        dotenv.load_dotenv()
        
        # Get Azure OpenAI credentials
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY", "")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", model_name)
        
        if not all([self.azure_openai_endpoint, self.azure_openai_key]):
            raise ValueError(
                "Azure OpenAI configuration missing. Set AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_KEY environment variables."
            )
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_key,
            api_version="2024-02-01"
        )
        
        self.model_name = model_name
        self.deployment_name = self.embedding_deployment
        
        # Set dimensions based on model
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        print(f"🔗 Loading Azure OpenAI embedding model: {model_name}")
        print(f"   Endpoint: {self.azure_openai_endpoint}")
        print(f"   Deployment: {self.deployment_name}")
        print(f"   Expected dimensions: {self.get_embedding_dimension()}")
        
        # Test the connection
        try:
            test_embedding = self._get_embeddings_from_api(["test"])
            actual_dim = len(test_embedding[0])
            print(f"   ✅ Connection successful! Actual dimensions: {actual_dim}")
        except Exception as e:
            print(f"   ❌ Connection test failed: {str(e)}")
            raise

    def embed_text(self, text: Union[str, List[str]], batch_size: int = 100) -> np.ndarray:
        """
        Embed single text or list of texts using Azure OpenAI
        
        Args:
            text: Single string or list of strings to embed
            batch_size: Number of texts to process in each API call
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        # Process in batches to handle API limits
        all_embeddings = []
        
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            
            try:
                batch_embeddings = self._get_embeddings_from_api(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Add small delay between batches to avoid rate limiting
                if len(text) > batch_size and i + batch_size < len(text):
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Error embedding batch {i//batch_size + 1}: {str(e)}")
                # Return zeros for failed batch (you might want different error handling)
                failed_embeddings = np.zeros((len(batch), self.get_embedding_dimension()))
                all_embeddings.extend(failed_embeddings.tolist())
        
        return np.array(all_embeddings)

    def _get_embeddings_from_api(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from Azure OpenAI API
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.deployment_name,
                input=texts
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            return embeddings
            
        except Exception as e:
            print(f"❌ API call failed: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        return self.dimension_map.get(self.model_name, 1536)  # Default to 1536 if unknown

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self.embed_text([text1, text2])
        
        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        return dot_product / (norm1 * norm2)

    def embed_texts_with_retry(self, 
                              texts: List[str], 
                              max_retries: int = 3, 
                              batch_size: int = 100) -> np.ndarray:
        """
        Embed texts with retry logic for better reliability
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retry attempts
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        for attempt in range(max_retries):
            try:
                return self.embed_text(texts, batch_size=batch_size)
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # This should never be reached due to the raise above
        return np.array([])

    def get_usage_stats(self) -> dict:
        """
        Get approximate usage statistics (tokens processed)
        Note: Azure OpenAI charges by tokens, roughly 1 token per 4 characters
        """
        # This is a simple approximation - you might want to track actual usage
        return {
            "model": self.model_name,
            "deployment": self.deployment_name,
            "dimension": self.get_embedding_dimension(),
            "note": "Track actual usage through Azure portal for billing"
        }



# Test the embedder
if __name__ == "__main__":
    # Test Azure OpenAI embedder
    print("🧪 Testing Azure OpenAI Embedder...")
    
    try:
        embedder = LocalEmbedder("text-embedding-3-small")
        
        # Test single text
        test_text = "This is a test document about machine learning."
        embedding = embedder.embed_text(test_text)
        print(f"✅ Single text embedding shape: {embedding.shape}")
        print(f"✅ Embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Test batch embedding
        test_texts = [
            "Machine learning is a field of artificial intelligence.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with many layers."
        ]
        
        batch_embeddings = embedder.embed_text(test_texts)
        print(f"✅ Batch embeddings shape: {batch_embeddings.shape}")
        
        # Test similarity
        text_a = "Machine learning is a field of artificial intelligence."
        text_b = "Artificial intelligence includes machine learning techniques."
        sim = embedder.similarity(text_a, text_b)
        print(f"✅ Similarity between text A and B: {sim:.4f}")
        
        # Usage stats
        stats = embedder.get_usage_stats()
        print(f"✅ Usage stats: {stats}")
        
    except Exception as e:
        print(f"❌ Azure OpenAI embedder failed: {str(e)}")
    
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import dotenv
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "Data_Ingestion"))
from LocalEmbedder import LocalEmbedder

class AzureSearchKnowledgeBase:
    def __init__(self, 
                 index_name: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 5):
        """
        Initialize the Azure Search Knowledge Base for RAG.
        """
        # Load environment variables
        dotenv.load_dotenv()
        self.azure_ai_search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT", "")
        self.azure_ai_search_key = os.getenv("AZURE_AI_SEARCH_KEY", "")
        
        if not self.azure_ai_search_endpoint or not self.azure_ai_search_key:
            raise ValueError("Azure Search credentials not found. Set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_KEY")
        
        self.index_name = index_name
        self.top_k = top_k
        
        # Initialize search client
        self.search_client = SearchClient(
            endpoint=self.azure_ai_search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.azure_ai_search_key)
        )
        
        # Initialize embedder for query encoding
        self.embedder = LocalEmbedder(embedding_model)
        
        print(f"🔍 Azure Search Knowledge Base initialized:")
        print(f"   Index: {index_name}")
        print(f"   Embedding model: {embedding_model}")
        print(f"   Default top_k: {top_k}")

    def _prepare_embedding(self, embedding) -> List[float]:
        """
        Convert embedding to the correct format for Azure Search.
        
        Args:
            embedding: Embedding from LocalEmbedder (could be various formats)
            
        Returns:
            List of floats ready for Azure Search
        """
        # Handle different embedding formats
        if hasattr(embedding, 'shape'):
            # It's a numpy array
            if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                # Shape is (1, 384) - flatten it
                embedding = embedding.flatten()
            elif len(embedding.shape) == 1:
                # Shape is (384,) - already correct
                pass
            else:
                raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
            
            # Convert to list
            return embedding.tolist()
        elif isinstance(embedding, list):
            # Already a list
            return embedding
        else:
            # Try to convert to list
            return list(embedding)

    def get_context_for_query(self, 
                             user_query: str, 
                             top_k: Optional[int] = None,
                             filters: Optional[str] = None,
                             include_tables: bool = True,
                             min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get relevant context chunks for a user query using vector similarity search.
        """
        if top_k is None:
            top_k = self.top_k
        
        print(f"🔍 Searching for context: '{user_query[:50]}...'")
        
        # Generate embedding for the query
        query_embedding = self.embedder.embed_text(user_query)
        
        # Convert embedding to correct format
        embedding_vector = self._prepare_embedding(query_embedding)
        
        print(f"   Prepared embedding: {len(embedding_vector)} dimensions")
        
        # Prepare filters
        search_filters = self._build_filters(filters, include_tables)
        
        try:
            vector_query = VectorizedQuery(
                vector=embedding_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            results = self.search_client.search(
                search_text=None,  # Pure vector search
                vector_queries=[vector_query],
                filter=search_filters,
                select=[
                    "id", "content", "title", "document_id", "section_title",
                    "chunk_type", "is_table", "token_count", "chunk_index",
                    "document_title", "document_filename"
                ]
            )
            
            # Process results
            context_chunks = []
            for result in results:
                score = result.get('@search.score', 0.0)
                
                if score >= min_score:
                    context_chunk = {
                        "content": result.get("content", ""),
                        "score": score,
                        "metadata": {
                            "id": result.get("id"),
                            "title": result.get("title", ""),
                            "document_id": result.get("document_id"),
                            "document_title": result.get("document_title", ""),
                            "document_filename": result.get("document_filename", ""),
                            "section_title": result.get("section_title", ""),
                            "chunk_type": result.get("chunk_type", ""),
                            "is_table": result.get("is_table", False),
                            "token_count": result.get("token_count", 0),
                            "chunk_index": result.get("chunk_index", 0)
                        }
                    }
                    context_chunks.append(context_chunk)
            
            print(f"   Found {len(context_chunks)} relevant chunks (scores: {min_score:.2f}+)")
            return context_chunks
            
        except Exception as e:
            print(f"❌ Error during vector search: {str(e)}")
            print(f"   Original embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else type(query_embedding)}")
            print(f"   Prepared embedding length: {len(embedding_vector) if 'embedding_vector' in locals() else 'N/A'}")
            return []

    def get_context_hybrid_search(self, 
                                 user_query: str, 
                                 top_k: Optional[int] = None,
                                 filters: Optional[str] = None,
                                 include_tables: bool = True,
                                 min_score: float = 0.0,
                                 text_weight: float = 0.3,
                                 vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get relevant context using hybrid search (text + vector).
        """
        if top_k is None:
            top_k = self.top_k
        
        print(f"🔍 Hybrid search for: '{user_query[:50]}...'")
        
        # Generate embedding for the query
        query_embedding = self.embedder.embed_text(user_query)
        
        # Convert embedding to correct format
        embedding_vector = self._prepare_embedding(query_embedding)
        
        # Prepare filters
        search_filters = self._build_filters(filters, include_tables)
        
        try:
            vector_query = VectorizedQuery(
                vector=embedding_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            results = self.search_client.search(
                search_text=user_query,  # Text search component
                vector_queries=[vector_query],  # Vector search component
                filter=search_filters,
                top=top_k,
                select=[
                    "id", "content", "title", "document_id", "section_title",
                    "chunk_type", "is_table", "token_count", "chunk_index",
                    "document_title", "document_filename"
                ]
            )
            
            # Process results
            context_chunks = []
            for result in results:
                score = result.get('@search.score', 0.0)
                
                if score >= min_score:
                    context_chunk = {
                        "content": result.get("content", ""),
                        "score": score,
                        "metadata": {
                            "id": result.get("id"),
                            "title": result.get("title", ""),
                            "document_id": result.get("document_id"),
                            "document_title": result.get("document_title", ""),
                            "document_filename": result.get("document_filename", ""),
                            "section_title": result.get("section_title", ""),
                            "chunk_type": result.get("chunk_type", ""),
                            "is_table": result.get("is_table", False),
                            "token_count": result.get("token_count", 0),
                            "chunk_index": result.get("chunk_index", 0)
                        }
                    }
                    context_chunks.append(context_chunk)
            
            print(f"   Found {len(context_chunks)} relevant chunks (hybrid search)")
            return context_chunks
            
        except Exception as e:
            print(f"❌ Error during hybrid search: {str(e)}")
            return []

    def get_table_context(self, 
                         user_query: str, 
                         top_k: Optional[int] = None,
                         min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get context specifically from table chunks.
        """
        filter_expression = "is_table eq true"
        
        return self.get_context_for_query(
            user_query=user_query,
            top_k=top_k,
            filters=filter_expression,
            min_score=min_score
        )

    def format_context_for_llm(self, 
                              context_chunks: List[Dict[str, Any]], 
                              include_metadata: bool = True,
                              max_tokens: int = 4000) -> str:
        """
        Format context chunks into a string suitable for LLM prompts.
        """
        if not context_chunks:
            return "No relevant context found."
        
        formatted_parts = []
        current_tokens = 0
        
        for i, chunk in enumerate(context_chunks):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            chunk_tokens = len(content) // 4
            
            if current_tokens + chunk_tokens > max_tokens:
                print(f"   Truncated context at {i} chunks due to token limit ({max_tokens})")
                break
            
            # Format chunk
            if include_metadata:
                source_info = f"[Source: {metadata.get('document_title', 'Unknown')} - {metadata.get('section_title', 'Unknown Section')}]"
                if metadata.get('is_table'):
                    source_info += " [TABLE DATA]"
                
                formatted_chunk = f"{source_info}\n{content}\n"
            else:
                formatted_chunk = f"{content}\n"
            
            formatted_parts.append(formatted_chunk)
            current_tokens += chunk_tokens
        
        context_text = "\n---\n".join(formatted_parts)
        
        print(f"   Formatted {len(formatted_parts)} chunks (~{current_tokens} tokens)")
        return context_text

    def _build_filters(self, 
                      user_filters: Optional[str], 
                      include_tables: bool) -> Optional[str]:
        """
        Build OData filter expression for Azure Search.
        """
        filters = []
        
        # Add table filter if needed
        if not include_tables:
            filters.append("is_table eq false")
        
        # Add user filters
        if user_filters:
            filters.append(f"({user_filters})")
        
        return " and ".join(filters) if filters else None

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        """
        try:
            # Get total document count
            count_result = self.search_client.search(search_text="*", include_total_count=True)
            total_docs = count_result.get_count()
            
            # Try faceting - handle if some fields aren't facetable
            try:
                facet_result = self.search_client.search(
                    search_text="*",
                    facets=["document_id", "chunk_type", "document_filename"]
                )
                facets = facet_result.get_facets()
            except:
                facets = {}
            
            # Count table chunks manually since is_table might not be facetable
            try:
                table_count_result = self.search_client.search(
                    search_text="*",
                    filter="is_table eq true",
                    include_total_count=True
                )
                table_chunks_count = table_count_result.get_count()
            except:
                table_chunks_count = 0
            
            stats = {
                "total_chunks": total_docs,
                "unique_documents": len(facets.get("document_id", [])) if "document_id" in facets else 0,
                "chunk_types": {f["value"]: f["count"] for f in facets.get("chunk_type", [])},
                "table_chunks": table_chunks_count,
                "documents": {f["value"]: f["count"] for f in facets.get("document_filename", [])}
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats: {str(e)}")
            return {"total_chunks": 0}

    def debug_vector_search(self, user_query: str):
        """
        Debug method to troubleshoot vector search issues.
        """
        print(f"🔧 Debugging vector search for: '{user_query}'")
        
        # Test embedding generation
        try:
            query_embedding = self.embedder.embed_text(user_query)
            print(f"   ✅ Original embedding: shape={query_embedding.shape if hasattr(query_embedding, 'shape') else len(query_embedding)}")
            print(f"   ✅ Embedding type: {type(query_embedding)}")
            
            # Convert to proper format
            embedding_vector = self._prepare_embedding(query_embedding)
            print(f"   ✅ Prepared embedding: length={len(embedding_vector)}")
            print(f"   ✅ First 5 values: {embedding_vector[:5]}")
            
            # Test vector query creation
            vector_query = VectorizedQuery(
                vector=embedding_vector,
                k_nearest_neighbors=3,
                fields="contentVector"
            )
            print(f"   ✅ VectorizedQuery created successfully")
            
            # Test simple search
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=3
            )
            
            result_count = 0
            for result in results:
                result_count += 1
                score = result.get('@search.score', 'N/A')
                content_preview = result.get('content', '')[:50] + "..." if result.get('content') else 'No content'
                print(f"   ✅ Result {result_count}: score={score}, preview='{content_preview}'")
            
            print(f"   ✅ Search completed with {result_count} results")
            
        except Exception as e:
            print(f"   ❌ Debug failed: {str(e)}")
            import traceback
            traceback.print_exc()


# Example usage and testing
if __name__ == "__main__":
    # Initialize knowledge base
    kb = AzureSearchKnowledgeBase(
        index_name="testing_index",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5
    )
    
    # Debug vector search first
    print("\n🔧 Running vector search debug...")
    kb.debug_vector_search("What are the steps of the keypoint alignment algorithm?")
    
    # Get knowledge base statistics
    print("\n📊 Knowledge Base Statistics:")
    stats = kb.get_knowledge_base_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test the actual query
    test_query = "What are the steps of the keypoint alignment algorithm?"
    print(f"\n🔍 Testing query: '{test_query}'")
    
    # Vector search
    context = kb.get_context_for_query(test_query, top_k=3)
    
    if context:
        print(f"   Found {len(context)} relevant chunks:")
        for i, chunk in enumerate(context):
            metadata = chunk["metadata"]
            print(f"     {i+1}. Score: {chunk['score']:.3f} | {metadata['document_title']} - {metadata['section_title']}")
            print(f"        Preview: {chunk['content'][:100]}...")
        
        # Format for LLM
        formatted = kb.format_context_for_llm(context, include_metadata=True)
        print(f"\n📝 Formatted context ({len(formatted)} chars):")
        print(formatted)
    else:
        print("   No relevant context found")
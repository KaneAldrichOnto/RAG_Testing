import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import dotenv
from openai import AzureOpenAI

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))

from GetContext import AzureSearchKnowledgeBase

class RAGQuestionAnswerer:
    def __init__(self,
                 search_index_name: str,
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_key: Optional[str] = None,
                 azure_openai_deployment: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 default_top_k: int = 5,
                 max_context_tokens: int = 4000,
                 temperature: float = 0.7):
        """
        Initialize the RAG Question Answerer with Azure AI Foundry.
        
        Args:
            search_index_name: Name of the Azure Search index
            azure_openai_endpoint: Azure OpenAI endpoint URL
            azure_openai_key: Azure OpenAI API key
            azure_openai_deployment: Name of your deployment in Azure AI Foundry
            embedding_model: Embedding model name (should match indexing)
            default_top_k: Default number of context chunks to retrieve
            max_context_tokens: Maximum tokens for context
            temperature: LLM temperature setting
        """
        # Load environment variables
        dotenv.load_dotenv()
        
        # Initialize Azure OpenAI client
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_openai_key = azure_openai_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_openai_deployment = azure_openai_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        
        if not all([self.azure_openai_endpoint, self.azure_openai_key, self.azure_openai_deployment]):
            raise ValueError(
                "Azure OpenAI configuration missing. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_KEY, and AZURE_OPENAI_DEPLOYMENT environment variables "
                "or pass them as parameters."
            )
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_key,
            api_version="2024-02-01"
        )
        
        # Initialize knowledge base
        self.knowledge_base = AzureSearchKnowledgeBase(
            index_name=search_index_name,
            embedding_model=embedding_model,
            top_k=default_top_k
        )
        
        # Configuration
        self.deployment_name = self.azure_openai_deployment
        self.default_top_k = default_top_k
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        
        # System prompt for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 

Instructions:
- Use the provided context to answer the user's question accurately and comprehensively
- If the context doesn't contain enough information to answer the question, say so clearly
- Always cite your sources by mentioning the document sections you're referencing
- If you find conflicting information in the context, mention it
- Be concise but thorough in your explanations
- When referencing tables or structured data, format them clearly

Context will be provided in the format:
[Source: Document Title - Section Title] [TABLE DATA if applicable]
Content...
---
[Source: Document Title - Section Title]
Content...
"""
        
        print(f"🤖 RAG Question Answerer initialized:")
        print(f"   Azure OpenAI Endpoint: {self.azure_openai_endpoint}")
        print(f"   Deployment: {self.deployment_name}")
        print(f"   Search Index: {search_index_name}")
        print(f"   Embedding Model: {embedding_model}")

    def ask_question(self,
                    question: str,
                    top_k: Optional[int] = None,
                    include_tables: bool = True,
                    min_score: float = 0.0,
                    custom_system_prompt: Optional[str] = None,
                    temperature: Optional[float] = None,
                    max_response_tokens: int = 1500,
                    search_filters: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG (Retrieval-Augmented Generation).
        
        Args:
            question: The user's question
            top_k: Number of context chunks to retrieve
            include_tables: Whether to include table data in context
            min_score: Minimum similarity score for context chunks
            custom_system_prompt: Custom system prompt (overrides default)
            temperature: LLM temperature (overrides default)
            max_response_tokens: Maximum tokens for the response
            search_filters: OData filters for Azure Search
            
        Returns:
            Dictionary containing the answer, context used, and metadata
        """
        if top_k is None:
            top_k = self.default_top_k
        
        if temperature is None:
            temperature = self.temperature
        
        print(f"❓ Question: {question}")
        print(f"🔍 Retrieving context (top_k={top_k}, include_tables={include_tables})...")
        
        # Step 1: Retrieve relevant context
        context_chunks = self.knowledge_base.get_context_for_query(
            user_query=question,
            top_k=top_k,
            filters=search_filters,
            include_tables=include_tables,
            min_score=min_score
        )
        
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or check if the information exists in the indexed documents.",
                "context_used": [],
                "sources": [],
                "confidence": "low",
                "context_found": False,
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0}
            }
        
        # Step 2: Format context for LLM
        formatted_context = self.knowledge_base.format_context_for_llm(
            context_chunks, 
            include_metadata=True,
            max_tokens=self.max_context_tokens
        )
        
        # Step 3: Create the prompt
        system_prompt = custom_system_prompt or self.system_prompt
        
        user_prompt = f"""Based on the following context, please answer this question: {question}

CONTEXT:
{formatted_context}

Please provide a comprehensive answer based on the context above."""
        
        print(f"🤖 Generating response with {len(context_chunks)} context chunks...")
        
        # Step 4: Call Azure OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_response_tokens,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract usage information
            usage = response.usage
            tokens_used = {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
                "total": usage.total_tokens if usage else 0
            }
            
            # Determine confidence based on context quality
            confidence = self._assess_confidence(context_chunks, question)
            
            # Extract sources
            sources = self._extract_sources(context_chunks)
            
            print(f"✅ Response generated successfully!")
            print(f"   Tokens used: {tokens_used['total']} (prompt: {tokens_used['prompt']}, completion: {tokens_used['completion']})")
            print(f"   Confidence: {confidence}")
            
            return {
                "answer": answer,
                "context_used": context_chunks,
                "sources": sources,
                "confidence": confidence,
                "context_found": True,
                "tokens_used": tokens_used,
                "question": question,
                "search_params": {
                    "top_k": top_k,
                    "include_tables": include_tables,
                    "min_score": min_score,
                    "filters": search_filters
                }
            }
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return {
                "answer": f"I encountered an error while generating the response: {str(e)}",
                "context_used": context_chunks,
                "sources": self._extract_sources(context_chunks),
                "confidence": "error",
                "context_found": True,
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0},
                "error": str(e)
            }

    def ask_question_with_hybrid_search(self,
                                      question: str,
                                      top_k: Optional[int] = None,
                                      include_tables: bool = True,
                                      min_score: float = 0.0,
                                      **kwargs) -> Dict[str, Any]:
        """
        Answer a question using hybrid search (text + vector) for better context retrieval.
        """
        if top_k is None:
            top_k = self.default_top_k
        
        print(f"❓ Question: {question}")
        print(f"🔍 Retrieving context with hybrid search (top_k={top_k})...")
        
        # Use hybrid search instead of pure vector search
        context_chunks = self.knowledge_base.get_context_hybrid_search(
            user_query=question,
            top_k=top_k,
            include_tables=include_tables,
            min_score=min_score
        )
        
        if not context_chunks:
            return self.ask_question(question, top_k=0, **kwargs)  # Return no-context response
        
        # Continue with normal RAG pipeline
        return self._generate_response_from_context(question, context_chunks, **kwargs)

    def get_table_information(self,
                            question: str,
                            top_k: Optional[int] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Answer questions specifically about table/structured data.
        """
        if top_k is None:
            top_k = self.default_top_k
        
        print(f"📊 Table-focused question: {question}")
        
        context_chunks = self.knowledge_base.get_table_context(
            user_query=question,
            top_k=top_k
        )
        
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant table data to answer your question.",
                "context_used": [],
                "sources": [],
                "confidence": "low",
                "context_found": False,
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0}
            }
        
        # Use a table-specific system prompt
        table_system_prompt = """You are a helpful AI assistant specialized in analyzing and explaining table data and structured information.

Instructions:
- Focus on the tabular/structured data provided in the context
- Present table information in a clear, formatted way
- Explain relationships and patterns you see in the data
- If asked for specific values, extract them accurately from the tables
- When summarizing tables, maintain the structure and highlight key insights
- Always cite which table or data source you're referencing
"""
        
        return self._generate_response_from_context(
            question, 
            context_chunks, 
            custom_system_prompt=table_system_prompt,
            **kwargs
        )

    def _generate_response_from_context(self,
                                      question: str,
                                      context_chunks: List[Dict[str, Any]],
                                      **kwargs) -> Dict[str, Any]:
        """
        Generate response from provided context chunks.
        """
        # Format context for LLM
        formatted_context = self.knowledge_base.format_context_for_llm(
            context_chunks, 
            include_metadata=True,
            max_tokens=self.max_context_tokens
        )
        
        # Create the prompt
        system_prompt = kwargs.get('custom_system_prompt', self.system_prompt)
        temperature = kwargs.get('temperature', self.temperature)
        max_response_tokens = kwargs.get('max_response_tokens', 1500)
        
        user_prompt = f"""Based on the following context, please answer this question: {question}

CONTEXT:
{formatted_context}

Please provide a comprehensive answer based on the context above."""
        
        print(f"🤖 Generating response with {len(context_chunks)} context chunks...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_response_tokens,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            return {
                "answer": answer,
                "context_used": context_chunks,
                "sources": self._extract_sources(context_chunks),
                "confidence": self._assess_confidence(context_chunks, question),
                "context_found": True,
                "tokens_used": {
                    "prompt": usage.prompt_tokens if usage else 0,
                    "completion": usage.completion_tokens if usage else 0,
                    "total": usage.total_tokens if usage else 0
                }
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "context_used": context_chunks,
                "sources": self._extract_sources(context_chunks),
                "confidence": "error",
                "context_found": True,
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0},
                "error": str(e)
            }

    def _assess_confidence(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Assess confidence in the answer based on context quality.
        """
        if not context_chunks:
            return "low"
        
        # Simple heuristics for confidence
        avg_score = sum(chunk["score"] for chunk in context_chunks) / len(context_chunks)
        num_chunks = len(context_chunks)
        
        if avg_score > 0.8 and num_chunks >= 3:
            return "high"
        elif avg_score > 0.6 and num_chunks >= 2:
            return "medium"
        else:
            return "low"

    def _extract_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from context chunks.
        """
        sources = []
        seen_sources = set()
        
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            source_key = f"{metadata.get('document_title', '')}-{metadata.get('section_title', '')}"
            
            if source_key not in seen_sources:
                sources.append({
                    "document_title": metadata.get("document_title", "Unknown"),
                    "section_title": metadata.get("section_title", "Unknown Section"),
                    "document_filename": metadata.get("document_filename", "Unknown File"),
                    "chunk_type": metadata.get("chunk_type", "unknown"),
                    "is_table": metadata.get("is_table", False)
                })
                seen_sources.add(source_key)
        
        return sources

    def chat_conversation(self,
                         conversation_history: List[Dict[str, str]],
                         current_question: str,
                         maintain_context: bool = True) -> Dict[str, Any]:
        """
        Handle multi-turn conversations with maintained context.
        
        Args:
            conversation_history: List of {"role": "user/assistant", "content": "..."}
            current_question: The current user question
            maintain_context: Whether to use previous context for current question
            
        Returns:
            Response dictionary with conversation context
        """
        # For now, treat each question independently
        # You could enhance this to maintain context across turns
        
        result = self.ask_question(current_question)
        
        # Add conversation history to the result
        result["conversation_history"] = conversation_history + [
            {"role": "user", "content": current_question},
            {"role": "assistant", "content": result["answer"]}
        ]
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGQuestionAnswerer(
        search_index_name="rag-test-index",
        embedding_model="all-MiniLM-L6-v2",
        default_top_k=3,
        temperature=0.7
    )
    
    # Test questions
    test_questions = [
        "What are the steps of the keypoint alignment algorithm?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)
        
        # Get answer
        result = rag.ask_question(question, top_k=3)
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Context chunks used: {len(result.get('context_used', []))}")
        print(f"Tokens used: {result.get('tokens_used', {})}")
        
        # Show sources
        print(f"\nSources:")
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"  {i}. {source['document_title']} - {source['section_title']} ({'Table' if source['is_table'] else 'Text'})")
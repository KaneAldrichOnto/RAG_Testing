import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
import json

from AzureDocumentIntelligence import AzureDocumentIntelligenceBridge
from Tokenizer import Tokenizer
from LocalEmbedder import LocalEmbedder

class FilesToEmbeddings:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "./cache",
                 target_chunk_size: int = 512,
                 overlap_percentage: float = 0.0):
        """
        Initialize the FilesToEmbeddings pipeline.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            cache_dir: Directory to store cached ADI results
            target_chunk_size: Target size for text chunks in tokens
            overlap_percentage: Percentage overlap between chunks
        """
        self.adi_bridge = AzureDocumentIntelligenceBridge()
        self.tokenizer = Tokenizer("gpt-3.5-turbo")
        self.embedder = LocalEmbedder(embedding_model)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.target_chunk_size = target_chunk_size
        self.overlap_percentage = overlap_percentage
        
        print(f"FilesToEmbeddings initialized with:")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - Embedding dimension: {self.embedder.get_embedding_dimension()}")
        print(f"  - Cache directory: {self.cache_dir}")
        print(f"  - Target chunk size: {target_chunk_size} tokens")
        print(f"  - Overlap percentage: {overlap_percentage}")

    def process_document(self, 
                        file_path: Union[str, Path], 
                        document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a document from file path to embeddings with metadata.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom document ID, otherwise uses filename
            
        Returns:
            List of dictionaries containing embeddings and metadata for vector database
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = file_path.stem
        
        print(f"\n🔄 Processing document: {file_path.name}")
        print(f"   Document ID: {document_id}")
        
        # Step 1: Get or create ADI analysis
        analysis_result = self._get_or_create_adi_analysis(file_path)
        
        # Step 2: Format into structured document
        print("📋 Formatting document structure...")
        structured_doc = self.adi_bridge.format_result_as_structured_document(analysis_result)
        
        # Step 3: Generate chunks
        print("✂️  Generating intelligent chunks...")
        chunks = self.tokenizer.chunk_adi_document(
            structured_doc,
            target_chunk_size=self.target_chunk_size,
            overlap_percentage=self.overlap_percentage
        )
        
        print(f"   Generated {len(chunks)} chunks")
        
        # Step 4: Generate embeddings
        print("🧠 Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_text(chunk_texts)
        
        # Step 5: Prepare final results with metadata
        print("📦 Preparing final results...")
        results = self._prepare_vector_db_entries(
            chunks, embeddings, document_id, file_path, structured_doc
        )
        
        print(f"✅ Successfully processed {file_path.name}")
        print(f"   Total embeddings generated: {len(results)}")
        
        return results

    def process_multiple_documents(self, 
                                 file_paths: List[Union[str, Path]],
                                 document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents and return combined embeddings list.
        
        Args:
            file_paths: List of paths to document files
            document_ids: Optional list of custom document IDs
            
        Returns:
            Combined list of all embeddings with metadata
        """
        all_results = []
        
        if document_ids is None:
            document_ids = [None] * len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                doc_id = document_ids[i] if i < len(document_ids) else None
                results = self.process_document(file_path, doc_id)
                all_results.extend(results)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                continue
        
        print(f"\n🎉 Batch processing complete!")
        print(f"   Total documents processed: {len(file_paths)}")
        print(f"   Total embeddings generated: {len(all_results)}")
        
        return all_results

    def _get_or_create_adi_analysis(self, file_path: Path):
        """Get cached ADI analysis or create new one."""
        # Generate cache filename based on file content hash for consistency
        file_hash = self._get_file_hash(file_path)
        cache_filename = f"{file_path.stem}_{file_hash}.pkl"
        cache_path = self.cache_dir / cache_filename
        
        if cache_path.exists():
            print(f"📁 Loading cached ADI analysis: {cache_filename}")
            return self.adi_bridge.load_analysis_result(str(cache_path))
        else:
            print(f"🔍 Running ADI analysis on new document...")
            analysis_result = self.adi_bridge.analyze_local_document(
                str(file_path), 
                str(cache_path)
            )
            return analysis_result

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content for cache consistency."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:8]  # Use first 8 chars for brevity

    def _prepare_vector_db_entries(self, 
                                  chunks: List[Dict[str, Any]], 
                                  embeddings, 
                                  document_id: str,
                                  file_path: Path,
                                  structured_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare final entries for vector database with comprehensive metadata.
        """
        results = []
        
        # Get file metadata
        file_stats = file_path.stat()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique chunk ID
            chunk_id = f"{document_id}_chunk_{i:04d}"
            
            # Prepare comprehensive metadata
            metadata = {
                # Document-level metadata
                "document_id": document_id,
                "document_title": structured_doc.get('title', file_path.stem),
                "document_filename": file_path.name,
                "document_pages": structured_doc.get('total_pages', 0),
                "file_size_bytes": file_stats.st_size,
                "file_modified": file_stats.st_mtime,
                
                # Chunk-level metadata
                "chunk_id": chunk_id,
                "chunk_index": i,
                "chunk_type": chunk['metadata']['chunk_type'],
                "token_count": chunk['metadata']['token_count'],
                "character_count": len(chunk['text']),
                
                # Section metadata
                "section_title": chunk['metadata'].get('section_title', ''),
                "subsection_title": chunk['metadata'].get('subsection_title', ''),
                "page_number": chunk['metadata'].get('page', 0),
                
                # Special handling for tables
                "is_table": chunk['metadata']['chunk_type'] in ['table', 'subsection_table'],
                "table_index": chunk['metadata'].get('table_index'),
                "exceeds_target": chunk['metadata'].get('exceeds_target', False),
                
                # Overlap information
                "has_overlap": chunk.get('has_overlap', False),
                "overlap_from_chunk": chunk.get('overlap_from_chunk'),
                
                # Document statistics for context
                "total_chunks_in_document": len(chunks),
                "total_sections_in_document": structured_doc.get('statistics', {}).get('total_sections', 0),
                "total_tables_in_document": structured_doc.get('statistics', {}).get('total_tables', 0),
            }
            
            # Add any document-specific metadata from ADI
            doc_metadata = structured_doc.get('metadata', {})
            for key, value in doc_metadata.items():
                metadata[f"doc_meta_{key}"] = value
            
            # Create final entry
            entry = {
                "id": chunk_id,
                "text": chunk['text'],
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                "metadata": metadata
            }
            
            results.append(entry)
        
        return results

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for vector database configuration."""
        return self.embedder.get_embedding_dimension()

    def save_results_to_json(self, results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """
        Save embedding results to JSON file for inspection or backup.
        
        Args:
            results: Results from process_document or process_multiple_documents
            output_path: Path where to save the JSON file
        """
        output_path = Path(output_path)
        
        # Create summary for the JSON file
        summary = {
            "total_entries": len(results),
            "embedding_dimension": len(results[0]["embedding"]) if results else 0,
            "documents_processed": len(set(r["metadata"]["document_id"] for r in results)),
            "chunk_types": list(set(r["metadata"]["chunk_type"] for r in results)),
            "total_tokens": sum(r["metadata"]["token_count"] for r in results),
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = FilesToEmbeddings(
        embedding_model="all-MiniLM-L6-v2",
        target_chunk_size=512,
        overlap_percentage=0.25
    )
    
    # Test with a single document
    test_file = "./Data/KeypointSiteAlignment.pdf"
    
    if os.path.exists(test_file):
        print("🔬 Testing single document processing...")
        
        # Process the document
        results = pipeline.process_document(test_file, "KeypointSiteAlignment")
        
        # Print statistics
        print(f"\n📊 Results Summary:")
        print(f"   Total embeddings: {len(results)}")
        print(f"   Embedding dimension: {pipeline.get_embedding_dimension()}")
        print(f"   Average tokens per chunk: {sum(r['metadata']['token_count'] for r in results) / len(results):.1f}")
        
        # Show chunk type distribution
        chunk_types = {}
        for result in results:
            ct = result['metadata']['chunk_type']
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
        
        print(f"   Chunk types:")
        for ct, count in chunk_types.items():
            print(f"     {ct}: {count}")
        
        # Show first result structure
        print(f"\n🔍 First result structure:")
        first_result = results[0]
        print(f"   ID: {first_result['id']}")
        print(f"   Text length: {len(first_result['text'])} chars")
        print(f"   Embedding shape: {len(first_result['embedding'])}")
        print(f"   Metadata keys: {list(first_result['metadata'].keys())}")
        print(f"   Text preview: {first_result['text'][:200]}...")
        
        # Save to JSON for inspection
        pipeline.save_results_to_json(results, "./test_embeddings_output.json")
        
    else:
        print(f"❌ Test file not found: {test_file}")
        print("   Please ensure the test PDF exists or update the path")
    
    # Example of batch processing
    # test_files = ["./Data/doc1.pdf", "./Data/doc2.pdf"]
    # batch_results = pipeline.process_multiple_documents(test_files)
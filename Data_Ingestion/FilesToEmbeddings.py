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
        
        # Save structured document for debugging
        debug_path = Path(f"./debug_structured_{document_id}.json")
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(structured_doc, f, indent=2, ensure_ascii=False)
        print(f"   📝 Saved structured document to: {debug_path}")
        
        # Analyze table distribution for debugging
        total_tables = structured_doc.get('statistics', {}).get('total_tables', 0)
        sections_with_tables = 0
        table_count_in_sections = 0
        
        for section in structured_doc.get('sections', []):
            if section.get('tables') and len(section['tables']) > 0:
                sections_with_tables += 1
                table_count_in_sections += len(section['tables'])
                print(f"   📊 Section '{section.get('sectionTitle', 'Unknown')}' has {len(section['tables'])} tables")
        
        print(f"\n   ⚠️  Table Analysis:")
        print(f"      - Total tables reported: {total_tables}")
        print(f"      - Tables found in sections: {table_count_in_sections}")
        print(f"      - Sections with tables: {sections_with_tables}")
        
        if total_tables != table_count_in_sections:
            print(f"   ⚠️  WARNING: Table count mismatch! {total_tables} total vs {table_count_in_sections} in sections")
            print(f"      This means {total_tables - table_count_in_sections} tables are not properly associated with sections")
            print(f"      Tables may be processed as regular content (which is fine if they're in markdown)")
        
        # Step 3: Generate chunks using the correct method name
        print("\n✂️  Generating intelligent chunks...")
        chunks = self.tokenizer.chunk_structured_adi_document(
            structured_doc,
            target_tokens_per_chunk=self.target_chunk_size,
            overlap_percentage=self.overlap_percentage
        )
        
        print(f"   Generated {len(chunks)} chunks")
        
        # Analyze chunks for table content
        chunks_with_table_type = [c for c in chunks if c['metadata'].get('chunk_type') == 'table']
        chunks_with_table_markers = [c for c in chunks if '|' in c['content'] and '---|' in c['content']]
        
        print(f"   📊 Chunk Analysis:")
        print(f"      - Chunks marked as 'table' type: {len(chunks_with_table_type)}")
        print(f"      - Chunks containing markdown tables: {len(chunks_with_table_markers)}")
        
        # Step 4: Generate embeddings
        print("\n🧠 Generating embeddings...")
        chunk_texts = [chunk['content'] for chunk in chunks]
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
        Optimized for Azure AI Search and RAG utilization.
        """
        results = []
        
        # Get file metadata
        file_stats = file_path.stat()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique chunk ID
            chunk_id = f"{document_id}_chunk_{i:04d}"
            
            # Extract chunk metadata (handle missing keys gracefully)
            chunk_metadata = chunk.get('metadata', {})
            
            # Create comprehensive metadata dictionary
            full_metadata = {
                "document_title": structured_doc.get('title', file_path.stem),
                "document_filename": file_path.name,
                "file_size_bytes": file_stats.st_size,
                "file_modified": file_stats.st_mtime,
                "position_in_document": i / len(chunks) if len(chunks) > 0 else 0,
                "total_chunks": len(chunks),
                "total_sections_in_document": structured_doc.get('statistics', {}).get('total_sections', 0),
                "total_tables_in_document": structured_doc.get('statistics', {}).get('total_tables', 0),
                **chunk_metadata  # Unpack all chunk metadata
            }
            
            # Create FLAT entry for Azure AI Search with all metadata unpacked
            entry = {
                # Required fields for Azure AI Search
                "id": chunk_id,
                "content": chunk.get('content', ''),
                "contentVector": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                
                # Core fields
                "title": f"{structured_doc.get('title', '')} - {chunk_metadata.get('section_title', '')}",
                "document_id": document_id,
                "chunk_index": i,
                "chunk_type": chunk_metadata.get('chunk_type', 'content'),
                "section_title": chunk_metadata.get('section_title', ''),
                "is_table": chunk_metadata.get('chunk_type') == 'table',
                "token_count": chunk_metadata.get('token_count', 0),
                "table_page_number": chunk_metadata.get('table_page_number'),
                
                # Unpack ALL metadata fields to top level (no nested metadata object)
                **full_metadata
            }
            
            results.append(entry)
        
        return results

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for vector database configuration."""
        return self.embedder.get_embedding_dimension()

    def save_results_for_azure_search(self, results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """
        Save embedding results in a format ready for Azure AI Search upload.
        
        Args:
            results: Results from process_document or process_multiple_documents
            output_path: Path where to save the JSON file
        """
        output_path = Path(output_path)
        
        # Format for Azure AI Search batch upload
        azure_search_format = {
            "value": results  # Azure AI Search expects documents in a "value" array
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(azure_search_format, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Azure AI Search ready file saved to: {output_path}")
        print(f"   Ready to upload {len(results)} documents to your index")

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
            "embedding_dimension": len(results[0]["contentVector"]) if results else 0,
            "documents_processed": len(set(r["document_id"] for r in results)),
            "chunk_types": list(set(r["chunk_type"] for r in results)),
            "total_tokens": sum(r["token_count"] for r in results),  # Fixed: removed nested metadata access
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
        print(f"   Total entries: {len(results)}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = FilesToEmbeddings(
        embedding_model="text-embedding-3-small",
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
        
        # Calculate average tokens using the flattened structure
        avg_tokens = sum(r['token_count'] for r in results) / len(results) if results else 0
        print(f"   Average tokens per chunk: {avg_tokens:.1f}")
        
        # Show chunk type distribution
        chunk_types = {}
        for result in results:
            ct = result['chunk_type']  # Access directly from result
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
        
        print(f"   Chunk types:")
        for ct, count in chunk_types.items():
            print(f"     {ct}: {count}")
        
        # Show tables count
        table_chunks = [r for r in results if r['is_table']]
        print(f"   Table chunks: {len(table_chunks)}")
        
        # Detailed table analysis
        print(f"\n📊 Table Content Analysis:")
        chunks_with_tables = []
        for r in results:
            if '|' in r['content'] and '---|' in r['content']:
                chunks_with_tables.append(r)
        print(f"   Chunks containing markdown tables: {len(chunks_with_tables)}")
        
        # Show which chunks contain tables
        if chunks_with_tables:
            print(f"\n   Chunks with table content:")
            for chunk in chunks_with_tables[:3]:  # Show first 3
                print(f"     - {chunk['id']} ({chunk['chunk_type']})")
                # Show snippet of table
                lines = chunk['content'].split('\n')
                table_lines = [l for l in lines if '|' in l][:3]
                for line in table_lines:
                    print(f"       {line[:80]}...")
        
        # Show first result structure
        print(f"\n🔍 First result structure:")
        first_result = results[0]
        print(f"   ID: {first_result['id']}")
        print(f"   Title: {first_result['title']}")
        print(f"   Content length: {len(first_result['content'])} chars")
        print(f"   Embedding dimension: {len(first_result['contentVector'])}")
        print(f"   Section: {first_result['section_title']}")
        print(f"   Chunk type: {first_result['chunk_type']}")
        print(f"   Token count: {first_result['token_count']}")
        print(f"   Is table: {first_result['is_table']}")
        print(f"   Table page: {first_result.get('table_page_number', 'N/A')}")
        
        # Show text preview
        print(f"\n📄 Content preview (first 300 chars):")
        print(f"   {first_result['content'][:300]}...")
        
        # Save both formats for inspection
        print(f"\n💾 Saving results...")
        
        # Save regular JSON for inspection
        pipeline.save_results_to_json(results, "./test_embeddings_output.json")
        
        # Save Azure AI Search format
        pipeline.save_results_for_azure_search(results, "./azure_search_upload.json")
        
        # Show some sample chunks
        print(f"\n📑 Sample chunks:")
        for i, result in enumerate(results[:3]):
            print(f"\n   Chunk {i + 1}:")
            print(f"     ID: {result['id']}")
            print(f"     Type: {result['chunk_type']}")
            print(f"     Section: {result['section_title']}")
            print(f"     Tokens: {result['token_count']}")
            print(f"     Preview: {result['content'][:100]}...")
        
        print(f"\n✅ Complete! Check these files for inspection:")
        print(f"   - debug_structured_KeypointSiteAlignment.json (structured document)")
        print(f"   - test_embeddings_output.json (embedding results)")
        print(f"   - azure_search_upload.json (Azure AI Search format)")
        
    else:
        print(f"❌ Test file not found: {test_file}")
        print("   Please ensure the test PDF exists or update the path")
    
    # Example of batch processing
    print("\n📚 Example batch processing (commented out):")
    print("   # test_files = ['./Data/doc1.pdf', './Data/doc2.pdf']")
    print("   # batch_results = pipeline.process_multiple_documents(test_files)")
    print("   # pipeline.save_results_to_json(batch_results, './batch_embeddings.json')")
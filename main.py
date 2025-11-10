import os
import dotenv
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List

# Add subdirectories to path for imports
sys.path.append(str(Path(__file__).parent / "Data_Ingestion"))
sys.path.append(str(Path(__file__).parent / "Data_Upsert"))

from FilesToEmbeddings import FilesToEmbeddings
from Upsert import AzureSearchDataUploader

class DocumentRAGPipeline:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 target_chunk_size: int = 512,
                 overlap_percentage: float = 0.25,
                 cache_dir: str = "./Data_Ingestion/cache"):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            embedding_model: Sentence transformer model name
            target_chunk_size: Target size for text chunks in tokens
            overlap_percentage: Percentage overlap between chunks
            cache_dir: Directory for caching ADI results
        """
        self.embedding_model = embedding_model
        self.target_chunk_size = target_chunk_size
        self.overlap_percentage = overlap_percentage
        self.cache_dir = cache_dir
        
        # Initialize pipeline components
        self.embeddings_processor = FilesToEmbeddings(
            embedding_model=embedding_model,
            cache_dir=cache_dir,
            target_chunk_size=target_chunk_size,
            overlap_percentage=overlap_percentage
        )
        
        # Initialize uploader (will be configured when needed)
        self.search_uploader = None
        
        print(f"🚀 RAG Pipeline Initialized:")
        print(f"   Embedding model: {embedding_model}")
        print(f"   Chunk size: {target_chunk_size} tokens")
        print(f"   Overlap: {overlap_percentage * 100:.0f}%")
        print(f"   Cache directory: {cache_dir}")

    def process_document(self, 
                        document_path: str,
                        document_id: Optional[str] = None,
                        save_intermediate: bool = True,
                        output_dir: str = "./output") -> List[dict]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            document_path: Path to the document file
            document_id: Optional custom document ID
            save_intermediate: Whether to save intermediate JSON files
            output_dir: Directory to save output files
            
        Returns:
            List of processed chunks with embeddings
        """
        document_path = Path(document_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = document_path.stem
        
        print(f"\n📄 Processing Document: {document_path.name}")
        print(f"   Document ID: {document_id}")
        print(f"   Output directory: {output_dir}")
        
        # Process document through embeddings pipeline
        results = self.embeddings_processor.process_document(
            file_path=document_path,
            document_id=document_id
        )
        
        if save_intermediate:
            # Save results in both formats
            regular_output = output_dir / f"{document_id}_embeddings.json"
            azure_output = output_dir / f"{document_id}_azure_search.json"
            
            self.embeddings_processor.save_results_to_json(results, regular_output)
            self.embeddings_processor.save_results_for_azure_search(results, azure_output)
            
            print(f"\n💾 Saved intermediate files:")
            print(f"   Regular format: {regular_output}")
            print(f"   Azure Search format: {azure_output}")
        
        return results

    def upload_to_azure_search(self, 
                              results: List[dict],
                              index_name: str,
                              create_index_if_missing: bool = True) -> bool:
        """
        Upload processed results to Azure AI Search.
        
        Args:
            results: Processed document chunks with embeddings
            index_name: Name of the Azure Search index
            create_index_if_missing: Whether to create index if it doesn't exist
            
        Returns:
            True if upload was successful
        """
        # Initialize search uploader if not already done
        if self.search_uploader is None:
            embedding_dim = self.embeddings_processor.get_embedding_dimension()
            self.search_uploader = AzureSearchDataUploader(embedding_dimension=embedding_dim)
        
        print(f"\n☁️  Uploading to Azure AI Search:")
        print(f"   Index: {index_name}")
        print(f"   Documents: {len(results)}")
        
        try:
            # Use the upsert_data method which handles index creation automatically
            result = self.search_uploader.upsert_data(results, index_name)
            print(f"✅ Successfully uploaded to Azure AI Search")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading to Azure Search: {str(e)}")
            return False

    def upload_from_json_file(self, 
                             json_file_path: str,
                             index_name: str) -> bool:
        """
        Upload documents directly from a JSON file to Azure AI Search.
        
        Args:
            json_file_path: Path to the JSON file with embeddings
            index_name: Name of the Azure Search index
            
        Returns:
            True if upload was successful
        """
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        # Initialize search uploader if not already done
        if self.search_uploader is None:
            # Default to 384 dimensions for all-MiniLM-L6-v2
            embedding_dim = 384  # You may want to detect this from the JSON
            self.search_uploader = AzureSearchDataUploader(embedding_dimension=embedding_dim)
        
        print(f"\n📁 Loading from JSON file: {json_file_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: direct list or wrapped in 'value' key
        documents = data if isinstance(data, list) else data.get('value', data.get('results', []))
        
        if not documents:
            print("❌ No documents found in JSON file")
            return False
        
        print(f"   Found {len(documents)} documents")
        
        try:
            result = self.search_uploader.upsert_data(documents, index_name)
            print(f"✅ Successfully uploaded from JSON file")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading from JSON: {str(e)}")
            return False

    def process_and_upload(self, 
                          document_path: str,
                          index_name: str,
                          document_id: Optional[str] = None,
                          save_intermediate: bool = True,
                          output_dir: str = "./output") -> bool:
        """
        Complete pipeline: process document and upload to Azure AI Search.
        
        Args:
            document_path: Path to the document file
            index_name: Name of the Azure Search index
            document_id: Optional custom document ID
            save_intermediate: Whether to save intermediate JSON files
            output_dir: Directory to save output files
            
        Returns:
            True if the entire pipeline was successful
        """
        try:
            # Step 1: Process document
            results = self.process_document(
                document_path=document_path,
                document_id=document_id,
                save_intermediate=save_intermediate,
                output_dir=output_dir
            )
            
            # Step 2: Upload to Azure Search
            success = self.upload_to_azure_search(
                results=results,
                index_name=index_name
            )
            
            if success:
                print(f"\n🎉 Pipeline Complete!")
                print(f"   Document '{document_id or Path(document_path).stem}' successfully processed and uploaded")
                return True
            else:
                print(f"\n❌ Pipeline failed during upload phase")
                return False
                
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            return False

    def batch_process_and_upload(self,
                                document_paths: List[str],
                                index_name: str,
                                document_ids: Optional[List[str]] = None,
                                save_intermediate: bool = True,
                                output_dir: str = "./output") -> dict:
        """
        Process multiple documents and upload to Azure AI Search.
        
        Args:
            document_paths: List of document file paths
            index_name: Name of the Azure Search index
            document_ids: Optional list of custom document IDs
            save_intermediate: Whether to save intermediate files
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with success/failure counts and details
        """
        results = {
            "total": len(document_paths),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        print(f"\n📚 Batch Processing {len(document_paths)} documents...")
        
        for i, doc_path in enumerate(document_paths):
            doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
            
            try:
                success = self.process_and_upload(
                    document_path=doc_path,
                    index_name=index_name,
                    document_id=doc_id,
                    save_intermediate=save_intermediate,
                    output_dir=output_dir
                )
                
                if success:
                    results["successful"] += 1
                    results["details"].append({"path": doc_path, "status": "success"})
                else:
                    results["failed"] += 1
                    results["details"].append({"path": doc_path, "status": "failed", "error": "Upload failed"})
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"path": doc_path, "status": "failed", "error": str(e)})
        
        print(f"\n📊 Batch Processing Complete:")
        print(f"   Total: {results['total']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        
        return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="RAG Document Processing Pipeline")
    
    # Required arguments
    parser.add_argument("command", choices=["process", "upload", "process-and-upload", "batch"], 
                       help="Command to execute")
    
    # Document processing arguments
    parser.add_argument("--document", "-d", type=str, help="Path to document file")
    parser.add_argument("--documents", type=str, nargs="+", help="Paths to multiple document files")
    parser.add_argument("--document-id", type=str, help="Custom document ID")
    
    # Azure Search arguments
    parser.add_argument("--index", "-i", type=str, help="Azure Search index name")
    parser.add_argument("--json-file", "-j", type=str, help="JSON file with embeddings to upload")
    
    # Pipeline configuration
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", 
                       help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=512, help="Target chunk size in tokens")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap percentage")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save intermediate files")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentRAGPipeline(
        embedding_model=args.embedding_model,
        target_chunk_size=args.chunk_size,
        overlap_percentage=args.overlap
    )
    
    # Execute command
    if args.command == "process":
        if not args.document:
            print("❌ --document is required for 'process' command")
            return 1
        
        pipeline.process_document(
            document_path=args.document,
            document_id=args.document_id,
            save_intermediate=not args.no_save,
            output_dir=args.output_dir
        )
        
    elif args.command == "upload":
        if not args.json_file or not args.index:
            print("❌ --json-file and --index are required for 'upload' command")
            return 1
        
        pipeline.upload_from_json_file(
            json_file_path=args.json_file,
            index_name=args.index
        )
        
    elif args.command == "process-and-upload":
        if not args.document or not args.index:
            print("❌ --document and --index are required for 'process-and-upload' command")
            return 1
        
        pipeline.process_and_upload(
            document_path=args.document,
            index_name=args.index,
            document_id=args.document_id,
            save_intermediate=not args.no_save,
            output_dir=args.output_dir
        )
        
    elif args.command == "batch":
        if not args.documents or not args.index:
            print("❌ --documents and --index are required for 'batch' command")
            return 1
        
        pipeline.batch_process_and_upload(
            document_paths=args.documents,
            index_name=args.index,
            save_intermediate=not args.no_save,
            output_dir=args.output_dir
        )
    
    return 0


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("🔬 Running in demo mode...")
        
        # Initialize pipeline
        pipeline = DocumentRAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            target_chunk_size=512,
            overlap_percentage=0.25
        )
        
        # Example: Process existing test file
        test_file = "./Data_Ingestion/Data/KeypointSiteAlignment.pdf"
        test_index = "rag-test-index"
        
        if os.path.exists(test_file):
            print(f"\n🧪 Demo: Processing test document...")
            success = pipeline.process_and_upload(
                document_path=test_file,
                index_name=test_index,
                document_id="KeypointSiteAlignment",
                save_intermediate=True,
                output_dir="./output"
            )
            
            if success:
                print("\n✅ Demo completed successfully!")
            else:
                print("\n❌ Demo failed")
        else:
            print(f"❌ Test file not found: {test_file}")
            print("\nUsage examples:")
            print("  python main.py process-and-upload -d ./document.pdf -i my-index")
            print("  python main.py upload -j ./embeddings.json -i my-index")
            print("  python main.py batch --documents ./doc1.pdf ./doc2.pdf -i my-index")
    else:
        exit(main())
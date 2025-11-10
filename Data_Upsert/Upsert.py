import os
import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.core.exceptions import ResourceNotFoundError

class AzureSearchDataUploader:
    def __init__(self, embedding_dimension: int):
        dotenv.load_dotenv()
        self.azure_ai_search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT", "")
        self.azure_ai_search_key = os.getenv("AZURE_AI_SEARCH_KEY", "")
        self.embedding_dimension = embedding_dimension  

        self.search_index_client = SearchIndexClient(
            endpoint=self.azure_ai_search_endpoint,
            credential=AzureKeyCredential(self.azure_ai_search_key)
        )

    def create_index(self, index_name: str) -> bool:
        """
        Create a new search index with the schema for FilesToEmbeddings output.
        
        Args:
            index_name: Name of the index to create
            
        Returns:
            True if creation was successful, False otherwise
        """
        try:
            print(f"🔨 Creating index '{index_name}'...")
            
            # Define the fields based on your FilesToEmbeddings JSON structure
            # All fields are flattened - no nested objects
            fields = [
                # Primary key
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                
                # Main content fields
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
                SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
                
                # Vector field for semantic search
                SearchField(
                    name="contentVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=self.embedding_dimension,
                    vector_search_profile_name="myHnswProfile"
                ),
                
                # Core metadata fields
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="section_title", type=SearchFieldDataType.String),
                SimpleField(name="is_table", type=SearchFieldDataType.Boolean, filterable=True),
                SimpleField(name="token_count", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="table_page_number", type=SearchFieldDataType.Int32, filterable=True),
                
                # Document-level metadata fields
                SearchableField(name="document_title", type=SearchFieldDataType.String),
                SimpleField(name="document_filename", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="file_size_bytes", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
                SimpleField(name="file_modified", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="position_in_document", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="total_chunks", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="total_sections_in_document", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="total_tables_in_document", type=SearchFieldDataType.Int32, filterable=True),
            ]
            
            # Configure vector search (required for contentVector field)
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw"
                    )
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            result = self.search_index_client.create_index(index)
            print(f"✅ Successfully created index '{index_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Error creating index: {str(e)}")
            return False

    def upsert_data(self, data, index_name):
        if self.azure_ai_search_endpoint == "" or self.azure_ai_search_key == "":
            raise ValueError("Azure Search endpoint or key is not configured properly.")
        if index_name is None or index_name.strip() == "":
            raise ValueError("Index name must be provided and cannot be empty.")
        
        # Check if Index Exists
        try:
            self.search_index_client.get_index(index_name)
            print(f"✅ Index '{index_name}' already exists")
        except ResourceNotFoundError:
            # Create Index
            print(f"📋 Index '{index_name}' not found, creating...")
            if not self.create_index(index_name):
                raise ValueError(f"Failed to create index '{index_name}'")

        # Upload documents
        search_client = SearchClient(
            endpoint=self.azure_ai_search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.azure_ai_search_key)
        )
        
        print(f"⬆️  Uploading {len(data)} documents...")
        result = search_client.upload_documents(documents=data)
        
        # Check for failures
        failed_uploads = [r for r in result if not r.succeeded]
        if failed_uploads:
            print(f"⚠️  {len(failed_uploads)} documents failed to upload")
            for failed in failed_uploads[:3]:  # Show first 3 failures
                print(f"     Failed: {failed.key} - {failed.error_message}")
        
        successful_uploads = len(data) - len(failed_uploads)
        print(f"✅ {successful_uploads}/{len(data)} documents uploaded successfully")
        
        return result

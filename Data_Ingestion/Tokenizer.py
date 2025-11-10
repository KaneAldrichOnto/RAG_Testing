import tiktoken
from typing import List, Union, Tuple, Dict, Any
import json

class Tokenizer:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
            print(f"Loaded tokenizer for {model}")
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-3.5/4)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print(f"Model {model} not found, using cl100k_base encoding")
    
    def count_tokens(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(text, str):
            return len(self.tokenizer.encode(text))
        else:
            return [len(self.tokenizer.encode(t)) for t in text]
    
    def tokenize_with_count(self, text: str) -> Tuple[List[str], int]:
        # Get token IDs
        token_ids = self.tokenizer.encode(text)
        # Convert token IDs back to strings
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]
        return tokens, len(tokens)

    def chunk_adi_document(self, 
                          structured_doc: Dict[str, Any], 
                          target_chunk_size: int = 512,
                          overlap_percentage: float = 0.25,
                          min_chunk_size: int = 100) -> List[Dict[str, Any]]:
        """
        Intelligently chunk Azure Document Intelligence output for RAG.
        
        Args:
            structured_doc: Output from format_result_as_structured_document()
            target_chunk_size: Target size for chunks in tokens (default 512)
            overlap_percentage: Percentage of overlap for beginning of chunks (default 25%)
            min_chunk_size: Minimum chunk size in tokens
            
        Returns:
            List of chunk dictionaries containing text and metadata
        """
        chunks = []
        overlap_size = int(target_chunk_size * overlap_percentage)
        
        # Process each section
        for section_idx, section in enumerate(structured_doc.get('sections', [])):
            section_title = section.get('title', f'Section {section_idx + 1}')
            section_metadata = {
                'section_title': section_title,
                'page_start': section.get('page_start', 0),
                'document_title': structured_doc.get('title', 'Unknown Document')
            }
            
            # Build section content with proper structure
            section_chunks = self._chunk_section(
                section=section,
                section_metadata=section_metadata,
                target_chunk_size=target_chunk_size,
                overlap_size=overlap_size,
                min_chunk_size=min_chunk_size
            )
            
            chunks.extend(section_chunks)
        
        # Add overlap between chunks (except for the first chunk)
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap_size > 0:
                # Get overlap text from previous chunk
                prev_text = chunks[i-1]['text']
                prev_tokens = self.tokenizer.encode(prev_text)
                
                if len(prev_tokens) > overlap_size:
                    # Take last 25% of previous chunk
                    overlap_tokens = prev_tokens[-overlap_size:]
                    overlap_text = self.tokenizer.decode(overlap_tokens)
                    
                    # Add overlap to current chunk if it doesn't already have it
                    if not chunk['text'].startswith(overlap_text):
                        chunk['text'] = overlap_text + "\n\n" + chunk['text']
                        chunk['has_overlap'] = True
                        chunk['overlap_from_chunk'] = i - 1
            
            final_chunks.append(chunk)
        
        return final_chunks

    def _chunk_section(self, 
                    section: Dict[str, Any],
                    section_metadata: Dict[str, Any],
                    target_chunk_size: int,
                    overlap_size: int,
                    min_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk a single section while respecting semantic boundaries.
        """
        chunks = []
        current_chunk_text = []
        current_chunk_tokens = 0
        
        # Always start with section title for context
        section_header = f"# {section['title']}\n\n"
        header_tokens = self.count_tokens(section_header)
        
        # Process subsections if they exist
        if section.get('subsections'):
            for subsection in section['subsections']:
                subsection_chunks = self._process_subsection(
                    subsection=subsection,
                    parent_header=section_header,
                    section_metadata=section_metadata,
                    target_chunk_size=target_chunk_size,
                    min_chunk_size=min_chunk_size
                )
                chunks.extend(subsection_chunks)
        
        # Process main section content
        current_chunk_text = [section_header]
        current_chunk_tokens = header_tokens
        
        # Get table text to filter out
        table_texts = set()
        for table in section.get('tables', []):
            # Collect all text from table cells to filter out
            for row in table.get('rows', []):
                for cell in row:
                    if cell and str(cell).strip():
                        table_texts.add(str(cell).strip())
            # Also add headers
            for header in table.get('headers', []):
                if header and str(header).strip():
                    table_texts.add(str(header).strip())
        
        # Process paragraphs (filtering out table text)
        for content_item in section.get('content', []):
            paragraph_text = content_item.get('text', '')
            
            # Skip if this is empty, marked as table, or contains table data
            if not paragraph_text.strip():
                continue
            
            # Check if this text is actually from a table
            is_table_content = (
                content_item.get('type') == 'table' or
                paragraph_text.strip() in table_texts or
                any(table_text in paragraph_text for table_text in table_texts if len(table_text) > 20)
            )
            
            if is_table_content:
                continue
                
            paragraph_tokens = self.count_tokens(paragraph_text)
            
            # Check if adding this paragraph would exceed target size
            if current_chunk_tokens + paragraph_tokens > target_chunk_size:
                # If current chunk is big enough, save it
                if current_chunk_tokens >= min_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **section_metadata,
                            'chunk_type': 'section_content',
                            'page': content_item.get('page', 0),
                            'token_count': current_chunk_tokens
                        }
                    })
                    
                    # Start new chunk with section header for context
                    current_chunk_text = [section_header]
                    current_chunk_tokens = header_tokens
            
            # Add paragraph to current chunk
            current_chunk_text.append(paragraph_text)
            current_chunk_tokens += paragraph_tokens
        
        # Process tables - try to keep with current content if it fits
        for table_idx, table in enumerate(section.get('tables', [])):
            table_text = self._format_table_for_chunk(table, table_idx)
            table_tokens = self.count_tokens(table_text)
            
            # Check if table can fit in current chunk (allow some overflow)
            # Allow up to 150% of target size if including a table
            max_with_table = int(target_chunk_size * 1.5)
            
            if current_chunk_tokens + table_tokens <= max_with_table and len(current_chunk_text) > 1:
                # Add table to current chunk
                current_chunk_text.append(table_text)
                current_chunk_tokens += table_tokens
            else:
                # Save current chunk if it has content
                if len(current_chunk_text) > 1:  # More than just header
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **section_metadata,
                            'chunk_type': 'section_content',
                            'token_count': current_chunk_tokens
                        }
                    })
                
                # Create new chunk with table
                table_chunk_text = section_header + table_text
                chunks.append({
                    'text': table_chunk_text,
                    'metadata': {
                        **section_metadata,
                        'chunk_type': 'table',
                        'table_index': table_idx,
                        'token_count': header_tokens + table_tokens,
                        'exceeds_target': (header_tokens + table_tokens) > target_chunk_size
                    }
                })
                
                # Reset for next content
                current_chunk_text = [section_header]
                current_chunk_tokens = header_tokens
        
        # Save any remaining content
        if len(current_chunk_text) > 1:  # More than just header
            chunk_text = '\n\n'.join(current_chunk_text)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **section_metadata,
                    'chunk_type': 'section_content',
                    'token_count': current_chunk_tokens
                }
            })
        
        return chunks

    def _process_subsection(self,
                        subsection: Dict[str, Any],
                        parent_header: str,
                        section_metadata: Dict[str, Any],
                        target_chunk_size: int,
                        min_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Process a subsection, keeping it with its parent section context.
        """
        chunks = []
        
        # Create subsection header with parent context
        subsection_header = f"{parent_header}## {subsection['title']}\n\n"
        header_tokens = self.count_tokens(subsection_header)
        
        current_chunk_text = [subsection_header]
        current_chunk_tokens = header_tokens
        
        # Get table text to filter out
        table_texts = set()
        for table in subsection.get('tables', []):
            for row in table.get('rows', []):
                for cell in row:
                    if cell and str(cell).strip():
                        table_texts.add(str(cell).strip())
            for header in table.get('headers', []):
                if header and str(header).strip():
                    table_texts.add(str(header).strip())
        
        # Process subsection content (filtering out table text)
        for content_item in subsection.get('content', []):
            paragraph_text = content_item.get('text', '')
            
            # Skip if this is empty
            if not paragraph_text.strip():
                continue
            
            # Check if this text is actually from a table
            is_table_content = (
                content_item.get('type') == 'table' or
                paragraph_text.strip() in table_texts or
                any(table_text in paragraph_text for table_text in table_texts if len(table_text) > 20)
            )
            
            if is_table_content:
                continue
                
            paragraph_tokens = self.count_tokens(paragraph_text)
            
            if current_chunk_tokens + paragraph_tokens > target_chunk_size:
                if current_chunk_tokens >= min_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **section_metadata,
                            'subsection_title': subsection['title'],
                            'chunk_type': 'subsection_content',
                            'page': content_item.get('page', 0),
                            'token_count': current_chunk_tokens
                        }
                    })
                    
                    current_chunk_text = [subsection_header]
                    current_chunk_tokens = header_tokens
            
            current_chunk_text.append(paragraph_text)
            current_chunk_tokens += paragraph_tokens
        
        # Process subsection tables - try to keep with content
        for table_idx, table in enumerate(subsection.get('tables', [])):
            table_text = self._format_table_for_chunk(table, table_idx)
            table_tokens = self.count_tokens(table_text)
            
            # Allow up to 150% of target size if including a table
            max_with_table = int(target_chunk_size * 1.5)
            
            if current_chunk_tokens + table_tokens <= max_with_table and len(current_chunk_text) > 1:
                # Add table to current chunk
                current_chunk_text.append(table_text)
                current_chunk_tokens += table_tokens
            else:
                # Save current content if any
                if len(current_chunk_text) > 1:
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **section_metadata,
                            'subsection_title': subsection['title'],
                            'chunk_type': 'subsection_content',
                            'token_count': current_chunk_tokens
                        }
                    })
                
                # Add table as separate chunk with context
                table_chunk_text = subsection_header + table_text
                chunks.append({
                    'text': table_chunk_text,
                    'metadata': {
                        **section_metadata,
                        'subsection_title': subsection['title'],
                        'chunk_type': 'subsection_table',
                        'table_index': table_idx,
                        'token_count': self.count_tokens(table_chunk_text),
                        'exceeds_target': self.count_tokens(table_chunk_text) > target_chunk_size
                    }
                })
                
                current_chunk_text = [subsection_header]
                current_chunk_tokens = header_tokens
        
        # Save remaining content
        if len(current_chunk_text) > 1:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **section_metadata,
                    'subsection_title': subsection['title'],
                    'chunk_type': 'subsection_content',
                    'token_count': current_chunk_tokens
                }
            })
        
        return chunks

    def _format_table_for_chunk(self, table: Dict[str, Any], table_idx: int) -> str:
        """
        Format a table for inclusion in a chunk.
        """
        table_text = f""
        
        # Add headers
        if table.get('headers'):
            table_text += "| " + " | ".join(str(h) for h in table['headers']) + " |\n"
            table_text += "|" + "---|" * len(table['headers']) + "\n"
        
        # Add rows
        for row in table.get('rows', []):
            table_text += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return table_text

    def split_adi_formatted_text(self, formatted_text : dict) -> List[str]:
        """Legacy method - now use chunk_adi_document instead"""
        chunks = self.chunk_adi_document(formatted_text)
        return [chunk['text'] for chunk in chunks]


# Test the tokenizer
if __name__ == "__main__":
    import os
    import sys
    import pickle
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from Data_Ingestion.AzureDocumentIntelligence import AzureDocumentIntelligenceBridge
    
    # Initialize tokenizer
    tokenizer = Tokenizer("gpt-3.5-turbo")
    
    # Load or create ADI result
    bridge = AzureDocumentIntelligenceBridge()
    
    pkl_path = "./KeypointSiteAlignment.pkl"
    if os.path.exists(pkl_path):
        print("Loading cached ADI result...")
        cached_result = bridge.load_analysis_result(pkl_path)
    else:
        print("Running ADI analysis...")
        cached_result = bridge.analyze_local_document(
            "./Data/KeypointSiteAlignment.pdf", 
            pkl_path
        )
    
    # Format the document
    structured_doc = bridge.format_result_as_structured_document(cached_result)
    
    # Chunk the document
    print("\nChunking document...")
    chunks = tokenizer.chunk_adi_document(
        structured_doc,
        target_chunk_size=512,
        overlap_percentage=0
    )
    
    # Print chunk statistics
    print(f"\nChunking Results:")
    print(f"Total chunks: {len(chunks)}")
    
    token_counts = [chunk['metadata']['token_count'] for chunk in chunks]
    print(f"Average tokens per chunk: {sum(token_counts) / len(token_counts):.1f}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    
    # Show chunk types distribution
    chunk_types = {}
    for chunk in chunks:
        ct = chunk['metadata']['chunk_type']
        chunk_types[ct] = chunk_types.get(ct, 0) + 1
    
    print(f"\nChunk types:")
    for ct, count in chunk_types.items():
        print(f"  {ct}: {count}")
    
    # Show first few chunks as examples
    print(f"\nFirst 3 chunks preview:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Type: {chunk['metadata']['chunk_type']}")
        print(f"Tokens: {chunk['metadata']['token_count']}")
        print(f"Section: {chunk['metadata'].get('section_title', 'N/A')}")
        print(f"Has overlap: {chunk.get('has_overlap', False)}")
        print(f"Text preview:\n{chunk['text']}")
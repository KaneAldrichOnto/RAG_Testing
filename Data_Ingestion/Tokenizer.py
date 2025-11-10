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
    
    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def _split_markdown_table(self, markdown_table: str, division_factor: int) -> List[str]:
        """
        Split a markdown table into division_factor parts of equal or close to equal size.
        Only splits on row boundaries, never within a row.
        
        Args:
            markdown_table: The markdown table as a string
            division_factor: Number of parts to split the table into
            
        Returns:
            List of markdown table strings, each with headers and portion of rows
        """
        if division_factor <= 1:
            return [markdown_table]
        
        lines = markdown_table.strip().split('\n')
        
        # Find header and separator lines
        header_line = None
        separator_line = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if '|' in line and header_line is None:
                header_line = line
            elif '---' in line and '|' in line and separator_line is None:
                separator_line = line
                data_start_idx = i + 1
                break
        
        if header_line is None or separator_line is None:
            # Not a valid markdown table, return as single part
            return [markdown_table]
        
        # Get data rows (everything after header and separator)
        data_rows = lines[data_start_idx:]
        
        # Filter out empty lines
        data_rows = [row for row in data_rows if row.strip() and '|' in row]
        
        if len(data_rows) == 0:
            # No data rows, return original table
            return [markdown_table]
        
        if len(data_rows) < division_factor:
            # Can't split into more parts than we have rows
            division_factor = len(data_rows)
        
        # Calculate rows per part
        rows_per_part = len(data_rows) // division_factor
        remainder_rows = len(data_rows) % division_factor
        
        split_tables = []
        current_row_idx = 0
        
        for part_num in range(division_factor):
            # Calculate how many rows for this part
            rows_in_this_part = rows_per_part
            if part_num < remainder_rows:
                rows_in_this_part += 1  # Distribute remainder rows to first parts
            
            # Build this part of the table
            table_part = header_line + '\n' + separator_line + '\n'
            
            # Add the rows for this part
            for row_idx in range(rows_in_this_part):
                if current_row_idx < len(data_rows):
                    table_part += data_rows[current_row_idx] + '\n'
                    current_row_idx += 1
            
            split_tables.append(table_part.strip())
        
        return split_tables
    
    def chunk_structured_adi_document(self, 
                                      structured_document : dict, 
                                      target_tokens_per_chunk : int,
                                      overlap_percentage : float) -> List[dict]:
        chunks = []
        for document_section in structured_document.get('sections', []):
            section_title = document_section['sectionTitle']

            current_chunk = section_title + "\n"
            current_token_count = self.count_tokens(current_chunk)

            for content_index, content_string in enumerate(document_section.get('content', [])):
                content_token_count = self.count_tokens(content_string)
                
                if current_token_count + content_token_count <= target_tokens_per_chunk:
                    current_chunk += content_string + "\n"
                    current_token_count += content_token_count
                elif current_token_count + content_token_count < target_tokens_per_chunk * 1.2:
                    current_chunk += content_string + "\n"
                    current_token_count += content_token_count
                else:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            'section_title': section_title,
                            'token_count': current_token_count
                        }})
                    current_chunk = section_title + "\n" + content_string + "\n"
                    current_token_count = self.count_tokens(current_chunk)
                
            # Add remaining content as chunk
            if current_token_count > 0:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        'section_title': section_title,
                        'token_count': current_token_count,
                        'chunk_type': 'text'
                    }})
                current_chunk = section_title + "\n"
                current_token_count = self.count_tokens(current_chunk)
            
            for table_index, table_section in enumerate(document_section['tables']):
                markdown_table = table_section['tableMarkdown']
                table_token_count = self.count_tokens(markdown_table)

                if current_token_count + table_token_count <= target_tokens_per_chunk * 2:
                    current_chunk += markdown_table + "\n"
                    current_token_count += table_token_count
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            'section_title': section_title,
                            'token_count': current_token_count,
                            'chunk_type': 'table',
                            'table_page_number': table_section['tablePageNumber']
                        }})
                else:
                    addedSuccessfully = False
                    for division_factor in range(1, 20):
                        # Split Markdown Table into Smaller Parts
                        split_tables = self._split_markdown_table(
                            markdown_table, division_factor)
                        
                        split_table_token_count = self.count_tokens(split_tables[0])
                        if current_token_count + split_table_token_count <= target_tokens_per_chunk * 2:
                            for split_table in split_tables:
                                split_table_token_count = self.count_tokens(split_table)

                                if current_token_count + split_table_token_count <= target_tokens_per_chunk * 2:
                                    current_chunk += markdown_table + "\n"
                                    current_token_count += table_token_count
                                    chunks.append({
                                        'content': current_chunk.strip(),
                                        'metadata': {
                                            'section_title': section_title,
                                            'token_count': current_token_count,
                                            'chunk_type': 'table',
                                            'table_page_number': table_section['tablePageNumber']
                                        }})
                                addedSuccessfully = True
                        if addedSuccessfully:
                            break
        return chunks
                            

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
    chunks = tokenizer.chunk_structured_adi_document(
        structured_doc,
        target_tokens_per_chunk=512,
        overlap_percentage=.25
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
        print(f"Section: {chunk['metadata']['section_title']}")
        print(f"Text preview:\n{chunk['content']}")
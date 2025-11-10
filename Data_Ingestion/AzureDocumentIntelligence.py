import dotenv
import os
import pickle
from pathlib import Path
import json

from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from azure.core.credentials import AzureKeyCredential

from Tokenizer import Tokenizer
from LocalEmbedder import LocalEmbedder

class AzureDocumentIntelligenceBridge:
    def __init__(self):
        dotenv.load_dotenv("../.env")
        self.adi_endpoint = os.getenv("AZURE_ADI_ENDPOINT")
        self.adi_key = os.getenv("AZURE_ADI_KEY")

        if not self.adi_endpoint or not self.adi_key:
            raise ValueError("Azure Document Intelligence endpoint or key not found in environment variables.")
    
    def save_analysis_result(self, analysis_result: AnalyzeResult, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(analysis_result, file)
        print(f"Analysis result saved to {file_path}")
    
    def load_analysis_result(self, file_path: str) -> AnalyzeResult:
        with open(file_path, 'rb') as file:
            analysis_result = pickle.load(file)
        print(f"Analysis result loaded from {file_path}")
        return analysis_result
        
    def analyze_document_at_url(self, document_url: str, save_path: str = "") -> AnalyzeResult:
        # Create Document Analysis Client Object
        assert self.adi_endpoint is not None and self.adi_key is not None, "Azure ADI endpoint and key must be set"
        document_analysis_client = DocumentAnalysisClient(
            endpoint=self.adi_endpoint,
            credential=AzureKeyCredential(self.adi_key)
        )

        # Analyze Document at URL
        poller = document_analysis_client.begin_analyze_document_from_url(
            "prebuilt-layout", document_url
        )

        # Get Result
        analysis_result = poller.result()
        
        # Optionally save the result
        if save_path != "":
            self.save_analysis_result(analysis_result, save_path)
        
        return analysis_result
    
    def analyze_local_document(self, file_path: str, save_path: str = "") -> AnalyzeResult:
        """Analyze a local document file"""
        document_analysis_client = DocumentAnalysisClient(
            endpoint=self.adi_endpoint,
            credential=AzureKeyCredential(self.adi_key)
        )

        with open(file_path, "rb") as file:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-layout", file
            )

        analysis_result = poller.result()
        
        if save_path != "":
            self.save_analysis_result(analysis_result, save_path)
        
        return analysis_result

    def format_result_as_structured_document(self, analysis_result: AnalyzeResult) -> dict:
        """
        Format ADI output into a hierarchical document structure with sections.
        Each section contains its title, content, tables, and other elements.
        """
        
        # First, create a mapping of page elements by their position
        page_elements = {}
        
        # Collect all elements with their page positions
        if analysis_result.paragraphs:
            for paragraph in analysis_result.paragraphs:
                if paragraph.bounding_regions:
                    for region in paragraph.bounding_regions:
                        page_num = region.page_number
                        if page_num not in page_elements:
                            page_elements[page_num] = []
                        
                        # Get the top Y coordinate for sorting
                        min_y = min(point.y for point in region.polygon) if region.polygon else 0
                        
                        page_elements[page_num].append({
                            "type": "paragraph",
                            "content": paragraph.content,
                            "role": paragraph.role,
                            "position": min_y,
                            "page": page_num,
                            "data": paragraph
                        })
        
        # Add tables with their positions
        if analysis_result.tables:
            for table in analysis_result.tables:
                if table.bounding_regions:
                    for region in table.bounding_regions:
                        page_num = region.page_number
                        if page_num not in page_elements:
                            page_elements[page_num] = []
                        
                        min_y = min(point.y for point in region.polygon) if region.polygon else 0
                        
                        # Format table data
                        formatted_table = self._format_table(table)
                        
                        page_elements[page_num].append({
                            "type": "table",
                            "position": min_y,
                            "page": page_num,
                            "data": formatted_table
                        })
        
        # Sort elements by position on each page
        for page_num in page_elements:
            page_elements[page_num].sort(key=lambda x: x["position"])
        
        # Now build the document structure
        document = {
            "title": None,
            "metadata": {},
            "sections": [],
            "appendices": [],
            "total_pages": len(analysis_result.pages) if analysis_result.pages else 0
        }
        
        # Extract key-value pairs as metadata
        if analysis_result.key_value_pairs:
            for kv_pair in analysis_result.key_value_pairs:
                if kv_pair.key and kv_pair.value:
                    key = kv_pair.key.content
                    value = kv_pair.value.content
                    document["metadata"][key] = value
        
        # Build sections from the ordered elements
        current_section = None
        current_subsection = None
        
        # Process elements in page order
        for page_num in sorted(page_elements.keys()):
            for element in page_elements[page_num]:
                
                if element["type"] == "paragraph":
                    role = element["role"]
                    content = element["content"]
                    
                    # Document title (usually appears once at the beginning)
                    if role == "title" and not document["title"]:
                        document["title"] = content
                        continue
                    
                    # Section heading - start a new section
                    elif role == "sectionHeading":
                        # Save current section if exists
                        if current_section:
                            if current_subsection:
                                current_section["subsections"].append(current_subsection)
                                current_subsection = None
                            document["sections"].append(current_section)
                        
                        # Create new section
                        current_section = {
                            "title": content,
                            "page_start": element["page"],
                            "content": [],
                            "tables": [],
                            "figures": [],
                            "subsections": []
                        }
                    
                    # Subsection heading (if you want to capture nested structure)
                    elif role in ["pageHeader", "subheading"] and current_section:
                        # This could be a subsection
                        if current_subsection:
                            current_section["subsections"].append(current_subsection)
                        
                        current_subsection = {
                            "title": content,
                            "content": [],
                            "tables": []
                        }
                    
                    # Regular content
                    else:
                        # Skip page headers/footers unless you want them
                        if role in ["pageFooter", "pageNumber"]:
                            continue
                        
                        # Add content to appropriate container
                        if current_subsection:
                            current_subsection["content"].append({
                                "text": content,
                                "page": element["page"],
                                "type": role or "body"
                            })
                        elif current_section:
                            current_section["content"].append({
                                "text": content,
                                "page": element["page"],
                                "type": role or "body"
                            })
                        else:
                            # Content before first section - could be introduction/preface
                            if not document["sections"]:
                                # Create an introduction section
                                current_section = {
                                    "title": "Introduction",
                                    "page_start": element["page"],
                                    "content": [{
                                        "text": content,
                                        "page": element["page"],
                                        "type": role or "body"
                                    }],
                                    "tables": [],
                                    "figures": [],
                                    "subsections": []
                                }
                
                elif element["type"] == "table":
                    # Add table to current section or subsection
                    if current_subsection:
                        current_subsection["tables"].append(element["data"])
                    elif current_section:
                        current_section["tables"].append(element["data"])
                    else:
                        # Table before first section
                        if not document["sections"]:
                            current_section = {
                                "title": "Document Start",
                                "page_start": element["page"],
                                "content": [],
                                "tables": [element["data"]],
                                "figures": [],
                                "subsections": []
                            }
        
        # Don't forget to add the last section
        if current_section:
            if current_subsection:
                current_section["subsections"].append(current_subsection)
            document["sections"].append(current_section)
        
        # Add summary statistics
        document["statistics"] = {
            "total_sections": len(document["sections"]),
            "total_tables": sum(len(s.get("tables", [])) for s in document["sections"]),
            "total_paragraphs": len(analysis_result.paragraphs) if analysis_result.paragraphs else 0,
            "total_pages": document["total_pages"]
        }
        
        return document

    def _format_table(self, table) -> dict:
        """Helper function to format a table into a more usable structure"""
        
        # Initialize table as 2D array
        formatted_table = {
            "row_count": table.row_count,
            "column_count": table.column_count,
            "headers": [],
            "rows": []
        }
        
        # Create empty 2D array
        table_array = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]
        
        # Fill the array with cell contents
        for cell in table.cells:
            row_idx = cell.row_index
            col_idx = cell.column_index
            
            # Handle cell spans if needed
            for r in range(cell.row_span or 1):
                for c in range(cell.column_span or 1):
                    if row_idx + r < table.row_count and col_idx + c < table.column_count:
                        table_array[row_idx + r][col_idx + c] = cell.content
        
        # Identify headers (usually first row or cells marked as headers)
        header_cells = [cell for cell in table.cells if cell.kind == "columnHeader" or cell.row_index == 0]
        
        if header_cells:
            # Use first row as headers
            formatted_table["headers"] = table_array[0]
            formatted_table["rows"] = table_array[1:]
        else:
            # No clear headers, use generic column names
            formatted_table["headers"] = [f"Column {i+1}" for i in range(table.column_count)]
            formatted_table["rows"] = table_array
        
        return formatted_table

    def save_structured_document_as_json(self, structured_doc: dict, file_path: str) -> None:
        """Save the structured document as a JSON file for inspection"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(structured_doc, f, indent=2, ensure_ascii=False)
            print(f"Structured document saved as JSON to: {file_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
    
    def save_raw_adi_as_json(self, analysis_result: AnalyzeResult, file_path: str) -> None:
        """Save the raw ADI result as JSON (converted from the object)"""
        try:
            # Convert the ADI result to a serializable format
            adi_data = {
                "content": analysis_result.content,
                "pages": [],
                "paragraphs": [],
                "tables": [],
                "key_value_pairs": [],
                "styles": []
            }
            
            # Convert pages
            if analysis_result.pages:
                for page in analysis_result.pages:
                    page_data = {
                        "page_number": page.page_number,
                        "angle": page.angle,
                        "width": page.width,
                        "height": page.height,
                        "unit": page.unit,
                        "lines": [],
                        "words": []
                    }
                    
                    if page.lines:
                        for line in page.lines:
                            page_data["lines"].append({
                                "content": line.content,
                                "polygon": [(p.x, p.y) for p in line.polygon] if line.polygon else []
                            })
                    
                    if page.words:
                        for word in page.words:
                            page_data["words"].append({
                                "content": word.content,
                                "confidence": word.confidence,
                                "polygon": [(p.x, p.y) for p in word.polygon] if word.polygon else []
                            })
                    
                    adi_data["pages"].append(page_data)
            
            # Convert paragraphs
            if analysis_result.paragraphs:
                for para in analysis_result.paragraphs:
                    para_data = {
                        "content": para.content,
                        "role": para.role,
                        "bounding_regions": []
                    }
                    
                    if para.bounding_regions:
                        for region in para.bounding_regions:
                            para_data["bounding_regions"].append({
                                "page_number": region.page_number,
                                "polygon": [(p.x, p.y) for p in region.polygon] if region.polygon else []
                            })
                    
                    adi_data["paragraphs"].append(para_data)
            
            # Convert tables
            if analysis_result.tables:
                for table in analysis_result.tables:
                    table_data = {
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cells": [],
                        "bounding_regions": []
                    }
                    
                    for cell in table.cells:
                        cell_data = {
                            "content": cell.content,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "row_span": cell.row_span,
                            "column_span": cell.column_span,
                            "kind": cell.kind
                        }
                        table_data["cells"].append(cell_data)
                    
                    if table.bounding_regions:
                        for region in table.bounding_regions:
                            table_data["bounding_regions"].append({
                                "page_number": region.page_number,
                                "polygon": [(p.x, p.y) for p in region.polygon] if region.polygon else []
                            })
                    
                    adi_data["tables"].append(table_data)
            
            # Convert key-value pairs
            if analysis_result.key_value_pairs:
                for kv in analysis_result.key_value_pairs:
                    kv_data = {
                        "key": kv.key.content if kv.key else None,
                        "value": kv.value.content if kv.value else None,
                        "confidence": kv.confidence
                    }
                    adi_data["key_value_pairs"].append(kv_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(adi_data, f, indent=2, ensure_ascii=False)
            print(f"Raw ADI result saved as JSON to: {file_path}")
            
        except Exception as e:
            print(f"Error saving raw ADI JSON: {e}")


if __name__ == "__main__":
    # Example usage
    bridge = AzureDocumentIntelligenceBridge()
    if not os.path.exists("./KeypointSiteAlignment.pkl"):
        print(f"Running ADI Call")
        result = bridge.analyze_local_document("./Data/KeypointSiteAlignment.pdf", "KeypointSiteAlignment.pkl")
    
    # Load from cache
    cached_result = bridge.load_analysis_result("KeypointSiteAlignment.pkl")
    
    # Save the raw ADI result as JSON
    print("Saving raw ADI result as JSON...")
    bridge.save_raw_adi_as_json(cached_result, "./raw_adi_result.json")
    
    # Format and save the structured document
    print("Formatting and saving structured document...")
    structured_doc = bridge.format_result_as_structured_document(cached_result)
    bridge.save_structured_document_as_json(structured_doc, "./structured_document.json")
    
    print(f"\nFiles created:")
    print(f"  - raw_adi_result.json: Complete raw ADI output")
    print(f"  - structured_document.json: Processed hierarchical document")
    
    print(f"\nDocument Summary:")
    print(f"  Title: {structured_doc.get('title', 'N/A')}")
    print(f"  Sections: {structured_doc['statistics']['total_sections']}")
    print(f"  Tables: {structured_doc['statistics']['total_tables']}")
    print(f"  Pages: {structured_doc['total_pages']}")


    # Use the new structured formatting
    # structured_doc = bridge.format_result_as_structured_document(cached_result)
    
    # # Print document structure
    # print(f"\nDocument Title: {structured_doc['title']}")
    # print(f"Total Pages: {structured_doc['total_pages']}")
    # print(f"Total Sections: {structured_doc['statistics']['total_sections']}")
    # print(f"Total Tables: {structured_doc['statistics']['total_tables']}")
    
    # # Print out Document Keys
    # print("\nDocument Keys:")
    # for key in structured_doc.keys():
    #     print(f"- {key} -> {type(structured_doc[key])}")

    # print("\n\n\n")

    # print(f"Title: {structured_doc['title']}")
    # print(f"Metadata: {structured_doc['metadata']}")
    # print(f"Sections:")
    # for section in structured_doc['sections'][:5]:
    #     print(f"  Section Title: {section['title']}")
    #     print(f"  Page Start: {section['page_start']}")
    #     print(f"  Number of Paragraphs: {len(section['content'])}")
    #     print(f"  Number of Tables: {len(section['tables'])}")
    #     print(f"  Paragraph 1: {section['content'][0] if section['content'] else 'N/A'}")
    #     for subsection in section['subsections']:
    #         print(f"    Subsection Title: {subsection['title']}")
    #         print(f"    Number of Paragraphs: {len(subsection['content'])}")
    #         print(f"    Number of Tables: {len(subsection['tables'])}")
    #     print("\n")
    # print(f"Appendices: {len(structured_doc['appendices'])}")
    # print(f"Total Pages: {structured_doc['total_pages']}")
    # print(f"Statistics: {structured_doc['statistics']}")

    print(f"\n\n\n")

    
    
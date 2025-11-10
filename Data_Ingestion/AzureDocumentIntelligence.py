import dotenv
import os
import pickle
from pathlib import Path
import json
from collections import defaultdict

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
        # Objects to Hold Document Structure
        document_sections = defaultdict(dict)
        current_section_title = "Document_Opening"
        document_sections[current_section_title] = {
                        "sectionTitle": current_section_title,
                        "pageNumber": [],
                        "content": [],
                        "tables": []    
                    }
        document_title = ""
        page_to_table_data = defaultdict(list)

        # Iterate Over Tables to Map to Pages to Avoid Adding as Raw Text. Add Tables to Document Structure Table Section
        if analysis_result.tables:
            for table_index, table in enumerate(analysis_result.tables):
                table_markdown_string = ""
                table_headers = []   
                table_markdown_string = self._format_table(table)    
                table_page_number = table.bounding_regions[0].page_number if table.bounding_regions else None
                
                if table_page_number is not None:
                    # Create a dictionary for each table with all its data
                    table_data = {
                        "boundingBox": table.bounding_regions[0].polygon if table.bounding_regions else [],
                        "tableMarkdown": table_markdown_string,
                        "tableAddedToSection": False
                    }
                    
                    # Append to the list for this page (handles multiple tables per page)
                    page_to_table_data[table_page_number].append(table_data)
                else:
                    raise ValueError("Table does not have a valid bounding region with page number.")

        if analysis_result.paragraphs:
            for section in analysis_result.paragraphs:
                # Just Skipped for Table Overlap
                skipSectionDueToTableOverlap = False
                # Get Page Number
                current_page_number = section.bounding_regions[0].page_number if section.bounding_regions else None
                # Get Section Bounding Box
                section_bounding_box = section.bounding_regions[0].polygon if section.bounding_regions else []
                # Get Section Type
                section_role = section.role if section.role else "content"
                # Get Section Content
                section_content = section.content.strip() if section.content else ""
                # Skip Unimportant Sections
                if section_role in ["pageFooter", "pageHeader", "pageNumber"]:
                    continue  # Skip non-content sections
                
                # Check Overlap With Tables on This Page, Add Markdown Table if Overlap is Present
                if current_page_number in page_to_table_data.keys():
                    # Check against all tables on this page
                    for table_data in page_to_table_data[current_page_number]:
                        table_bounding_box = table_data["boundingBox"]
                        
                        if table_bounding_box:  # Make sure bounding box exists
                            table_top_left = table_bounding_box[0]
                            table_bottom_right = table_bounding_box[2]

                            # Simple Bounding Box Check
                            if table_top_left[0] - 0.1 <= section_bounding_box[0][0] <= table_bottom_right[0] + 0.1 and \
                            table_top_left[1] - 0.1 <= section_bounding_box[0][1] <= table_bottom_right[1] + 0.1:
                                # Add Table as Markdown to Current Section
                                if not table_data["tableAddedToSection"]:
                                    document_sections[current_section_title]["tables"].append({
                                        "tableMarkdown": table_data["tableMarkdown"],
                                        "tablePageNumber": current_page_number
                                    })
                                    table_data["tableAddedToSection"] = True
                                skipSectionDueToTableOverlap = True
                                break  # Only add each overlapping table once per section
                
                # Did we just skip adding content because of table overlap?
                if skipSectionDueToTableOverlap:
                    continue

                # If Title, Set or Append to Document Title
                if section_role == "title":
                    if document_title == "":
                        document_title = section_content
                    else:
                        document_title += " " + section_content
                # If Section Heading, Create New Section Entry
                elif section_role == "sectionHeading":
                    document_sections[section_content + "_" + str(current_page_number)] = {
                        "sectionTitle": section_content,
                        "pageNumber": [current_page_number],
                        "content": [],
                        "tables": []    
                    }
                    # Set Title of Last Section So We Can Add Content to It
                    current_section_title = section_content + "_" + str(current_page_number)
                # If Section Content, Prepare to Append to Current Section
                elif section_role == "content":
                    document_sections[current_section_title]["content"].append(section_content)
                    if current_page_number not in document_sections[current_section_title]["pageNumber"]:
                        document_sections[current_section_title]["pageNumber"].append(current_page_number)

        return {
            "title": document_title,
            "sections": list(document_sections.values()),
            "total_pages": len(analysis_result.pages) if analysis_result.pages else 0,
            "statistics": {
                "total_sections": len(document_sections),
                "total_tables": len(analysis_result.tables) if analysis_result.tables else 0
            }      
        }

    def _format_table(self, table) -> str:
        """Helper function to format a table into markdown"""
        row_count = table.row_count
        column_count = table.column_count
        table_matrix = [["" for _ in range(column_count)] for _ in range(row_count)]
        
        # Fill the matrix with cell content
        for cell in table.cells:
            table_matrix[cell.row_index][cell.column_index] = cell.content
            if cell.row_span > 1 or cell.column_span > 1:
                # Handle merged cells if necessary
                for r in range(cell.row_index, cell.row_index + cell.row_span):
                    for c in range(cell.column_index, cell.column_index + cell.column_span):
                        if r == cell.row_index and c == cell.column_index:
                            continue
                        table_matrix[r][c] = ""  # Mark merged cells as Empty
        
        # Convert matrix to markdown string
        markdown_lines = []
        
        # Add header row (first row)
        if row_count > 0:
            header_row = "| " + " | ".join(str(cell).strip() for cell in table_matrix[0]) + " |"
            markdown_lines.append(header_row)
            
            # Add separator row
            separator = "|" + "---|" * column_count
            markdown_lines.append(separator)
            
            # Add data rows (remaining rows)
            for row in table_matrix[1:]:
                data_row = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
                markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)  

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

    
    
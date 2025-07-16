# This is script 4! code works - it takes all the xml in fulltext downloaded by paper_retriever.py and saved in retrieved_papers, 
# extracts the content, saves in chunks of 5'ves/jsonl along with meta data - N.B ALL IN ONE JSONL I.E 1F YOU STARTED WITH 1M XML all will be in 1 jsonl
import os
import re
import json
import xml.etree.ElementTree as ET
import logging
import spacy
from habanero import Crossref

# Configure logging
logging.basicConfig(
    filename='process_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Logging initialized. Script started.")

# Load spaCy model for sentence segmentation
nlp = spacy.load("en_core_web_md")

# Function to extract DOI from the filename
def extract_doi_from_filename(file_path):
    """Extract DOI from the filename."""
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    if '_' in name_without_ext and name_without_ext.startswith("10."):
        prefix, suffix = name_without_ext.split('_', 1)
        return f"{prefix}/{suffix}"
    return name_without_ext.replace('_', '.')

def fetch_journal_metadata(doi):
    """Fetch journal metadata including title, journal name, and DOI from Crossref."""
    cr = Crossref()
    try:
        result = cr.works(ids=doi)
        if 'message' in result:
            message = result['message']
            title = message['title'][0] if 'title' in message and isinstance(message['title'], list) else ""
            container = message['container-title'][0] if 'container-title' in message and message['container-title'] else ""
            return {
                'title': title,
                'journal_name': container,
                'doi': message.get('DOI', doi)
            }
    except Exception as e:
        logging.error(f"Error fetching journal metadata for DOI {doi}: {e}")
        return None
    
# Function to extract metadata (title, journal name) from the XML itself
def extract_metadata_from_xml(xml_path):
    """Extract title and journal name from the XML itself."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        title = None
        journal_name = None

        # Try different common tag patterns for title
        title_tags = [
            ".//title-group/article-title",
            ".//article-meta/title-group/article-title",
            ".//front/article-meta/title-group/article-title", 
            ".//title"
        ]
        
        # Try different common tag patterns for journal name
        journal_tags = [
            ".//journal-meta/journal-title-group/journal-title",
            ".//journal-meta/journal-title",
            ".//front/journal-meta/journal-title-group/journal-title",
            ".//journal-name"
        ]

        # Try to find title
        for tag in title_tags:
            title_tag = root.find(tag)
            if title_tag is not None and title_tag.text:
                title = title_tag.text.strip()
                break
                
        # Try to find journal name
        for tag in journal_tags:
            journal_tag = root.find(tag)
            if journal_tag is not None and journal_tag.text:
                journal_name = journal_tag.text.strip()
                break

        # Call the enhanced page number extraction function
        page_range = extract_page_numbers(root)

        return {
            'title': title,
            'journal_name': journal_name,
            'pages': page_range
        }
    except Exception as e:
        logging.error(f"Error extracting metadata from XML '{xml_path}': {e}")
        return None
    
def extract_page_numbers(root):
    """Extract page numbers with multiple fallback patterns."""
    
    def clean_page_number(page_str):
        """Clean page number string by removing common prefixes and formatting."""
        if not page_str:
            return None
            
        # Remove common prefixes like "p.", "pp.", "page", "pages", etc.
        page_str = re.sub(r'^(p\.?\s*|pp\.?\s*|page\s*|pages\s*)', '', page_str, flags=re.IGNORECASE)
        
        # Remove any remaining leading/trailing whitespace
        page_str = page_str.strip()
        
        # Remove any trailing periods or commas
        page_str = re.sub(r'[.,]+$', '', page_str)
        
        return page_str if page_str else None
    
    try:
        # Method 1: fpage + lpage
        fpage = root.find(".//fpage")
        lpage = root.find(".//lpage")
        if fpage is not None and lpage is not None:
            fpage_text = clean_page_number(fpage.text) if fpage.text else ""
            lpage_text = clean_page_number(lpage.text) if lpage.text else ""
            if fpage_text and lpage_text:
                return f"{fpage_text}-{lpage_text}"

         # Method 2: fpage only
        if fpage is not None and fpage.text:
            cleaned_fpage = clean_page_number(fpage.text)
            if cleaned_fpage:
                return cleaned_fpage

        # Method 3: pub-page
        pub_page = root.find(".//pub-page")
        if pub_page is not None and pub_page.text:
            cleaned_pub_page = clean_page_number(pub_page.text)
            if cleaned_pub_page:
                return cleaned_pub_page

        # Method 4: other page tags
        page_patterns = [
            ".//page-range",
            ".//pages",
            ".//page",
            ".//article-meta/page-range",
            ".//front/article-meta/page-range"
        ]
        for pattern in page_patterns:
            page_element = root.find(pattern)
            if page_element is not None and page_element.text:
                cleaned_page = clean_page_number(page_element.text)
                if cleaned_page:
                    return cleaned_page

        # Method 5: elocation-id (electronic only)
        elocation = root.find(".//elocation-id")
        if elocation is not None and elocation.text:
            cleaned_elocation = clean_page_number(elocation.text)
            if cleaned_elocation:
                return cleaned_elocation

        # Method 6: attributes on article-meta
        article_meta = root.find(".//article-meta")
        if article_meta is not None:
            for attr in ['page-start', 'page-end', 'pages']:
                if attr in article_meta.attrib:
                    cleaned_attr = clean_page_number(article_meta.attrib[attr])
                    if cleaned_attr:
                        return cleaned_attr

        # Log if no page info found
        logging.warning("No page number found in XML.")
        return None

    except Exception as e:
        logging.error(f"Error extracting page numbers: {e}")
        return None

# Function to clean the extracted text (remove unwanted LaTeX components and normalize text)
def clean_text(text: str):
    """
    Clean PMC article text.
    - Remove LaTeX equations, packages, and document structure.
    - Normalize whitespace.
    """
    # Remove LaTeX packages (e.g., \usepackage{...})
    tags_pattern = r"\\usepackage\{.{1,10}\}"
    text = re.sub(tags_pattern, "", text)
    
    # Remove document class and related information (e.g., \documentclass[12pt]{...})
    equation_pattern = r"\\documentclass\[12pt\].{1,350}\\documentclass\[12pt\]\{minimal\}"
    text = re.sub(equation_pattern, "", text)
    
    # Remove LaTeX document body content (e.g., \begin{document} ... \end{document})
    begin_end_pattern = r"\\begin\{document\}.{1,55}\\end\{document\}"
    text = re.sub(begin_end_pattern, "", text)

    # For author-year citations
    text = re.sub(r'\([A-Za-z]+ et al\. \d{4}\)', '', text)
    text = re.sub(r'\([A-Za-z]+, \d{4}\)', '', text)

    # For numeric citations
    text = re.sub(r'\[\d+(?:, ?\d+)*\]', '', text)  # Handles [23], [23,45], [23, 45, 89] etc.

    # Normalize whitespace (convert multiple spaces to a single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Trim any leading/trailing whitespace
    return text.strip()

# Function to extract sections (e.g., Introduction, Background, Results, etc.) from the text
def extract_sections(text):
    """Extract sections from the full text using regular expressions."""
    
    sections = {
        "introduction": None,
        "background": None,
        #"methods": None,
        "results": None,
        "discussion": None,
        "conclusion": None,
    }

    # Define regex patterns to detect section headers (case insensitive)
    patterns = {
        "introduction": r"\b(introduction|intro)\b",
        "background": r"\b(background|literature review)\b",
        #"methods": r"\b(methods|methodology|materials and methods)\b",
        "results": r"\b(results|findings)\b",
        "discussion": r"\b(discussion)\b",
        "conclusion": r"\b(conclusion|conclusions|concluding remarks)\b"
    }

    paragraphs = text.split("\n\n")
    current_section = None
    section_content = {key: [] for key in sections}  # Create an empty list for each section

    # Iterate through paragraphs and assign to the correct section
    for para in paragraphs:
        para = para.strip()  # Remove leading/trailing whitespaces
        if not para:  # Skip empty paragraphs
            continue

        section_found = False
        for section, pattern in patterns.items():
            if re.search(pattern, para, re.IGNORECASE):
                current_section = section
                section_found = True
                break
        
        if current_section and not section_found:  # Only add if not a section header
            section_content[current_section].append(para)

    # Store only non-empty sections
    for section in section_content:
        if section_content[section]:
            sections[section] = "\n\n".join(section_content[section])

    return sections
# Function to extract text from XML
def extract_xml_text(xml_path):
    """Extract text content from XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        text = ""
        # Try to find all possible paragraph elements
        for paragraph_tag in ["p", "par", "paragraph"]:
            paragraphs = root.findall(f".//{paragraph_tag}")
            for paragraph in paragraphs:
                if paragraph.text:
                    text += paragraph.text.strip() + "\n\n"
                # Also check for nested text in children elements
                for child in paragraph:
                    if child.text:
                        text += child.text.strip() + " "
                    if child.tail:
                        text += child.tail.strip() + " "
                
        return text
    except Exception as e:
        logging.error(f"Error parsing XML file '{xml_path}': {e}")
        return ""

# function for XML-specific extraction
def extract_sections_from_xml_tags(xml_path):
    """Extract sections directly from XML structure using sec tags."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Initialize an ordered dictionary to preserve section order
        xml_sections = {}
        
        # Find all sec elements
        sec_elements = root.findall(".//sec")
        
        for sec in sec_elements:
            # Try to get section type from sec-type attribute
            sec_type = sec.get("sec-type", "")
            sec_id = sec.get("id", "")
            
            # Get section title
            title_element = sec.find("./title")
            title_text = title_element.text.strip() if title_element is not None and title_element.text else ""
            
            # Extract text content from all paragraphs in this section
            content = ""
            for p in sec.findall(".//p"):
                if p.text:
                    content += p.text + " "
                # Also process any nested elements within paragraphs
                for child in p:
                    if child.text:
                        content += child.text + " "
                    if child.tail:
                        content += child.tail + " "
                content += "\n\n"
            
            # Add to sections dictionary with title as key and content as value
            if title_text:
                xml_sections[title_text] = content.strip()
            elif sec_id:
                xml_sections[sec_id] = content.strip()
        
        return xml_sections
    except Exception as e:
        logging.error(f"Error extracting XML sections: {e}")
        return {}
    
# Function to process XML files and save to JSONL
def process_xml(xml_path, output_file):
    try:
        # Extract DOI from filename
        doi = extract_doi_from_filename(xml_path)
        
        # Get metadata from XML and Crossref
        xml_metadata = extract_metadata_from_xml(xml_path)
        crossref_metadata = fetch_journal_metadata(doi) if doi else None
        
        # Combine metadata, with Crossref taking precedence if available
        metadata = {
            'pages': None,
            'doi': doi,
            'title': None,
            'journal_name': None
        }
        
        if xml_metadata:
            metadata['pages'] = xml_metadata.get('pages')
            metadata['title'] = xml_metadata.get('title')
            metadata['journal_name'] = xml_metadata.get('journal_name')
            
        if crossref_metadata:
            metadata['title'] = crossref_metadata.get('title') or metadata['title']
            metadata['journal_name'] = crossref_metadata.get('journal_name') or metadata['journal_name']
            metadata['doi'] = crossref_metadata.get('doi') or metadata['doi']
            
        # Extract and clean text
        text = extract_xml_text(xml_path)
        if not text:
            logging.warning(f"No text extracted from {xml_path}")
            return
        
        # First extract section structure directly from XML
        xml_section_map = extract_sections_from_xml_tags(xml_path)  

        # Clean the extracted text
        cleaned_text = clean_text(text)
        
        # Extract sections after cleaning
        sections = extract_sections(cleaned_text)

        # Enhance sections with XML structure if available
        if xml_section_map:
            # Map the XML sections to our standard section categories
            for title, content in xml_section_map.items():
                title_lower = title.lower()
                
                # Map to appropriate section based on title
                if "introduction" in title_lower or title.startswith("1."):
                    sections["introduction"] = content
                elif "background" in title_lower:
                    sections["background"] = content
                # elif "method" in title_lower or "materials" in title_lower:
                #     sections["methods"] = content
                elif "result" in title_lower:
                    sections["results"] = content
                elif "discussion" in title_lower:
                    sections["discussion"] = content
                elif "conclusion" in title_lower:
                    sections["conclusion"] = content

        # Add metadata to sections
        sections.update(metadata)
        
        # Chunk the text into sentences and stack them in chunks of 5
        sentences = chunk_into_sentences(cleaned_text)
        sentence_chunks = stack_sentences(sentences, window_size=5)
        
        # Save the chunks and metadata to JSONL
        save_to_jsonl(sentence_chunks, sections, output_file)
        logging.info(f"Successfully processed {xml_path}")
        
    except Exception as e:
        logging.error(f"Error processing XML file '{xml_path}': {e}")

# Function to chunk text into sentences using spaCy
def chunk_into_sentences(text):
    """Chunk the cleaned text into individual sentences using spaCy."""
    try:
        # Handle very large texts by breaking them into chunks
        max_chars = 100000  # Limit text size for spaCy processing
        if len(text) > max_chars:
            sentences = []
            for i in range(0, len(text), max_chars):
                text_chunk = text[i:i+max_chars]
                doc = nlp(text_chunk)
                sentences.extend([sent.text.strip() for sent in doc.sents])
            return sentences
        else:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
    except Exception as e:
        logging.error(f"Error chunking text into sentences: {e}")
        # Fallback to simple sentence splitting
        return [s.strip() for s in text.split('.') if s.strip()]

# Function to stack sentences into chunks of a specified window size
def stack_sentences(sentences, window_size=5):
    """Stack sentences into chunks of a specified size while preserving metadata."""
    chunks = []
    for i in range(0, len(sentences), window_size):
        chunk = sentences[i:i + window_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(" ".join(chunk))
    return chunks

# Function to save extracted sections and chunked sentences to JSONL file
def save_to_jsonl(chunks, sections, output_file):
    try:
        # Make sure we have something to write
        if not chunks:
            logging.warning(f"No chunks to save for sections: {list(sections.keys())}")
            return
        with open(output_file, "a") as f:
            for chunk in chunks:
                # Create a new object for each chunk
                chunk_section = get_section_from_chunk(chunk, sections)
                
                json_obj = {
                    'input': chunk,  # The chunked sentences
                    'metadata': {
                        'pages': sections.get('pages', ''),
                        'section': chunk_section,
                        'doi': sections.get('doi', ''),
                        'title': sections.get('title', ''),
                        'journal': sections.get('journal_name', '')
                    }
                }
                # Add debug print
                print(f"Writing chunk to {output_file}: {json_obj['input'][:50]}...")
                # Write to JSONL file
                f.write(json.dumps(json_obj) + '\n')
        # Verify file was written
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logging.info(f"JSONL file size after writing: {file_size} bytes")
        else:
            logging.error(f"Output file {output_file} does not exist after writing!")
    except Exception as e:
        logging.error(f"Error saving to JSONL: {e}")

# Function to determine which section the chunk belongs to
def get_section_from_chunk(chunk, sections):
    """Determine which section the chunk belongs to by checking content overlap."""
    try:
        # Get meaningful sections (exclude metadata keys)
        content_sections = {k: v for k, v in sections.items() if k not in ['doi', 'title', 'journal_name']}
        
        # For each section, check if chunk content exists in the section
        for section_name, section_content in content_sections.items():
            if section_content and section_content.find(chunk) != -1:
                return section_name
                
        # If no matching section is found, determine by keyword analysis
        keywords = {
            "introduction": ["introduce", "background", "context", "overview"],
            "background": ["previous", "literature", "review", "background"],
            #"methods": ["method", "approach", "technique", "procedure", "experiment"],
            "results": ["result", "finding", "outcome", "data show", "analysis reveal"],
            "discussion": ["discuss", "implication", "interpret", "suggest"],
            "conclusion": ["conclude", "summary", "future work", "recommendation"]
        }
        
        chunk_lower = chunk.lower()
        for section, words in keywords.items():
            for word in words:
                if word in chunk_lower:
                    return section
                    
        return "unknown"
    except Exception as e:
        logging.error(f"Error determining section for chunk: {e}")
        return "unknown"

# Function to process multiple XML files in a directory
def process_xmls_in_directory(directory_path, output_file):
    # Create output file or clear existing content
    with open(output_file, 'w') as f:
        pass  # Just create or clear the file
        
    files_processed = 0
    for xml_file in os.listdir(directory_path):
        if xml_file.endswith(".xml"):
            try:
                xml_path = os.path.join(directory_path, xml_file)
                process_xml(xml_path, output_file)
                files_processed += 1
                logging.info(f"Processed XML file ({files_processed}): {xml_file}")
            except Exception as e:
                logging.error(f"Failed to process {xml_file}: {e}")
    
    logging.info(f"Completed processing {files_processed} XML files")
    return files_processed

# Main function to run the process
def main():
    xml_directory = "/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/retrieved_papers/fulltext"  # Replace with your XML folder path
    output_file = "/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/retrieved_papers/fulltext_output.jsonl"  # The output JSONL file to store the extracted data

    try:
        total_processed = process_xmls_in_directory(xml_directory, output_file)
        logging.info(f"XML extraction complete. Processed {total_processed} files. Data saved to {output_file}")
        print(f"Processing complete! {total_processed} files processed and saved to {output_file}")
    except Exception as e:
        logging.critical(f"Critical error in main process: {e}")
        print(f"Error occurred during processing: {e}")

if __name__ == "__main__":
    main()
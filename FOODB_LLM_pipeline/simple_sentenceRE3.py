# Modified simple_sentencesRE2.py - Enhanced to provide chunk context to classifier
from unsloth import FastLanguageModel
import numpy as np
import torch
import os
import re
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from triple_classifier3 import process_triples_for_classification, write_classified_triples_to_csv

# Configuration
GLOBAL_OUTPUT_MAX_TOKENS = 512
MAX_SEQ_LENGTH = GLOBAL_OUTPUT_MAX_TOKENS
DTYPE = None
LOAD_IN_4BIT = True
DUPLICATE_THRESHOLD = 0.85

# Paths
INPUT_FOLDER = "/home/otfatoku/omolola/foodb/LLM_PROJECT/simple_sentences_collab_code/FINAL_WORKFLOW_V1/Simple_Sentences/CHUNKED_XML"
OUTPUT_FOLDER = "/home/otfatoku/omolola/foodb/LLM_PROJECT/simple_sentences_collab_code/FINAL_WORKFLOW_V1/Triples/CHUNKED_XML"
LOG_FILE = "simple_sentenceRE.log"
SOURCE_ARTICLE_FOLDER = "/home/otfatoku/omolola/foodb/LLM_PROJECT/simple_sentences_collab_code/FINAL_WORKFLOW_V1/CHUNKED_XML"

# INPUT_FOLDER = "/home/otfatoku/omolola/foodb/LLM_PROJECT/collab_preprocessor/ChunkingPDF/Health_PDFs/Test/Chunked_DOIs/Simple_Sentences_Output"
# OUTPUT_FOLDER = "/home/otfatoku/omolola/foodb/LLM_PROJECT/simple_sentences_collab_code/Simple_Sentences_RE"
# LOG_FILE = "simple_sentenceRE.log"
# SOURCE_ARTICLE_FOLDER = "/home/otfatoku/omolola/foodb/LLM_PROJECT/collab_preprocessor/ChunkingPDF/Health_PDFs/Test2/Chunked_PMIDs/Test2/jsonl"

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    log_path = os.path.join(OUTPUT_FOLDER, LOG_FILE)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model():
    """Load the language model for inference"""
    logger = logging.getLogger(__name__)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model_name = 'unsloth/gemma-3-27b-it-unsloth-bnb-4bit'
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN environment variable not set.")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            token=token
        )
        FastLanguageModel.for_inference(model)
        logger.info(f"Successfully loaded model: {model_name}")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

def load_sentence_transformer():
    """Load sentence transformer for duplicate detection"""
    logger = logging.getLogger(__name__)
    
    try:
        # Using a good sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded sentence transformer model")
        return model
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {e}")
        return None

def read_jsonl_file(file_path: str) -> List[Dict]:
    """Read and parse a JSONL file"""
    logger = logging.getLogger(__name__)
    
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                    continue
        
        logger.info(f"Successfully read {len(data)} lines from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

def create_article_text(file_path: str) -> str:
    """Create article text from JSONL file"""
    joined_sentences = ""
    with open(file_path, "r", encoding="utf-8") as file:
        paragraphs = [json.loads(line)["input"] for line in file]  # Extract all paragraphs
        joined_sentences = "\n".join(paragraphs)  # Join all paragraphs

    return joined_sentences

def create_chunk_mapping(file_path: str) -> Dict[int, str]:
    """
    Create a mapping from chunk_index to chunk text
    This allows us to retrieve the original chunk for each simple sentence
    """
    logger = logging.getLogger(__name__)
    
    try:
        chunk_mapping = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    chunk_data = json.loads(line.strip())
                    chunk_index = chunk_data.get('chunk_index', line_num - 1)
                    chunk_text = chunk_data.get('input', '')

                    # Store with both int and string keys for flexibility
                    chunk_mapping[chunk_index] = chunk_text
                    if isinstance(chunk_index, int):
                        chunk_mapping[str(chunk_index)] = chunk_text
                    elif isinstance(chunk_index, str) and chunk_index.isdigit():
                        chunk_mapping[int(chunk_index)] = chunk_text
        
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing chunk line {line_num} in {file_path}: {e}")
                    continue
        
        logger.info(f"Successfully created chunk mapping with {len(chunk_mapping)} chunks")
        return chunk_mapping
    except Exception as e:
        logger.error(f"Error creating chunk mapping from {file_path}: {e}")
        return {}

def create_prompt(sample_input: str, entities: List[str]) -> str:
    """Create the prompt for triple extraction"""
    entities_str = ", ".join(entities)
    
    prompt = f"""Extract explicit or clearly implied subject–predicate–object triples from the sentence below.

    Each triple must express a meaningful biomedical or food/food constituent-related relationship.
    1. Ensure the entire context and meaning of the sentence is expressed through the extracted triples — especially ALL cause-effect relationships, relevant symptoms, disease associations, 
    2. Extract ALL relationships that contribute to understanding the biomedical or nutritional significance of the sentence.
    3. Ignore non-biomedical or meta-scientific statements.
    4. Do not extract triples from sentences that: Are keyword listings (e.g., index terms, MeSH headings) OR Describe methods or study design needs OR Have no clear subject-verb-object structure

    For each triple, format as: TRIPLE_START|Subject|Predicate|Object|TRIPLE_END

    For multiple triples, put each on a new line:
    TRIPLE_START|Subject1|Predicate1|Object1|TRIPLE_END
    TRIPLE_START|Subject2|Predicate2|Object2|TRIPLE_END

    5. If no valid triple is found, return: NO_TRIPLES
    6. no additional text, no explanations, no markdown, no code blocks, no quadripules.

    7. Ensure that the entity/enities given is the subject of triple relationship if other entities are present, do multiple triples and provide them as objects. 
    8. Valid objects are entities, properties, or other relevant biomedical/ pharmacological activity related information that can be derived from the sentence. 
    9. Ensure triples depict the meaning and biomedical context of the sentence. 
    10. Extract all possible triples from the sentence that are related to the entity with predicates that are verbs or action phrases.

    Special consideration:
    - where Taxonomy, scientific/botanical names is stated - use the terms "Scientific name", "Family", "Cultivar" as predicates as applicable. e.g |cranberry|Scientific name|Vaccinium macrocarpon| or |Pilgrim|Cultivar|Cranberry|
    - where constituents with quantity are stated, add quantity/concentration with the units to the object e.g Ascorbic acid (550mg/100g) or Potassium (10%) etc.
    - where sensory property is stated use the terms "Colour", "Taste", "Aroma", "Flavor", "Texture" as predicates as applicable. e.g |Phycocyanin|Colour|blue-green| or |Kefir|Taste|Sour| or |Cranberry|Aroma|Fruity|
    - where stated that constituent can be found in biofluid use "found in" as predicate e.g |Peonidin 3-o-galactoside|found in|urine|
    - Exclude redundant triples like |Mango|is a|fruit| OR |Resveratrol|is a|bioactive compound|
    ---

    Examples:

    Sentence: "Resveratrol activates SIRT1"
    Entities: ["Resveratrol"]
    Output: 
    TRIPLE_START|Resveratrol|activates|SIRT1|TRIPLE_END
    TRIPLE_START|Resveratrol|inhibits|inflammation|TRIPLE_END

    Sentence: "Pumpkin seeds (PS) contain high levels of magnesium, which may help lower blood pressure in hypertensive patients."
    Entities: ["pumpkin seeds"]
    Output: 
    TRIPLE_START|pumpkin seeds (PS)|contain|magnesium|TRIPLE_END
    TRIPLE_START|magnesium|helps lower|blood pressure|TRIPLE_END
    TRIPLE_START|blood pressure|associated with|hypertensive patients|TRIPLE_END
    ---

    Sentence: {sample_input}
    Entities: {entities_str}

    Output: """
    
    return prompt

def check_entities_in_article(triple, article):
    """Check if entities in triple exist in the article"""
    if len(triple) != 3:
        return False
    first_string = triple[0]
    last_string = triple[-1]
    
    # Try to extract number (integer or decimal) from last_string
    number_match = re.search(r'\d*\.?\d+', last_string)
    if number_match:
        # If a number is found, use only the number with a leading space
        last_string = f" {number_match.group()}"
    else:
        # If no number, try to extract acronym in parentheses
        acronym_match = re.search(r'\(([A-Za-z]+)\)', last_string)
        if acronym_match:
            # If an acronym is found, use only the text in parentheses
            last_string = acronym_match.group(1)
            print(f"last_string: {last_string}")

    subject_acronym_match = re.search(r'\(([A-Za-z]+)\)', first_string)
    if subject_acronym_match:
        # If an acronym is found in the subject, use only the text in parentheses
        first_string = subject_acronym_match.group(1)
    
    return first_string.lower() in article.lower() and last_string.lower() in article.lower()

def extract_triples(file_text, model, tokenizer, device, sample_input: str, entities: List[str], debug_mode: bool = False) -> List[List[str]]:
    """Extract triples from a sentence using the model"""
    logger = logging.getLogger(__name__)
    
    try:
        prompt = create_prompt(sample_input, entities)
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate - INCREASED TEMPERATURE
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GLOBAL_OUTPUT_MAX_TOKENS,
                temperature=0.7,  # Changed from 0.01
                top_p=0.9
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated tokens (model's response)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Debug logging for first few responses
        if debug_mode:
            logger.info(f"DEBUG - Input sentence: {sample_input}")
            logger.info(f"DEBUG - Entities: {entities}")
            logger.info(f"DEBUG - Raw model response: {repr(model_response)}")
        
        # Parse using delimiters - NO MORE JSON!
        triples = []
        
        # Check if no triples found
        if "NO_TRIPLES" in model_response:
            if debug_mode:
                logger.info("DEBUG - Model returned NO_TRIPLES")
            return []
        
        # Extract triples using delimiter pattern
        lines = model_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('TRIPLE_START|') and line.endswith('|TRIPLE_END'):
                # Remove the delimiters
                content = line[13:-11]  # Remove 'TRIPLE_START|' and '|TRIPLE_END'
                
                # Split by pipe
                parts = content.split('|')
                
                if len(parts) == 3:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    
                    # Make sure none are empty
                    if subject and predicate and obj:
                        triples.append([subject, predicate, obj])
                        if debug_mode:
                            logger.info(f"DEBUG - Extracted triple: [{subject}, {predicate}, {obj}]")
                else:
                    if debug_mode:
                        logger.warning(f"DEBUG - Invalid triple format: {line}")
        
        if debug_mode:
            logger.info(f"DEBUG - Final extracted triples: {triples}")

        print(f"Triples before checking if they are in article: {triples}")
        final_triples = []
        for triple in triples:
            if check_entities_in_article(triple, file_text):
                final_triples.append(triple)
            else:
                if debug_mode:
                    logger.warning(f"DEBUG - Triple not found in article: {triple}")

        print(f"Final triples after checking if they are in article: {final_triples}\n")
        
        return final_triples
            
    except Exception as e:
        logger.error(f"Error extracting triples: {e}")
        return []

def remove_duplicate_triples(triples_with_context: List[Tuple[List[str], str, Dict, str]], 
                           sentence_model, threshold: float = DUPLICATE_THRESHOLD) -> List[Tuple[List[str], str, Dict, str]]:
    """Remove duplicate triples using sentence similarity"""
    logger = logging.getLogger(__name__)
    
    if len(triples_with_context) <= 1:
        return triples_with_context
    
    try:
        # Extract triples and create text representations
        triple_texts = []
        for triple, _, _, _ in triples_with_context:
            if len(triple) == 3:
                triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
                triple_texts.append(triple_text)
            else:
                triple_texts.append("")
        
        # Get embeddings
        embeddings = sentence_model.encode(triple_texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicates
        to_remove = set()
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] >= threshold:
                    to_remove.add(j)  # Remove the later occurrence
        
        # Keep only non-duplicates
        unique_triples = [triples_with_context[i] for i in range(len(triples_with_context)) if i not in to_remove]
        
        removed_count = len(triples_with_context) - len(unique_triples)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate triples")
        
        return unique_triples
        
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        return triples_with_context

def save_to_csv(triples_with_context: List[Tuple[List[str], str, Dict, str]], output_file: str):
    """Save triples to CSV file (without chunk text to keep it clean)"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['simple_sentence', 'subject', 'predicate', 'object', 'pages', 'journal', 'doi', 'chunk_index']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for triple, sentence, metadata, _ in triples_with_context:  # Note: ignoring chunk_text for CSV
                if len(triple) == 3:
                    row = {
                        'simple_sentence': sentence,
                        'subject': triple[0],
                        'predicate': triple[1],
                        'object': triple[2],
                        'pages': metadata.get('pages', ''),
                        'journal': metadata.get('journal', ''),
                        'doi': metadata.get('doi', ''),
                        'chunk_index': metadata.get('chunk_index', '')
                    }
                    writer.writerow(row)
        
        logger.info(f"Successfully saved {len(triples_with_context)} triples to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

def construct_new_filepath(input_filepath: str, source_folder: str, target_folder: str) -> str:
    """Construct the corresponding file path in the target folder"""
    # Get the relative path from the source folder
    relative_path = os.path.relpath(input_filepath, source_folder)
    
    # Construct the new file path in the target folder
    new_filepath = os.path.join(target_folder, relative_path)
    
    return new_filepath

def find_source_article_path(simple_sentence_file: str) -> str:
    """Find the corresponding source article file path"""
    logger = logging.getLogger(__name__)
    
    # Extract the base filename without extension
    base_name = os.path.splitext(os.path.basename(simple_sentence_file))[0]
    
    # Remove the "_simple_sentences" suffix if present
    if base_name.endswith("_simple_sentences"):
        base_name = base_name[:-len("_simple_sentences")]
    
    logger.info(f"Looking for source file with base name: {base_name}")
    
    # Look for the corresponding JSONL file in the source article folder
    source_file = os.path.join(SOURCE_ARTICLE_FOLDER, f"{base_name}.jsonl")
    logger.info(f"Direct match attempt: {source_file}")
    
    # If the direct match exists, return it
    if os.path.exists(source_file):
        logger.info(f"Found direct match: {source_file}")
        return source_file
    
    # If the direct match doesn't exist, try to find a file that matches
    logger.warning(f"Direct match not found, searching for alternatives...")
    
    if os.path.exists(SOURCE_ARTICLE_FOLDER):
        available_files = [f for f in os.listdir(SOURCE_ARTICLE_FOLDER) if f.endswith('.jsonl')]
        logger.info(f"Available files in source folder: {available_files}")
        
        for file in available_files:
            file_base = os.path.splitext(file)[0]
            
            # Try different matching strategies
            if (file_base == base_name or 
                file_base.startswith(base_name) or 
                base_name.startswith(file_base)):
                candidate_file = os.path.join(SOURCE_ARTICLE_FOLDER, file)
                logger.info(f"Found potential match: {candidate_file}")
                return candidate_file
        
        # If no match found, try more flexible matching
        # Look for files that contain key parts of the filename
        base_parts = base_name.split()
        for file in available_files:
            file_base = os.path.splitext(file)[0]
            # Check if the file contains significant parts of the base name
            if len(base_parts) > 2:  # Only do this for longer filenames
                key_parts = [part for part in base_parts if len(part) > 3]  # Skip short words
                matches = sum(1 for part in key_parts if part.lower() in file_base.lower())
                if matches >= len(key_parts) * 0.7:  # At least 70% of key parts match
                    candidate_file = os.path.join(SOURCE_ARTICLE_FOLDER, file)
                    logger.info(f"Found fuzzy match: {candidate_file} (matched {matches}/{len(key_parts)} key parts)")
                    return candidate_file
    
    logger.warning(f"No matching source file found for: {simple_sentence_file}")
    logger.warning(f"Expected file: {source_file}")
    return source_file

def process_single_file(file_path: str, model, tokenizer, device, sentence_model):
    """Process a single JSONL file with triple extraction AND classification"""
    logger = logging.getLogger(__name__)
    
    file_name = Path(file_path).stem
    logger.info(f"Processing file: {file_name}")
    
    # Read the JSONL file
    data = read_jsonl_file(file_path)
    if not data:
        logger.warning(f"No data found in {file_path}")
        return
    
    # Find corresponding source article file
    source_article_path = find_source_article_path(file_path)
    if not os.path.exists(source_article_path):
        logger.warning(f"Source article file not found: {source_article_path}")
        return
    
    # Create article text and chunk mapping
    file_text = create_article_text(source_article_path)
    chunk_mapping = create_chunk_mapping(source_article_path)
    
    print(f"File text: {file_text[:100]}...")  # Print first 100 characters for debugging
    
    # Extract triples for all sentences
    all_triples_with_context = []
    successful_extractions = 0
    failed_extractions = 0
    
    for i, item in enumerate(data):
        simple_sentence = item.get('simple_sentence', '')
        entities = item.get('entities', [])
        
        # Extract metadata (handle both formats)
        if 'metadata' in item:
            metadata = item['metadata']
        else:
            metadata = {
                'pages': item.get('pages', ''),
                'journal': item.get('journal', ''),
                'doi': item.get('doi', ''),
                'chunk_index': item.get('chunk_index', '')
            }
        
        # FIX: Handle chunk_index type conversion
        chunk_index = item.get('chunk_index', metadata.get('chunk_index', ''))
        
        # Convert chunk_index to int if it's a string number
        if isinstance(chunk_index, str) and chunk_index.isdigit():
            chunk_index = int(chunk_index)
        elif isinstance(chunk_index, str) and chunk_index == '':
            chunk_index = i  # Use item index as fallback
        
        # Get chunk text - handle both int and string keys
        chunk_text = chunk_mapping.get(chunk_index, '')
        if not chunk_text and isinstance(chunk_index, int):
            # Try string version
            chunk_text = chunk_mapping.get(str(chunk_index), '')
        if not chunk_text and isinstance(chunk_index, str):
            # Try int version
            try:
                chunk_text = chunk_mapping.get(int(chunk_index), '')
            except (ValueError, TypeError):
                pass
        
        # If still no chunk text, use a default or log warning
        if not chunk_text:
            logger.warning(f"No chunk text found for index {chunk_index} in item {i}")
            chunk_text = simple_sentence  # Fallback to simple sentence
        
        if not simple_sentence or not entities:
            logger.warning(f"Skipping item {i}: missing sentence or entities")
            continue
        
        # Progress logging every 10 sentences
        if i % 10 == 0:
            logger.info(f"Processing sentence {i+1}/{len(data)}: {simple_sentence[:100]}...")
        
        # Debug mode for first 3 sentences
        debug_mode = i < 3
        
        # Extract triples
        triples = extract_triples(file_text, model, tokenizer, device, simple_sentence, entities, debug_mode)
        
        if triples:
            successful_extractions += 1
            for triple in triples:
                all_triples_with_context.append((triple, simple_sentence, metadata, chunk_text))
        else:
            failed_extractions += 1
            if debug_mode:
                logger.warning(f"No triples extracted for sentence {i}: {simple_sentence}")
    
    logger.info(f"Extracted triples from {successful_extractions}/{len(data)} sentences")
    logger.info(f"Failed extractions: {failed_extractions}")
    logger.info(f"Total triples before deduplication: {len(all_triples_with_context)}")
    
    # Remove duplicates
    unique_triples = remove_duplicate_triples(all_triples_with_context, sentence_model)
    logger.info(f"Total triples after deduplication: {len(unique_triples)}")
    
    # Save to original deduplicated triples to CSV
    output_file = os.path.join(OUTPUT_FOLDER, f"{file_name}_triples.csv")
    save_to_csv(unique_triples, output_file)
    
    # Classify triples and save to separate CSV for KG
    logger.info("Starting triple classification...")
    
    # Run classification - Data structure is now correct
    classified_triples = process_triples_for_classification(unique_triples, model, tokenizer, device)
    
    # Save classified triples to KG-ready CSV
    kg_output_file = os.path.join(OUTPUT_FOLDER, f"{file_name}_kg_ready.csv")
    write_classified_triples_to_csv(classified_triples, kg_output_file)

    return len(data), successful_extractions, len(all_triples_with_context), len(unique_triples), len(classified_triples)

def main():
    """Main processing function"""
    logger = setup_logging()
    
    try:
        # Load models
        logger.info("Loading models...")
        model, tokenizer, device = load_model()
        if model is None:
            logger.error("Failed to load language model")
            return
        
        sentence_model = load_sentence_transformer()
        if sentence_model is None:
            logger.error("Failed to load sentence transformer")
            return
        
        # Process all files in the input folder
        input_files = []
        for root, dirs, files in os.walk(INPUT_FOLDER):
            for file in files:
                if file.endswith('.jsonl'):
                    input_files.append(os.path.join(root, file))
        
        if not input_files:
            logger.warning(f"No JSONL files found in {INPUT_FOLDER}")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Track overall statistics
        total_sentences = 0
        total_successful = 0
        total_triples_before = 0
        total_triples_after = 0
        total_classified = 0
        
        # Process each file
        for file_path in input_files:
            logger.info(f"Processing file: {file_path}")
            
            # Process the file
            result = process_single_file(file_path, model, tokenizer, device, sentence_model)
            
            if result:
                sentences, successful, triples_before, triples_after, classified = result
                total_sentences += sentences
                total_successful += successful
                total_triples_before += triples_before
                total_triples_after += triples_after
                total_classified += classified
                
                logger.info(f"File {Path(file_path).stem} - Sentences: {sentences}, "
                           f"Successful: {successful}, Triples: {triples_before}→{triples_after}, "
                           f"Classified: {classified}")
            else:
                logger.warning(f"Failed to process {file_path}")
        
        # Log final statistics
        logger.info("=" * 50)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {len(input_files)}")
        logger.info(f"Total sentences: {total_sentences}")
        logger.info(f"Successful extractions: {total_successful}")
        logger.info(f"Total triples before deduplication: {total_triples_before}")
        logger.info(f"Total triples after deduplication: {total_triples_after}")
        logger.info(f"Total classified triples: {total_classified}")
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    print("=== SCRIPT STARTED ===")
    main()
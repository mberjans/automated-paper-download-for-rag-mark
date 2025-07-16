"""
API-based version of simple_sentenceRE3.py
This script replaces the local Gemma model with API-based models for triple extraction
"""

import numpy as np
import os
import re
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import the API wrapper
from llm_wrapper import LLMWrapper

# Configuration
GLOBAL_OUTPUT_MAX_TOKENS = 512
DUPLICATE_THRESHOLD = 0.85

# Paths (modify these for your setup)
INPUT_FOLDER = "sample_input"
OUTPUT_FOLDER = "sample_output_api"
LOG_FILE = "simple_sentenceRE_api.log"

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

def load_api_wrapper():
    """Load the API wrapper for inference"""
    logger = logging.getLogger(__name__)
    
    try:
        api_wrapper = LLMWrapper()
        
        # Test the wrapper
        test_response = api_wrapper.generate_single("Test", max_tokens=5)
        if not test_response:
            raise ValueError("API wrapper test failed")
        
        logger.info(f"Successfully loaded API wrapper with model: {api_wrapper.current_model.get('model_name', 'Unknown')}")
        return api_wrapper
        
    except Exception as e:
        logger.error(f"Error loading API wrapper: {e}")
        return None

def load_sentence_transformer():
    """Load sentence transformer for duplicate detection"""
    logger = logging.getLogger(__name__)
    
    try:
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
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    continue
        
        logger.info(f"Successfully read {len(data)} entries from {file_path}")
        return data
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

def extract_triples_from_sentence(api_wrapper, sentence: str, chunk_context: str = "") -> List[List[str]]:
    """Extract subject-predicate-object triples from a sentence using API"""
    logger = logging.getLogger(__name__)
    
    # Create prompt for triple extraction
    prompt = f"""Extract subject-predicate-object triples from the given sentence. Focus on relationships between food compounds, bioactivities, and health effects.

Context (if available): {chunk_context}

Sentence: {sentence}

Instructions:
1. Extract clear subject-predicate-object relationships
2. Focus on food compounds, biological targets, and health effects
3. Return each triple as [subject, predicate, object]
4. Only extract factual relationships present in the sentence
5. Return one triple per line

Triples:"""
    
    try:
        response = api_wrapper.generate_single(
            prompt=prompt,
            max_tokens=GLOBAL_OUTPUT_MAX_TOKENS,
            temperature=0.1
        )
        
        # Parse triples from response
        triples = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for patterns like [subject, predicate, object] or "subject, predicate, object"
            if line.startswith('[') and line.endswith(']'):
                try:
                    # Remove brackets and split by comma
                    triple_str = line.strip('[]')
                    parts = [part.strip().strip('"\'') for part in triple_str.split(',')]
                    if len(parts) == 3:
                        triples.append(parts)
                except:
                    continue
            elif ',' in line and not line.startswith('#'):
                # Try to parse comma-separated values
                parts = [part.strip().strip('"\'') for part in line.split(',')]
                if len(parts) == 3:
                    triples.append(parts)
        
        logger.debug(f"Extracted {len(triples)} triples from sentence: {sentence[:50]}...")
        return triples
        
    except Exception as e:
        logger.error(f"Error extracting triples: {e}")
        return []

def classify_triple_api(api_wrapper, chunk_text: str, simple_sentence: str, triple: List[str]) -> Dict[str, str]:
    """Classify a triple using the API"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not chunk_text or not simple_sentence or len(triple) != 3:
            logger.warning(f"Invalid inputs for classification")
            return {
                "subject_entity": triple[0] if len(triple) > 0 else "Unknown",
                "subject_entity_type": "Other",
                "predicate": triple[1] if len(triple) > 1 else "Unknown",
                "object_entity": triple[2] if len(triple) > 2 else "Unknown",
                "object_entity_type": "Other",
                "label": "Other"
            }
        
        triple_str = f'["{triple[0]}", "{triple[1]}", "{triple[2]}"]'
        
        prompt = f"""You are given an original text chunk, a simple sentence extracted from it, and a triple (subject, predicate, object) extracted from that sentence.

Use the ORIGINAL CHUNK for context to better understand the entities and relationships.

For the subject and object entities, classify their entity types using these broad categories:
- Food (e.g plant species, processed food etc)
- Chemical constituent
- Category (e.g.,categories such as fruit, beverage)
- Geographical Location
- Biological location (e.g. organ, tissue, cell) 
- Appearance/colour
- Taste
- Flavor
- Aroma
- Texture
- Disease
- Biological process
- Pathway (e.g., glycolysis, MAPK pathway)
- Enzyme
- Protein
- Hormone
- Biofluid e.g plasma, urine
- Biomarker (e.g measurable indicators, can be chemical/protein)
- Microbe (e.g bacterial and fungi)
- Other (if none of the above)

Then, assign a label to the triple from these:
- Taxonomic (for botanical classifications only)
- Compositional
- Geographical source
- Health effect
- Sensory/organoleptic (e.g colour, taste, aroma, texture)
- Descriptive
- Mechanism of action
- Use
- Other (if none of the above)

Return ONLY in this format (no JSON, no extra text or explanation):
CLASSIFICATION_START|subject_entity|subject_entity_type|predicate|object_entity|object_entity_type|label|CLASSIFICATION_END

Original Chunk: "{chunk_text}"
Simple Sentence: "{simple_sentence}"
Triple: {triple_str}"""
        
        response = api_wrapper.generate_single(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1
        )
        
        # Parse classification result
        if "CLASSIFICATION_START|" in response and "|CLASSIFICATION_END" in response:
            start_idx = response.find("CLASSIFICATION_START|") + len("CLASSIFICATION_START|")
            end_idx = response.find("|CLASSIFICATION_END")
            classification_part = response[start_idx:end_idx]
            
            parts = classification_part.split('|')
            if len(parts) >= 6:
                return {
                    "subject_entity": parts[0],
                    "subject_entity_type": parts[1],
                    "predicate": parts[2],
                    "object_entity": parts[3],
                    "object_entity_type": parts[4],
                    "label": parts[5]
                }
        
        # Fallback if parsing fails
        return {
            "subject_entity": triple[0],
            "subject_entity_type": "Other",
            "predicate": triple[1],
            "object_entity": triple[2],
            "object_entity_type": "Other",
            "label": "Other"
        }
        
    except Exception as e:
        logger.error(f"Error classifying triple: {e}")
        return {
            "subject_entity": triple[0],
            "subject_entity_type": "Other",
            "predicate": triple[1],
            "object_entity": triple[2],
            "object_entity_type": "Other",
            "label": "Other"
        }

def process_simple_sentences_file(api_wrapper, input_file: str, output_file: str):
    """Process a file of simple sentences to extract and classify triples"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing file: {input_file}")
    
    # Read input data
    data = read_jsonl_file(input_file)
    if not data:
        logger.error(f"No data to process from {input_file}")
        return
    
    results = []
    
    for entry_num, entry in enumerate(data, 1):
        try:
            # Get the simple sentences from API processing results
            api_results = entry.get('api_processing_results', {})
            simple_sentences = api_results.get('simple_sentences', [])
            original_text = entry.get('text', entry.get('content', ''))
            
            if not simple_sentences:
                logger.warning(f"No simple sentences found in entry {entry_num}")
                continue
            
            logger.info(f"Processing entry {entry_num} with {len(simple_sentences)} sentences")
            
            entry_results = {
                **entry,
                'triple_extraction_results': {
                    'extracted_triples': [],
                    'classified_triples': []
                }
            }
            
            # Process each simple sentence
            for sentence in simple_sentences:
                if not sentence or not sentence.strip():
                    continue
                
                # Extract triples
                triples = extract_triples_from_sentence(api_wrapper, sentence, original_text)
                
                for triple in triples:
                    # Classify triple
                    classification = classify_triple_api(api_wrapper, original_text, sentence, triple)
                    
                    triple_result = {
                        'sentence': sentence,
                        'triple': triple,
                        'classification': classification
                    }
                    
                    entry_results['triple_extraction_results']['extracted_triples'].append({
                        'sentence': sentence,
                        'triple': triple
                    })
                    
                    entry_results['triple_extraction_results']['classified_triples'].append(triple_result)
            
            results.append(entry_results)
            
            if entry_num % 5 == 0:
                logger.info(f"Processed {entry_num}/{len(data)} entries")
                
        except Exception as e:
            logger.error(f"Error processing entry {entry_num}: {e}")
            continue
    
    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary
        total_triples = sum(len(r['triple_extraction_results']['extracted_triples']) for r in results)
        logger.info(f"Processing complete. Extracted {total_triples} triples from {len(results)} entries.")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main function"""
    print("🔗 FOODB Triple Extraction (API Version)")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging()
    
    # Load API wrapper
    api_wrapper = load_api_wrapper()
    if not api_wrapper:
        print("❌ Failed to load API wrapper")
        return
    
    print(f"✅ API wrapper loaded: {api_wrapper.current_model.get('model_name', 'Unknown')}")
    
    # Create directories
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Look for processed simple sentence files
    input_files = list(Path(INPUT_FOLDER).glob("processed_*.jsonl"))
    
    if not input_files:
        print(f"No processed JSONL files found in {INPUT_FOLDER}")
        print("Please run the simple sentence generation script first.")
        return
    
    # Process each file
    for input_file in input_files:
        output_file = Path(OUTPUT_FOLDER) / f"triples_{input_file.name}"
        
        print(f"\nProcessing: {input_file}")
        process_simple_sentences_file(api_wrapper, str(input_file), str(output_file))
        print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

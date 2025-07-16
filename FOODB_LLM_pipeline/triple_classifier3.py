# triple_classifier.py - Modified to use original chunks for better context - Current working code.... along with simple_sentenceRE3.
import json
import re
import csv
import os
import logging
from typing import List, Dict, Tuple, Any
import torch

def prepare_classifier_prompt(chunk_text: str, simple_sentence: str, triple: List[str]) -> str:
    """
    Prepare prompt to classify a triple extracted from a sentence, using original chunk for context
    """
    # Format triple as a readable string
    triple_str = f'["{triple[0]}", "{triple[1]}", "{triple[2]}"]'
    
    prompt = f"""Below is an instruction that describes a task.

    ### Instruction:
    You are given an original text chunk, a simple sentence extracted from it, and a triple (subject, predicate, object) extracted from that sentence.

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
    
    Examples:
    chunk_text: "Resveratrol is a polyphenolic compound found in red wine that exhibits various beneficial effects on metabolic health. Resveratrol activates SIRT1, a protein that plays a key role in cellular stress resistance and longevity. This interaction has been extensively studied in the context of aging and neurodegenerative diseases, highlighting its potential therapeutic applications. Understanding how resveratrol influences SIRT1 activity helps researchers explore new strategies for promoting healthy aging and preventing disease progression."
    simple_sentence: "Resveratrol activates SIRT1"
    triple_str: ["Resveratrol", "Activates", "SIRT1"]
    Output: CLASSIFICATION_START|Resveratrol|Chemical constituent|Activates|SIRT1|protein|Mechanism of action|CLASSIFICATION_END

    ---

    ### Input:
    Original Chunk: "{chunk_text}"
    Simple Sentence: "{simple_sentence}"
    Triple: {triple_str}

    ### Response:
    """
    return prompt

def classify_triple(model, tokenizer, device, chunk_text: str, simple_sentence: str, triple: List[str]) -> Dict[str, str]:
    """
    Classify a triple using the original chunk for better context
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not chunk_text or not simple_sentence or len(triple) != 3:
            logger.warning(f"Invalid inputs for classification: chunk_text={bool(chunk_text)}, sentence={bool(simple_sentence)}, triple_len={len(triple) if triple else 0}")
            return {
                "subject_entity": triple[0] if len(triple) > 0 else "Unknown",
                "subject_entity_type": "Other",
                "predicate": triple[1] if len(triple) > 1 else "Unknown",
                "object_entity": triple[2] if len(triple) > 2 else "Unknown",
                "object_entity_type": "Other",
                "label": "Other"
            }
        prompt = prepare_classifier_prompt(chunk_text, simple_sentence, triple)
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Shorter for classification
                temperature=0.3,     # Lower temperature for classification
                top_p=0.9
            )
        
        # Decode
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Parse using delimiters
        if "CLASSIFICATION_START|" in model_response and "|CLASSIFICATION_END" in model_response:
            start_idx = model_response.find("CLASSIFICATION_START|") + len("CLASSIFICATION_START|")
            end_idx = model_response.find("|CLASSIFICATION_END")
            content = model_response[start_idx:end_idx]
            
            parts = content.split('|')
            if len(parts) == 6:
                return {
                    "subject_entity": parts[0].strip(),
                    "subject_entity_type": parts[1].strip(),
                    "predicate": parts[2].strip(),
                    "object_entity": parts[3].strip(),
                    "object_entity_type": parts[4].strip(),
                    "label": parts[5].strip()
                }
        
        # Fallback if parsing fails
        logger.warning(f"Failed to parse classification response: {model_response}")
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

def write_classified_triples_to_csv(classified_triples: List[Tuple[Dict, str, Dict]], base_output_file: str):
    """
    Save classified triples to separate CSV files:
    - One for ontology-labeled triples (taxonomic, compositional, etc.)
    - One for "Other" labeled triples
    Note: We only save the simple sentence, not the full chunk to keep CSV manageable
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Define ontology labels (case-insensitive matching)
        ontology_labels = {
            'taxonomic', 'compositional', 'health effect', 'Use',
            'sensory/organoleptic', 'descriptive', 'mechanism of action', 'geographical source'
        }
        
        # Separate triples based on label
        ontology_triples = []
        other_triples = []
        
        for classification, sentence, metadata in classified_triples:
            label = classification.get('label', '').lower().strip()
            
            if label in ontology_labels:
                ontology_triples.append((classification, sentence, metadata))
            else:
                other_triples.append((classification, sentence, metadata))
        
        # Create output file names
        base_name = base_output_file.rsplit('.', 1)[0]  # Remove .csv extension
        ontology_file = f"{base_name}_ontology_labeled.csv"
        other_file = f"{base_name}_other_labeled.csv"
        
        # Common fieldnames - keeping simple sentence only, not chunk
        fieldnames = [
            'simple_sentence', 'subject_entity', 'subject_entity_type', 
            'predicate', 'object_entity', 'object_entity_type', 'label',
            'pages', 'journal', 'doi', 'chunk_index'
        ]
        
        # Write ontology-labeled triples
        if ontology_triples:
            with open(ontology_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for classification, sentence, metadata in ontology_triples:
                    row = {
                        'simple_sentence': sentence,  # Only the simple sentence, not the chunk
                        'subject_entity': classification['subject_entity'],
                        'subject_entity_type': classification['subject_entity_type'],
                        'predicate': classification['predicate'],
                        'object_entity': classification['object_entity'],
                        'object_entity_type': classification['object_entity_type'],
                        'label': classification['label'],
                        'pages': metadata.get('pages', ''),
                        'journal': metadata.get('journal', ''),
                        'doi': metadata.get('doi', ''),
                        'chunk_index': metadata.get('chunk_index', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Successfully saved {len(ontology_triples)} ontology-labeled triples to {ontology_file}")
        else:
            logger.info("No ontology-labeled triples found")
        
        # Write "Other" labeled triples
        if other_triples:
            with open(other_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for classification, sentence, metadata in other_triples:
                    row = {
                        'simple_sentence': sentence,  # Only the simple sentence, not the chunk
                        'subject_entity': classification['subject_entity'],
                        'subject_entity_type': classification['subject_entity_type'],
                        'predicate': classification['predicate'],
                        'object_entity': classification['object_entity'],
                        'object_entity_type': classification['object_entity_type'],
                        'label': classification['label'],
                        'pages': metadata.get('pages', ''),
                        'journal': metadata.get('journal', ''),
                        'doi': metadata.get('doi', ''),
                        'chunk_index': metadata.get('chunk_index', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Successfully saved {len(other_triples)} other-labeled triples to {other_file}")
        else:
            logger.info("No other-labeled triples found")
        
        # Log statistics
        total_triples = len(classified_triples)
        logger.info(f"Classification breakdown: {len(ontology_triples)} ontology-labeled, {len(other_triples)} other-labeled out of {total_triples} total")
        
    except Exception as e:
        logger.error(f"Error saving classified triples to CSV: {e}")

def get_classification_statistics(classified_triples: List[Tuple[Dict, str, Dict]]) -> Dict[str, int]:
    """
    Get statistics about the classification labels
    """
    label_counts = {}
    entity_type_counts = {'subject': {}, 'object': {}}
    
    for classification, _, _ in classified_triples:
        # Count labels
        label = classification.get('label', 'Unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Count entity types
        subj_type = classification.get('subject_entity_type', 'Unknown')
        obj_type = classification.get('object_entity_type', 'Unknown')
        
        entity_type_counts['subject'][subj_type] = entity_type_counts['subject'].get(subj_type, 0) + 1
        entity_type_counts['object'][obj_type] = entity_type_counts['object'].get(obj_type, 0) + 1
    
    return {
        'label_counts': label_counts,
        'entity_type_counts': entity_type_counts
    }

def process_triples_for_classification(triples_with_context: List[Tuple[List[str], str, Dict, str]], 
                                     model, tokenizer, device) -> List[Tuple[Dict, str, Dict]]:
    """
    Process all triples for classification using the original chunks for better context
    Enhanced to use chunks for classification while keeping simple sentences for output
    """
    logger = logging.getLogger(__name__)
    
    classified_triples = []
    total_triples = len(triples_with_context)
    
    logger.info(f"Starting classification of {total_triples} triples using chunk context...")
    
    for i, (triple, sentence, metadata, chunk_text) in enumerate(triples_with_context):
        # Progress logging
        if i % 10 == 0:
            logger.info(f"Classifying triple {i+1}/{total_triples}: {triple}")
        
        # Classify the triple using chunk context
        classification = classify_triple(model, tokenizer, device, chunk_text, sentence, triple)
        
        # Store with simple sentence (not chunk) for final output
        classified_triples.append((classification, sentence, metadata))
    
    logger.info(f"Completed classification of {len(classified_triples)} triples")
    
    # Log classification statistics
    stats = get_classification_statistics(classified_triples)
    logger.info("Classification Statistics:")
    logger.info(f"Label distribution: {stats['label_counts']}")
    logger.info(f"Subject entity types: {stats['entity_type_counts']['subject']}")
    logger.info(f"Object entity types: {stats['entity_type_counts']['object']}")
    
    return classified_triples
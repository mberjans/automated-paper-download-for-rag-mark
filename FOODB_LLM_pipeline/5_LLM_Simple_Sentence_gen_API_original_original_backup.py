"""
API-based version of 5_LLM_Simple_Sentence_gen.py
This script replaces the local Gemma model with API-based models for testing
"""

import torch
import json
import pandas as pd
import re
import logging
import os
import glob
import warnings
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from global_prompt import global_output_max_new_tokens, simple_sentence_prompt, food_entity_prompt

# Import the API wrapper
from llm_wrapper import LLMWrapper

logging.basicConfig(
    filename='sentence_processor_api.log',  
    level=logging.DEBUG,        
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filemode='a'
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Initialize API wrapper instead of local model
print("Initializing API-based LLM wrapper...")
api_wrapper = LLMWrapper()

# Load sentence transformer for deduplication (keep this local)
model_to_find_unique_sentences = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Prompts (same as original)
given_prompt1 = simple_sentence_prompt
given_prompt2 = food_entity_prompt

# Alpaca prompt format (same as original)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Statistics tracking class (same as original)
class ProcessingStats:
    def __init__(self):
        self.stats_data = []
    
    def add_article_stats(self, article_name, input_file_path, processing_timestamp, 
                         total_chunks, chunks_processed, raw_sentences_total, 
                         cleaned_sentences_total, sentences_with_entities_total,
                         unique_sentences_after_dedup, total_entities_extracted,
                         unique_entities_count, deduplication_reduction_ratio,
                         processing_status, error_message=""):
        
        self.stats_data.append({
            'article_name': article_name,
            'input_file_path': input_file_path,
            'processing_timestamp': processing_timestamp,
            'total_chunks': total_chunks,
            'chunks_processed': chunks_processed,
            'raw_sentences_total': raw_sentences_total,
            'cleaned_sentences_total': cleaned_sentences_total,
            'sentences_with_entities_total': sentences_with_entities_total,
            'unique_sentences_after_dedup': unique_sentences_after_dedup,
            'total_entities_extracted': total_entities_extracted,
            'unique_entities_count': unique_entities_count,
            'deduplication_reduction_ratio': deduplication_reduction_ratio,
            'processing_status': processing_status,
            'error_message': error_message
        })
    
    def save_stats(self, output_path):
        df = pd.DataFrame(self.stats_data)
        df.to_csv(output_path, index=False)
        print(f"Statistics saved to: {output_path}")
        logging.info(f"Statistics saved to: {output_path}")

# Global stats tracker
stats_tracker = ProcessingStats()

def generate_simple_sentences(text_chunk, prompt):
    """Generate simple sentences from a text chunk using API."""
    # Create the full prompt
    full_prompt = alpaca_prompt.format(prompt, text_chunk, "")
    
    try:
        # Use API wrapper instead of local model
        generated_response = api_wrapper.generate_single(
            prompt=full_prompt,
            max_tokens=global_output_max_new_tokens,
            temperature=0.1
        )
        
        # Clean up the response (same as original)
        response = generated_response.strip()
        
        # Remove any remaining "### Response:" headers if they appear
        if response.startswith("### Response:"):
            response = response[13:].strip()
        
        return response
        
    except Exception as e:
        logging.error(f"Error in API generation: {e}")
        return ""

def check_entities_present(sentence, entity_prompt):
    """Check if a sentence contains relevant entities using API."""
    # Create the full prompt
    full_prompt = alpaca_prompt.format(entity_prompt, sentence, "")
    
    try:
        # Use API wrapper instead of local model
        generated_response = api_wrapper.generate_single(
            prompt=full_prompt,
            max_tokens=300,
            temperature=0.1
        )
        
        # Clean up the response (same as original)
        response = generated_response.strip()
        
        # Remove any remaining "### Response:" headers if they appear
        if response.startswith("### Response:"):
            response = response[13:].strip()
        
        return response
        
    except Exception as e:
        logging.error(f"Error in API entity check: {e}")
        return ""

def clean_sentence(sentence):
    """Clean and normalize a sentence (same as original)."""
    if not sentence or not sentence.strip():
        return ""
    
    sentence = sentence.strip()
    
    # Remove leading numbers and bullets
    sentence = re.sub(r'^\d+\.\s*', '', sentence)
    sentence = re.sub(r'^[•\-\*]\s*', '', sentence)
    
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Ensure sentence ends with punctuation
    if sentence and not sentence[-1] in '.!?':
        sentence += '.'
    
    return sentence.strip()

def remove_duplicates_with_embeddings(sentences, threshold=0.85):
    """Remove duplicate sentences using embeddings (same as original)."""
    if not sentences:
        return []
    
    try:
        # Generate embeddings
        embeddings = model_to_find_unique_sentences.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        
        # Find unique sentences
        unique_indices = []
        for i in range(len(sentences)):
            is_unique = True
            for j in range(i):
                if similarity_matrix[i][j] > threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_indices.append(i)
        
        unique_sentences = [sentences[i] for i in unique_indices]
        return unique_sentences
        
    except Exception as e:
        logging.error(f"Error in deduplication: {e}")
        return sentences

def process_jsonl_file(input_file_path, output_file_path):
    """Process a JSONL file with API-based models."""
    print(f"Processing file: {input_file_path}")
    logging.info(f"Starting processing of file: {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            total_chunks = 0
            processed_chunks = 0
            
            for line_num, line in enumerate(infile, 1):
                try:
                    chunk_data = json.loads(line.strip())
                    total_chunks += 1
                    
                    # Extract text content
                    text_content = chunk_data.get('text', chunk_data.get('content', ''))
                    
                    if text_content:
                        print(f"Processing chunk {line_num}...")
                        
                        # Generate simple sentences
                        simple_sentences_text = generate_simple_sentences(text_content, given_prompt1)
                        
                        if simple_sentences_text:
                            # Split into individual sentences
                            raw_sentences = [s.strip() for s in simple_sentences_text.split('\n') if s.strip()]
                            
                            # Clean sentences
                            cleaned_sentences = [clean_sentence(s) for s in raw_sentences if clean_sentence(s)]
                            
                            # Remove duplicates
                            unique_sentences = remove_duplicates_with_embeddings(cleaned_sentences)
                            
                            # Check for entities in each sentence
                            sentences_with_entities = []
                            for sentence in unique_sentences:
                                entities = check_entities_present(sentence, given_prompt2)
                                if entities and entities.strip():
                                    sentences_with_entities.append({
                                        'sentence': sentence,
                                        'entities': entities
                                    })
                            
                            # Create output data
                            output_data = {
                                **chunk_data,
                                'api_processing_results': {
                                    'simple_sentences': unique_sentences,
                                    'sentences_with_entities': sentences_with_entities,
                                    'processing_stats': {
                                        'raw_sentences_count': len(raw_sentences),
                                        'cleaned_sentences_count': len(cleaned_sentences),
                                        'unique_sentences_count': len(unique_sentences),
                                        'sentences_with_entities_count': len(sentences_with_entities)
                                    }
                                }
                            }
                            
                            # Write to output file
                            outfile.write(json.dumps(output_data) + '\n')
                            processed_chunks += 1
                            
                            if line_num % 10 == 0:
                                print(f"Processed {line_num} chunks...")
                        
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error at line {line_num}: {e}")
                except Exception as e:
                    logging.error(f"Processing error at line {line_num}: {e}")
        
        print(f"Processing complete. Processed {processed_chunks}/{total_chunks} chunks.")
        logging.info(f"Processing complete. Processed {processed_chunks}/{total_chunks} chunks.")
        
    except FileNotFoundError:
        error_msg = f"Input file not found: {input_file_path}"
        print(error_msg)
        logging.error(error_msg)
    except Exception as e:
        error_msg = f"Error processing file: {e}"
        print(error_msg)
        logging.error(error_msg)

def main():
    """Main function to process files."""
    print("🧬 FOODB Simple Sentence Generation (API Version)")
    print("=" * 60)
    
    # Check if API wrapper is working
    try:
        test_response = api_wrapper.generate_single("Test", max_tokens=5)
        if not test_response:
            print("❌ API wrapper not working. Please check your API keys.")
            return
        print(f"✅ API wrapper working. Using model: {api_wrapper.current_model.get('model_name', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error testing API wrapper: {e}")
        return
    
    # Example usage - process a sample file
    input_dir = "sample_input"
    output_dir = "sample_output_api"
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for JSONL files to process
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        print("Please add JSONL files to process or modify the input directory path.")
        return
    
    for input_file in jsonl_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"processed_{filename}")
        
        print(f"\nProcessing: {input_file}")
        process_jsonl_file(input_file, output_file)
        print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

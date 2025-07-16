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


logging.basicConfig(
    filename='sentence_processor.log',  
    level=logging.DEBUG,        
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filemode='a'  # 'w' for overwrite (use 'a' to append to existing file)
)
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Initialize CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
max_seq_length = global_output_max_new_tokens
dtype = None
load_in_4bit = True

# Load finetuned model
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='unsloth/gemma-3-27b-it-unsloth-bnb-4bit',
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=os.environ.get("HF_TOKEN")
    )
    FastLanguageModel.for_inference(model)

# Load sentence transformer for deduplication
model_to_find_unique_sentences = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Prompts
given_prompt1 = simple_sentence_prompt
given_prompt2 = food_entity_prompt

# Alpaca prompt format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Statistics tracking class
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
    """Generate simple sentences from a text chunk."""
    # Create the full prompt
    full_prompt = alpaca_prompt.format(prompt, text_chunk, "")
    
    # Tokenize the prompt
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    
    # Get the input length to know where new generation starts
    input_length = inputs.input_ids.shape[1]
    
    # Generate without text streamer to avoid printing prompt
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=global_output_max_new_tokens,
        do_sample=False,  # For deterministic output
        pad_token_id=tokenizer.eos_token_id  # Handle padding
    )
    
    # Only decode the newly generated tokens (skip the input prompt)
    new_tokens = output_ids[0][input_length:]
    generated_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up the response
    response = generated_response.strip()
    
    # Remove any remaining "### Response:" headers if they appear
    if response.startswith("### Response:"):
        response = response[13:].strip()
    
    return response

def check_entities_present(sentence, entity_prompt):
    """Check if a sentence contains relevant entities using the entity prompt."""
    # Create the full prompt
    full_prompt = alpaca_prompt.format(entity_prompt, sentence, "")
    
    # Tokenize
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    
    # Get input length
    input_length = inputs.input_ids.shape[1]
    
    # Generate without streaming
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=300, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Only decode new tokens
    new_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

     ##############################################
    # DEBUG: Log ONLY problematic cases
    if "entities" not in response.lower() or len(response) > 200:
        print("\n--- DEBUG: Suspicious Response ---")
        print(f"Sentence: {sentence[:50]}...")
        print(f"Raw JSON: {response[:200]}...")  # Show more for debugging
    ##############################################
    
    # Clean up response
    # Only add this if responses include ```json or ``` markers:
    response = response.replace("```json", "").replace("```", "").strip()  # Remove Markdown wrappers
    if response.startswith("### Response:"):
        response = response[13:].strip()
    
    try:
        json_response = json.loads(response)
        entities = json_response.get('entities', [])
        return len(entities) > 0, entities
    except json.JSONDecodeError:
        # Enhanced fallback for malformed JSON
        if '"entities"' in response.lower():
            try:
                import re
                entities_match = re.search(r'"entities"\s*:\s*\[([^\]]*)\]', response, re.DOTALL)
                if entities_match:
                    entities_str = entities_match.group(1)
                    entities = []
                    for entity in re.findall(r'"([^"]*)"', entities_str):
                        if entity.strip():
                            entities.append(entity.strip())
                    return len(entities) > 0, entities
            except:
                pass
        
        print(f"Warning: Could not parse entity response: {response[:100]}...")
        return False, []

def is_meaningful_sentence(sentence, min_words=4):
    """Filter out fragments, references, and non-meaningful sentences."""
    sentence = sentence.strip()
    
    # Skip if too short
    if len(sentence.split()) < min_words:
        return False
    
    # Skip standalone references (e.g., "Singh et al. 79")
    if re.match(r'^[A-Za-z]+\s+et\s+al\.\s*\d*\.?$', sentence):
        return False
    
    # Skip pure citation fragments (e.g., "References")
    meaningless_patterns = [
        r'^\d+$',
        r'^[A-Z\s]+$'  # All caps fragments
    ]
    
    for pattern in meaningless_patterns:
        if re.match(pattern, sentence, re.IGNORECASE):
            return False
    
    return True

def find_unique_sentences(list_of_sentences):
    """Remove duplicate sentences using embeddings."""
    if not list_of_sentences:
        return []
    
    embeddings = model_to_find_unique_sentences.encode(list_of_sentences, convert_to_tensor=True)
    unique_sentences = []
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    added = set()
    threshold = 0.85

    for i in range(len(list_of_sentences)):
        if i in added:
            continue
        unique_sentences.append(list_of_sentences[i])
        for j in range(i + 1, len(list_of_sentences)):
            if cosine_scores[i][j] > threshold:
                added.add(j)
    return unique_sentences

def clean_sentence(sentence):
    """Clean and format a sentence."""
    sentence = sentence.strip()
    if sentence.startswith('(') and sentence.endswith(')'):
        sentence = sentence[1:-1].strip()
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1].strip()
    return sentence

def deduplicate_with_entities(filtered_sentences):
    """Enhanced deduplication considering both sentence similarity and entity overlap."""
    if not filtered_sentences:
        return []
    
    sentence_texts = [item['sentence'] for item in filtered_sentences]
    embeddings = model_to_find_unique_sentences.encode(sentence_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    
    unique_sentences = []
    added_indices = set()
    threshold = 0.85  # Slightly lower threshold for scientific content
    
    for i in range(len(filtered_sentences)):
        if i in added_indices:
            continue
            
        current_item = filtered_sentences[i]
        unique_sentences.append(current_item)
        
        # Mark similar sentences as duplicates
        for j in range(i + 1, len(filtered_sentences)):
            if j in added_indices:
                continue
                
            # Check both semantic similarity and entity overlap
            semantic_similar = cosine_scores[i][j] > threshold
            entity_overlap = len(set(current_item['entities']) & set(filtered_sentences[j]['entities'])) > 0
            
            if semantic_similar and entity_overlap:
                added_indices.add(j)
    
    return unique_sentences

def process_single_article(jsonl_file_path, output_folder):
    """Enhanced processing with better context handling, filtering, and statistics tracking."""
    
    # Initialize tracking variables
    processing_timestamp = datetime.now().isoformat()
    article_name = os.path.splitext(os.path.basename(jsonl_file_path))[0]
    total_chunks = 0
    chunks_processed = 0
    raw_sentences_total = 0
    cleaned_sentences_total = 0
    sentences_with_entities_total = 0
    unique_sentences_after_dedup = 0
    total_entities_extracted = 0
    unique_entities_count = 0
    deduplication_reduction_ratio = 0.0
    processing_status = "SUCCESS"
    error_message = ""
    
    try:
        chunks = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_data = json.loads(line.strip())
                chunks.append(chunk_data)
        
        total_chunks = len(chunks)
        
        if not chunks:
            print(f"No chunks found in {jsonl_file_path}")
            processing_status = "FAILED"
            error_message = "No chunks found in file"
            return
        
        first_chunk = chunks[0]
        article_metadata = first_chunk.get('metadata', {})
        
        print(f"Processing article: {article_name}")
        print(f"Found {len(chunks)} chunks")
        logging.info(f"Found {len(chunks)} chunks in article: {article_name}")
        
        all_filtered_sentences = []
        all_entities = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                chunk_text = chunk.get('input', '')
                chunk_metadata = chunk.get('metadata', {})
                
                print(f"Processing chunk {i+1}/{len(chunks)}")
                logging.info(f"Processing chunk {i+1}/{len(chunks)} of article: {article_name}")
                
                # Generate simple sentences
                simple_sentences_response = generate_simple_sentences(chunk_text, given_prompt1)
                
                # Parse and clean sentences
                lines = simple_sentences_response.splitlines()
                raw_sentences_chunk = len(lines)
                raw_sentences_total += raw_sentences_chunk

                # DEBUG: Print first few lines
                print(f"DEBUG - First 3 lines: {lines[:3]}")

                cleaned_sentences = []
                for line in lines:
                    sentence = clean_sentence(line)
                    if sentence and is_meaningful_sentence(sentence):  # Added meaningfulness check
                        cleaned_sentences.append(sentence)
                
                cleaned_sentences_chunk = len(cleaned_sentences)
                cleaned_sentences_total += cleaned_sentences_chunk

                # DEBUG: Print first few cleaned sentences
                print(f"DEBUG - First 3 cleaned sentences: {cleaned_sentences[:3]}")
                
                # Filter sentences with entities and collect entity information
                filtered_sentences = []
                for sentence in cleaned_sentences:
                    has_entities, entities = check_entities_present(sentence, given_prompt2)
                    
                    # DEBUG: Log entities for the first 5 chunks
                    if i < 5:  # Only for first 5 chunks (adjust as needed)
                        logging.debug(f"Chunk {i+1} | Sentence: {sentence[:50]}...")
                        logging.debug(f"Chunk {i+1} | Entities: {entities}")

                    if has_entities:
                        filtered_sentences.append({
                            'sentence': sentence,
                            'entities': entities,  # Store extracted entities
                            'chunk_metadata': chunk_metadata,
                            'chunk_index': i
                        })
                        all_entities.extend(entities)
                
                sentences_with_entities_chunk = len(filtered_sentences)
                sentences_with_entities_total += sentences_with_entities_chunk
                
                all_filtered_sentences.extend(filtered_sentences)
                chunks_processed += 1
                
                print(f"Chunk {i+1}: {raw_sentences_chunk} raw -> {cleaned_sentences_chunk} cleaned -> {sentences_with_entities_chunk} with entities")
                logging.info(f"Chunk {i+1}: {raw_sentences_chunk} raw -> {cleaned_sentences_chunk} cleaned -> {sentences_with_entities_chunk} with entities")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                logging.error(f"Error processing chunk {i+1} of article {article_name}: {e}")
                error_message += f"Chunk {i+1}: {str(e)}; "
                continue
        
        # Calculate entity statistics
        total_entities_extracted = len(all_entities)
        unique_entities_count = len(set(all_entities))
        
        # Enhanced deduplication with entity context
        sentences_before_dedup = len(all_filtered_sentences)
        unique_filtered_sentences = deduplicate_with_entities(all_filtered_sentences)
        unique_sentences_after_dedup = len(unique_filtered_sentences)
        
        # Calculate deduplication ratio
        if sentences_before_dedup > 0:
            deduplication_reduction_ratio = (sentences_before_dedup - unique_sentences_after_dedup) / sentences_before_dedup
        
        # Create enhanced output JSONL
        output_file = os.path.join(output_folder, f"{article_name}_simple_sentences.jsonl")
        os.makedirs(output_folder, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in unique_filtered_sentences:
                json_obj = {
                    'simple_sentence': item['sentence'],
                    'entities': item['entities'],  # Include extracted entities
                    'metadata': {
                        **article_metadata,
                        **item['chunk_metadata'],
                        'chunk_index': item['chunk_index']
                    }
                }
                f.write(json.dumps(json_obj) + '\n')
        
        print(f"Article {article_name} complete:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Chunks processed: {chunks_processed}")
        print(f"  Raw sentences: {raw_sentences_total}")
        print(f"  Cleaned sentences: {cleaned_sentences_total}")
        print(f"  Sentences with entities: {sentences_with_entities_total}")
        print(f"  Unique sentences after dedup: {unique_sentences_after_dedup}")
        print(f"  Total entities extracted: {total_entities_extracted}")
        print(f"  Unique entities: {unique_entities_count}")
        print(f"  Deduplication reduction: {deduplication_reduction_ratio:.2%}")
        print(f"  Output saved to: {output_file}")
        logging.info(f"Article {article_name} complete: Unique sentences with entities: {unique_sentences_after_dedup}, Output saved to: {output_file}")
        print("-" * 80)
        
    except Exception as e:
        processing_status = "FAILED"
        error_message = str(e)
        print(f"Error processing article {article_name}: {e}")
        logging.error(f"Error processing article {article_name}: {e}")
    
    finally:
        # Add statistics to tracker
        stats_tracker.add_article_stats(
            article_name=article_name,
            input_file_path=jsonl_file_path,
            processing_timestamp=processing_timestamp,
            total_chunks=total_chunks,
            chunks_processed=chunks_processed,
            raw_sentences_total=raw_sentences_total,
            cleaned_sentences_total=cleaned_sentences_total,
            sentences_with_entities_total=sentences_with_entities_total,
            unique_sentences_after_dedup=unique_sentences_after_dedup,
            total_entities_extracted=total_entities_extracted,
            unique_entities_count=unique_entities_count,
            deduplication_reduction_ratio=deduplication_reduction_ratio,
            processing_status=processing_status,
            error_message=error_message
        )

def process_all_articles(input_folder, output_folder):
    """Process all JSONL files from the chunking algorithm."""
    
    # Find all JSONL files in the input folder
    jsonl_pattern = os.path.join(input_folder, "**", "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern, recursive=True)
    
    # Filter out metadata-only files
    jsonl_files = [f for f in jsonl_files if not os.path.basename(f).endswith('_metadata.jsonl')]
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_folder}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    logging.info(f"Found {len(jsonl_files)} JSONL files to process in {input_folder}")
    
    # Process each file
    for jsonl_file in jsonl_files:
        try:
            process_single_article(jsonl_file, output_folder)
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            logging.error(f"Error processing {jsonl_file}: {e}")
            # Add failed entry to stats
            article_name = os.path.splitext(os.path.basename(jsonl_file))[0]
            stats_tracker.add_article_stats(
                article_name=article_name,
                input_file_path=jsonl_file,
                processing_timestamp=datetime.now().isoformat(),
                total_chunks=0,
                chunks_processed=0,
                raw_sentences_total=0,
                cleaned_sentences_total=0,
                sentences_with_entities_total=0,
                unique_sentences_after_dedup=0,
                total_entities_extracted=0,
                unique_entities_count=0,
                deduplication_reduction_ratio=0.0,
                processing_status="FAILED",
                error_message=str(e)
            )
            continue
    
    # Save statistics
    stats_file = os.path.join(output_folder, "simple_sentences_stats.csv")
    stats_tracker.save_stats(stats_file)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(stats_tracker.stats_data)
    if not df.empty:
        total_articles = len(df)
        successful_articles = len(df[df['processing_status'] == 'SUCCESS'])
        failed_articles = len(df[df['processing_status'] == 'FAILED'])
        
        print(f"Total articles processed: {total_articles}")
        print(f"Successfully processed: {successful_articles}")
        print(f"Failed processing: {failed_articles}")
        
        if successful_articles > 0:
            success_df = df[df['processing_status'] == 'SUCCESS']
            print(f"Total chunks processed: {success_df['chunks_processed'].sum()}")
            print(f"Total raw sentences: {success_df['raw_sentences_total'].sum()}")
            print(f"Total cleaned sentences: {success_df['cleaned_sentences_total'].sum()}")
            print(f"Total sentences with entities: {success_df['sentences_with_entities_total'].sum()}")
            print(f"Total unique sentences after dedup: {success_df['unique_sentences_after_dedup'].sum()}")
            print(f"Total entities extracted: {success_df['total_entities_extracted'].sum()}")
            print(f"Average deduplication reduction: {success_df['deduplication_reduction_ratio'].mean():.2%}")
        
        print(f"\nDetailed statistics saved to: {stats_file}")

if __name__ == "__main__":
    # Configuration
    input_folder = "/home/otfatoku/omolola/foodb/LLM_PROJECT/simple_sentences_collab_code/FINAL_WORKFLOW_V1/CHUNKED_XML"
    output_folder = "/home/otfatoku/omolola/foodb/LLM_PROJECT/simple_sentences_collab_code/FINAL_WORKFLOW_V1/Simple_Sentences/CHUNKED_XML"
    
    # Process all articles
    process_all_articles(input_folder, output_folder)
# This is script 6! code works - it takes all the xml in dedupfulltext downloaded by paper_retriever.py/depluicated after vectorization and saved in retrieved_papers
import json
import re
import ahocorasick
import pandas as pd
import logging

# Step 1: Set up logging configuration
logging.basicConfig(
    filename='CheckCompounds_log.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Logging initialized. Script started.")

# Step 2: Load the compounds CSV file
logging.info("Loading compounds data from CSV.")
compound_df = pd.read_csv('/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/normalized_compounds/compounds_with_synonyms.csv')

# Helper: Normalize quotes
def normalize_quotes(text):
    return text.replace("''", '"').replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")

# Build a compound variant → normalized name map
compound_lookup = {}
skipped_short_synonyms = 0  # Initialize counter

for _, row in compound_df.iterrows():
    normalized = normalize_quotes(str(row['normalized_name']).strip())

    # Original name
    original = row['original_name']
    if pd.notna(original):
        cleaned_original = normalize_quotes(original.strip().lower())
        if len(cleaned_original) >= 3:
            compound_lookup[cleaned_original] = normalized
        else:
            skipped_short_synonyms += 1
            logging.warning(f"⚠️ Skipping short original name: '{cleaned_original}' for {normalized}")

    # Synonyms
    synonyms = row['synonyms']
    if pd.notna(synonyms):
        for syn in synonyms.split('|'):
            cleaned_syn = normalize_quotes(syn.strip().lower())
            if len(cleaned_syn) >= 3:
                compound_lookup[cleaned_syn] = normalized
            else:
                skipped_short_synonyms += 1
                logging.warning(f"⚠️ Skipping short synonym: '{cleaned_syn}' for {normalized}")

logging.info(f"✅ Built compound lookup with {len(compound_lookup)} entries.")
logging.info(f"Skipped {skipped_short_synonyms} short synonyms (under 5 chars).")

# Step 3a: Build Aho-Corasick automaton for compounds
logging.info("Building Aho-Corasick automaton for compounds.")
compound_automaton = ahocorasick.Automaton()
for idx, variant in enumerate(compound_lookup.keys()):
    compound_automaton.add_word(variant, (idx, variant))
compound_automaton.make_automaton()
logging.info(f"✅ Built automaton with {len(compound_lookup)} compound name variants.")

# step 3b: Load health effects from CSV and build Aho-Corasick automaton
logging.info("Loading health effect terms and building Aho-Corasick automaton.")
health_effect_df = pd.read_csv('HealthEffects_table.csv')
health_effect_terms = health_effect_df['health_effect'].dropna().str.lower().tolist()

# Build the automaton
health_effect_automaton = ahocorasick.Automaton()
for idx, term in enumerate(health_effect_terms):
    health_effect_automaton.add_word(term, (idx, term))
health_effect_automaton.make_automaton()

logging.info("Health effects automaton setup complete.")

# Step 4: Define keywords for USE and MECHANISM OF ACTION
use_keywords = [
    'use', 'used', 'using', 'application', 'applied', 'utilize', 'utilization',
    'consumption', 'consume', 'intake', 'dietary', 'supplement', 'supplementation',
    'administration', 'administer', 'treatment', 'therapy', 'therapeutic',
    'functional food', 'nutraceutical', 'food additive', 'preservative',
    'ingredient', 'component', 'added to', 'incorporated', 'formulated'
]

mechanism_keywords = [
    'mechanism', 'mechanism of action', 'mode of action', 'pathway',
    'signaling pathway', 'metabolic pathway', 'biochemical pathway',
    'molecular mechanism', 'cellular mechanism', 'targets', 'receptor',
    'enzyme', 'inhibits', 'inhibition', 'activates', 'activation',
    'upregulates', 'downregulates', 'modulates', 'regulates', 'regulation',
    'binds to', 'binding', 'interaction', 'metabolism', 'metabolized',
    'bioavailability', 'absorption', 'distribution', 'excretion',
    'pharmacokinetics', 'pharmacodynamics', 'dose-response', 'IC50', 'EC50',
    'gene expression', 'protein expression', 'transcription', 'translation',
    'cell signaling', 'apoptosis', 'proliferation', 'differentiation',
    'oxidative stress', 'inflammation', 'immune response'
]

# Create regex patterns for each category
use_pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in use_keywords) + r')\b'
mechanism_pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in mechanism_keywords) + r')\b'

logging.info("Keywords and patterns setup complete.")

def validate_compound_match(text, compound_variant, start_idx, end_idx):
    """
    Validate that a compound match is a real whole-word match, not a substring.
    Returns True if valid, False if it's a false positive.
    """
    if len(compound_variant) < 3:
        return False
    # Check if the match is surrounded by word boundaries
    before_char = text[start_idx - 1] if start_idx > 0 else ' '
    after_char = text[end_idx + 1] if end_idx < len(text) - 1 else ' '
    
    # Define word boundary characters
    word_boundary_chars = set(' \t\n\r.,;:!?()[]{}"\'-/\\|')
    
    # Check if both sides have word boundaries
    is_valid = (before_char in word_boundary_chars and 
                after_char in word_boundary_chars)
    
    return is_valid

# Function to check relevance with detailed categorization
# def check_paragraph_relevance(text):
#     """
#     Check if paragraph discusses use, health effects, or mechanism of action of food compounds.
#     Returns dict with relevance type and boolean result.
#     """
#     # Apply quote normalization to search text as well
#     text_normalized = normalize_quotes(text)
#     text_lower = text_normalized.lower()
    
#     # Find compound matches using Aho-Corasick
#     compound_matches = set()
#     for end_idx, (_, matched_variant) in compound_automaton.iter(text_lower):
#         start_idx = end_idx - len(matched_variant) + 1
#         matched_text = text_lower[start_idx:end_idx+1]
#         canonical_name = compound_lookup[matched_variant]
#         compound_matches.add(canonical_name)

#     has_compound = len(compound_matches) > 0
#     if not has_compound:
#         return {
#             'relevant': False,
#             'type': None,
#             'reason': 'No compound found',
#             'matched_health_effects': [],
#             'matched_compounds': []
#         }
def check_paragraph_relevance_with_validation(text):
    """
    Enhanced version with compound validation
    """
    # Apply quote normalization to search text
    text_normalized = normalize_quotes(text)
    text_lower = text_normalized.lower()
    
    # Find compound matches using Aho-Corasick with validation
    compound_matches = set()
    validated_matches = []
    
    for end_idx, (_, matched_variant) in compound_automaton.iter(text_lower):
        start_idx = end_idx - len(matched_variant) + 1
        
        # Validate the match
        if validate_compound_match(text_lower, matched_variant, start_idx, end_idx):
            canonical_name = compound_lookup[matched_variant]
            compound_matches.add(canonical_name)
            validated_matches.append({
                'variant': matched_variant,
                'canonical': canonical_name,
                'start': start_idx,
                'end': end_idx,
                'matched_text': text_lower[start_idx:end_idx+1]
            })
        else:
            # Log false positives for debugging
            logging.debug(f"False positive filtered: '{matched_variant}' in context: "
                         f"'{text_lower[max(0, start_idx-10):end_idx+10]}'")

    has_compound = len(compound_matches) > 0
    if not has_compound:
        return {
            'relevant': False,
            'type': None,
            'reason': 'No compound found',
            'matched_health_effects': [],
            'matched_compounds': [],
            'validated_compound_matches': []
        }
     # Rest of the function remains the same - swapped above function check relvace with validation
    # Use the regex patterns that were created
    has_use = bool(re.search(use_pattern, text_lower, re.IGNORECASE))
    has_mechanism = bool(re.search(mechanism_pattern, text_lower, re.IGNORECASE))

    health_effect_matches = set()
    for _, (_, matched_term) in health_effect_automaton.iter(text_lower):
        health_effect_matches.add(matched_term)

    has_health_effect = len(health_effect_matches) > 0

    relevance_types = []
    if has_use:
        relevance_types.append('use')
    if has_health_effect:
        relevance_types.append('health_effect')
    if has_mechanism:
        relevance_types.append('mechanism')

    is_relevant = len(relevance_types) > 0

    return {
        'relevant': is_relevant,
        'type': relevance_types,
        'reason': f"Compound found with: {', '.join(relevance_types)}" if is_relevant else "Compound found but no use/health/mechanism keywords",
        'matched_compounds': list(compound_matches),
        'matched_health_effects': list(health_effect_matches)
    }

# Enhanced processing function with detailed categorization
def process_jsonl_enhanced(input_jsonl_file, output_dir):
    """Process JSONL with enhanced relevance checking"""
    
    logging.info(f"Processing JSONL file: {input_jsonl_file}")
    
    # Open output files
    files = {
        'all_relevant': open(f'{output_dir}/relevant_paragraphs_enhanced.jsonl', 'w'),
        'use_only': open(f'{output_dir}/relevant_use_only.jsonl', 'w'),
        'health_effect_only': open(f'{output_dir}/relevant_health_effect_only.jsonl', 'w'),
        'mechanism_only': open(f'{output_dir}/relevant_mechanism_only.jsonl', 'w'),
        'multiple_types': open(f'{output_dir}/relevant_multiple_types.jsonl', 'w'),
        'irrelevant': open(f'{output_dir}/irrelevant_paragraphs_enhanced.jsonl', 'w')
    }
    
    # Counters for statistics
    counts = {
        'total_processed': 0,
        'relevant_total': 0,
        'use': 0,
        'health_effect': 0,
        'mechanism': 0,
        'multiple_types': 0,
        'compound_but_irrelevant': 0,
        'no_compound': 0
    }
    
    try:
        with open(input_jsonl_file, 'r') as f_in:
            for line in f_in:
                data = json.loads(line.strip())
                text = data['input']
                counts['total_processed'] += 1
                
                # Check relevance
                relevance_result = check_paragraph_relevance_with_validation(text)
                
                # Add relevance metadata to the data
                data['relevance_info'] = relevance_result
                
                if relevance_result['relevant']:
                    counts['relevant_total'] += 1
                    files['all_relevant'].write(json.dumps(data) + '\n')
                    
                    # Count individual types
                    for rel_type in relevance_result['type']:
                        counts[rel_type] += 1
                    
                    # Categorize by specificity
                    if len(relevance_result['type']) == 1:
                        # Single type
                        single_type = relevance_result['type'][0]
                        files[f'{single_type}_only'].write(json.dumps(data) + '\n')
                    else:
                        # Multiple types
                        counts['multiple_types'] += 1
                        files['multiple_types'].write(json.dumps(data) + '\n')
                
                else:
                    files['irrelevant'].write(json.dumps(data) + '\n')
                    if 'No compound found' in relevance_result['reason']:
                        counts['no_compound'] += 1
                    else:
                        counts['compound_but_irrelevant'] += 1
                
                # Log progress every 1000 lines
                if counts['total_processed'] % 1000 == 0:
                    logging.info(f"Processed {counts['total_processed']} lines. Relevant so far: {counts['relevant_total']}")
    
    finally:
        # Close all files
        for file_handle in files.values():
            file_handle.close()
    
    # Calculate and log final statistics
    total = counts['total_processed']
    logging.info(f"Processing complete. Final statistics:")
    logging.info(f"Total processed: {total}")
    logging.info(f"Relevant paragraphs: {counts['relevant_total']} ({counts['relevant_total']/total*100:.2f}%)")
    logging.info(f"  - Use-related: {counts['use']} ({counts['use']/total*100:.2f}%)")
    logging.info(f"  - Health effect-related: {counts['health_effect']} ({counts['health_effect']/total*100:.2f}%)")
    logging.info(f"  - Mechanism-related: {counts['mechanism']} ({counts['mechanism']/total*100:.2f}%)")
    logging.info(f"  - Multiple types: {counts['multiple_types']} ({counts['multiple_types']/total*100:.2f}%)")
    logging.info(f"Compound found but irrelevant: {counts['compound_but_irrelevant']} ({counts['compound_but_irrelevant']/total*100:.2f}%)")
    logging.info(f"No compound found: {counts['no_compound']} ({counts['no_compound']/total*100:.2f}%)")
    
    # Print summary to console
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total processed: {total:,}")
    print(f"Relevant paragraphs: {counts['relevant_total']:,} ({counts['relevant_total']/total*100:.2f}%)")
    print(f"  └─ Use-related: {counts['use']:,}")
    print(f"  └─ Health effect-related: {counts['health_effect']:,}")
    print(f"  └─ Mechanism-related: {counts['mechanism']:,}")
    print(f"  └─ Multiple types: {counts['multiple_types']:,}")
    print(f"Compound found but irrelevant: {counts['compound_but_irrelevant']:,}")
    print(f"No compound found: {counts['no_compound']:,}")
    
    return counts

# Simple function for just the enhanced relevant/irrelevant split
def process_jsonl_simple(input_jsonl_file, relevant_output_file, irrelevant_output_file):
    """Simple processing function with enhanced keywords"""
    
    logging.info(f"Processing JSONL file: {input_jsonl_file}")
    
    with open(input_jsonl_file, 'r') as f_in, \
         open(relevant_output_file, 'w') as f_relevant, \
         open(irrelevant_output_file, 'w') as f_irrelevant:
        
        relevant_count = 0
        irrelevant_count = 0
        
        for line in f_in:
            data = json.loads(line.strip())
            text = data['input']
            
            # Check if paragraph is relevant using enhanced criteria
            relevance_result = check_paragraph_relevance_with_validation(text)
            
            if relevance_result['relevant']:
                f_relevant.write(json.dumps(data) + '\n')
                relevant_count += 1
            else:
                f_irrelevant.write(json.dumps(data) + '\n')
                irrelevant_count += 1
        
        logging.info(f"Simple processing complete. Relevant: {relevant_count}, Irrelevant: {irrelevant_count}")
        print(f"Simple processing complete. Relevant: {relevant_count:,}, Irrelevant: {irrelevant_count:,}")

# Run the processing
if __name__ == "__main__":
    input_jsonl_file = '/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/deduped_fulltext.jsonl'
    output_dir = '/Users/OTFatokun/AI_LLM_PROJECTS/ChunkingPDF/XML_PIPELINE/retrieved_papers'
    
    # For automatic execution, use enhanced processing
    logging.info("Starting enhanced processing")
    process_jsonl_enhanced(input_jsonl_file, output_dir)
    
    logging.info("Script finished.")

    # Choose processing method:
    print("Choose processing method:")
    print("1. Enhanced processing (detailed categorization)")
    print("2. Simple processing (just relevant/irrelevant like original)")
    
    
    # If you want simple processing instead, uncomment these lines:
    # relevant_output_file = f'{output_dir}/relevant_paragraphs_simple.jsonl'
    # irrelevant_output_file = f'{output_dir}/irrelevant_paragraphs_simple.jsonl'
    # process_jsonl_simple(input_jsonl_file, relevant_output_file, irrelevant_output_file)
    
    logging.info("Script finished.")
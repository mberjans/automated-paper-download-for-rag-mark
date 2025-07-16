# FOODB LLM Pipeline Documentation

## Overview

The FOODB LLM Pipeline is a comprehensive system for automated paper download, processing, and knowledge extraction focused on food compounds and their bioactivities. The pipeline integrates multiple components to search PubMed, retrieve full-text papers, extract structured information, and generate knowledge graphs from scientific literature.

## Repository Structure

```
automated-paper-download-for-rag-and-llm-fine-tuning/
├── FOODB_LLM_pipeline/           # Main pipeline components
│   ├── compound_normalizer.py    # Script 1: Compound normalization
│   ├── pubmed_searcher.py        # Script 2: PubMed search and metadata
│   ├── paper_retriever.py        # Script 3: Full-text retrieval
│   ├── Chunk_XML.py              # Script 4: XML processing and chunking
│   ├── fulltext_deduper.py       # Script 5: Deduplication using embeddings
│   ├── 5_LLM_Simple_Sentence_gen.py  # Script 6: Simple sentence generation
│   ├── simple_sentenceRE3.py     # Script 7: Triple extraction
│   ├── triple_classifier3.py     # Script 8: Triple classification
│   ├── check_for_compds2.py      # Script 9: Compound relevance filtering
│   ├── global_prompt.py          # Shared prompts and configurations
│   └── requirement.txt           # Python dependencies
├── ADHDPIPELINE.py               # Alternative pipeline for ADHD research
├── CHUNKPDF.py                   # PDF processing utility
├── Execute_Search_Query.py       # Query execution utility
├── Download.sh                   # Shell script for paper downloading
├── Pipeline_generate_keywords_by_groq  # Groq-based keyword generation
└── llm_integration               # LLM integration utilities
```

## Core Pipeline Components

### 1. Compound Normalizer (`compound_normalizer.py`)

**Purpose**: Normalizes compound names and fetches synonyms from PubChem to improve search coverage.

**Key Features**:
- Standardizes chemical nomenclature formatting
- Fetches synonyms from PubChem API
- Handles special characters and chemical notation
- Caches results to avoid redundant API calls

**Input**: CSV file with compound data (columns: `public_id`, `name`, `klass`)
**Output**: CSV with normalized names and synonyms (`compounds_with_synonyms.csv`)

**Usage**:
```python
normalizer = CompoundNormalizer("input_compounds.csv")
normalizer.load_data()
normalizer.process_compounds()
normalizer.batch_fetch_synonyms(batch_size=50, max_compounds=1000)
```

### 2. PubMed Searcher (`pubmed_searcher.py`)

**Purpose**: Searches PubMed/PMC for papers related to food compounds and bioactivity.

**Key Features**:
- Constructs complex search queries combining compound names and bioactivity terms
- Filters by publication types (Review, Systematic Review, RCT)
- Checks PMC Open Access availability
- Handles rate limiting and API errors
- Generates comprehensive search reports

**Input**: CSV with compound synonyms
**Output**:
- `search_summary.csv`: Search statistics per compound
- `papers_to_retrieve.csv`: Papers available for download

**Search Terms Include**:
- Disease-specific mechanisms: "apoptosis induction", "COX-2 inhibition", "AMPK activation"
- General mechanisms: "enzyme inhibition", "receptor binding", "pathway modulation"
- Action verbs: "inhibits", "binds", "downregulates", "upregulates"

### 3. Paper Retriever (`paper_retriever.py`)

**Purpose**: Downloads full-text XML from PMC or abstracts from PubMed.

**Key Features**:
- Parallel processing with ThreadPoolExecutor
- Prioritizes full-text over abstracts
- Handles DOI sanitization for file naming
- Comprehensive error handling and logging
- Generates retrieval statistics

**Input**: CSV with papers to retrieve
**Output**:
- `fulltext/`: Directory with full-text XML files
- `abstracts/`: Directory with abstract XML files
- `retrieval_summary.csv`: Download statistics

### 4. XML Chunk Processor (`Chunk_XML.py`)

**Purpose**: Processes XML files to extract and chunk text content with metadata.

**Key Features**:
- Extracts text from PMC XML structure
- Removes references and citations
- Segments into sections (Introduction, Results, Discussion, etc.)
- Creates 5-sentence chunks with metadata
- Fetches additional metadata from Crossref
- Grammar correction using LanguageTool

**Input**: Directory of XML files
**Output**: JSONL file with chunked text and metadata

**Metadata Extracted**:
- DOI, title, journal name, page numbers
- Section classification
- Chunk index and context

### 5. Fulltext Deduplicator (`fulltext_deduper.py`)

**Purpose**: Removes duplicate content using semantic similarity.

**Key Features**:
- Uses sentence transformers for embedding generation
- Cosine similarity-based duplicate detection
- Configurable similarity threshold (default: 0.85)
- Preserves first occurrence of duplicates
- Detailed deduplication statistics

**Input**: JSONL file with text chunks
**Output**: Deduplicated JSONL file

### 6. Simple Sentence Generator (`5_LLM_Simple_Sentence_gen.py`)

**Purpose**: Converts complex scientific text into simple, clear sentences using LLM.

**Key Features**:
- Uses Gemma-3-27B model for text simplification
- Extracts food entities from sentences
- Deduplicates sentences using embeddings
- Comprehensive statistics tracking
- Batch processing with progress monitoring

**Input**: JSONL files with text chunks
**Output**: JSONL files with simple sentences and extracted entities

**Entity Types Extracted**:
- Specific foods (e.g., "green tea", "grape skin extract")
- Food constituents (e.g., "epicatechin", "curcumin", "resveratrol")

### 7. Triple Extractor (`simple_sentenceRE3.py`)

**Purpose**: Extracts subject-predicate-object triples from simple sentences.

**Key Features**:
- Uses Gemma-3-27B for triple extraction
- Validates triples against source text
- Removes duplicates using sentence similarity
- Integrates with triple classifier
- Comprehensive error handling and logging

**Input**: JSONL files with simple sentences and entities
**Output**:
- CSV files with extracted triples
- Classified triples ready for knowledge graph construction

**Triple Format**: `[Subject, Predicate, Object]`
**Example**: `["Resveratrol", "activates", "SIRT1"]`

### 8. Triple Classifier (`triple_classifier3.py`)

**Purpose**: Classifies extracted triples by entity types and relationship categories.

**Key Features**:
- Uses original chunk context for better classification
- Classifies entity types (Food, Chemical constituent, Disease, etc.)
- Assigns relationship labels (Taxonomic, Compositional, Health effect, etc.)
- Separates ontology-labeled from other triples
- Detailed classification statistics

**Entity Types**:
- Food, Chemical constituent, Category, Geographical Location
- Biological location, Disease, Biological process, Pathway
- Enzyme, Protein, Hormone, Biofluid, Biomarker, Microbe

**Relationship Labels**:
- Taxonomic, Compositional, Geographical source, Health effect
- Sensory/organoleptic, Descriptive, Mechanism of action, Use

### 9. Compound Relevance Filter (`check_for_compds2.py`)

**Purpose**: Filters text chunks for relevance to food compounds and bioactivity.

**Key Features**:
- Uses Aho-Corasick algorithm for efficient pattern matching
- Validates compound matches with word boundaries
- Categorizes by relevance type (use, health effect, mechanism)
- Comprehensive keyword dictionaries
- Detailed relevance statistics

**Relevance Categories**:
- **Use**: consumption, dietary, supplement, treatment
- **Health Effect**: disease prevention, therapeutic benefits
- **Mechanism**: pathway modulation, enzyme inhibition, receptor binding

## Dependencies

### Core Requirements
```
sentence-transformers==2.6.1
torch==2.2.0
transformers==4.41.2
unsloth==0.2.3
pandas==2.2.2
scikit-learn==1.4.2
numpy==1.26.4
accelerate==0.30.1
bitsandbytes==0.42.0
```

### Additional Libraries
- `spacy`: Text processing and sentence segmentation
- `language_tool_python`: Grammar correction
- `habanero`: Crossref API integration
- `ahocorasick`: Efficient string matching
- `requests`: HTTP requests for APIs
- `tqdm`: Progress bars
- `xml.etree.ElementTree`: XML parsing

## Configuration

### Environment Variables
- `HF_TOKEN`: Hugging Face token for model access

### Key Parameters
- `GLOBAL_OUTPUT_MAX_TOKENS`: 512 (LLM output limit)
- `DUPLICATE_THRESHOLD`: 0.85 (similarity threshold)
- `MAX_SEQ_LENGTH`: 2048 (model sequence length)

## Usage Examples

### Complete Pipeline Execution
```bash
# 1. Normalize compounds
python compound_normalizer.py

# 2. Search PubMed
python pubmed_searcher.py

# 3. Retrieve papers
python paper_retriever.py

# 4. Process XML and chunk
python Chunk_XML.py

# 5. Deduplicate content
python fulltext_deduper.py

# 6. Generate simple sentences
python 5_LLM_Simple_Sentence_gen.py

# 7. Extract and classify triples
python simple_sentenceRE3.py
```

### Individual Component Usage
```python
# Compound normalization
from compound_normalizer import CompoundNormalizer
normalizer = CompoundNormalizer("compounds.csv")
result = normalizer.batch_fetch_synonyms()

# PubMed search
from pubmed_searcher import PubMedSearcher
searcher = PubMedSearcher("compounds_with_synonyms.csv")
results = searcher.process_compounds()

# Paper retrieval
from paper_retriever import PaperRetriever
retriever = PaperRetriever("papers_to_retrieve.csv")
retriever.process_papers()
```

## Output Files

### Primary Outputs
- `compounds_with_synonyms.csv`: Normalized compounds with synonyms
- `papers_to_retrieve.csv`: Papers available for download
- `fulltext_output.jsonl`: Processed and chunked text
- `deduped_fulltext.jsonl`: Deduplicated content
- `*_simple_sentences.jsonl`: Simple sentences with entities
- `*_kg_ready.csv`: Knowledge graph-ready triples

### Statistics and Logs
- `search_summary.csv`: Search statistics
- `retrieval_summary.csv`: Download statistics
- `simple_sentences_stats.csv`: Processing statistics
- Various log files for debugging and monitoring

## Error Handling

The pipeline includes comprehensive error handling:
- API rate limiting and retry logic
- File validation and existence checks
- Memory management for large datasets
- Graceful degradation for failed components
- Detailed logging for debugging

## Performance Considerations

- **Memory Usage**: Large models require significant GPU memory
- **Processing Time**: Full pipeline can take hours for large datasets
- **API Limits**: PubMed and PubChem have rate limits
- **Storage**: XML files and embeddings require substantial disk space

## Future Enhancements

- Integration with additional databases (ChEBI, FooDB)
- Support for more document formats (PDF, HTML)
- Advanced entity linking and normalization
- Real-time processing capabilities
- Web interface for pipeline management

---

# Detailed Script Documentation

## Script 1: Compound Normalizer (`compound_normalizer.py`)

### Class: CompoundNormalizer

#### Methods

**`__init__(self, input_csv, output_dir="normalized_compounds")`**
- Initializes the normalizer with input CSV and output directory
- Creates output directory if it doesn't exist

**`load_data(self)`**
- Loads compound data from CSV file
- Returns pandas DataFrame with compound information

**`normalize_name(self, name)`**
- Normalizes compound names by:
  - Removing unnecessary spaces
  - Standardizing brackets and quotes
  - Converting various dash types to hyphens
  - Handling unmatched parentheses

**`fetch_synonyms(self, compound_name, compound_id)`**
- Fetches synonyms from PubChem API
- Returns list of synonyms for the compound
- Handles API errors gracefully

**`batch_fetch_synonyms(self, batch_size=50, max_compounds=None)`**
- Processes compounds in batches to fetch synonyms
- Implements caching to avoid redundant API calls
- Respects API rate limits with sleep delays

### Input Format
```csv
public_id,name,klass
FOOD00001,Caffeine,Alkaloid
FOOD00002,Resveratrol,Polyphenol
```

### Output Format
```csv
public_id,original_name,normalized_name,class,synonyms
FOOD00001,Caffeine,caffeine,Alkaloid,caffeine|1,3,7-trimethylxanthine|methyltheobromine
```

## Script 2: PubMed Searcher (`pubmed_searcher.py`)

### Class: PubMedSearcher

#### Key Methods

**`create_search_query(self, compound_names, max_synonyms=5, pub_types=None)`**
- Creates complex PubMed search queries
- Combines compound names with bioactivity terms
- Limits synonyms to avoid overly complex queries

**`search_pubmed(self, query, retmax=100)`**
- Searches PubMed using E-utilities API
- Returns list of PubMed IDs
- Handles API errors and rate limiting

**`fetch_paper_metadata(self, pmid_list)`**
- Fetches metadata for PubMed IDs
- Extracts DOI, PMC ID, title, journal, authors
- Returns structured metadata dictionary

**`check_pmc_availability(self, pmc_id)`**
- Checks if paper is available in PMC Open Access
- Returns boolean indicating availability

### Bioactivity Terms
The searcher includes comprehensive bioactivity terms:
- **Disease-specific**: "apoptosis induction", "cell cycle arrest", "COX-2 inhibition"
- **General mechanisms**: "enzyme inhibition", "molecular target", "gene expression"
- **Action verbs**: "inhibits", "binds", "downregulates", "upregulates"

### Output Structure
```csv
compound_id,compound_name,papers_found,pmc_available
FOOD00001,Caffeine,45,12
```

## Script 3: Paper Retriever (`paper_retriever.py`)

### Class: PaperRetriever

#### Key Methods

**`fetch_pmc_fulltext(self, pmc_id)`**
- Downloads full-text XML from PMC
- Handles PMC ID formatting
- Returns XML content as string

**`fetch_pubmed_abstract(self, pmid)`**
- Downloads abstract XML from PubMed
- Fallback when full-text unavailable
- Returns XML content as string

**`process_paper(self, paper_row)`**
- Processes individual paper for download
- Prioritizes full-text over abstracts
- Handles file naming and storage

**`process_papers(self, max_workers=5)`**
- Parallel processing using ThreadPoolExecutor
- Configurable number of worker threads
- Comprehensive error handling and logging

### File Naming Convention
- Full-text files: `{sanitized_doi}.xml` in `fulltext/` directory
- Abstract files: `{sanitized_doi}.xml` in `abstracts/` directory
- DOI sanitization replaces invalid filename characters with underscores

## Script 4: XML Chunk Processor (`Chunk_XML.py`)

### Key Functions

**`extract_xml_text(xml_path)`**
- Extracts text content from PMC XML structure
- Handles various paragraph tags (`p`, `par`, `paragraph`)
- Processes nested elements and text content

**`extract_sections_from_xml_tags(xml_path)`**
- Extracts sections directly from XML structure
- Uses `sec` tags with `sec-type` attributes
- Maps section titles to content

**`clean_text(text)`**
- Removes LaTeX equations and document structure
- Normalizes whitespace and removes citations
- Handles both numeric and author-year citations

**`chunk_into_sentences(text)`**
- Uses spaCy for sentence segmentation
- Handles large texts by chunking
- Fallback to simple splitting if spaCy fails

**`stack_sentences(sentences, window_size=5)`**
- Groups sentences into chunks of specified size
- Preserves sentence boundaries
- Returns list of concatenated sentence chunks

### Section Mapping
The processor maps XML sections to standard categories:
- Introduction, Background, Results, Discussion, Conclusion
- Handles various section naming conventions
- Preserves original section structure when possible

### Metadata Extraction
- **From XML**: Title, journal name, page numbers
- **From Crossref**: Additional metadata validation
- **Enhanced page extraction**: Multiple fallback patterns for page numbers

## Script 5: Fulltext Deduplicator (`fulltext_deduper.py`)

### Class: ParagraphChecker

#### Key Methods

**`load_jsonl_file(self, file_path, text_field='input')`**
- Loads data from JSONL file
- Validates JSON format for each line
- Returns list of dictionaries

**`vectorize_texts(self, data, text_field='input')`**
- Generates embeddings using sentence transformers
- Uses 'all-MiniLM-L6-v2' model by default
- Configurable batch size for memory management

**`find_duplicates(self, similarity_threshold=0.85)`**
- Calculates cosine similarity matrix
- Identifies pairs above threshold
- Returns sorted list of duplicate pairs

**`remove_duplicates(self, similarity_threshold=0.85)`**
- Removes duplicates while preserving first occurrence
- Tracks removed items for analysis
- Returns unique items and duplicate information

### Deduplication Process
1. Load JSONL data and extract text fields
2. Generate embeddings for all text chunks
3. Calculate pairwise cosine similarities
4. Identify duplicates above threshold
5. Remove later occurrences, keep first
6. Save deduplicated results

## Script 6: Simple Sentence Generator (`5_LLM_Simple_Sentence_gen.py`)

### Key Functions

**`generate_simple_sentences(text_chunk, prompt)`**
- Uses Gemma-3-27B model for text simplification
- Applies Alpaca prompt format
- Generates deterministic output (do_sample=False)

**`check_entities_present(sentence, entity_prompt)`**
- Identifies food entities in sentences
- Returns boolean and list of entities
- Handles malformed JSON responses

**`is_meaningful_sentence(sentence, min_words=4)`**
- Filters out fragments and references
- Checks minimum word count
- Removes standalone citations

**`deduplicate_with_entities(filtered_sentences)`**
- Enhanced deduplication considering entity overlap
- Uses both semantic similarity and entity matching
- Preserves sentences with unique entity combinations

### Entity Extraction
The system extracts specific entity types:
- **Foods**: "green tea", "grape skin extract", "melon soup"
- **Compounds**: "epicatechin", "curcumin", "resveratrol", "EGCG"
- **Exclusions**: Generic terms like "polyphenols", "flavonoids"

### Processing Statistics
Tracks comprehensive statistics:
- Raw sentences generated
- Cleaned sentences after filtering
- Sentences with entities
- Unique sentences after deduplication
- Entity extraction counts

## Script 7: Triple Extractor (`simple_sentenceRE3.py`)

### Key Functions

**`create_prompt(sample_input, entities)`**
- Creates detailed prompt for triple extraction
- Includes specific instructions for biomedical relationships
- Provides examples for different triple types

**`extract_triples(file_text, model, tokenizer, device, sample_input, entities)`**
- Extracts subject-predicate-object triples
- Uses delimiter-based parsing (TRIPLE_START|...|TRIPLE_END)
- Validates triples against source text

**`check_entities_in_article(triple, article)`**
- Validates that triple entities exist in source text
- Handles acronyms and numerical values
- Prevents hallucinated relationships

**`remove_duplicate_triples(triples_with_context, sentence_model)`**
- Uses sentence similarity for duplicate detection
- Considers both semantic similarity and entity overlap
- Preserves contextual information

### Triple Validation
- Entities must exist in source article
- Handles acronym expansion and numerical extraction
- Filters out non-biomedical relationships
- Validates subject-verb-object structure

### Output Format
```csv
simple_sentence,subject,predicate,object,pages,journal,doi,chunk_index
"Resveratrol activates SIRT1",Resveratrol,activates,SIRT1,123-130,Nature,10.1038/...,5
```

## Script 8: Triple Classifier (`triple_classifier3.py`)

### Key Functions

**`prepare_classifier_prompt(chunk_text, simple_sentence, triple)`**
- Uses original chunk for better context
- Provides comprehensive entity type categories
- Includes relationship label definitions

**`classify_triple(model, tokenizer, device, chunk_text, simple_sentence, triple)`**
- Classifies entity types and relationship labels
- Uses lower temperature for consistent classification
- Handles parsing errors gracefully

**`write_classified_triples_to_csv(classified_triples, base_output_file)`**
- Separates ontology-labeled from other triples
- Creates two output files for different use cases
- Maintains metadata and provenance information

### Entity Type Categories
- **Food**: Plant species, processed foods
- **Chemical constituent**: Bioactive compounds
- **Biological**: Organs, tissues, cells, processes
- **Disease**: Medical conditions
- **Sensory**: Taste, aroma, texture, color

### Relationship Labels
- **Taxonomic**: Botanical classifications
- **Compositional**: Chemical composition
- **Health effect**: Therapeutic benefits
- **Mechanism of action**: Molecular mechanisms
- **Sensory/organoleptic**: Sensory properties

## Script 9: Compound Relevance Filter (`check_for_compds2.py`)

### Key Functions

**`validate_compound_match(text, compound_variant, start_idx, end_idx)`**
- Ensures whole-word matches only
- Checks word boundaries to prevent false positives
- Handles various punctuation and spacing

**`check_paragraph_relevance_with_validation(text)`**
- Uses Aho-Corasick for efficient pattern matching
- Validates compound matches with context
- Categorizes relevance by type (use, health effect, mechanism)

**`process_jsonl_enhanced(input_jsonl_file, output_dir)`**
- Processes JSONL files with detailed categorization
- Creates separate outputs for different relevance types
- Generates comprehensive statistics

### Keyword Categories
- **Use keywords**: consumption, dietary, supplement, treatment
- **Mechanism keywords**: pathway, enzyme, inhibition, activation
- **Health effect terms**: Loaded from external CSV file

### Output Categories
- `relevant_use_only.jsonl`: Use-related content only
- `relevant_health_effect_only.jsonl`: Health effect content
- `relevant_mechanism_only.jsonl`: Mechanism content
- `relevant_multiple_types.jsonl`: Multiple relevance types
- `irrelevant_paragraphs_enhanced.jsonl`: Non-relevant content

---

# Supporting Scripts and Utilities

## Global Prompt Configuration (`global_prompt.py`)

### Constants
- `global_output_max_new_tokens = 2048`: Maximum tokens for LLM generation
- `simple_sentence_prompt`: Detailed prompt for text simplification
- `food_entity_prompt`: Prompt for food entity extraction

### Simple Sentence Prompt Features
- Converts complex scientific text to simple sentences
- Preserves scientific measurements and terminology
- Maintains subject-verb-object structure
- Includes specific examples for different text types

### Entity Extraction Prompt Features
- Focuses on edible foods and specific compounds
- Excludes generic categories and plant parts
- Includes IUPAC names linkable to ChEBI/PubChem
- Returns structured JSON output

## Alternative Pipelines

### ADHD Pipeline (`ADHDPIPELINE.py`)

**Purpose**: Specialized pipeline for ADHD and neurological disorder research.

**Key Features**:
- Disease-specific keyword sets
- Automated CSV retrieval and processing
- Integration with Download.sh for paper acquisition
- Biomarker-focused search strategies

**Disease Keywords**:
```python
disease_keywords = [
    {"name": "ADHD", "keywords": ["ADHD", "biomarkers", "compounds"]},
    {"name": "Alzheimer Disease", "keywords": ["Alzheimer Disease", "biomarkers", "urine"]},
    {"name": "Dementia", "keywords": ["Dementia", "biomarkers", "plasma"]}
]
```

### PDF Chunk Processor (`CHUNKPDF.py`)

**Purpose**: Processes PDF files for text extraction and chunking.

**Key Features**:
- PDF text extraction using pdfminer
- Reference and citation removal
- Grammar correction with LanguageTool
- Sentence segmentation with spaCy
- 5-sentence chunk creation
- JSONL output format

**Processing Pipeline**:
1. Extract text from PDF using pdfminer
2. Remove references and numbered citations
3. Clean hyphenated words and formatting
4. Segment into sentences using spaCy
5. Correct grammar with LanguageTool
6. Group into 5-sentence chunks
7. Convert to JSONL format

### Query Execution Utility (`Execute_Search_Query.py`)

**Purpose**: Executes search queries with pattern matching for food categories.

**Key Features**:
- Pattern-based query parsing
- Food category recognition
- Integration with Download.sh
- Formatted query generation

**Supported Categories**:
- Food Categories: Fruits, Vegetables, Grains, Dairy, Meats
- Organoleptic Properties: Taste, Texture, Aroma, Color
- Food Sources: Plant-based, Animal-based, Synthetic
- Functional Foods: Probiotics, Prebiotics, Nutraceuticals

## LLM Integration Utilities

### Groq Integration (`Pipeline_generate_keywords_by_groq`)

**Purpose**: Uses Groq AI API to generate dynamic keywords for search queries.

**Key Features**:
- CSV input processing
- Dynamic keyword generation
- Groq API integration
- Enhanced search stringency

**Workflow**:
1. Read food-chemical pairs from CSV
2. Generate alternative terms using Groq API
3. Create multiple search strategies
4. Execute PubMed searches with generated keywords

### Local LLM Integration (`llm_integration`)

**Purpose**: Integrates with local Gemma:7b model for keyword expansion.

**Key Features**:
- Local model deployment
- Keyword synonym generation
- API-based interaction
- Reduced dependency on external services

## Shell Scripts

### Download Script (`Download.sh`)

**Purpose**: Automates paper downloading using PyPaperBot.

**Key Features**:
- DOI file processing
- Directory management
- Background process handling
- Progress monitoring

**Usage**:
```bash
./Download.sh input_file.txt download_directory "search_query"
```

**Process Flow**:
1. Validate input file and create download directory
2. Launch PyPaperBot with nohup for background execution
3. Monitor process completion
4. Handle multiple file processing

## Installation and Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv foodb_pipeline
source foodb_pipeline/bin/activate  # Linux/Mac
# foodb_pipeline\Scripts\activate  # Windows

# Install dependencies
pip install -r FOODB_LLM_pipeline/requirement.txt

# Install additional dependencies
pip install spacy language_tool_python habanero ahocorasick
python -m spacy download en_core_web_md
```

### Configuration Requirements
1. **Hugging Face Token**: Set `HF_TOKEN` environment variable
2. **Email for NCBI**: Configure email in searcher scripts
3. **API Keys**: Optional NCBI API key for higher rate limits
4. **Model Access**: Ensure access to Gemma-3-27B model

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM for Gemma-3-27B
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 100GB+ for papers, models, and intermediate files
- **Network**: Stable internet for API calls and model downloads

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch sizes or use smaller models
2. **API Rate Limits**: Implement longer delays between requests
3. **Model Loading Errors**: Verify HF_TOKEN and model access
4. **File Permission Errors**: Check directory write permissions
5. **JSON Parsing Errors**: Validate input file formats

### Performance Optimization
- Use GPU acceleration for model inference
- Implement parallel processing where possible
- Cache API results to avoid redundant calls
- Monitor memory usage during processing
- Use appropriate batch sizes for available hardware

### Logging and Debugging
- All scripts include comprehensive logging
- Log files created in output directories
- Debug modes available for development
- Progress tracking for long-running processes
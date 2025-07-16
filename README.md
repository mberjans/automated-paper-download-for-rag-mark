# FOODB LLM Pipeline for Automated Paper Download and Knowledge Extraction

A comprehensive pipeline for automated scientific literature retrieval, processing, and knowledge graph construction focused on food compounds and their bioactivities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with 16GB+ VRAM (for LLM processing)
- 32GB+ RAM recommended
- Hugging Face account and token

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd automated-paper-download-for-rag-and-llm-fine-tuning
```

2. **Create virtual environment**
```bash
python -m venv foodb_pipeline
source foodb_pipeline/bin/activate  # Linux/Mac
# foodb_pipeline\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r FOODB_LLM_pipeline/requirement.txt
pip install spacy language_tool_python habanero ahocorasick
python -m spacy download en_core_web_md
```

4. **Set environment variables**
```bash
export HF_TOKEN="your_huggingface_token"
```

### Basic Usage

**Run the complete pipeline:**
```bash
cd FOODB_LLM_pipeline

# 1. Normalize compounds and fetch synonyms
python compound_normalizer.py

# 2. Search PubMed for relevant papers
python pubmed_searcher.py

# 3. Download full-text papers
python paper_retriever.py

# 4. Process XML and create chunks
python Chunk_XML.py

# 5. Remove duplicate content
python fulltext_deduper.py

# 6. Generate simple sentences
python 5_LLM_Simple_Sentence_gen.py

# 7. Extract knowledge triples
python simple_sentenceRE3.py
```

## 📋 Pipeline Overview

The FOODB LLM Pipeline consists of 9 main components:

1. **Compound Normalizer** - Standardizes compound names and fetches synonyms
2. **PubMed Searcher** - Searches for relevant scientific papers
3. **Paper Retriever** - Downloads full-text XML or abstracts
4. **XML Chunk Processor** - Extracts and chunks text with metadata
5. **Fulltext Deduplicator** - Removes duplicate content using embeddings
6. **Simple Sentence Generator** - Converts complex text to simple sentences
7. **Triple Extractor** - Extracts subject-predicate-object relationships
8. **Triple Classifier** - Classifies entities and relationship types
9. **Compound Relevance Filter** - Filters content for bioactivity relevance

## 📁 Key Input/Output Files

### Inputs
- `compounds.csv` - List of food compounds to search
- `HealthEffects_table.csv` - Health effect terms for filtering

### Outputs
- `compounds_with_synonyms.csv` - Normalized compounds with synonyms
- `papers_to_retrieve.csv` - Papers available for download
- `fulltext_output.jsonl` - Processed and chunked text
- `*_kg_ready.csv` - Knowledge graph-ready triples

## 🔧 Configuration

### Key Parameters (in scripts)
- `GLOBAL_OUTPUT_MAX_TOKENS`: 512 (LLM output limit)
- `DUPLICATE_THRESHOLD`: 0.85 (similarity threshold)
- `MAX_SEQ_LENGTH`: 2048 (model sequence length)

### Required API Keys
- Hugging Face token for model access
- Optional: NCBI API key for higher rate limits

## 📊 Features

- **Automated Literature Search**: Intelligent PubMed queries with compound synonyms
- **Full-text Processing**: XML parsing with section extraction and metadata
- **LLM-powered Extraction**: Uses Gemma-3-27B for text simplification and triple extraction
- **Semantic Deduplication**: Embedding-based duplicate removal
- **Knowledge Graph Ready**: Structured triples with entity classification
- **Comprehensive Logging**: Detailed logs and statistics for monitoring

## 🛠️ Alternative Pipelines

- **ADHD Pipeline** (`ADHDPIPELINE.py`) - Specialized for neurological research
- **PDF Processor** (`CHUNKPDF.py`) - Direct PDF processing utility
- **Groq Integration** - Dynamic keyword generation using Groq AI

## 📖 Documentation

For detailed documentation, see `FOODB_LLM_Pipeline_Documentation.md` which includes:
- Complete API reference for all classes and methods
- Input/output format specifications
- Configuration options and parameters
- Troubleshooting guide
- Performance optimization tips

## 🔍 Example Output

**Simple Sentence:**
```
"Resveratrol activates SIRT1 protein in human cells."
```

**Extracted Triple:**
```csv
simple_sentence,subject,predicate,object,subject_type,object_type,label
"Resveratrol activates SIRT1 protein",Resveratrol,activates,SIRT1,Chemical constituent,Protein,Mechanism of action
```

## ⚠️ System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ for papers and models
- **Network**: Stable internet for API calls

## 🐛 Troubleshooting

**Common Issues:**
- CUDA out of memory → Reduce batch sizes
- API rate limits → Increase delays between requests
- Model loading errors → Verify HF_TOKEN
- JSON parsing errors → Check input file formats

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

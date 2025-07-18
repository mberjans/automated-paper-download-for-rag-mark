# FOODB Pipeline - Technical Documentation

## 🏗️ Architecture Overview

The FOODB Pipeline is a modern, CLI-based system for extracting metabolites and biomarkers from scientific PDF documents using multiple LLM providers with **enhanced intelligent fallback mechanisms**.

## ✨ **NEW: Enhanced Fallback System (V4.0)**

The pipeline now features a **revolutionary fallback system** with 30x faster recovery and intelligent provider switching:

### **🚀 Technical Improvements**
- **Intelligent Rate Limiting**: Switch providers after 2 consecutive failures
- **V4 Priority-Based Selection**: 25 models ranked by F1 scores and performance
- **Real-time Health Monitoring**: Provider status tracking with automatic recovery
- **Optimized Model Selection**: Best models automatically chosen for each provider
- **Comprehensive Statistics**: Detailed performance metrics and monitoring

### Core Components

```
foodb_pipeline_cli.py          # Main CLI interface with argument parsing
foodb_pipeline_runner.py       # Pipeline execution engine
llm_wrapper_enhanced.py        # Enhanced LLM provider management with V4 fallback
llm_usage_priority_list.json   # V4 priority list (25 models ranked by performance)
free_models_reasoning_ranked_v4.json  # Complete V4 ranking data
FOODB_LLM_pipeline/           # Legacy pipeline components
```

### Processing Flow

```mermaid
graph TD
    A[PDF Input] --> B[Text Extraction]
    B --> C[Text Chunking]
    C --> D[LLM Processing]
    D --> E[Metabolite Extraction]
    E --> F[Database Matching]
    F --> G[Results Analysis]
    G --> H[Output Generation]
    
    D --> I[Provider Fallback]
    I --> J[Exponential Backoff]
    J --> D
```

## 🔧 Technical Specifications

### 🚀 Enhanced LLM Provider Integration (V4.0)

#### V4 Priority-Based Provider Hierarchy
1. **Cerebras** (Speed Priority): Ultra-fast inference (0.56-0.62s)
   - **Best Model**: `llama-4-scout-17b-16e-instruct` (Score: 9.8, Speed: 0.59s)
   - **Use Case**: Real-time applications, batch processing

2. **Groq** (Accuracy Priority): Best F1 scores (0.40-0.51)
   - **Best Model**: `meta-llama/llama-4-maverick-17b-128e-instruct` (F1: 0.5104, Recall: 83%)
   - **Use Case**: High-accuracy metabolite extraction

3. **OpenRouter** (Diversity Priority): Most model options (15 models)
   - **Best Model**: `mistralai/mistral-nemo:free` (F1: 0.5772, Recall: 73%)
   - **Use Case**: Specialized capabilities, final fallback

#### Enhanced Fallback Mechanism
```python
# V4.0 Enhanced Configuration
RetryConfig(
    max_attempts=2,        # Reduced for faster switching
    base_delay=1.0,        # Shorter initial delay
    max_delay=10.0,        # Capped maximum delay
    exponential_base=2.0,
    jitter=True
)

# V4 Enhanced Multi-Tier Rate Limiting Logic
def enhanced_rate_limiting_flow():
    """
    Comprehensive multi-tier fallback with intelligent model rotation
    """
    # Tier 1: Cerebras Models (Speed Priority)
    cerebras_models = [
        "llama-4-scout-17b-16e-instruct",  # 0.59s, Score: 9.8
        "llama-3.3-70b",                   # 0.62s, Score: 9.5
        "llama3.1-8b",                     # 0.56s, Score: 8.5
        "qwen-3-32b"                       # 0.57s, Score: 8.2
    ]

    # Tier 2: Groq Models (Accuracy Priority)
    groq_models = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",  # F1: 0.5104
        "meta-llama/llama-4-scout-17b-16e-instruct",      # F1: 0.5081
        "qwen/qwen3-32b",                                 # F1: 0.5056
        "llama-3.1-8b-instant",                          # F1: 0.5000
        "llama-3.3-70b-versatile",                       # F1: 0.4706
        "moonshotai/kimi-k2-instruct"                     # F1: 0.4053
    ]

    # Tier 3: OpenRouter Models (Diversity Priority)
    openrouter_models = [
        "mistralai/mistral-nemo:free",                    # F1: 0.5772
        "tngtech/deepseek-r1t-chimera:free",             # F1: 0.4372
        # ... all 15 OpenRouter models in priority order
    ]

    # Execute multi-tier fallback
    for provider_models in [cerebras_models, groq_models, openrouter_models]:
        for model in provider_models:
            if try_model_with_retries(model, max_attempts=5):
                return success

    return failure  # Only after all 25 models exhausted
```

#### Enhanced Rate Limiting Handling
- **Multi-tier model rotation** within each provider before escalation
- **Intelligent model selection** based on V4 priority rankings
- **Comprehensive fallback coverage** across 25 models and 3 providers
- **Exponential backoff** with user-defined retry limits per model
- **Provider escalation** only after exhausting all models in current provider
- **Real-time statistics tracking** for performance monitoring and optimization

### Text Processing Pipeline

#### PDF Text Extraction
```python
def extract_pdf_text(input_file: str) -> Dict[str, Any]:
    """
    Extracts text from PDF using PyPDF2
    Returns: {
        'text': str,
        'pages': int,
        'characters': int,
        'extraction_time': float
    }
    """
```

#### Text Chunking
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into overlapping chunks
    - chunk_size: Target size in characters (default: 1500)
    - overlap: Overlap between chunks (default: 0)
    - min_chunk_size: Minimum chunk size to process (default: 100)
    """
```

#### Metabolite Extraction
```python
def extract_metabolites_document_only(chunk: str, max_tokens: int) -> str:
    """
    Document-only extraction to prevent training data contamination
    Uses specific prompts that focus only on document content
    """
```

### Database Matching System

#### Matching Algorithm
```python
def match_database(metabolites: List[str], csv_biomarkers: Set[str]) -> Dict:
    """
    Multi-level matching strategy:
    1. Exact string match (case-insensitive)
    2. Meaningful partial match (length > 5 characters)
    3. Bidirectional substring matching
    """
```

#### Performance Metrics
```python
def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int):
    """
    Calculates:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
```

## 📁 Output Structure Design

### Standard Mode
```
output_directory/
├── {filename}_{timestamp}_results.json      # Complete results
├── {filename}_{timestamp}_results.csv       # Metabolites table
├── {filename}_{timestamp}_results.xlsx      # Multi-sheet workbook
└── {filename}_{timestamp}_timing.json       # Performance analysis
```

### Directory Mode
```
output_directory/
├── individual_papers/                       # Timestamped individual results
│   ├── paper1_{timestamp}.json
│   ├── paper1_{timestamp}.csv
│   └── paper1_{timestamp}_timing.json
├── consolidated/                            # Append-mode consolidated results
│   ├── consolidated_results.json           # Historical runs
│   ├── consolidated_metabolites.csv        # All metabolites
│   ├── consolidated_results.xlsx           # Multi-sheet analysis
│   └── processing_summary.json            # Run summaries
└── batch_summary_{timestamp}.json          # Current batch summary
```

### Consolidated Data Structure
```json
{
  "processing_runs": [
    {
      "timestamp": "2025-07-16 23:19:47",
      "summary_stats": {
        "total_papers": 2,
        "total_unique_metabolites": 190,
        "average_detection_rate": 0.728
      },
      "paper_results": [...],
      "all_unique_metabolites": [...],
      "all_unique_matches": [...]
    }
  ]
}
```

## 🔄 Error Handling & Recovery

### Retry Mechanisms
```python
class RetryConfig:
    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
```

### V4 Enhanced Provider Switching Logic
```python
def enhanced_provider_switching():
    """
    V4 Multi-Tier Fallback with Intelligent Model Rotation

    Flow:
    1. Start with best model from current provider (V4 priority list)
    2. Apply exponential backoff for rate limits (up to max_attempts)
    3. If model exhausted, switch to next model in same provider
    4. If all models in provider exhausted, escalate to next provider
    5. Continue until all 25 models across 3 providers are exhausted
    6. Only then return failure
    """

    providers = {
        'cerebras': {
            'models': ['llama-4-scout-17b-16e-instruct', 'llama-3.3-70b',
                      'llama3.1-8b', 'qwen-3-32b'],
            'priority': 'speed'
        },
        'groq': {
            'models': ['meta-llama/llama-4-maverick-17b-128e-instruct',
                      'meta-llama/llama-4-scout-17b-16e-instruct',
                      'qwen/qwen3-32b', 'llama-3.1-8b-instant',
                      'llama-3.3-70b-versatile', 'moonshotai/kimi-k2-instruct'],
            'priority': 'accuracy'
        },
        'openrouter': {
            'models': ['mistralai/mistral-nemo:free',
                      'tngtech/deepseek-r1t-chimera:free',
                      # ... all 15 models in V4 priority order],
            'priority': 'diversity'
        }
    }

    for provider_name, provider_config in providers.items():
        for model in provider_config['models']:
            result = try_model_with_exponential_backoff(
                provider=provider_name,
                model=model,
                max_attempts=5,
                base_delay=1.0,
                max_delay=60.0
            )
            if result.success:
                return result

    # All 25 models exhausted
    return failure_response()
```

### Recovery Strategies
- **Chunk-level recovery**: Resume from specific chunk
- **Provider exhaustion**: Switch to next provider
- **Complete failure**: Detailed error logging and graceful degradation

## 📊 Performance Optimization

### Chunking Strategy
```python
# Optimal chunk sizes for different scenarios
CHUNK_SIZES = {
    'speed': 2000,      # Faster processing, fewer API calls
    'accuracy': 1000,   # Better extraction, more context
    'balanced': 1500    # Default balanced approach
}
```

### API Call Optimization
```python
# Rate limiting prevention
DELAY_STRATEGIES = {
    'conservative': {'base_delay': 3.0, 'max_delay': 120.0},
    'balanced': {'base_delay': 2.0, 'max_delay': 60.0},
    'aggressive': {'base_delay': 1.0, 'max_delay': 30.0}
}
```

### Memory Management
```python
# Memory-efficient processing
def process_large_documents():
    """
    - Stream processing for large files
    - Chunk-by-chunk processing
    - Garbage collection between chunks
    - Optional chunk saving to disk
    """
```

## 🧪 Testing Framework

### Unit Tests
```python
# Test individual components
test_pdf_extraction()
test_text_chunking()
test_metabolite_parsing()
test_database_matching()
test_metrics_calculation()
```

### Integration Tests
```python
# Test complete workflows
test_single_file_processing()
test_directory_processing()
test_provider_fallback()
test_append_mode()
test_timestamp_functionality()
```

### Performance Tests
```python
# Benchmark performance
test_processing_speed()
test_memory_usage()
test_api_rate_limits()
test_large_document_handling()
```

## 🔐 Security Considerations

### API Key Management
```python
# Secure API key handling
load_dotenv()  # Load from .env file
api_keys = {
    'cerebras': os.getenv('CEREBRAS_API_KEY'),
    'groq': os.getenv('GROQ_API_KEY'),
    'openrouter': os.getenv('OPENROUTER_API_KEY')
}
```

### Data Privacy
- **Document-only mode**: Prevents training data contamination
- **Local processing**: Text extraction and chunking done locally
- **API data**: Only text chunks sent to LLM providers
- **No data retention**: Providers don't store processed content

### Input Validation
```python
def validate_inputs():
    """
    - PDF file validation
    - Path sanitization
    - Parameter bounds checking
    - Configuration validation
    """
```

## 📈 Monitoring & Analytics

### Statistics Tracking
```python
class ProcessingStatistics:
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    provider_usage: Dict[str, int]
    retry_events: List[Dict]
    processing_times: List[float]
```

### Performance Metrics
```python
def generate_performance_report():
    """
    - Processing speed analysis
    - Provider performance comparison
    - Error rate tracking
    - Resource utilization metrics
    """
```

### Logging Configuration
```python
# Comprehensive logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
```

## 🔧 Configuration Management

### CLI Arguments
```python
# 40+ configurable parameters organized in groups:
- Output Configuration (8 parameters)
- Database Configuration (2 parameters)
- Text Processing (3 parameters)
- LLM Configuration (4 parameters)
- Retry/Fallback (5 parameters)
- Provider Configuration (2 parameters)
- Processing Configuration (4 parameters)
- Analysis Configuration (3 parameters)
- Logging/Debugging (5 parameters)
- Configuration File (2 parameters)
```

### Configuration File Format
```json
{
  "output_dir": "./results",
  "directory_mode": true,
  "document_only": true,
  "chunk_size": 2000,
  "max_tokens": 300,
  "providers": ["cerebras", "groq", "openrouter"],
  "max_attempts": 5,
  "base_delay": 2.0,
  "export_format": "all",
  "timestamp_files": true
}
```

## 🚀 Deployment Considerations

### Production Setup
```bash
# Environment setup
python -m venv foodb_production
source foodb_production/bin/activate
pip install -r requirements.txt

# Configuration
cp example_config.json production_config.json
# Edit production_config.json with production settings

# API keys
echo "CEREBRAS_API_KEY=your_key" > .env
echo "GROQ_API_KEY=your_key" >> .env
echo "OPENROUTER_API_KEY=your_key" >> .env
```

### Scaling Strategies
```python
# Horizontal scaling options
- Multiple worker processes
- Load balancing across API providers
- Distributed processing for large document sets
- Caching for repeated document processing
```

### Monitoring Setup
```python
# Production monitoring
- API usage tracking
- Error rate monitoring
- Performance metrics collection
- Resource utilization alerts
```

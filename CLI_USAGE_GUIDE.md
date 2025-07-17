# FOODB Pipeline CLI Usage Guide

## Overview
The FOODB Pipeline CLI provides comprehensive command-line access to the metabolite extraction pipeline with full configurability of all parameters.

## Quick Start

### Basic Usage
```bash
# Process a single PDF with default settings
python foodb_pipeline_cli.py input.pdf

# Process with document-only extraction (recommended)
python foodb_pipeline_cli.py input.pdf --document-only
```

### Recommended Production Settings
```bash
python foodb_pipeline_cli.py input.pdf \
  --document-only \
  --verify-compounds \
  --save-timing \
  --export-format all \
  --output-dir ./results \
  --verbose
```

## Command Line Arguments

### Required Arguments
- `input_files`: One or more PDF files to process

### Output Configuration
- `--output-dir, -o`: Output directory (default: ./foodb_results)
- `--output-prefix`: Prefix for output files
- `--timestamp-files`: Add timestamp to filenames to preserve old files (default: True)
- `--no-timestamp`: Disable timestamp in filenames (will overwrite existing files)
- `--timestamp-format`: Timestamp format for filenames (default: %Y%m%d_%H%M%S)
- `--custom-timestamp`: Custom timestamp string instead of current time
- `--directory-mode`: Process directory of PDFs with organized output structure
- `--consolidated-output`: Create consolidated output files (append mode, no timestamps) (default: True)
- `--individual-output`: Create individual paper-specific output files (timestamped) (default: True)
- `--individual-subdir`: Subdirectory for individual paper results (default: individual_papers)
- `--consolidated-subdir`: Subdirectory for consolidated results (default: consolidated)
- `--save-chunks`: Save text chunks to separate files
- `--save-timing`: Save detailed timing analysis
- `--save-raw-responses`: Save raw LLM responses for debugging

### Database Configuration
- `--csv-database`: CSV database file (default: urinary_wine_biomarkers.csv)
- `--csv-column`: Column name for compound names (default: Compound Name)

### Text Processing
- `--chunk-size`: Chunk size in characters (default: 1500)
- `--chunk-overlap`: Overlap between chunks (default: 0)
- `--min-chunk-size`: Minimum chunk size to process (default: 100)

### LLM Configuration
- `--max-tokens`: Maximum tokens for responses (default: 200)
- `--document-only`: Use document-only extraction (prevents training data contamination)
- `--verify-compounds`: Verify extracted compounds against original text
- `--custom-prompt`: Custom extraction prompt

### Retry and Fallback
- `--max-attempts`: Maximum retry attempts (default: 5)
- `--base-delay`: Base delay for exponential backoff (default: 2.0s)
- `--max-delay`: Maximum delay (default: 60.0s)
- `--exponential-base`: Exponential base (default: 2.0)
- `--disable-jitter`: Disable jitter in backoff

### Provider Configuration
- `--providers`: LLM providers in fallback order (default: cerebras groq openrouter)
- `--primary-provider`: Primary provider to use
- `--groq-model`: Specific Groq model to use (see Available Models below)

### Processing Options
- `--batch-mode`: Process multiple files in batch mode
- `--parallel-chunks`: Number of chunks to process in parallel (default: 1)
- `--skip-existing`: Skip files that already have results
- `--resume-from-chunk`: Resume from specific chunk number

### Analysis Configuration
- `--calculate-metrics`: Calculate precision, recall, F1 scores (default: True)
- `--generate-report`: Generate comprehensive report (default: True)
- `--export-format`: Export format: json, csv, xlsx, all (default: json)

### Logging and Debugging
- `--debug`: Enable debug logging
- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Suppress non-essential output
- `--log-file`: Log file path
- `--progress-bar`: Show progress bar (default: True)

### Configuration File
- `--config`: Load configuration from JSON file
- `--save-config`: Save current configuration to JSON file

## Usage Examples

### 1. Basic Processing
```bash
python foodb_pipeline_cli.py paper.pdf
```

### 2. Document-Only Extraction (Recommended)
```bash
python foodb_pipeline_cli.py paper.pdf --document-only --verify-compounds
```

### 3. Full Configuration
```bash
python foodb_pipeline_cli.py paper.pdf \
  --output-dir ./results \
  --csv-database biomarkers.csv \
  --chunk-size 2000 \
  --max-tokens 300 \
  --document-only \
  --verify-compounds \
  --max-attempts 5 \
  --base-delay 2.0 \
  --export-format all \
  --save-timing \
  --verbose
```

### 4. Batch Processing
```bash
python foodb_pipeline_cli.py *.pdf \
  --batch-mode \
  --output-dir ./batch_results \
  --document-only \
  --skip-existing
```

### 5. Debug Mode
```bash
python foodb_pipeline_cli.py paper.pdf \
  --debug \
  --save-chunks \
  --save-timing \
  --save-raw-responses \
  --log-file debug.log
```

### 6. Using Configuration File
```bash
# Save current configuration
python foodb_pipeline_cli.py paper.pdf --save-config production_config.json

# Load configuration
python foodb_pipeline_cli.py paper.pdf --config production_config.json
```

### 7. Resume Processing
```bash
# Resume from chunk 25 if processing was interrupted
python foodb_pipeline_cli.py paper.pdf --resume-from-chunk 25
```

### 8. Custom Provider Configuration
```bash
python foodb_pipeline_cli.py paper.pdf \
  --primary-provider groq \
  --providers groq cerebras \
  --max-attempts 3
```

### 9. Groq Model Selection
```bash
# Use the fastest Groq model (recommended)
python foodb_pipeline_cli.py paper.pdf --groq-model "moonshotai/kimi-k2-instruct"

# Use Llama 4 Scout model
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-scout-17b-16e-instruct"

# Use Llama 4 Maverick model
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-maverick-17b-128e-instruct"

# Use Qwen model
python foodb_pipeline_cli.py paper.pdf --groq-model "qwen/qwen3-32b"
```

### 9. Timestamp Configuration
```bash
# Default timestamp (preserves old files)
python foodb_pipeline_cli.py paper.pdf --timestamp-files

# Custom timestamp format
python foodb_pipeline_cli.py paper.pdf --timestamp-format "%Y-%m-%d_%H-%M-%S"

# Custom timestamp string
python foodb_pipeline_cli.py paper.pdf --custom-timestamp "experiment_v1"

# Disable timestamps (overwrites existing files)
python foodb_pipeline_cli.py paper.pdf --no-timestamp
```

### 10. File Preservation Examples
```bash
# Run multiple times - each creates new timestamped files
python foodb_pipeline_cli.py paper.pdf --document-only
# Creates: paper_20241217_143022_results.json

python foodb_pipeline_cli.py paper.pdf --document-only
# Creates: paper_20241217_143156_results.json (preserves previous)

# Custom experiment naming
python foodb_pipeline_cli.py paper.pdf --custom-timestamp "baseline"
# Creates: paper_baseline_results.json

python foodb_pipeline_cli.py paper.pdf --custom-timestamp "optimized"
# Creates: paper_optimized_results.json
```

### 11. Directory Processing
```bash
# Process entire directory of PDFs
python foodb_pipeline_cli.py /path/to/pdf_directory --directory-mode --document-only

# Directory with custom output structure
python foodb_pipeline_cli.py ./papers/ \
  --directory-mode \
  --output-dir ./results \
  --individual-subdir "papers" \
  --consolidated-subdir "combined" \
  --document-only

# Process directory with all export formats
python foodb_pipeline_cli.py ./papers/ \
  --directory-mode \
  --export-format all \
  --save-timing \
  --document-only
```

### 12. Consolidated vs Individual Output
```bash
# Both consolidated and individual (default)
python foodb_pipeline_cli.py ./papers/ --directory-mode

# Individual papers only (timestamped files)
python foodb_pipeline_cli.py ./papers/ \
  --directory-mode \
  --individual-output \
  --consolidated-output false

# Consolidated only (append mode, no timestamps)
python foodb_pipeline_cli.py ./papers/ \
  --directory-mode \
  --consolidated-output \
  --individual-output false
```

### 13. Multiple Directory Runs (Append Mode)
```bash
# First batch of papers
python foodb_pipeline_cli.py ./batch1/ \
  --directory-mode \
  --custom-timestamp "batch1" \
  --output-dir ./study_results

# Second batch - appends to consolidated files
python foodb_pipeline_cli.py ./batch2/ \
  --directory-mode \
  --custom-timestamp "batch2" \
  --output-dir ./study_results

# Results: consolidated files contain data from both batches
# Individual files are timestamped separately
```

## Available Groq Models

The pipeline supports multiple Groq models, tested and ranked by performance:

### Recommended Models (Tested and Verified)
| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| **moonshotai/kimi-k2-instruct** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Recommended** - Fastest with excellent accuracy |
| **meta-llama/llama-4-scout-17b-16e-instruct** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High accuracy, good speed |
| **meta-llama/llama-4-maverick-17b-128e-instruct** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Excellent accuracy, moderate speed |
| **llama-3.1-8b-instant** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Original default model |
| **qwen/qwen3-32b** | ⭐⭐⭐ | ⭐⭐⭐ | Alternative option |

### Performance Benchmarks (Wine Biomarkers Test)
- **moonshotai/kimi-k2-instruct**: 42.7s, 185 metabolites, 77.9% recall, 28.4% precision
- **meta-llama/llama-4-scout-17b-16e-instruct**: Similar accuracy, slightly slower
- **meta-llama/llama-4-maverick-17b-128e-instruct**: Excellent accuracy, good for complex documents

### Usage Examples
```bash
# Use recommended model (fastest)
python foodb_pipeline_cli.py paper.pdf --groq-model "moonshotai/kimi-k2-instruct"

# Use for maximum accuracy
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-scout-17b-16e-instruct"
```

## Configuration File Format

Create a JSON configuration file to save frequently used settings:

```json
{
  "output_dir": "./production_results",
  "csv_database": "urinary_wine_biomarkers.csv",
  "chunk_size": 2000,
  "max_tokens": 300,
  "document_only": true,
  "verify_compounds": true,
  "max_attempts": 5,
  "base_delay": 2.0,
  "providers": ["cerebras", "groq", "openrouter"],
  "groq_model": "moonshotai/kimi-k2-instruct",
  "export_format": "all",
  "save_timing": true
}
```

## Output Files

The pipeline generates different output structures depending on the processing mode:

### Standard Mode (Single Files)
- `{filename}_results.json`: Complete results with all data
- `{filename}_results.csv`: Metabolites with database match status (if CSV format)
- `{filename}_results.xlsx`: Multi-sheet workbook (if Excel format)
- `{filename}_results_timing.json`: Detailed timing breakdown (if --save-timing)

### Directory Mode Output Structure
```
output_directory/
├── individual_papers/          # Individual paper results (timestamped)
│   ├── paper1_20241217_143022.json
│   ├── paper1_20241217_143022.csv
│   ├── paper1_20241217_143022_timing.json
│   ├── paper2_20241217_143156.json
│   └── ...
├── consolidated/               # Consolidated results (append mode)
│   ├── consolidated_results.json      # All runs combined
│   ├── consolidated_metabolites.csv   # All metabolites from all papers
│   ├── consolidated_results.xlsx      # Multi-sheet workbook
│   └── processing_summary.json       # Summary of all processing runs
└── batch_summary_20241217_143200.json # Current batch summary
```

### Consolidated Files (Directory Mode)
- **consolidated_results.json**: Contains all processing runs with timestamps
  - `processing_runs[]`: Array of all runs
  - Each run contains: summary stats, paper results, unique metabolites
- **consolidated_metabolites.csv**: All metabolites from all papers with metadata
  - Columns: Processing_Timestamp, Paper_Name, Metabolite, In_Database, etc.
- **consolidated_results.xlsx**: Multi-sheet Excel workbook
  - `All_Runs`: Historical data from all processing runs
  - `Current_Run`: Data from the current processing run
  - `Summary`: Summary statistics
  - `All_Metabolites`: Unique metabolites across all papers
- **processing_summary.json**: High-level summaries of all runs

### Individual Files (Directory Mode)
- Each paper gets its own timestamped files in `individual_papers/`
- Same format as standard mode but with timestamps
- Preserves individual paper analysis for detailed review

## Performance Optimization

### For Speed
```bash
python foodb_pipeline_cli.py paper.pdf \
  --chunk-size 2000 \
  --max-tokens 150 \
  --max-attempts 3 \
  --base-delay 1.0
```

### For Accuracy
```bash
python foodb_pipeline_cli.py paper.pdf \
  --document-only \
  --verify-compounds \
  --chunk-size 1000 \
  --chunk-overlap 100 \
  --max-tokens 300
```

### For Large Documents
```bash
python foodb_pipeline_cli.py large_paper.pdf \
  --chunk-size 2500 \
  --max-attempts 7 \
  --base-delay 3.0 \
  --max-delay 120.0
```

## Troubleshooting

### Rate Limiting Issues
- Increase `--base-delay` and `--max-delay`
- Reduce `--chunk-size` to make fewer API calls
- Use `--primary-provider` to start with a less rate-limited provider

### Memory Issues
- Reduce `--chunk-size`
- Use `--save-chunks false` to avoid storing chunks in memory
- Process files individually instead of batch mode

### API Errors
- Check API keys in .env file
- Use `--debug` for detailed error information
- Try different `--providers` order

### Verification Failures
- Disable `--verify-compounds` if causing issues
- Increase `--max-tokens` for verification responses
- Check `--save-raw-responses` for debugging

## Best Practices

1. **Always use `--document-only`** to prevent training data contamination
2. **Enable `--verify-compounds`** for higher accuracy
3. **Use `--save-timing`** to monitor performance
4. **Set appropriate `--chunk-size`** based on document complexity
5. **Use configuration files** for consistent settings
6. **Enable `--verbose`** for monitoring progress
7. **Use `--export-format all`** for comprehensive output formats

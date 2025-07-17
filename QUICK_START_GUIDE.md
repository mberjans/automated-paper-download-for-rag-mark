# FOODB Pipeline - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Installation
```bash
# Clone repository
git clone <repository-url>
cd automated-paper-download-for-rag-mark

# Create virtual environment
python -m venv foodb_wrapper_env
source foodb_wrapper_env/bin/activate  # Linux/Mac
# foodb_wrapper_env\Scripts\activate  # Windows

# Install dependencies
pip install PyPDF2 pandas requests python-dotenv openpyxl
```

### Step 2: API Keys Setup
Create a `.env` file in the project root:
```bash
CEREBRAS_API_KEY=your_cerebras_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Step 3: Basic Usage
```bash
# Process a single PDF (recommended settings)
python foodb_pipeline_cli.py paper.pdf --document-only --verify-compounds

# View all options
python foodb_pipeline_cli.py --help
```

## 📋 Common Use Cases

### 1. Single Paper Analysis
```bash
# Basic extraction
python foodb_pipeline_cli.py research_paper.pdf

# Full analysis with all outputs
python foodb_pipeline_cli.py research_paper.pdf \
  --document-only \
  --verify-compounds \
  --export-format all \
  --save-timing \
  --verbose
```

### 2. Directory Processing
```bash
# Process all PDFs in a directory
python foodb_pipeline_cli.py /path/to/pdf_directory --directory-mode --document-only

# Organized output structure
python foodb_pipeline_cli.py ./papers/ \
  --directory-mode \
  --output-dir ./results \
  --export-format all \
  --document-only
```

### 3. Batch Processing with Timestamps
```bash
# First batch
python foodb_pipeline_cli.py ./batch1/ \
  --directory-mode \
  --custom-timestamp "batch1" \
  --output-dir ./study_results

# Second batch (appends to consolidated files)
python foodb_pipeline_cli.py ./batch2/ \
  --directory-mode \
  --custom-timestamp "batch2" \
  --output-dir ./study_results
```

### 4. Configuration File Usage
```bash
# Save current settings
python foodb_pipeline_cli.py paper.pdf --save-config my_settings.json

# Use saved configuration
python foodb_pipeline_cli.py paper.pdf --config my_settings.json
```

## 📊 Understanding Output

### Standard Mode Output
```
output_directory/
├── paper_20241217_143022_results.json     # Complete results
├── paper_20241217_143022_results.csv      # Metabolites table
├── paper_20241217_143022_results.xlsx     # Excel workbook
└── paper_20241217_143022_timing.json      # Performance data
```

### Directory Mode Output
```
output_directory/
├── individual_papers/          # Individual paper results
│   ├── study1_20241217_143022.json
│   └── study2_20241217_143156.json
├── consolidated/               # Combined results
│   ├── consolidated_results.json
│   ├── consolidated_metabolites.csv
│   └── processing_summary.json
└── batch_summary_20241217_143200.json
```

## 🔧 Key Parameters

### Essential Arguments
```bash
--document-only          # Prevents training data contamination (RECOMMENDED)
--verify-compounds       # Verifies extracted compounds (RECOMMENDED)
--export-format all      # Exports JSON, CSV, and Excel
--save-timing           # Saves performance analysis
--verbose               # Shows detailed progress
```

### Output Control
```bash
--output-dir ./results          # Custom output directory
--timestamp-files              # Add timestamps (default: enabled)
--custom-timestamp "exp1"      # Custom experiment name
--no-timestamp                 # Disable timestamps (overwrite mode)
```

### Processing Options
```bash
--chunk-size 2000              # Larger chunks for speed
--max-tokens 300               # More detailed extraction
--max-attempts 5               # Retry attempts per chunk
--providers cerebras groq      # Provider preference order
```

## 🐛 Troubleshooting

### Rate Limiting Issues
```bash
# Increase delays
python foodb_pipeline_cli.py paper.pdf --base-delay 3.0 --max-delay 120.0

# Change provider order
python foodb_pipeline_cli.py paper.pdf --providers groq cerebras
```

### Processing Failures
```bash
# Debug mode
python foodb_pipeline_cli.py paper.pdf --debug --log-file debug.log

# Resume from specific chunk
python foodb_pipeline_cli.py paper.pdf --resume-from-chunk 25
```

### Memory Issues
```bash
# Reduce chunk size
python foodb_pipeline_cli.py paper.pdf --chunk-size 1000

# Disable chunk saving
python foodb_pipeline_cli.py paper.pdf --save-chunks false
```

## 📈 Performance Tips

### For Speed
```bash
python foodb_pipeline_cli.py paper.pdf \
  --chunk-size 2500 \
  --max-tokens 200 \
  --max-attempts 3
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
  --chunk-size 2000 \
  --max-attempts 7 \
  --base-delay 3.0 \
  --max-delay 120.0
```

## 🔍 Example Results

### Performance Metrics
- **Processing Time**: ~1 minute per page
- **Success Rate**: 100% with retry mechanisms
- **Accuracy**: 79.7% recall, 25.5% precision (wine biomarkers)

### Sample Output
```json
{
  "input_file": "research_paper.pdf",
  "processing_time": 46.2,
  "extraction_result": {
    "unique_metabolites": 184,
    "metabolites": ["resveratrol", "quercetin", "catechin"]
  },
  "matching_result": {
    "matches_found": 47,
    "detection_rate": 0.797
  },
  "metrics": {
    "precision": 0.255,
    "recall": 0.797,
    "f1_score": 0.387
  }
}
```

## 📚 Next Steps

### Learn More
- **CLI_USAGE_GUIDE.md**: Comprehensive usage examples
- **TECHNICAL_DOCUMENTATION.md**: Detailed technical specs
- **CHANGELOG.md**: Recent updates and features

### Advanced Features
- **Configuration files**: Save and reuse settings
- **Directory processing**: Batch analysis workflows
- **Timestamp management**: Organize experiment results
- **Multiple export formats**: Choose output format

### Get Help
```bash
# View all options
python foodb_pipeline_cli.py --help

# Test with sample data
python foodb_pipeline_cli.py Wine-consumptionbiomarkers-HMDB.pdf --document-only --verbose
```

## 🎯 Best Practices

1. **Always use `--document-only`** to prevent training data contamination
2. **Enable `--verify-compounds`** for higher accuracy
3. **Use `--save-timing`** to monitor performance
4. **Set appropriate `--chunk-size`** based on document complexity
5. **Use configuration files** for consistent settings
6. **Enable `--verbose`** for monitoring progress
7. **Use `--export-format all`** for comprehensive analysis

## 🚀 Ready to Start!

You're now ready to extract metabolites from scientific PDFs! Start with a simple command and explore the advanced features as needed.

```bash
# Your first extraction
python foodb_pipeline_cli.py your_paper.pdf --document-only --verify-compounds --verbose
```

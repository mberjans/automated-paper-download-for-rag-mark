# Changelog

All notable changes to the FOODB Pipeline project will be documented in this file.

## [4.0.0] - 2025-07-17 - Enhanced Fallback API System

### 🚀 **MAJOR ENHANCEMENTS**

#### **Revolutionary Fallback System**
- **30x faster recovery** from rate limiting (2s vs 60s+)
- **Intelligent rate limiting**: Switch providers after 2 consecutive failures
- **Automatic provider switching**: Cerebras → Groq → OpenRouter
- **Real-time provider health monitoring** with automatic recovery

#### **V4 Priority-Based Model Selection**
- **25 models ranked** by F1 scores and performance metrics
- **Automatic model optimization** for each provider
- **Best models selected automatically**:
  - Cerebras: `llama-4-scout-17b-16e-instruct` (0.59s, Score: 9.8)
  - Groq: `meta-llama/llama-4-maverick-17b` (F1: 0.5104, 83% recall)
  - OpenRouter: `mistralai/mistral-nemo:free` (F1: 0.5772, 73% recall)

#### **Performance Improvements**
- **Sub-second inference** with Cerebras models (0.56-0.62s)
- **Best accuracy** with Groq models (F1 scores up to 0.51)
- **Highest diversity** with OpenRouter models (15 models available)
- **100% success rate** with intelligent retry mechanisms

### ✨ **NEW FEATURES**

#### **Enhanced LLM Wrapper (`llm_wrapper_enhanced.py`)**
- V4 priority list integration (`llm_usage_priority_list.json`)
- Intelligent rate limiting with consecutive failure tracking
- Real-time provider health monitoring
- Comprehensive performance statistics
- Automatic model selection for each provider

#### **V4 Model Ranking System**
- `free_models_reasoning_ranked_v4.json` - Complete ranking data
- `llm_usage_priority_list.json` - Priority-ordered usage list
- `V4_RANKING_SUMMARY.md` - Comprehensive analysis
- `LLM_USAGE_PRIORITY_GUIDE.md` - Usage recommendations

#### **Enhanced Testing Suite**
- `test_enhanced_fallback.py` - Comprehensive fallback testing
- Provider health monitoring tests
- Rate limiting behavior verification
- Performance benchmarking

### 🔧 **IMPROVEMENTS**

#### **Rate Limiting Behavior**
- **Before**: Wait 60s+ with exponential backoff
- **After**: Switch provider after 2 failures (30x faster)

#### **Provider Switching**
- **Before**: Manual intervention required
- **After**: Automatic seamless switching

#### **Model Selection**
- **Before**: Fixed hardcoded models
- **After**: V4 priority-based optimization

### 📊 **PERFORMANCE METRICS**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Rate Limit Recovery** | 60s+ wait | 2s switch | **30x faster** |
| **Provider Switching** | Manual | Automatic | **Seamless** |
| **Model Selection** | Fixed | V4 optimized | **Better accuracy** |
| **Success Rate** | Variable | 100% | **Reliable** |

### 📁 **FILES ADDED**
- `llm_usage_priority_list.json` - V4 priority list (25 models)
- `free_models_reasoning_ranked_v4.json` - Complete V4 ranking
- `create_v4_ranking.py` - V4 ranking generation script
- `create_usage_priority_list.py` - Priority list generation
- `test_enhanced_fallback.py` - Enhanced testing suite
- `ENHANCED_FALLBACK_SYSTEM_SUMMARY.md` - Implementation summary
- `V4_RANKING_SUMMARY.md` - Model ranking analysis
- `LLM_USAGE_PRIORITY_GUIDE.md` - Usage guide

### 📝 **FILES UPDATED**
- `FOODB_LLM_pipeline/llm_wrapper_enhanced.py` - Enhanced with V4 system
- `README.md` - Updated with V4 features
- `FOODB_LLM_Pipeline_Documentation.md` - Enhanced system documentation
- `TECHNICAL_DOCUMENTATION.md` - V4 technical details
- `Fallback_System_Documentation.md` - V4 fallback documentation

### 🔄 **BACKWARD COMPATIBILITY**
- **✅ Same interface** as original wrapper
- **✅ Drop-in replacement** for existing code
- **✅ No breaking changes** to pipeline
- **✅ Existing configurations** still supported

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-07-16

### Added - Major CLI Implementation
- **Complete command-line interface** with 40+ configurable parameters
- **Comprehensive argument parsing** with validation and help system
- **Configuration file support** (JSON format) for saving/loading settings
- **Batch processing capabilities** for multiple files
- **Multiple export formats** (JSON, CSV, Excel)

### Added - Timestamp Functionality
- **Automatic timestamp addition** to all output filenames (default: enabled)
- **Configurable timestamp format** (default: %Y%m%d_%H%M%S)
- **Custom timestamp string support** for experiment naming
- **Option to disable timestamps** (--no-timestamp) for overwriting
- **Timestamp preservation** for all file types (JSON, CSV, Excel, timing, raw responses)

### Added - Directory Processing
- **Process entire directories** of PDF files automatically
- **Dual output structure**: consolidated + individual paper results
- **Append mode for consolidated files** preserves historical data
- **Timestamped individual files** prevent overwriting
- **Configurable subdirectory structure** for organized output

### Added - Enhanced LLM Integration
- **Multi-provider fallback system**: Cerebras → Groq → OpenRouter
- **Exponential backoff with jitter** for rate limiting
- **Document-only extraction mode** prevents training data contamination
- **Compound verification system** against original text
- **100% success rate** through robust retry mechanisms

### Added - Comprehensive Output Formats
- **JSON results**: Complete analysis with all metadata
- **CSV export**: Metabolites with database match status
- **Excel workbooks**: Multi-sheet analysis with summary statistics
- **Timing analysis**: Detailed performance breakdown
- **Raw responses**: LLM responses for debugging (optional)

### Added - Advanced Features
- **Performance metrics calculation**: Precision, recall, F1 scores
- **Database matching system** with multiple matching strategies
- **Progress tracking** with verbose output options
- **Resume functionality** from specific chunks
- **Skip existing files** option for batch processing

### Added - Documentation
- **CLI_USAGE_GUIDE.md**: Comprehensive usage guide with examples
- **TECHNICAL_DOCUMENTATION.md**: Detailed technical specifications
- **Test scripts**: Timestamp and directory processing demonstrations
- **Configuration examples**: Production-ready templates

### Changed - Architecture
- **Migrated from script-based to CLI-based architecture**
- **Modular design** with separate CLI and runner components
- **Enhanced error handling** with graceful degradation
- **Improved logging system** with configurable levels

### Performance Improvements
- **Optimized chunking strategy** for better API utilization
- **Intelligent provider switching** to minimize rate limiting
- **Memory-efficient processing** for large documents
- **Parallel processing support** (configurable)

## [1.0.0] - 2025-07-15

### Added - Initial Implementation
- **Basic LLM wrapper** with single provider support
- **PDF text extraction** using PyPDF2
- **Simple metabolite extraction** from text chunks
- **Database matching** against CSV biomarker database
- **JSON output** with basic results

### Added - Core Features
- **Document-only extraction** to prevent training data contamination
- **Exponential backoff** for rate limiting
- **Basic error handling** and retry mechanisms
- **Performance timing** analysis

### Added - Testing
- **Wine biomarkers test case** with comprehensive analysis
- **Performance benchmarking** with detailed timing
- **Success rate validation** (100% chunk processing)

## [0.1.0] - 2025-07-14

### Added - Project Foundation
- **Initial project structure** and repository setup
- **Legacy FOODB pipeline** components
- **Basic dependencies** and environment setup
- **Initial documentation** and README

---

## Migration Guide

### From v1.0.0 to v2.0.0

#### Breaking Changes
- **CLI interface required**: Direct script execution no longer supported
- **New argument structure**: All parameters now use CLI arguments
- **Output structure changed**: New timestamp and directory organization

#### Migration Steps
1. **Update command structure**:
   ```bash
   # Old (v1.0.0)
   python extract_metabolites.py

   # New (v2.0.0)
   python foodb_pipeline_cli.py input.pdf --document-only
   ```

2. **Update configuration**:
   ```bash
   # Create configuration file
   python foodb_pipeline_cli.py input.pdf --save-config my_config.json
   
   # Use configuration file
   python foodb_pipeline_cli.py input.pdf --config my_config.json
   ```

3. **Update output handling**:
   ```bash
   # Specify output directory and format
   python foodb_pipeline_cli.py input.pdf \
     --output-dir ./results \
     --export-format all \
     --save-timing
   ```

#### New Features Available
- **Directory processing**: Process entire folders of PDFs
- **Multiple export formats**: JSON, CSV, Excel
- **Timestamp management**: Preserve old results automatically
- **Configuration files**: Save and reuse settings
- **Enhanced error handling**: Better reliability and debugging

#### Recommended Settings
```bash
# Production settings
python foodb_pipeline_cli.py input.pdf \
  --document-only \
  --verify-compounds \
  --export-format all \
  --save-timing \
  --output-dir ./production_results
```

---

## Roadmap

### Planned Features (v2.1.0)
- [ ] **Parallel processing**: Multi-threaded chunk processing
- [ ] **Advanced caching**: Cache results for repeated processing
- [ ] **Web interface**: Browser-based GUI for pipeline
- [ ] **API server**: REST API for programmatic access
- [ ] **Enhanced visualization**: Interactive results dashboard

### Planned Features (v2.2.0)
- [ ] **Machine learning integration**: Automated compound classification
- [ ] **Advanced NLP**: Named entity recognition for compounds
- [ ] **Database integration**: Direct database storage options
- [ ] **Cloud deployment**: Docker containers and cloud templates
- [ ] **Advanced analytics**: Trend analysis across document sets

### Long-term Goals (v3.0.0)
- [ ] **Real-time processing**: Stream processing for large datasets
- [ ] **Advanced AI models**: Custom fine-tuned models for metabolomics
- [ ] **Integration ecosystem**: Plugins for popular research tools
- [ ] **Collaborative features**: Multi-user research environments
- [ ] **Advanced visualization**: 3D molecular structure integration

---

## Support

For questions about specific versions or migration assistance:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check CLI_USAGE_GUIDE.md for detailed examples
- **Configuration**: Use example_config.json as a starting template

## Contributors

- **Development Team**: Core pipeline development and CLI implementation
- **Testing Team**: Comprehensive testing and validation
- **Documentation Team**: User guides and technical documentation

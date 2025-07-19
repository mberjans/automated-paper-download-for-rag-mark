# 🚩 CLI Flags Analysis: Current vs Missing Relevance Methods

## ❓ **Your Question: What are the flags for each method? What is the default?**

**Answer: There are currently NO flags for relevance evaluation methods because relevance evaluation is not yet integrated into the main pipeline.**

---

## 📊 **Current CLI Flags (No Relevance Methods)**

### **Text Processing Configuration**
```bash
--chunk-size 1500          # Size of text chunks (default: 1500)
--chunk-overlap 0          # Overlap between chunks (default: 0)
--min-chunk-size 100       # Minimum chunk size (default: 100)
```

### **LLM Configuration**
```bash
--max-tokens 200           # Maximum tokens for responses (default: 200)
--document-only            # Use document-only extraction
--verify-compounds         # Verify extracted compounds
--custom-prompt "..."      # Custom extraction prompt
```

### **Provider Configuration**
```bash
--providers cerebras groq openrouter  # Fallback order (default)
--primary-provider cerebras           # Primary provider
--groq-model kimi-k2-instruct        # Specific Groq model
```

### **Retry Configuration**
```bash
--max-attempts 5           # Maximum retry attempts (default: 5)
--base-delay 2.0          # Base delay in seconds (default: 2.0)
--max-delay 60.0          # Maximum delay (default: 60.0)
--exponential-base 2.0    # Exponential base (default: 2.0)
--disable-jitter          # Disable jitter in backoff
```

### **Database Configuration**
```bash
--csv-database urinary_wine_biomarkers.csv  # CSV database (default)
--csv-column "Compound Name"                 # Column name (default)
```

---

## ❌ **Missing: Relevance Evaluation Flags**

**The current pipeline has NO relevance evaluation flags because the system I developed is not yet integrated.**

---

## ✅ **Proposed Relevance Evaluation Flags**

Here are the flags that SHOULD be added to enable relevance evaluation:

### **1. Relevance Method Selection**
```bash
# Method selection flags
--relevance-method [none|keywords|csv|text|ontology|ml|combined]
  # none: No relevance filtering (current default)
  # keywords: Use hardcoded keywords (current approach)
  # csv: Generate from CSV database
  # text: Generate from text analysis
  # ontology: Generate from chemical ontologies
  # ml: Generate using machine learning
  # combined: Combine multiple methods (recommended)

# Default method
--relevance-method none  # Current default (no filtering)
```

### **2. Relevance Configuration**
```bash
# Threshold and scoring
--relevance-threshold 0.3        # Relevance threshold (default: 0.3)
--relevance-weights 0.4,0.3,0.2,0.1  # Keyword,Chemical,Section,Context weights

# Keyword sources
--keyword-file keywords.json     # Load keywords from file
--save-keywords keywords.json    # Save generated keywords
--update-keywords               # Update keywords from sources
```

### **3. Method-Specific Flags**

#### **CSV Method Flags**
```bash
--csv-keyword-source urinary_wine_biomarkers.csv  # CSV file for keywords
--csv-keyword-column "Compound Name"               # Column for compounds
--csv-include-synonyms                            # Include compound synonyms
```

#### **Text Analysis Flags**
```bash
--text-analysis-corpus input.pdf    # Text corpus for analysis
--chemical-patterns-file patterns.json  # Custom chemical patterns
--min-term-frequency 2              # Minimum term frequency
```

#### **Ontology Method Flags**
```bash
--ontology-source chebi             # Ontology source (chebi, pubchem)
--ontology-depth 3                  # Hierarchy depth to include
--include-synonyms                  # Include ontology synonyms
```

#### **Machine Learning Flags**
```bash
--ml-training-data training.json    # Labeled training data
--ml-model-file relevance_model.pkl # Saved ML model
--ml-feature-method tfidf           # Feature extraction method
```

### **4. Reporting and Analysis**
```bash
--relevance-report                  # Generate relevance analysis report
--show-filtered-chunks             # Show which chunks were filtered
--relevance-stats                  # Show detailed relevance statistics
--save-relevance-scores scores.json # Save chunk relevance scores
```

---

## 🎯 **Recommended Default Configuration**

### **Conservative Default (Backward Compatible)**
```bash
# Current behavior - no relevance filtering
python foodb_pipeline_cli.py input.pdf
# Equivalent to: --relevance-method none
```

### **Recommended Default (With Relevance)**
```bash
# Recommended default with CSV-based relevance
python foodb_pipeline_cli.py input.pdf --relevance-method csv --relevance-threshold 0.3
```

### **Advanced Usage Examples**
```bash
# Combined method with custom threshold
python foodb_pipeline_cli.py input.pdf \
  --relevance-method combined \
  --relevance-threshold 0.4 \
  --relevance-report

# Text analysis method with custom corpus
python foodb_pipeline_cli.py input.pdf \
  --relevance-method text \
  --text-analysis-corpus corpus.txt \
  --save-keywords generated_keywords.json

# Machine learning method with training data
python foodb_pipeline_cli.py input.pdf \
  --relevance-method ml \
  --ml-training-data training.json \
  --relevance-threshold 0.35
```

---

## 📊 **Implementation Priority**

### **Phase 1: Basic Integration**
```bash
--relevance-method [none|csv]      # Basic CSV method
--relevance-threshold 0.3          # Configurable threshold
--relevance-report                 # Basic reporting
```

### **Phase 2: Advanced Methods**
```bash
--relevance-method [text|combined] # Text analysis and combined
--keyword-file keywords.json       # Keyword management
--relevance-weights 0.4,0.3,0.2,0.1 # Weight configuration
```

### **Phase 3: Full Feature Set**
```bash
--relevance-method [ontology|ml]   # Advanced methods
--ml-training-data training.json   # ML integration
--ontology-source chebi           # Ontology integration
```

---

## 🔧 **Proposed CLI Integration**

### **Modified Help Output**
```bash
python foodb_pipeline_cli.py --help

Relevance Evaluation Configuration:
  --relevance-method {none,csv,text,combined}
                        Method for chunk relevance evaluation (default: none)
  --relevance-threshold FLOAT
                        Relevance threshold for filtering (default: 0.3)
  --csv-keyword-source FILE
                        CSV file for keyword generation (default: urinary_wine_biomarkers.csv)
  --relevance-report    Generate detailed relevance analysis report
  --save-keywords FILE  Save generated keywords to file
```

### **Example Usage**
```bash
# Enable CSV-based relevance filtering
python foodb_pipeline_cli.py Wine.pdf --relevance-method csv

# Use combined method with custom threshold
python foodb_pipeline_cli.py Wine.pdf \
  --relevance-method combined \
  --relevance-threshold 0.4 \
  --relevance-report

# Generate and save keywords
python foodb_pipeline_cli.py Wine.pdf \
  --relevance-method csv \
  --save-keywords wine_keywords.json
```

---

## 📈 **Expected Impact**

### **With Default (No Relevance)**
```bash
python foodb_pipeline_cli.py Wine.pdf
# Processes all 86 chunks (current behavior)
```

### **With CSV Relevance Method**
```bash
python foodb_pipeline_cli.py Wine.pdf --relevance-method csv
# Processes ~60 relevant chunks (30% reduction)
# 30% faster processing, 30% cost savings
```

### **With Combined Method**
```bash
python foodb_pipeline_cli.py Wine.pdf --relevance-method combined
# Processes ~55 relevant chunks (35% reduction)
# 35% faster processing, higher precision
```

---

## 🚀 **Conclusion**

**Current Status:**
- ❌ **No relevance evaluation flags** in current CLI
- ❌ **No relevance methods** implemented in main pipeline
- ❌ **Default method**: None (processes all chunks)

**Recommended Implementation:**
- ✅ **Add relevance method flags** with `none` as default (backward compatible)
- ✅ **Implement CSV method first** (easiest, highest impact)
- ✅ **Add combined method** for advanced users
- ✅ **Include reporting flags** for analysis and debugging

**This would transform the pipeline from a "process everything" approach to an intelligent, relevance-aware system with significant efficiency and accuracy improvements!** 🚩📊🚀

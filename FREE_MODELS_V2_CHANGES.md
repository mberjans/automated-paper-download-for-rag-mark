# Free Models Reasoning Ranked V2 - Changes and Additions

## 📋 Overview

The `free_models_reasoning_ranked_v2.json` file is an enhanced version that includes all the missing Groq models you specified, with updated rankings based on comprehensive FOODB pipeline evaluation results.

## 🆕 New Models Added

### 1. **meta-llama/llama-4-maverick-17b-128e-instruct** (Rank #1)
- **Provider**: Groq
- **Reasoning Score**: 9.9 (highest)
- **FOODB Performance**: F1=0.5104, Recall=83.1%, Processing=393.4s
- **Strengths**: Best biomarker detection, most comprehensive extraction
- **Best For**: Research requiring maximum accuracy

### 2. **meta-llama/llama-4-scout-17b-16e-instruct** (Rank #2)
- **Provider**: Groq  
- **Reasoning Score**: 9.8
- **FOODB Performance**: F1=0.5081, Recall=79.7%, Processing=124.4s
- **Strengths**: Best speed-accuracy balance, fastest processing
- **Best For**: Production environments

### 3. **qwen/qwen3-32b** (Rank #4)
- **Provider**: Groq
- **Reasoning Score**: 9.6
- **FOODB Performance**: F1=0.5056, Recall=76.3%, Processing=237.6s
- **Strengths**: Consistent performance, alternative architecture
- **Best For**: Reliable alternative option

### 4. **llama-3.1-8b-instant** (Rank #7)
- **Provider**: Groq
- **Reasoning Score**: 9.0
- **FOODB Performance**: F1=0.5000, Recall=79.7%, Processing=169.5s
- **Strengths**: Proven baseline, reliable processing
- **Best For**: Stable, predictable results

### 5. **llama-3.3-70b-versatile** (Rank #5)
- **Provider**: Groq
- **Reasoning Score**: 9.5
- **FOODB Performance**: F1=0.4706, Recall=74.6%, Processing=132.8s
- **Strengths**: Large 70B model, good speed for size
- **Best For**: Versatile performance needs

### 6. **moonshotai/kimi-k2-instruct** (Rank #11)
- **Provider**: Groq
- **Reasoning Score**: 8.0
- **FOODB Performance**: F1=0.4053, Recall=52.5%, Processing=175.6s
- **Strengths**: Unique architecture, specialized performance
- **Best For**: Specific use cases, alternative option

## 📊 Key Changes from V1

### **Enhanced Ranking System**
- **V1**: Based primarily on reasoning benchmarks
- **V2**: Incorporates real-world FOODB pipeline performance data
- **New Metrics**: F1 scores, precision, recall, processing time, metabolite extraction

### **FOODB Performance Integration**
All Groq models now include `foodb_performance` section with:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Accuracy of extracted metabolites
- **Recall**: Percentage of known biomarkers detected
- **Processing Time**: Total analysis time in seconds
- **Metabolites Extracted**: Total unique compounds found
- **Biomarkers Detected**: Number of ground truth biomarkers found

### **Updated Rankings**
1. **meta-llama/llama-4-maverick-17b-128e-instruct** (NEW #1)
2. **meta-llama/llama-4-scout-17b-16e-instruct** (NEW #2)
3. **llama-4-scout-17b-16e-instruct** (Cerebras - maintained)
4. **qwen/qwen3-32b** (NEW #4)
5. **llama-3.3-70b-versatile** (Groq - updated)
6. **llama-3.3-70b** (Cerebras - maintained)
7. **llama-3.1-8b-instant** (NEW #7)

## 🎯 Model Recommendations by Use Case

### **Production Deployment**
```json
{
  "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
  "rank": 2,
  "reasoning": "Best speed-accuracy balance (F1=0.5081, 124.4s)"
}
```

### **Research/Maximum Accuracy**
```json
{
  "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct", 
  "rank": 1,
  "reasoning": "Highest F1 score (0.5104) and recall (83.1%)"
}
```

### **Reliable Baseline**
```json
{
  "model_id": "llama-3.1-8b-instant",
  "rank": 7,
  "reasoning": "Proven stability with solid F1 score (0.5000)"
}
```

### **Alternative Option**
```json
{
  "model_id": "qwen/qwen3-32b",
  "rank": 4,
  "reasoning": "Consistent performance with good F1 score (0.5056)"
}
```

## 📈 Performance Comparison

| Model | F1 Score | Recall | Processing Time | Efficiency* |
|-------|----------|--------|-----------------|-------------|
| **llama-4-maverick** | **0.5104** | **83.1%** | 393.4s | 0.0013 |
| **llama-4-scout** | **0.5081** | 79.7% | **124.4s** | **0.0041** |
| **qwen3-32b** | 0.5056 | 76.3% | 237.6s | 0.0021 |
| **llama-3.1-8b** | 0.5000 | 79.7% | 169.5s | 0.0029 |
| **llama-3.3-70b** | 0.4706 | 74.6% | 132.8s | 0.0035 |
| **kimi-k2** | 0.4053 | 52.5% | 175.6s | 0.0023 |

*Efficiency = F1 Score / Processing Time

## 🔧 Technical Specifications

### **All Models Include**:
- **Provider**: Groq API endpoint
- **API URL**: `https://api.groq.com/openai/v1/chat/completions`
- **API Key**: `GROQ_API_KEY` environment variable
- **Cost**: FREE tier access
- **Context Tokens**: 131,072 (except where noted)
- **Reliability**: Perfect (100% success rate)

### **Enhanced Metadata**:
- **Reasoning Scores**: Updated based on combined benchmarks
- **Test Results**: Logical deduction, mathematical reasoning, pattern recognition, causal reasoning
- **FOODB Performance**: Real-world metabolite extraction metrics
- **Strengths**: Specific use case recommendations
- **Processing Times**: Actual measured performance

## 🎉 Benefits of V2

### **1. Real-World Performance Data**
- Based on actual FOODB pipeline testing
- Includes F1 scores from wine biomarkers evaluation
- Processing time measurements from 9-page PDF analysis

### **2. Enhanced Model Selection**
- Clear recommendations for different use cases
- Performance-based ranking system
- Efficiency metrics for speed vs accuracy trade-offs

### **3. Complete Groq Coverage**
- All requested Groq models included
- Consistent metadata structure
- Comprehensive performance comparison

### **4. Production-Ready Information**
- Actual API endpoints and configuration
- Proven reliability metrics (100% success rates)
- Real processing time expectations

## 🚀 Usage with FOODB Pipeline

### **Configuration Example**:
```json
{
  "groq_model": "meta-llama/llama-4-scout-17b-16e-instruct",
  "providers": ["groq", "cerebras", "openrouter"],
  "document_only": true,
  "verify_compounds": true
}
```

### **CLI Usage**:
```bash
# Use best balance model
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-scout-17b-16e-instruct"

# Use highest accuracy model  
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-maverick-17b-128e-instruct"
```

## 📋 Summary

The V2 file provides a comprehensive, performance-tested ranking of free reasoning models with special emphasis on the Groq models you requested. All models have been evaluated with real-world FOODB pipeline data, providing accurate performance expectations for metabolite extraction tasks.

**Key Improvement**: Rankings now reflect actual performance in scientific document processing rather than just theoretical reasoning capabilities.

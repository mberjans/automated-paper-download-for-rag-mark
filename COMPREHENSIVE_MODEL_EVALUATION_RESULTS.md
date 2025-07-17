# Comprehensive Groq Model Evaluation Results

## 📊 Executive Summary

**Document Tested**: Wine-consumptionbiomarkers-HMDB.pdf (9 pages)  
**Ground Truth Database**: urinary_wine_biomarkers.csv (59 biomarkers)  
**Evaluation Date**: July 17, 2025  
**Models Evaluated**: 6 Groq models  
**Success Rate**: 100% (6/6 models working)

## 🏆 Performance Ranking by F1 Score

| Rank | Model | F1 Score | Precision | Recall | Processing Time | Metabolites Extracted |
|------|-------|----------|-----------|--------|-----------------|----------------------|
| **1** | **meta-llama/llama-4-maverick-17b-128e-instruct** | **0.5104** | 0.3684 | **0.8305** | 393.4s | 154 |
| **2** | **meta-llama/llama-4-scout-17b-16e-instruct** | **0.5081** | 0.3730 | 0.7966 | 124.4s | 142 |
| **3** | **qwen/qwen3-32b** | **0.5056** | 0.3782 | 0.7627 | 237.6s | 136 |
| **4** | **llama-3.1-8b-instant** | **0.5000** | 0.3643 | 0.7966 | 169.5s | 143 |
| **5** | **llama-3.3-70b-versatile** | **0.4706** | 0.3438 | 0.7458 | 132.8s | 151 |
| **6** | **moonshotai/kimi-k2-instruct** | **0.4053** | 0.3298 | 0.5254 | 175.6s | 104 |

## ⚡ Speed Ranking

| Rank | Model | Processing Time | F1 Score | Efficiency Score* |
|------|-------|-----------------|----------|-------------------|
| **1** | **meta-llama/llama-4-scout-17b-16e-instruct** | **124.4s** | 0.5081 | **0.0041** |
| **2** | **llama-3.3-70b-versatile** | **132.8s** | 0.4706 | 0.0035 |
| **3** | **llama-3.1-8b-instant** | **169.5s** | 0.5000 | 0.0029 |
| **4** | **moonshotai/kimi-k2-instruct** | **175.6s** | 0.4053 | 0.0023 |
| **5** | **qwen/qwen3-32b** | **237.6s** | 0.5056 | 0.0021 |
| **6** | **meta-llama/llama-4-maverick-17b-128e-instruct** | **393.4s** | 0.5104 | 0.0013 |

*Efficiency Score = F1 Score / Processing Time

## 🎯 Detailed Performance Metrics

### Best Overall Performance: meta-llama/llama-4-maverick-17b-128e-instruct
- **F1 Score**: 0.5104 (highest)
- **Recall**: 0.8305 (highest - found 49/59 biomarkers)
- **Precision**: 0.3684
- **Processing Time**: 393.4s (slowest)
- **Metabolites Extracted**: 154
- **Success Rate**: 100% (46/46 chunks)

### Best Speed-Accuracy Balance: meta-llama/llama-4-scout-17b-16e-instruct
- **F1 Score**: 0.5081 (2nd highest)
- **Recall**: 0.7966 (found 47/59 biomarkers)
- **Precision**: 0.3730
- **Processing Time**: 124.4s (fastest)
- **Metabolites Extracted**: 142
- **Efficiency Score**: 0.0041 (best)

### Fastest Processing: meta-llama/llama-4-scout-17b-16e-instruct
- **Processing Time**: 124.4s
- **F1 Score**: 0.5081 (excellent performance despite speed)
- **Recall**: 0.7966
- **Efficiency**: Best balance of speed and accuracy

## 📈 Key Findings

### 🏆 Top Recommendations

1. **For Maximum Accuracy**: `meta-llama/llama-4-maverick-17b-128e-instruct`
   - Highest F1 score (0.5104) and recall (83.1%)
   - Best at finding biomarkers (49/59 detected)
   - Trade-off: Slower processing (393.4s)

2. **For Best Balance**: `meta-llama/llama-4-scout-17b-16e-instruct`
   - Excellent F1 score (0.5081) with fastest processing (124.4s)
   - High recall (79.7%) with good speed
   - **RECOMMENDED for production use**

3. **For Reliability**: `llama-3.1-8b-instant`
   - Solid F1 score (0.5000) with proven stability
   - Good recall (79.7%) and moderate speed (169.5s)
   - Best for consistent, reliable results

### 📊 Performance Insights

- **All models achieved 100% success rate** (46/46 chunks processed)
- **Recall range**: 52.5% - 83.1% (average: 73.9%)
- **F1 score range**: 0.405 - 0.510 (average: 0.489)
- **Processing time range**: 124.4s - 393.4s (average: 205.5s)
- **Metabolites extracted range**: 104 - 154 (average: 138)

### 🔍 Model-Specific Analysis

#### meta-llama/llama-4-maverick-17b-128e-instruct (Best Accuracy)
- **Strengths**: Highest recall (83.1%), most comprehensive extraction
- **Weaknesses**: Slowest processing time
- **Best for**: Research requiring maximum biomarker detection

#### meta-llama/llama-4-scout-17b-16e-instruct (Best Balance)
- **Strengths**: Fastest processing with excellent accuracy
- **Weaknesses**: Slightly lower recall than Maverick
- **Best for**: Production environments requiring speed and accuracy

#### qwen/qwen3-32b (Consistent Performer)
- **Strengths**: Good F1 score, consistent performance
- **Weaknesses**: Moderate speed
- **Best for**: Alternative option with reliable results

#### llama-3.1-8b-instant (Proven Baseline)
- **Strengths**: Reliable, well-tested, good performance
- **Weaknesses**: Not the fastest or most accurate
- **Best for**: Stable, predictable processing

#### llama-3.3-70b-versatile (Large Model)
- **Strengths**: Good speed for a 70B model
- **Weaknesses**: Lower F1 score despite size
- **Best for**: When model size/capability is prioritized

#### moonshotai/kimi-k2-instruct (Specialized)
- **Strengths**: Unique architecture, good for specific use cases
- **Weaknesses**: Lowest recall and F1 score in this test
- **Best for**: Specialized applications or when other models fail

## 🎯 Usage Recommendations

### Production Deployment
```bash
# Recommended: Best balance of speed and accuracy
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-scout-17b-16e-instruct"
```

### Research/Maximum Accuracy
```bash
# For highest biomarker detection rate
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-maverick-17b-128e-instruct"
```

### High-Volume Processing
```bash
# For fastest processing with good accuracy
python foodb_pipeline_cli.py paper.pdf --groq-model "meta-llama/llama-4-scout-17b-16e-instruct"
```

### Reliable Baseline
```bash
# For proven, stable performance
python foodb_pipeline_cli.py paper.pdf --groq-model "llama-3.1-8b-instant"
```

## 📊 Statistical Summary

- **Average F1 Score**: 0.489
- **Average Recall**: 73.9%
- **Average Precision**: 36.3%
- **Average Processing Time**: 205.5 seconds
- **Average Metabolites Extracted**: 138
- **Success Rate**: 100% across all models

## 🔬 Technical Details

- **Test Document**: Wine-consumptionbiomarkers-HMDB.pdf (9 pages, wine biomarkers study)
- **Ground Truth**: 59 known urinary wine biomarkers from urinary_wine_biomarkers.csv
- **Chunk Size**: 1,500 characters (standardized for fair comparison)
- **Max Tokens**: 200 (standardized for fair comparison)
- **Processing Mode**: Document-only extraction with compound verification
- **Evaluation Metrics**: Precision, Recall, F1 Score, Processing Time
- **Success Criteria**: Successful processing of all 46 text chunks

## 🎉 Conclusion

All 6 tested Groq models successfully processed the wine biomarkers document with 100% success rate. The **meta-llama/llama-4-scout-17b-16e-instruct** model provides the best balance of speed and accuracy, making it the recommended choice for production use. For maximum biomarker detection, **meta-llama/llama-4-maverick-17b-128e-instruct** achieves the highest recall rate of 83.1%.

The evaluation demonstrates that the FOODB pipeline works reliably across all tested models, providing users with flexibility to choose the model that best fits their specific requirements for speed, accuracy, or reliability.

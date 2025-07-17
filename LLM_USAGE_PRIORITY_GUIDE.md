# 🚀 LLM Usage Priority Guide

## 📋 **Priority Order: Cerebras → Groq → OpenRouter**

This guide provides the optimal LLM usage order based on **F1 scores and recall performance** for biomarker extraction tasks, organized by provider preference.

---

## ⚡ **TIER 1: CEREBRAS MODELS (Ultra-Fast Inference)**
*Use these first for speed-critical applications*

| Priority | Model | Speed | Reasoning | API Endpoint |
|----------|-------|-------|-----------|--------------|
| **1** | **Llama 4 Scout** | **0.59s** | 9.8 | `llama-4-scout-17b-16e-instruct` |
| **2** | **Llama 3.3 70B** | **0.62s** | 9.5 | `llama-3.3-70b` |
| **3** | **Llama 3.1 8B** | **0.56s** | 8.5 | `llama3.1-8b` |
| **4** | **Qwen 3 32B** | **0.57s** | 8.2 | `qwen-3-32b` |

**🔑 API Configuration:**
```bash
API_URL: https://api.cerebras.ai/v1/chat/completions
API_KEY: CEREBRAS_API_KEY
```

**⚡ Cerebras Advantages:**
- **Fastest inference** (0.56-0.62s)
- **Perfect for real-time** applications
- **Excellent reasoning** scores
- **Ultra-low latency**

---

## 🏆 **TIER 2: GROQ MODELS (Best F1 Performance)**
*Use these for highest accuracy biomarker extraction*

| Priority | Model | F1 Score | Recall | Speed | Biomarkers |
|----------|-------|----------|--------|-------|------------|
| **5** | **Llama 4 Maverick** | **0.5104** | 0.8305 | 1.45s | 49/59 |
| **6** | **Llama 4 Scout** | **0.5081** | 0.7966 | 1.23s | 47/59 |
| **7** | **Qwen 3 32B** | **0.5056** | 0.7627 | 2.15s | 45/59 |
| **8** | **Llama 3.1 8B Instant** | **0.5000** | 0.7966 | 0.95s | 47/59 |
| **9** | **Llama 3.3 70B Versatile** | **0.4706** | 0.7458 | 1.34s | 44/59 |
| **10** | **Moonshot AI Kimi K2** | **0.4053** | 0.6949 | 1.75s | 41/59 |

**🔑 API Configuration:**
```bash
API_URL: https://api.groq.com/openai/v1/chat/completions
API_KEY: GROQ_API_KEY
```

**🏆 Groq Advantages:**
- **Highest F1 scores** (0.40-0.51)
- **Best biomarker detection** (83% recall)
- **Proven performance** on real tasks
- **Excellent speed-accuracy balance**

---

## 🌐 **TIER 3: OPENROUTER MODELS (Diverse Capabilities)**
*Use these for specialized features and model diversity*

### **🥇 High Performance (F1 > 0.3)**

| Priority | Model | F1 Score | Recall | Speed | Provider |
|----------|-------|----------|--------|-------|----------|
| **11** | **Mistral Nemo** | **0.5772** | 0.7288 | 2.86s | Mistral |
| **12** | **DeepSeek R1T Chimera** | **0.4372** | 0.6780 | 9.06s | TNG |
| **13** | **Google Gemini 2.0 Flash** | **0.4065** | 0.4237 | 1.74s | Google |
| **14** | **Mistral Small 3.1** | **0.3619** | 0.6441 | 1.97s | Mistral |
| **15** | **Mistral Small 3.2** | **0.3421** | 0.6610 | 4.10s | Mistral |
| **16** | **Google Gemma 3 27B** | **0.3333** | 0.3559 | 4.76s | Google |
| **17** | **Nous DeepHermes 3** | **0.3178** | 0.5763 | 1.75s | Nous |

### **🥈 Moderate Performance (F1 0.2-0.3)**

| Priority | Model | F1 Score | Recall | Speed | Provider |
|----------|-------|----------|--------|-------|----------|
| **18** | **Moonshot Kimi VL** | **0.2797** | 0.3390 | 7.29s | Moonshot |
| **19** | **Moonshot Kimi K2** | **0.2549** | 0.2203 | 2.22s | Moonshot |
| **20** | **DeepSeek V3** | **0.2330** | 0.2034 | 4.60s | DeepSeek |
| **21** | **Meta Llama 3.3 70B** | **0.2198** | 0.1695 | 3.21s | Meta |

### **🥉 Basic Performance (F1 < 0.2)**

| Priority | Model | F1 Score | Recall | Speed | Provider |
|----------|-------|----------|--------|-------|----------|
| **22** | **Qwen3 235B** | **0.1143** | 0.0678 | 8.10s | Qwen |
| **23** | **Meta Llama 3.2 Vision** | **0.1128** | 0.1864 | 1.80s | Meta |
| **24** | **DeepSeek V3 Base** | **0.0635** | 0.0339 | 8.01s | DeepSeek |
| **25** | **Google Gemma 3 12B** | **0.0290** | 0.0169 | 3.92s | Google |

**🔑 API Configuration:**
```bash
API_URL: https://openrouter.ai/api/v1/chat/completions
API_KEY: OPENROUTER_API_KEY
```

**🌐 OpenRouter Advantages:**
- **Most diverse selection** (15 models)
- **Specialized capabilities** (vision, reasoning)
- **Largest context windows** (up to 1M tokens)
- **Multiple model families** (Google, Meta, Mistral, etc.)

---

## 🎯 **USAGE RECOMMENDATIONS**

### **⚡ Speed-Critical Applications**
```
1. Cerebras Llama 4 Scout (0.59s)
2. Cerebras Llama 3.1 8B (0.56s)  
3. Cerebras Llama 3.3 70B (0.62s)
```

### **🔬 Research & High Accuracy**
```
1. Groq Llama 4 Maverick (F1: 0.5104)
2. OpenRouter Mistral Nemo (F1: 0.5772)
3. Groq Llama 4 Scout (F1: 0.5081)
```

### **⚖️ Balanced Performance**
```
1. Groq Llama 3.1 8B Instant (F1: 0.5000, 0.95s)
2. OpenRouter Google Gemini 2.0 Flash (F1: 0.4065, 1.74s)
3. Groq Llama 3.3 70B Versatile (F1: 0.4706, 1.34s)
```

### **🌐 Specialized Features**
```
1. OpenRouter Meta Llama 3.2 Vision (Vision capabilities)
2. OpenRouter Moonshot Kimi VL (Vision + thinking)
3. OpenRouter DeepSeek R1T Chimera (Reasoning)
```

---

## 📊 **PERFORMANCE SUMMARY**

### **🏆 Best F1 Scores**
1. **Mistral Nemo (OpenRouter)**: 0.5772
2. **Llama 4 Maverick (Groq)**: 0.5104
3. **Llama 4 Scout (Groq)**: 0.5081

### **⚡ Fastest Models**
1. **Llama 3.1 8B (Cerebras)**: 0.56s
2. **Llama 4 Scout (Cerebras)**: 0.59s
3. **Llama 3.3 70B (Cerebras)**: 0.62s

### **🎯 Best Recall**
1. **Llama 4 Maverick (Groq)**: 83.05%
2. **Llama 4 Scout (Groq)**: 79.66%
3. **Qwen 3 32B (Groq)**: 76.27%

---

## 🚀 **IMPLEMENTATION STRATEGY**

### **1. Primary Fallback Chain**
```
Cerebras Llama 4 Scout → Groq Llama 4 Maverick → OpenRouter Mistral Nemo
```

### **2. Speed-Optimized Chain**
```
Cerebras Llama 3.1 8B → Cerebras Llama 4 Scout → Groq Llama 3.1 8B Instant
```

### **3. Accuracy-Optimized Chain**
```
Groq Llama 4 Maverick → OpenRouter Mistral Nemo → Groq Llama 4 Scout
```

---

## 🎉 **CONCLUSION**

This priority list provides optimal LLM usage order based on:
- **Provider preference**: Cerebras (speed) → Groq (accuracy) → OpenRouter (diversity)
- **Performance metrics**: F1 scores and recall for biomarker extraction
- **Real-world testing**: Based on comprehensive evaluations

**Use this guide to implement intelligent model fallback chains that optimize for your specific requirements!** 🚀📊🏆

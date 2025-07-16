# 🚀 OpenRouter API Integration for FOODB Pipeline

## 🎯 **Immediate Speed Boost: Local Gemma → OpenRouter Free Models**

**Expected Speedup**: 10-50x faster inference, no GPU requirements, instant deployment

## 📋 **Available Free Models on OpenRouter**

### **Recommended Free Models for FOODB Pipeline:**

1. **Meta Llama 3.1 8B Instruct (Free)** - Best quality/speed balance
   - Model ID: `meta-llama/llama-3.1-8b-instruct:free`
   - Context: 128k tokens
   - Speed: Very fast
   - Quality: Excellent for scientific text

2. **Google Gemma 2 9B (Free)** - Similar to your current model
   - Model ID: `google/gemma-2-9b-it:free`
   - Context: 8k tokens
   - Speed: Fast
   - Quality: High

3. **Mistral 7B Instruct (Free)** - Lightweight and fast
   - Model ID: `mistralai/mistral-7b-instruct:free`
   - Context: 32k tokens
   - Speed: Very fast
   - Quality: Good

4. **Qwen 2.5 7B Instruct (Free)** - Excellent for reasoning
   - Model ID: `qwen/qwen-2.5-7b-instruct:free`
   - Context: 32k tokens
   - Speed: Fast
   - Quality: Very good

## 🔧 **Implementation Guide**

### **Step 1: Setup OpenRouter Account**

```bash
# Get your API key from https://openrouter.ai/
export OPENROUTER_API_KEY="your_api_key_here"
```

### **Step 2: Install Dependencies**

```bash
pip install openai  # OpenRouter uses OpenAI-compatible API
pip install requests
pip install asyncio aiohttp  # For async processing
```
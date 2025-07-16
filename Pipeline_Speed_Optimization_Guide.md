# 🚀 FOODB Pipeline Speed Optimization Guide

## 🎯 **Immediate High-Impact Optimizations (2-5x speedup)**

### 1. **LLM Model Optimization** (Biggest Impact: 3-5x speedup)

**Current Bottleneck**: Gemma-3-27B is extremely slow (27B parameters)

**Solutions**:

#### Option A: Smaller Gemma Model (Recommended)
```python
# Replace in 5_LLM_Simple_Sentence_gen.py and simple_sentenceRE3.py
MODEL_NAME = "google/gemma-2-9b-it"  # Instead of gemma-3-27b
BATCH_SIZE = 16  # Increase from 1-4
QUANTIZATION = "4bit"  # Reduce memory usage
```

#### Option B: Llama 3.1 8B (Best quality/speed balance)
```python
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BATCH_SIZE = 24
QUANTIZATION = "4bit"
```

#### Option C: vLLM Integration (2-5x inference speedup)
```bash
pip install vllm
```
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-2-9b-it",
    tensor_parallel_size=1,
    max_model_len=2048,
    gpu_memory_utilization=0.9
)

# Process in large batches instead of one-by-one
responses = llm.generate(batch_prompts, sampling_params)
```

### 2. **API Caching** (2-10x speedup for repeated runs)

**Current Issue**: Redundant API calls to PubChem/PubMed

**Solution**: Implement Redis or file-based caching

```python
# Add to compound_normalizer.py
import pickle
import os

class CachedPubChemAPI:
    def __init__(self, cache_file="pubchem_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def get_synonyms(self, compound_name):
        if compound_name in self.cache:
            return self.cache[compound_name]

        # Original API call
        synonyms = self.fetch_from_api(compound_name)
        self.cache[compound_name] = synonyms
        self.save_cache()
        return synonyms
```

### 3. **Increased Batch Processing** (2-3x speedup)

**Current Issue**: Small batch sizes in LLM processing

**Solution**: Optimize batch sizes per GPU memory

```python
# Memory-based batch size calculation
def calculate_optimal_batch_size(model_size_gb, available_gpu_gb):
    if model_size_gb <= 8:  # 8B model
        return min(32, available_gpu_gb // 2)
    elif model_size_gb <= 16:  # 16B model
        return min(16, available_gpu_gb // 3)
    else:
        return min(8, available_gpu_gb // 4)

# Dynamic batch sizing
BATCH_SIZE = calculate_optimal_batch_size(9, 24)  # For 24GB GPU
```

### 4. **Parallel Compound Processing** (2-4x speedup)

**Current Issue**: Sequential processing of compounds

**Solution**: Process multiple compounds simultaneously

```python
# Modify pubmed_searcher.py
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_compounds_parallel(compounds, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for compound in compounds:
            task = executor.submit(process_single_compound, compound)
            tasks.append(task)

        results = []
        for task in tasks:
            results.append(task.result())
        return results
```

## 🔧 **Medium-Impact Optimizations (3-8x speedup)**

### 5. **Pipeline Parallelism** (2-3x overall speedup)

**Current Issue**: Each step waits for previous to complete entirely

**Solution**: Start next step while previous is finishing

```python
# Pipeline coordinator
import threading
from queue import Queue

class PipelineCoordinator:
    def __init__(self):
        self.queues = {
            'xml_to_chunk': Queue(maxsize=100),
            'chunk_to_dedupe': Queue(maxsize=100),
            'dedupe_to_llm': Queue(maxsize=50)
        }

    def start_parallel_pipeline(self):
        # Start all stages simultaneously
        threading.Thread(target=self.xml_processor).start()
        threading.Thread(target=self.deduplicator).start()
        threading.Thread(target=self.llm_processor).start()
```

### 6. **Streaming Data Processing** (Reduces memory bottlenecks)

**Current Issue**: Loading entire datasets into memory

**Solution**: Stream processing with generators

```python
def stream_jsonl_processing(input_file, output_file, process_func):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            processed = process_func(data)
            outfile.write(json.dumps(processed) + '\n')
            outfile.flush()  # Immediate write
```

### 7. **Multi-GPU Processing** (Linear speedup with GPU count)

**Solution**: Distribute LLM work across multiple GPUs

```python
# For multiple GPUs
import torch.distributed as dist

def setup_multi_gpu_processing():
    if torch.cuda.device_count() > 1:
        # Use DataParallel or DistributedDataParallel
        model = torch.nn.DataParallel(model)
        batch_size *= torch.cuda.device_count()
```

## ⚡ **Advanced Optimizations (5-20x speedup)**

### 8. **API-Based LLM Services** (10-50x speedup)

**Trade-off**: Cost vs Speed

```python
# Replace local inference with API calls
import openai

def process_with_api_llm(texts, model="gpt-4o-mini"):
    # Much faster than local inference
    responses = []
    for batch in batch_texts(texts, 20):  # API batch processing
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1
        )
        responses.extend(response.choices)
    return responses
```

### 9. **Optimized File Formats** (2-3x I/O speedup)

**Current Issue**: JSON/CSV are slow for large datasets

**Solution**: Use Parquet or Arrow formats

```python
import pandas as pd
import pyarrow as pa

# Replace CSV with Parquet
df.to_parquet('compounds.parquet', compression='snappy')
df = pd.read_parquet('compounds.parquet')

# 2-5x faster than CSV for large files
```

### 10. **Async API Processing** (3-5x API throughput)

```python
import aiohttp
import asyncio

async def fetch_multiple_papers(session, paper_ids):
    tasks = []
    for paper_id in paper_ids:
        task = fetch_single_paper(session, paper_id)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Process 50-100 papers simultaneously instead of sequentially
```

## 📊 **Expected Performance Improvements**

| Optimization | Estimated Speedup | Implementation Effort | Risk Level |
|--------------|-------------------|----------------------|------------|
| Smaller Model (Gemma-2-9B) | 3-4x | Low | Low |
| vLLM Integration | 2-5x | Medium | Low |
| API Caching | 2-10x | Low | Low |
| Increased Batch Size | 2-3x | Low | Low |
| Pipeline Parallelism | 2-3x | Medium | Medium |
| Multi-GPU | 2-4x | Medium | Medium |
| API-based LLMs | 10-50x | Low | Medium (cost) |
| Async Processing | 3-5x | Medium | Low |

## 🎯 **Recommended Implementation Order**

### Phase 1: Quick Wins (1-2 days)
1. Switch to Gemma-2-9B with 4-bit quantization
2. Implement API result caching
3. Increase batch sizes
4. Add parallel compound processing

**Expected Combined Speedup**: 5-10x

### Phase 2: Architecture Improvements (1-2 weeks)
1. Integrate vLLM for inference
2. Implement pipeline parallelism
3. Add streaming data processing
4. Optimize file formats

**Expected Combined Speedup**: 10-20x

### Phase 3: Advanced Optimizations (2-4 weeks)
1. Multi-GPU processing
2. API-based LLM integration
3. Distributed processing
4. Custom CUDA optimizations

**Expected Combined Speedup**: 20-50x

## 🔍 **Quality Preservation Strategies**

### A/B Testing Framework
```python
def quality_comparison_test(original_output, optimized_output):
    # Compare triple extraction accuracy
    # Measure entity recognition precision/recall
    # Validate relationship classification
    pass
```

### Fallback Mechanisms
```python
def adaptive_processing(text, difficulty_threshold=0.8):
    if complexity_score(text) > difficulty_threshold:
        return process_with_large_model(text)  # Fallback to Gemma-27B
    else:
        return process_with_fast_model(text)   # Use Gemma-9B
```

### Quality Checkpoints
- Validate output quality at each optimization step
- Maintain accuracy metrics dashboard
- Implement automatic quality regression detection

## 🚀 **Implementation Priority**

**Start with these for immediate 5-10x speedup**:
1. Model downgrade to Gemma-2-9B + 4-bit quantization
2. API result caching
3. Increased batch processing
4. vLLM integration

These changes require minimal code modification but provide substantial performance improvements while maintaining output quality.
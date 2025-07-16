# Model Optimization Guide for FOODB Pipeline
# Recommended model replacements for speed without quality loss

# Option 1: Smaller Gemma Model (3-4x faster)
FAST_MODEL_CONFIG = {
    "model_name": "google/gemma-2-9b-it",  # Instead of gemma-3-27b
    "quantization": "4bit",  # Reduces memory and increases speed
    "batch_size": 16,  # Increase from current 1-4
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16"
}

# Option 2: Llama 3.1 8B (Similar quality, much faster)
LLAMA_CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "quantization": "4bit",
    "batch_size": 24,
    "max_seq_length": 2048
}

# Option 3: vLLM Integration (2-5x inference speedup)
VLLM_CONFIG = {
    "model": "google/gemma-2-9b-it",
    "tensor_parallel_size": 1,  # Use multiple GPUs if available
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.9,
    "batch_size": 32  # Much larger batches possible
}

# Implementation example for vLLM
"""
from vllm import LLM, SamplingParams

# Replace current model loading with:
llm = LLM(
    model="google/gemma-2-9b-it",
    tensor_parallel_size=1,
    max_model_len=2048,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=512,
    stop=["</s>"]
)

# Batch processing instead of one-by-one
responses = llm.generate(batch_prompts, sampling_params)
"""
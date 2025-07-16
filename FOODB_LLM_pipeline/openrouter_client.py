"""
OpenRouter API Client for FOODB LLM Pipeline
Replaces local Gemma model with fast API-based inference
"""

import openai
import asyncio
import aiohttp
import json
from typing import List, Dict, Any
import time
import os

class OpenRouterClient:
    def __init__(self, api_key: str = None, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

        # Configure OpenAI client for OpenRouter
        openai.api_key = self.api_key
        openai.api_base = self.base_url

    def generate_single(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate single response using OpenRouter API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                headers={
                    "HTTP-Referer": "https://github.com/foodb-pipeline",
                    "X-Title": "FOODB LLM Pipeline"
                }
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in OpenRouter API call: {e}")
            return ""

    async def generate_batch_async(self, prompts: List[str], max_tokens: int = 512,
                                 temperature: float = 0.1, max_concurrent: int = 10) -> List[str]:
        """Generate multiple responses concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_one(prompt):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/foodb-pipeline",
                        "X-Title": "FOODB LLM Pipeline"
                    }

                    data = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }

                    try:
                        async with session.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=data
                        ) as response:
                            result = await response.json()
                            if 'choices' in result and len(result['choices']) > 0:
                                return result['choices'][0]['message']['content'].strip()
                            else:
                                print(f"Unexpected response format: {result}")
                                return ""
                    except Exception as e:
                        print(f"Error in async API call: {e}")
                        return ""

        tasks = [generate_one(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Exception in batch processing: {result}")
                clean_results.append("")
            else:
                clean_results.append(result)

        return clean_results

    def generate_batch_sync(self, prompts: List[str], max_tokens: int = 512,
                           temperature: float = 0.1, max_concurrent: int = 10) -> List[str]:
        """Synchronous wrapper for batch processing"""
        return asyncio.run(self.generate_batch_async(prompts, max_tokens, temperature, max_concurrent))

# Available free models on OpenRouter
FREE_MODELS = {
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
    "gemma-2-9b": "google/gemma-2-9b-it:free",
    "mistral-7b": "mistralai/mistral-7b-instruct:free",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct:free"
}

def get_recommended_model(task_type: str = "general") -> str:
    """Get recommended model based on task type"""
    if task_type == "scientific":
        return FREE_MODELS["llama-3.1-8b"]  # Best for scientific text
    elif task_type == "fast":
        return FREE_MODELS["mistral-7b"]    # Fastest
    elif task_type == "reasoning":
        return FREE_MODELS["qwen-2.5-7b"]   # Best reasoning
    else:
        return FREE_MODELS["llama-3.1-8b"]  # Default

# Usage example
if __name__ == "__main__":
    client = OpenRouterClient(
        model="meta-llama/llama-3.1-8b-instruct:free"
    )

    # Single generation
    response = client.generate_single("Explain photosynthesis in simple terms.")
    print("Single response:", response)

    # Batch generation
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "What are neural networks?"
    ]
    responses = client.generate_batch_sync(prompts, max_concurrent=5)
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}")
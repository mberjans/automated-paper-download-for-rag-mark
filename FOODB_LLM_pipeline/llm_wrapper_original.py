"""
FOODB LLM Pipeline Wrapper for Testing with API-based Models
This wrapper provides a unified interface for testing the FOODB pipeline
with various API-based LLM providers instead of local models.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual loading if python-dotenv is not available
    from pathlib import Path
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

# Import API clients
try:
    from openrouter_client import OpenRouterClient, FREE_MODELS
except ImportError:
    print("Warning: openrouter_client not found. OpenRouter functionality will be limited.")
    OpenRouterClient = None
    FREE_MODELS = {}

# Import other API clients for different providers
import requests


class ModelProvider(Enum):
    """Supported model providers"""
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    NVIDIA = "nvidia"
    GROQ = "groq"
    PERPLEXITY = "perplexity"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_id: str
    api_key_env: str
    api_url: str
    max_tokens: int = 512
    temperature: float = 0.1
    context_length: int = 8192


class LLMWrapper:
    """
    Unified wrapper for FOODB LLM Pipeline testing with API-based models
    """
    
    def __init__(self, config_file: str = "free_models_reasoning_ranked.json"):
        """
        Initialize the LLM wrapper
        
        Args:
            config_file: Path to the model configuration file
        """
        self.logger = self._setup_logging()
        self.models_config = self._load_model_configs(config_file)
        self.current_model = None
        self.current_client = None
        
        # Load default model (highest ranked)
        if self.models_config:
            self.set_model(self.models_config[0])
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the wrapper"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_model_configs(self, config_file: str) -> List[Dict]:
        """Load model configurations from JSON file"""
        try:
            with open(config_file, 'r') as f:
                configs = json.load(f)
            self.logger.info(f"Loaded {len(configs)} model configurations")
            return configs
        except FileNotFoundError:
            self.logger.error(f"Config file {config_file} not found")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing config file: {e}")
            return []
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with their configurations"""
        return self.models_config
    
    def get_model_by_provider(self, provider: str) -> List[Dict]:
        """Get models filtered by provider"""
        return [model for model in self.models_config 
                if model.get('provider', '').lower() == provider.lower()]
    
    def get_top_models(self, n: int = 5) -> List[Dict]:
        """Get top N models by ranking"""
        return self.models_config[:n]
    
    def set_model(self, model_config: Dict) -> bool:
        """
        Set the current model for generation
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            bool: True if model was set successfully
        """
        try:
            provider = model_config.get('provider', '').lower()
            api_key_env = model_config.get('api_key_env')
            api_key = os.getenv(api_key_env)
            
            if not api_key:
                self.logger.error(f"API key not found for {api_key_env}")
                return False
            
            # Initialize the appropriate client
            if provider == 'cerebras':
                self.current_client = self._init_cerebras_client(model_config, api_key)
            elif provider == 'openrouter':
                self.current_client = self._init_openrouter_client(model_config, api_key)
            elif provider == 'nvidia':
                self.current_client = self._init_nvidia_client(model_config, api_key)
            else:
                self.logger.error(f"Unsupported provider: {provider}")
                return False
            
            self.current_model = model_config
            self.logger.info(f"Set model: {model_config.get('model_name', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting model: {e}")
            return False
    
    def _init_cerebras_client(self, model_config: Dict, api_key: str):
        """Initialize Cerebras API client"""
        return CerebrasClient(
            api_key=api_key,
            model_id=model_config['model_id'],
            api_url=model_config['api_url']
        )
    
    def _init_openrouter_client(self, model_config: Dict, api_key: str):
        """Initialize OpenRouter API client"""
        if OpenRouterClient:
            return OpenRouterClient(
                api_key=api_key,
                model=model_config['model_id']
            )
        else:
            raise ImportError("OpenRouter client not available")
    
    def _init_nvidia_client(self, model_config: Dict, api_key: str):
        """Initialize NVIDIA API client"""
        return NvidiaClient(
            api_key=api_key,
            model_id=model_config['model_id'],
            api_url=model_config['api_url']
        )
    
    def generate_single(self, prompt: str, max_tokens: int = None, 
                       temperature: float = None) -> str:
        """
        Generate a single response
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.current_client:
            raise ValueError("No model set. Call set_model() first.")
        
        max_tokens = max_tokens or self.current_model.get('max_tokens', 512)
        temperature = temperature or self.current_model.get('temperature', 0.1)
        
        try:
            return self.current_client.generate_single(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            return ""
    
    def generate_batch(self, prompts: List[str], max_tokens: int = None,
                      temperature: float = None, max_concurrent: int = 10) -> List[str]:
        """
        Generate multiple responses in batch
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of generated texts
        """
        if not self.current_client:
            raise ValueError("No model set. Call set_model() first.")
        
        max_tokens = max_tokens or self.current_model.get('max_tokens', 512)
        temperature = temperature or self.current_model.get('temperature', 0.1)
        
        try:
            if hasattr(self.current_client, 'generate_batch_sync'):
                return self.current_client.generate_batch_sync(
                    prompts=prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_concurrent=max_concurrent
                )
            else:
                # Fallback to sequential generation
                return [self.generate_single(prompt, max_tokens, temperature) 
                       for prompt in prompts]
        except Exception as e:
            self.logger.error(f"Error in batch generation: {e}")
            return [""] * len(prompts)
    
    def test_model_performance(self, test_prompts: List[str] = None) -> Dict:
        """
        Test the current model's performance
        
        Args:
            test_prompts: List of test prompts
            
        Returns:
            Performance metrics
        """
        if not test_prompts:
            test_prompts = [
                "Convert this to simple sentences: Resveratrol is a polyphenolic compound found in red wine.",
                "Extract entities from: Green tea contains EGCG which has antioxidant properties.",
                "Classify this triple: [\"Curcumin\", \"reduces\", \"inflammation\"]"
            ]
        
        start_time = time.time()
        responses = self.generate_batch(test_prompts)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_request = total_time / len(test_prompts)
        
        return {
            "model": self.current_model.get('model_name', 'Unknown'),
            "total_time": total_time,
            "avg_time_per_request": avg_time_per_request,
            "requests_per_second": len(test_prompts) / total_time,
            "successful_responses": sum(1 for r in responses if r.strip()),
            "success_rate": sum(1 for r in responses if r.strip()) / len(responses)
        }


class CerebrasClient:
    """Client for Cerebras API"""
    
    def __init__(self, api_key: str, model_id: str, api_url: str):
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = api_url
    
    def generate_single(self, prompt: str, max_tokens: int = 512, 
                       temperature: float = 0.1) -> str:
        """Generate single response using Cerebras API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error in Cerebras API call: {e}")
            return ""


class NvidiaClient:
    """Client for NVIDIA API"""
    
    def __init__(self, api_key: str, model_id: str, api_url: str):
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = api_url
    
    def generate_single(self, prompt: str, max_tokens: int = 512, 
                       temperature: float = 0.1) -> str:
        """Generate single response using NVIDIA API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error in NVIDIA API call: {e}")
            return ""


# Convenience functions for easy testing
def create_test_wrapper() -> LLMWrapper:
    """Create a wrapper instance for testing"""
    return LLMWrapper()


def quick_test(prompt: str = "Explain photosynthesis in simple terms.") -> str:
    """Quick test function"""
    wrapper = create_test_wrapper()
    return wrapper.generate_single(prompt)


if __name__ == "__main__":
    # Example usage
    wrapper = LLMWrapper()
    
    print("Available models:")
    for i, model in enumerate(wrapper.get_top_models()):
        print(f"{i+1}. {model.get('model_name')} ({model.get('provider')})")
    
    # Test generation
    test_prompt = "Convert this to simple sentences: Resveratrol is a polyphenolic compound found in red wine that exhibits antioxidant properties."
    response = wrapper.generate_single(test_prompt)
    print(f"\nTest response: {response}")
    
    # Performance test
    performance = wrapper.test_model_performance()
    print(f"\nPerformance metrics: {performance}")

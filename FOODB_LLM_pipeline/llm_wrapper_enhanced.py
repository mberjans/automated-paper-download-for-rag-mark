"""
FOODB LLM Pipeline Wrapper with Enhanced Fallback System
This wrapper provides rate limiting resilience and provider fallback for the FOODB pipeline
"""

import os
import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file only
def load_env_file():
    """Load environment variables from .env file only (not global environment)"""
    env_vars = {}

    # Look for .env file in current directory and parent directory
    env_paths = ['.env', '../.env']

    for env_path in env_paths:
        try:
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
                print(f"✅ Loaded API keys from {env_path}")
                return env_vars
        except Exception as e:
            print(f"❌ Error loading {env_path}: {e}")
            continue

    print("❌ No .env file found in current or parent directory")
    return env_vars

# Load environment variables
ENV_VARS = load_env_file()

class ProviderStatus(Enum):
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"

@dataclass
class RetryConfig:
    max_attempts: int = 5  # Increased to 5 attempts for better rate limit handling
    base_delay: float = 2.0  # Start with 2 seconds
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class ProviderHealth:
    status: ProviderStatus
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    rate_limit_reset_time: Optional[float] = None

class LLMWrapper:
    """Enhanced LLM Wrapper with fallback capabilities for FOODB pipeline"""
    
    def __init__(self, retry_config: RetryConfig = None, document_only_mode: bool = False, groq_model: str = None):
        self.logger = logging.getLogger(__name__)
        self.retry_config = retry_config or RetryConfig()
        self.document_only_mode = document_only_mode
        self.preferred_groq_model = groq_model  # Allow custom Groq model selection
        
        # Load API keys from .env file only
        self.api_keys = {
            'cerebras': ENV_VARS.get('CEREBRAS_API_KEY'),
            'groq': ENV_VARS.get('GROQ_API_KEY'),
            'openrouter': ENV_VARS.get('OPENROUTER_API_KEY')
        }
        
        # Provider management
        self.provider_health = {}
        self.fallback_order = ["cerebras", "groq", "openrouter"]
        
        # Initialize providers
        self._initialize_providers()
        
        # Set current model info for compatibility
        self.current_model = {
            'model_name': f'{self.current_provider}_model',
            'provider': self.current_provider
        }
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'fallback_switches': 0
        }
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        for provider in self.fallback_order:
            self.provider_health[provider] = ProviderHealth(ProviderStatus.HEALTHY)
        
        self.current_provider = self._get_best_available_provider()
        print(f"🎯 Primary provider: {self.current_provider}")
    
    def _get_best_available_provider(self) -> Optional[str]:
        """Get the best available provider"""
        for provider in self.fallback_order:
            health = self.provider_health.get(provider)
            if health and health.status == ProviderStatus.HEALTHY:
                if self.api_keys.get(provider):
                    return provider
        return None
    
    def _update_provider_health(self, provider: str, success: bool, is_rate_limit: bool = False):
        """Update provider health status"""
        current_time = time.time()
        health = self.provider_health[provider]
        
        if success:
            health.status = ProviderStatus.HEALTHY
            health.last_success = current_time
            health.consecutive_failures = 0
            health.rate_limit_reset_time = None
        else:
            health.last_failure = current_time
            health.consecutive_failures += 1
            
            if is_rate_limit:
                health.status = ProviderStatus.RATE_LIMITED
                health.rate_limit_reset_time = current_time + 60
                print(f"⚠️ {provider} rate limited, switching providers...")
            else:
                health.status = ProviderStatus.FAILED
    
    def _switch_provider(self) -> bool:
        """Switch to the next best available provider"""
        old_provider = self.current_provider
        new_provider = self._get_best_available_provider()

        if new_provider and new_provider != old_provider:
            self.current_provider = new_provider
            self.stats['fallback_switches'] += 1
            print(f"🔄 Switched provider: {old_provider} → {new_provider}")
            return True
        return False

    def _get_best_groq_model(self) -> str:
        """Get the best available Groq model based on testing results"""
        # If user specified a preferred model, use it
        if self.preferred_groq_model:
            return self.preferred_groq_model

        # Ranked by performance (speed and accuracy) from testing
        groq_models = [
            "llama-3.3-70b-versatile",              # Latest Llama 3.3 model
            "moonshotai/kimi-k2-instruct",           # Fastest (0.57s) with excellent extraction (13 compounds)
            "meta-llama/llama-4-scout-17b-16e-instruct",  # Fast (0.88s) with excellent extraction (13 compounds)
            "meta-llama/llama-4-maverick-17b-128e-instruct",  # Good (1.37s) with excellent extraction (13 compounds)
            "llama-3.1-8b-instant",                 # Original fallback model
            "qwen/qwen3-32b"                        # Slower but functional (1.34s, some parsing issues)
        ]

        # Return the first model (best performing)
        return groq_models[0]

    def get_available_groq_models(self) -> List[str]:
        """Get list of all available Groq models for metabolite extraction"""
        return [
            "moonshotai/kimi-k2-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b"
        ]
    
    def _make_api_request(self, provider: str, prompt: str, max_tokens: int = 500) -> tuple:
        """Make API request to specific provider"""
        try:
            if provider == "cerebras":
                return self._cerebras_request(prompt, max_tokens)
            elif provider == "groq":
                return self._groq_request(prompt, max_tokens)
            elif provider == "openrouter":
                return self._openrouter_request(prompt, max_tokens)
            else:
                return "", False, False
        except Exception as e:
            print(f"❌ Error in {provider} API request: {e}")
            return "", False, False
    
    def _cerebras_request(self, prompt: str, max_tokens: int) -> tuple:
        """Make request to Cerebras API"""
        api_key = self.api_keys.get('cerebras')
        if not api_key:
            return "", False, False
        
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return "", False, True  # Rate limited
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), True, False
    
    def _groq_request(self, prompt: str, max_tokens: int) -> tuple:
        """Make request to Groq API"""
        api_key = self.api_keys.get('groq')
        if not api_key:
            return "", False, False
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Select best available Groq model
        groq_model = self._get_best_groq_model()

        data = {
            "model": groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return "", False, True  # Rate limited
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), True, False
    
    def _openrouter_request(self, prompt: str, max_tokens: int) -> tuple:
        """Make request to OpenRouter API"""
        api_key = self.api_keys.get('openrouter')
        if not api_key:
            return "", False, False
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/foodb-pipeline",
            "X-Title": "FOODB Pipeline"
        }
        data = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return "", False, True  # Rate limited
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), True, False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff"""
        import random
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def generate_single(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate single response with fallback (compatible with original interface)"""
        return self.generate_single_with_fallback(prompt, max_tokens)
    
    def generate_single_with_fallback(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate single response with exponential backoff BEFORE provider switching"""
        self.stats['total_requests'] += 1

        for attempt in range(self.retry_config.max_attempts):
            if not self.current_provider:
                break

            # Make API request
            response, success, is_rate_limit = self._make_api_request(
                self.current_provider, prompt, max_tokens
            )

            # Update provider health
            self._update_provider_health(self.current_provider, success, is_rate_limit)

            if success:
                self.stats['successful_requests'] += 1
                return response

            # Handle rate limiting with exponential backoff FIRST
            if is_rate_limit:
                self.stats['rate_limited_requests'] += 1

                # Try exponential backoff with same provider BEFORE switching
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    print(f"⚠️ {self.current_provider} rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_config.max_attempts})...")
                    time.sleep(delay)
                    continue

                # Only switch provider after exhausting all retry attempts
                print(f"🔄 {self.current_provider} exhausted all {self.retry_config.max_attempts} retry attempts, switching providers...")
                if self._switch_provider():
                    # Reset attempt counter for new provider
                    for new_attempt in range(self.retry_config.max_attempts):
                        response, success, is_rate_limit = self._make_api_request(
                            self.current_provider, prompt, max_tokens
                        )
                        self._update_provider_health(self.current_provider, success, is_rate_limit)

                        if success:
                            self.stats['successful_requests'] += 1
                            return response

                        if is_rate_limit and new_attempt < self.retry_config.max_attempts - 1:
                            delay = self._calculate_delay(new_attempt)
                            print(f"⚠️ {self.current_provider} rate limited, retrying in {delay:.1f}s (attempt {new_attempt + 1}/{self.retry_config.max_attempts})...")
                            time.sleep(delay)
                        elif not is_rate_limit:
                            break  # Non-rate-limit error, try next provider

                    # If new provider also failed, try next one
                    continue

            else:
                # Non-rate-limit error, try exponential backoff first
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    print(f"❌ {self.current_provider} error, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_config.max_attempts})...")
                    time.sleep(delay)
                    continue

                # After exhausting retries, try switching provider
                print(f"🔄 {self.current_provider} exhausted all retry attempts, switching providers...")
                if not self._switch_provider():
                    break

        self.stats['failed_requests'] += 1
        return ""
    
    def generate_batch(self, prompts: List[str], max_tokens: int = None, 
                      temperature: float = None, max_concurrent: int = 10) -> List[str]:
        """Generate batch responses (compatible with original interface)"""
        return [self.generate_single(prompt, max_tokens or 500) for prompt in prompts]
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total = self.stats['total_requests']
        return {
            **self.stats,
            'success_rate': self.stats['successful_requests'] / total if total > 0 else 0,
            'failure_rate': self.stats['failed_requests'] / total if total > 0 else 0
        }

    def extract_metabolites_document_only(self, text_chunk: str, max_tokens: int = 200) -> str:
        """Extract metabolites using strict document-only prompt to prevent training data contamination"""

        prompt = f"""EXTRACT COMPOUNDS FROM TEXT ONLY

TASK: Find and list all chemical compound names that are explicitly mentioned in this text.

TEXT TO ANALYZE:
{text_chunk}

INSTRUCTIONS:
- Look for specific chemical compound names in the text
- Include any compound with a chemical name (e.g., "malvidin-3-glucoside", "caffeic acid")
- Include metabolites, biomarkers, and chemical substances mentioned by name
- DO NOT add compounds not mentioned in the text
- DO NOT use your general knowledge about wine or metabolites
- List each compound on a separate line
- If no compounds are mentioned, write "No compounds found"

RESPONSE FORMAT:
[compound name 1]
[compound name 2]
[etc.]"""

        return self.generate_single_with_fallback(prompt, max_tokens)

    def verify_compounds_in_text(self, text_chunk: str, compounds: List[str], max_tokens: int = 300) -> str:
        """Verify which compounds are actually mentioned in the text to eliminate training data contamination"""

        compounds_list = "\n".join(compounds)

        prompt = f"""COMPOUND VERIFICATION TASK

STRICT RULES:
1. Only mark as "FOUND" if the exact compound name appears in the text
2. Partial matches or similar compounds should be marked "NOT_FOUND"
3. Do not use your training knowledge - only what's in the text
4. Be extremely conservative in your verification

TEXT CHUNK:
{text_chunk}

COMPOUNDS TO VERIFY:
{compounds_list}

RESPONSE FORMAT:
For each compound, respond with:
COMPOUND_NAME: FOUND/NOT_FOUND"""

        return self.generate_single_with_fallback(prompt, max_tokens)

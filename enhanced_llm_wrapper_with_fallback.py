#!/usr/bin/env python3
"""
Enhanced LLM Wrapper with Rate Limiting Fallback and Provider Switching
This enhanced wrapper implements:
1. Exponential backoff for rate limiting
2. Automatic fallback to alternative API providers
3. Configurable retry attempts
4. Provider health monitoring
"""

import time
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class ProviderStatus(Enum):
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
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

class EnhancedLLMWrapper:
    """Enhanced LLM Wrapper with fallback capabilities"""
    
    def __init__(self, config_path: str = "free_models_reasoning_ranked.json", 
                 retry_config: RetryConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.retry_config = retry_config or RetryConfig()
        
        # Load configurations
        self.model_configs = self._load_model_configs()
        self.api_keys = self._load_api_keys()
        
        # Provider management
        self.provider_health = {}
        self.current_provider = None
        self.fallback_order = ["cerebras", "groq", "openrouter"]
        
        # Initialize providers
        self._initialize_providers()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'fallback_switches': 0,
            'retry_attempts': 0
        }
    
    def _load_model_configs(self) -> List[Dict]:
        """Load model configurations"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load model configs: {e}")
            return []
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from .env file only (not global environment)"""
        import os

        # Load API keys from .env file only
        env_vars = {}

        try:
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
                print("✅ Loaded API keys from .env file")
            else:
                print("❌ No .env file found")
        except Exception as e:
            print(f"❌ Error loading .env file: {e}")

        return {
            'cerebras': env_vars.get('CEREBRAS_API_KEY'),
            'groq': env_vars.get('GROQ_API_KEY'),
            'openrouter': env_vars.get('OPENROUTER_API_KEY')
        }
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        for provider in self.fallback_order:
            self.provider_health[provider] = ProviderHealth(ProviderStatus.HEALTHY)
        
        # Set initial provider
        self.current_provider = self._get_best_available_provider()
        self.logger.info(f"Initialized with provider: {self.current_provider}")
    
    def _get_best_available_provider(self) -> Optional[str]:
        """Get the best available provider based on health status"""
        for provider in self.fallback_order:
            health = self.provider_health.get(provider)
            if health and health.status == ProviderStatus.HEALTHY:
                if self.api_keys.get(provider):
                    return provider
        
        # If no healthy providers, try rate-limited ones that might have recovered
        current_time = time.time()
        for provider in self.fallback_order:
            health = self.provider_health.get(provider)
            if (health and health.status == ProviderStatus.RATE_LIMITED and
                health.rate_limit_reset_time and 
                current_time > health.rate_limit_reset_time):
                health.status = ProviderStatus.HEALTHY
                return provider
        
        return None
    
    def _update_provider_health(self, provider: str, success: bool, 
                              is_rate_limit: bool = False):
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
                # Estimate rate limit reset time (typically 1 minute for most APIs)
                health.rate_limit_reset_time = current_time + 60
                self.logger.warning(f"Provider {provider} rate limited, estimated reset: {health.rate_limit_reset_time}")
            else:
                health.status = ProviderStatus.FAILED
                
            # Mark as unavailable after too many consecutive failures
            if health.consecutive_failures >= 5:
                health.status = ProviderStatus.UNAVAILABLE
                self.logger.error(f"Provider {provider} marked as unavailable after {health.consecutive_failures} failures")
    
    def _switch_provider(self) -> bool:
        """Switch to the next best available provider"""
        old_provider = self.current_provider
        new_provider = self._get_best_available_provider()
        
        if new_provider and new_provider != old_provider:
            self.current_provider = new_provider
            self.stats['fallback_switches'] += 1
            self.logger.info(f"Switched provider: {old_provider} → {new_provider}")
            return True
        
        return False
    
    def _make_api_request(self, provider: str, prompt: str, max_tokens: int = 500) -> Tuple[str, bool, bool]:
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
            self.logger.error(f"Error in {provider} API request: {e}")
            return "", False, False
    
    def _cerebras_request(self, prompt: str, max_tokens: int) -> Tuple[str, bool, bool]:
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
    
    def _groq_request(self, prompt: str, max_tokens: int) -> Tuple[str, bool, bool]:
        """Make request to Groq API"""
        api_key = self.api_keys.get('groq')
        if not api_key:
            return "", False, False
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-8b-instant",
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
    
    def _openrouter_request(self, prompt: str, max_tokens: int) -> Tuple[str, bool, bool]:
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
        """Calculate delay for exponential backoff with jitter"""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    def generate_single_with_fallback(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate single response with fallback and retry logic"""
        self.stats['total_requests'] += 1
        
        for attempt in range(self.retry_config.max_attempts):
            if not self.current_provider:
                self.logger.error("No available providers")
                break
            
            self.stats['retry_attempts'] += 1 if attempt > 0 else 0
            
            # Make API request
            response, success, is_rate_limit = self._make_api_request(
                self.current_provider, prompt, max_tokens
            )
            
            # Update provider health
            self._update_provider_health(self.current_provider, success, is_rate_limit)
            
            if success:
                self.stats['successful_requests'] += 1
                return response
            
            # Handle rate limiting
            if is_rate_limit:
                self.stats['rate_limited_requests'] += 1
                self.logger.warning(f"Rate limited on {self.current_provider}, attempt {attempt + 1}")
                
                # Try to switch provider immediately on rate limit
                if self._switch_provider():
                    continue  # Try with new provider immediately
                
                # If no alternative provider, wait with exponential backoff
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
            else:
                # Non-rate-limit error, try switching provider
                self.logger.error(f"API error on {self.current_provider}, attempt {attempt + 1}")
                if not self._switch_provider():
                    # No alternative provider available
                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        self.logger.info(f"Waiting {delay:.2f}s before retry...")
                        time.sleep(delay)
        
        self.stats['failed_requests'] += 1
        self.logger.error(f"All retry attempts failed for prompt: {prompt[:100]}...")
        return ""
    
    def get_provider_status(self) -> Dict:
        """Get current status of all providers"""
        status = {
            'current_provider': self.current_provider,
            'providers': {}
        }
        
        for provider, health in self.provider_health.items():
            status['providers'][provider] = {
                'status': health.status.value,
                'has_api_key': bool(self.api_keys.get(provider)),
                'consecutive_failures': health.consecutive_failures,
                'last_success': health.last_success,
                'last_failure': health.last_failure,
                'rate_limit_reset_time': health.rate_limit_reset_time
            }
        
        return status
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total = self.stats['total_requests']
        return {
            **self.stats,
            'success_rate': self.stats['successful_requests'] / total if total > 0 else 0,
            'failure_rate': self.stats['failed_requests'] / total if total > 0 else 0,
            'rate_limit_rate': self.stats['rate_limited_requests'] / total if total > 0 else 0
        }
    
    def reset_provider_health(self, provider: str = None):
        """Reset health status for a provider or all providers"""
        if provider:
            if provider in self.provider_health:
                self.provider_health[provider] = ProviderHealth(ProviderStatus.HEALTHY)
                self.logger.info(f"Reset health for provider: {provider}")
        else:
            for p in self.provider_health:
                self.provider_health[p] = ProviderHealth(ProviderStatus.HEALTHY)
            self.logger.info("Reset health for all providers")
            
        # Update current provider
        self.current_provider = self._get_best_available_provider()

# Convenience function for easy integration
def create_enhanced_wrapper(max_attempts: int = 3, base_delay: float = 1.0) -> EnhancedLLMWrapper:
    """Create enhanced wrapper with custom retry configuration"""
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay
    )
    return EnhancedLLMWrapper(retry_config=retry_config)

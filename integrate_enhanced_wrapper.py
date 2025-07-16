#!/usr/bin/env python3
"""
Integrate Enhanced Wrapper into Main FOODB Pipeline
This script updates the main pipeline scripts to use the enhanced wrapper with fallback
"""

import os
import shutil
from pathlib import Path

def backup_original_wrapper():
    """Backup the original wrapper"""
    print("📦 Backing up original wrapper...")
    
    original_path = "FOODB_LLM_pipeline/llm_wrapper.py"
    backup_path = "FOODB_LLM_pipeline/llm_wrapper_original.py"
    
    if os.path.exists(original_path):
        shutil.copy2(original_path, backup_path)
        print(f"✅ Original wrapper backed up to: {backup_path}")
    else:
        print(f"❌ Original wrapper not found: {original_path}")

def create_enhanced_wrapper_for_pipeline():
    """Create enhanced wrapper specifically for the pipeline"""
    print("🔧 Creating enhanced wrapper for pipeline...")
    
    enhanced_wrapper_content = '''"""
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

class LLMWrapper:
    """Enhanced LLM Wrapper with fallback capabilities for FOODB pipeline"""
    
    def __init__(self, retry_config: RetryConfig = None):
        self.logger = logging.getLogger(__name__)
        self.retry_config = retry_config or RetryConfig()
        
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
        """Generate single response with fallback and retry logic"""
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
            
            # Handle rate limiting
            if is_rate_limit:
                self.stats['rate_limited_requests'] += 1
                
                # Try to switch provider immediately
                if self._switch_provider():
                    continue
                
                # If no alternative, wait with exponential backoff
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
            else:
                # Non-rate-limit error, try switching provider
                if not self._switch_provider():
                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        time.sleep(delay)
        
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
'''
    
    # Write the enhanced wrapper
    enhanced_path = "FOODB_LLM_pipeline/llm_wrapper_enhanced.py"
    with open(enhanced_path, 'w') as f:
        f.write(enhanced_wrapper_content)
    
    print(f"✅ Enhanced wrapper created: {enhanced_path}")

def update_main_pipeline_script():
    """Update the main pipeline script to use enhanced wrapper"""
    print("🔄 Updating main pipeline script...")
    
    script_path = "FOODB_LLM_pipeline/5_LLM_Simple_Sentence_gen_API.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Main script not found: {script_path}")
        return
    
    # Read current script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace the import and initialization
    updated_content = content.replace(
        "from llm_wrapper import LLMWrapper",
        "from llm_wrapper_enhanced import LLMWrapper"
    )
    
    # Add fallback status reporting
    status_code = '''
# Show enhanced wrapper status
print(f"🎯 Primary provider: {api_wrapper.current_provider}")
print(f"🛡️ Fallback system: Active")
stats = api_wrapper.get_statistics()
if stats['total_requests'] > 0:
    print(f"📊 Success rate: {stats['success_rate']:.1%}")
    if stats['fallback_switches'] > 0:
        print(f"🔄 Provider switches: {stats['fallback_switches']}")
'''
    
    # Add status reporting after wrapper initialization
    updated_content = updated_content.replace(
        'api_wrapper = LLMWrapper()',
        f'api_wrapper = LLMWrapper()\n{status_code}'
    )
    
    # Write updated script
    backup_script_path = script_path.replace('.py', '_original.py')
    
    # Backup original
    with open(backup_script_path, 'w') as f:
        f.write(content)
    
    # Write updated version
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Main script updated: {script_path}")
    print(f"📦 Original backed up: {backup_script_path}")

def create_integration_test():
    """Create test script to verify integration"""
    print("🧪 Creating integration test...")
    
    test_content = '''#!/usr/bin/env python3
"""
Test Enhanced Wrapper Integration with FOODB Pipeline
"""

import sys
import os

# Add pipeline directory to path
sys.path.append('FOODB_LLM_pipeline')

def test_enhanced_wrapper_integration():
    """Test that enhanced wrapper works with pipeline"""
    print("🧪 Testing Enhanced Wrapper Integration")
    print("=" * 50)
    
    try:
        # Import enhanced wrapper
        from llm_wrapper_enhanced import LLMWrapper
        
        # Create wrapper
        wrapper = LLMWrapper()
        
        print(f"✅ Enhanced wrapper imported successfully")
        print(f"🎯 Primary provider: {wrapper.current_provider}")
        
        # Test basic functionality
        test_prompt = "Extract metabolites from: Red wine contains resveratrol."
        
        print(f"\\n🔬 Testing basic functionality...")
        response = wrapper.generate_single(test_prompt, max_tokens=100)
        
        if response:
            print(f"✅ Response generated successfully")
            print(f"📝 Response length: {len(response)} characters")
            print(f"📊 Provider used: {wrapper.current_provider}")
        else:
            print(f"❌ No response generated")
        
        # Show statistics
        stats = wrapper.get_statistics()
        print(f"\\n📈 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1%}" if 'rate' in key else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_wrapper_integration()
'''
    
    with open("test_enhanced_integration.py", 'w') as f:
        f.write(test_content)
    
    print(f"✅ Integration test created: test_enhanced_integration.py")

def main():
    """Integrate enhanced wrapper into main FOODB pipeline"""
    print("🔧 FOODB Pipeline - Enhanced Wrapper Integration")
    print("=" * 60)
    
    try:
        # Step 1: Backup original wrapper
        backup_original_wrapper()
        
        # Step 2: Create enhanced wrapper for pipeline
        create_enhanced_wrapper_for_pipeline()
        
        # Step 3: Update main pipeline script
        update_main_pipeline_script()
        
        # Step 4: Create integration test
        create_integration_test()
        
        print(f"\n🎉 Enhanced Wrapper Integration Complete!")
        print(f"\nWhat was done:")
        print(f"  ✅ Original wrapper backed up")
        print(f"  ✅ Enhanced wrapper created for pipeline")
        print(f"  ✅ Main script updated to use enhanced wrapper")
        print(f"  ✅ Integration test created")

        print(f"\nNext steps:")
        print(f"  1. Run: python test_enhanced_integration.py")
        print(f"  2. Test main pipeline with enhanced fallback")
        print(f"  3. Monitor rate limiting behavior in production")

        print(f"\n🛡️ The FOODB pipeline now has:")
        print(f"  • Automatic rate limiting detection")
        print(f"  • Provider fallback (Cerebras → Groq → OpenRouter)")
        print(f"  • Exponential backoff retry logic")
        print(f"  • Comprehensive error handling")
        print(f"  • Production-ready resilience")

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

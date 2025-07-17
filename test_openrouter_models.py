#!/usr/bin/env python3
"""
OpenRouter Models Functionality Test
Tests all OpenRouter models from free_models_reasoning_ranked.json
Reads API keys from .env file for security (ignores global environment)
"""

import os
import json
import time
import requests
from typing import Dict, List, Any
import logging
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file (ignore global environment)"""
    env_vars = {}
    env_file = Path('.env')

    if not env_file.exists():
        print("❌ .env file not found")
        return env_vars

    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    env_vars[key.strip()] = value

        print(f"✅ Loaded {len(env_vars)} environment variables from .env file")
        return env_vars

    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return env_vars

def setup_logging():
    """Setup logging for OpenRouter model testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('openrouter_model_testing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_openrouter_models():
    """Load OpenRouter models from the ranking file"""
    try:
        with open('free_models_reasoning_ranked.json', 'r') as f:
            models = json.load(f)
        
        openrouter_models = [
            model for model in models 
            if model.get('provider') == 'OpenRouter'
        ]
        
        return openrouter_models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return []

def test_openrouter_api_key(env_vars):
    """Test if OpenRouter API key is available and working"""
    api_key = env_vars.get('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env file")
        print("   Please add your OpenRouter API key to .env file:")
        print("   OPENROUTER_API_KEY=your-api-key-here")
        return False, None

    print(f"✅ OpenRouter API key found in .env: {api_key[:8]}...")

    # Test API key by fetching available models
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/mberjans/automated-paper-download-for-rag-mark',
            'X-Title': 'FOODB Pipeline Model Testing',
            'User-Agent': 'FOODB-Pipeline/1.0'
        }

        response = requests.get(
            'https://openrouter.ai/api/v1/models',
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get('data', [])
            print(f"✅ API key valid: Found {len(available_models)} available models")
            return True, api_key
        else:
            print(f"❌ API key test failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None

    except Exception as e:
        print(f"❌ API key test error: {e}")
        return False, None

def test_single_openrouter_model(model: Dict[str, Any], api_key: str, logger) -> Dict[str, Any]:
    """Test a single OpenRouter model"""
    model_id = model['model_id']
    model_name = model['model_name']

    print(f"\n🔬 Testing: {model_name}")
    print(f"   Model ID: {model_id}")
    
    # Test prompt for metabolite extraction
    test_prompt = """Extract metabolites and biomarkers from this text:

"Wine consumption leads to the excretion of several urinary biomarkers including resveratrol, gallic acid, and catechin. These polyphenolic compounds are metabolized and appear in urine as glucuronide and sulfate conjugates."

Please list the metabolites mentioned in a simple format."""
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/mberjans/automated-paper-download-for-rag-mark',
        'X-Title': 'FOODB Pipeline Model Testing',
        'User-Agent': 'FOODB-Pipeline/1.0'
    }
    
    payload = {
        'model': model_id,
        'messages': [
            {
                'role': 'user',
                'content': test_prompt
            }
        ],
        'max_tokens': 200,
        'temperature': 0.1
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                
                # Count metabolites mentioned in response
                metabolites_found = []
                test_metabolites = ['resveratrol', 'gallic acid', 'catechin', 'glucuronide', 'sulfate']
                
                for metabolite in test_metabolites:
                    if metabolite.lower() in content.lower():
                        metabolites_found.append(metabolite)
                
                result = {
                    'success': True,
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'content_length': len(content),
                    'metabolites_found': metabolites_found,
                    'metabolites_count': len(metabolites_found),
                    'response_preview': content[:200] + "..." if len(content) > 200 else content,
                    'usage': data.get('usage', {}),
                    'model_used': data.get('model', model_id)
                }
                
                print(f"   ✅ SUCCESS: {response_time:.2f}s")
                print(f"   📊 Metabolites found: {len(metabolites_found)}/5")
                print(f"   📝 Response length: {len(content)} chars")
                
                logger.info(f"✅ {model_name}: SUCCESS - {response_time:.2f}s, {len(metabolites_found)} metabolites")
                
                return result
            else:
                error_msg = "No choices in response"
                print(f"   ❌ FAILED: {error_msg}")
                logger.error(f"❌ {model_name}: {error_msg}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'response_data': data
                }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"   ❌ FAILED: {error_msg}")
            logger.error(f"❌ {model_name}: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'response_time': response_time,
                'status_code': response.status_code,
                'response_text': response.text
            }
            
    except requests.exceptions.Timeout:
        error_msg = "Request timeout (60s)"
        print(f"   ❌ TIMEOUT: {error_msg}")
        logger.error(f"❌ {model_name}: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'response_time': 60.0,
            'status_code': None
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ ERROR: {error_msg}")
        logger.error(f"❌ {model_name}: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'response_time': time.time() - start_time,
            'status_code': None
        }

def test_all_openrouter_models():
    """Test all OpenRouter models"""
    logger = setup_logging()

    print("🧪 OPENROUTER MODELS FUNCTIONALITY TEST")
    print("=" * 60)
    print("🔒 Using API keys from .env file (ignoring global environment)")

    # Load environment variables from .env file
    env_vars = load_env_file()
    if not env_vars:
        return

    # Check API key
    api_key_valid, api_key = test_openrouter_api_key(env_vars)
    if not api_key_valid:
        return
    
    # Load models
    models = load_openrouter_models()
    if not models:
        print("❌ No OpenRouter models found")
        return
    
    print(f"📋 Found {len(models)} OpenRouter models to test")
    
    results = {}
    successful_models = []
    failed_models = []
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"Testing model {i}/{len(models)}")
        
        try:
            result = test_single_openrouter_model(model, api_key, logger)
            results[model['model_id']] = {
                'model_info': model,
                'test_result': result
            }
            
            if result['success']:
                successful_models.append(model)
            else:
                failed_models.append(model)
                
        except Exception as e:
            logger.error(f"❌ Exception testing {model['model_name']}: {e}")
            failed_models.append(model)
            results[model['model_id']] = {
                'model_info': model,
                'test_result': {
                    'success': False,
                    'error': str(e),
                    'response_time': 0
                }
            }
        
        # Add delay between requests to avoid rate limiting
        if i < len(models):
            time.sleep(2)
    
    # Generate report
    generate_openrouter_report(results, successful_models, failed_models, logger)
    
    return results

def generate_openrouter_report(results: Dict, successful_models: List, failed_models: List, logger):
    """Generate comprehensive OpenRouter test report"""
    print(f"\n{'='*80}")
    print("📊 OPENROUTER MODELS TEST REPORT")
    print(f"{'='*80}")
    
    total_models = len(successful_models) + len(failed_models)
    success_rate = (len(successful_models) / total_models * 100) if total_models > 0 else 0
    
    print(f"✅ Successful models: {len(successful_models)}/{total_models} ({success_rate:.1f}%)")
    print(f"❌ Failed models: {len(failed_models)}/{total_models}")
    
    if successful_models:
        print(f"\n🏆 WORKING OPENROUTER MODELS")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model Name':<40} {'Response Time':<12} {'Metabolites':<12}")
        print("-" * 80)
        
        # Sort by response time
        successful_sorted = sorted(
            successful_models,
            key=lambda x: results[x['model_id']]['test_result'].get('response_time', float('inf'))
        )
        
        for i, model in enumerate(successful_sorted, 1):
            result = results[model['model_id']]['test_result']
            model_name = model['model_name'][:35]
            response_time = f"{result.get('response_time', 0):.2f}s"
            metabolites = f"{result.get('metabolites_count', 0)}/5"
            
            print(f"{i:<4} {model_name:<40} {response_time:<12} {metabolites:<12}")
        
        print(f"\n⚡ SPEED RANKING")
        print("-" * 50)
        for i, model in enumerate(successful_sorted, 1):
            result = results[model['model_id']]['test_result']
            model_name = model['model_name'][:35]
            response_time = f"{result.get('response_time', 0):.2f}s"
            print(f"{i}. {model_name}: {response_time}")
    
    if failed_models:
        print(f"\n❌ FAILED OPENROUTER MODELS")
        print("-" * 50)
        for model in failed_models:
            result = results[model['model_id']]['test_result']
            error = result.get('error', 'Unknown error')[:60]
            print(f"   • {model['model_name']}: {error}")
    
    # Save detailed results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"openrouter_model_testing_results_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        logger.info(f"💾 Results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    # Summary statistics
    if successful_models:
        response_times = [
            results[model['model_id']]['test_result'].get('response_time', 0)
            for model in successful_models
        ]
        
        print(f"\n📈 PERFORMANCE STATISTICS")
        print("-" * 40)
        print(f"Average response time: {sum(response_times)/len(response_times):.2f}s")
        print(f"Fastest response: {min(response_times):.2f}s")
        print(f"Slowest response: {max(response_times):.2f}s")

def main():
    """Main testing function"""
    print("🧪 OPENROUTER MODELS FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        results = test_all_openrouter_models()
        
        print(f"\n🎉 TESTING COMPLETED!")
        print(f"📊 Check openrouter_model_testing_results_*.json for detailed results")
        print(f"📋 Check openrouter_model_testing.log for complete logs")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

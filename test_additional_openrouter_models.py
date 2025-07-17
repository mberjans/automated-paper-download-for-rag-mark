#!/usr/bin/env python3
"""
Additional OpenRouter Models Functionality Test
Tests the comprehensive list of OpenRouter free models
Uses .env file for secure API key management (ignores global environment)
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
    """Setup logging for additional OpenRouter model testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('additional_openrouter_testing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_additional_openrouter_models():
    """Get the comprehensive list of additional OpenRouter free models"""
    return [
        {
            "model_name": "MoonshotAI: Kimi K2 (free)",
            "model_id": "moonshotai/kimi-k2:free",
            "context_tokens": 65536,
            "category": "Moonshot AI"
        },
        {
            "model_name": "TNG: DeepSeek R1T2 Chimera (free)",
            "model_id": "tngtech/deepseek-r1t2-chimera:free",
            "context_tokens": 163840,
            "category": "DeepSeek R1"
        },
        {
            "model_name": "Mistral: Mistral Small 3.2 24B (free)",
            "model_id": "mistralai/mistral-small-3.2-24b-instruct:free",
            "context_tokens": 96000,
            "category": "Mistral"
        },
        {
            "model_name": "Kimi Dev 72b (free)",
            "model_id": "moonshotai/kimi-dev-72b:free",
            "context_tokens": 131072,
            "category": "Moonshot AI"
        },
        {
            "model_name": "DeepSeek: Deepseek R1 0528 Qwen3 8B (free)",
            "model_id": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "context_tokens": 131072,
            "category": "DeepSeek R1"
        },
        {
            "model_name": "DeepSeek: R1 0528 (free)",
            "model_id": "deepseek/deepseek-r1-0528:free",
            "context_tokens": 163840,
            "category": "DeepSeek R1"
        },
        {
            "model_name": "Qwen: Qwen3 235B A22B (free)",
            "model_id": "qwen/qwen3-235b-a22b:free",
            "context_tokens": 131072,
            "category": "Qwen"
        },
        {
            "model_name": "TNG: DeepSeek R1T Chimera (free)",
            "model_id": "tngtech/deepseek-r1t-chimera:free",
            "context_tokens": 163840,
            "category": "DeepSeek R1"
        },
        {
            "model_name": "Microsoft: MAI DS R1 (free)",
            "model_id": "microsoft/mai-ds-r1:free",
            "context_tokens": 163840,
            "category": "Microsoft"
        },
        {
            "model_name": "Agentica: Deepcoder 14B Preview (free)",
            "model_id": "agentica-org/deepcoder-14b-preview:free",
            "context_tokens": 96000,
            "category": "Agentica"
        },
        {
            "model_name": "Moonshot AI: Kimi VL A3B Thinking (free)",
            "model_id": "moonshotai/kimi-vl-a3b-thinking:free",
            "context_tokens": 131072,
            "category": "Moonshot AI"
        },
        {
            "model_name": "NVIDIA: Llama 3.1 Nemotron Ultra 253B v1 (free)",
            "model_id": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
            "context_tokens": 131072,
            "category": "NVIDIA"
        },
        {
            "model_name": "DeepSeek: DeepSeek V3 Base (free)",
            "model_id": "deepseek/deepseek-v3-base:free",
            "context_tokens": 163840,
            "category": "DeepSeek"
        },
        {
            "model_name": "Google: Gemini 2.5 Pro Experimental",
            "model_id": "google/gemini-2.5-pro-exp-03-25",
            "context_tokens": 1048576,
            "category": "Google"
        },
        {
            "model_name": "Mistral: Mistral Small 3.1 24B (free)",
            "model_id": "mistralai/mistral-small-3.1-24b-instruct:free",
            "context_tokens": 128000,
            "category": "Mistral"
        },
        {
            "model_name": "Google: Gemma 3 12B (free)",
            "model_id": "google/gemma-3-12b-it:free",
            "context_tokens": 96000,
            "category": "Google"
        },
        {
            "model_name": "Google: Gemma 3 27B (free)",
            "model_id": "google/gemma-3-27b-it:free",
            "context_tokens": 96000,
            "category": "Google"
        },
        {
            "model_name": "Nous: DeepHermes 3 Llama 3 8B Preview (free)",
            "model_id": "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "context_tokens": 131072,
            "category": "Nous Research"
        },
        {
            "model_name": "DeepSeek: R1 Distill Qwen 14B (free)",
            "model_id": "deepseek/deepseek-r1-distill-qwen-14b:free",
            "context_tokens": 64000,
            "category": "DeepSeek R1"
        },
        {
            "model_name": "DeepSeek: R1 (free)",
            "model_id": "deepseek/deepseek-r1:free",
            "context_tokens": 163840,
            "category": "DeepSeek R1"
        },
        {
            "model_name": "DeepSeek: DeepSeek V3 (free)",
            "model_id": "deepseek/deepseek-chat:free",
            "context_tokens": 163840,
            "category": "DeepSeek"
        },
        {
            "model_name": "Google: Gemini 2.0 Flash Experimental (free)",
            "model_id": "google/gemini-2.0-flash-exp:free",
            "context_tokens": 1048576,
            "category": "Google"
        },
        {
            "model_name": "Meta: Llama 3.3 70B Instruct (free)",
            "model_id": "meta-llama/llama-3.3-70b-instruct:free",
            "context_tokens": 65536,
            "category": "Meta"
        },
        {
            "model_name": "Meta: Llama 3.2 3B Instruct (free)",
            "model_id": "meta-llama/llama-3.2-3b-instruct:free",
            "context_tokens": 131072,
            "category": "Meta"
        },
        {
            "model_name": "Meta: Llama 3.2 11B Vision Instruct (free)",
            "model_id": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "context_tokens": 131072,
            "category": "Meta"
        },
        {
            "model_name": "Meta: Llama 3.1 405B Instruct (free)",
            "model_id": "meta-llama/llama-3.1-405b-instruct:free",
            "context_tokens": 65536,
            "category": "Meta"
        },
        {
            "model_name": "Mistral: Mistral Nemo (free)",
            "model_id": "mistralai/mistral-nemo:free",
            "context_tokens": 131072,
            "category": "Mistral"
        }
    ]

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
            'X-Title': 'FOODB Pipeline Additional Model Testing',
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

def test_single_model(model: Dict[str, Any], api_key: str, logger) -> Dict[str, Any]:
    """Test a single OpenRouter model"""
    model_id = model['model_id']
    model_name = model['model_name']
    
    print(f"\n🔬 Testing: {model_name}")
    print(f"   Model ID: {model_id}")
    print(f"   Context: {model['context_tokens']:,} tokens")
    print(f"   Category: {model['category']}")
    
    # Enhanced test prompt for metabolite extraction
    test_prompt = """Extract metabolites and biomarkers from this scientific text:

"Wine consumption leads to the excretion of several urinary biomarkers including resveratrol, gallic acid, catechin, and epicatechin. These polyphenolic compounds are metabolized in the liver and appear in urine as glucuronide and sulfate conjugates. Additional metabolites include tartaric acid, malic acid, and various anthocyanins."

Please list all metabolites mentioned in a clear format."""
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/mberjans/automated-paper-download-for-rag-mark',
        'X-Title': 'FOODB Pipeline Additional Model Testing',
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
        'max_tokens': 300,
        'temperature': 0.1
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=120  # Increased timeout for larger models
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                
                # Count metabolites mentioned in response
                metabolites_found = []
                test_metabolites = [
                    'resveratrol', 'gallic acid', 'catechin', 'epicatechin',
                    'glucuronide', 'sulfate', 'tartaric acid', 'malic acid', 'anthocyanins'
                ]
                
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
                    'total_possible': len(test_metabolites),
                    'detection_rate': len(metabolites_found) / len(test_metabolites),
                    'response_preview': content[:300] + "..." if len(content) > 300 else content,
                    'usage': data.get('usage', {}),
                    'model_used': data.get('model', model_id),
                    'context_tokens': model['context_tokens'],
                    'category': model['category']
                }
                
                print(f"   ✅ SUCCESS: {response_time:.2f}s")
                print(f"   📊 Metabolites: {len(metabolites_found)}/{len(test_metabolites)} ({len(metabolites_found)/len(test_metabolites)*100:.1f}%)")
                print(f"   📝 Response: {len(content)} chars")
                
                logger.info(f"✅ {model_name}: SUCCESS - {response_time:.2f}s, {len(metabolites_found)}/{len(test_metabolites)} metabolites")
                
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
                    'response_data': data,
                    'context_tokens': model['context_tokens'],
                    'category': model['category']
                }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"   ❌ FAILED: {error_msg}")
            logger.error(f"❌ {model_name}: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'response_time': response_time,
                'status_code': response.status_code,
                'response_text': response.text[:500],
                'context_tokens': model['context_tokens'],
                'category': model['category']
            }
            
    except requests.exceptions.Timeout:
        error_msg = "Request timeout (120s)"
        print(f"   ❌ TIMEOUT: {error_msg}")
        logger.error(f"❌ {model_name}: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'response_time': 120.0,
            'status_code': None,
            'context_tokens': model['context_tokens'],
            'category': model['category']
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ ERROR: {error_msg}")
        logger.error(f"❌ {model_name}: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'response_time': time.time() - start_time,
            'status_code': None,
            'context_tokens': model['context_tokens'],
            'category': model['category']
        }

def test_all_additional_models():
    """Test all additional OpenRouter models"""
    logger = setup_logging()

    print("🧪 ADDITIONAL OPENROUTER MODELS COMPREHENSIVE TEST")
    print("=" * 70)
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
    models = get_additional_openrouter_models()
    print(f"📋 Testing {len(models)} additional OpenRouter free models")

    # Group models by category for organized testing
    categories = {}
    for model in models:
        category = model['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(model)

    print(f"📊 Model categories: {', '.join(categories.keys())}")

    results = {}
    successful_models = []
    failed_models = []

    for i, model in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"Testing model {i}/{len(models)}")

        try:
            result = test_single_model(model, api_key, logger)
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
                    'response_time': 0,
                    'context_tokens': model['context_tokens'],
                    'category': model['category']
                }
            }

        # Add delay between requests to avoid rate limiting
        if i < len(models):
            print(f"   ⏳ Waiting 3 seconds before next test...")
            time.sleep(3)

    # Generate comprehensive report
    generate_comprehensive_report(results, successful_models, failed_models, categories, logger)

    return results

def generate_comprehensive_report(results: Dict, successful_models: List, failed_models: List, categories: Dict, logger):
    """Generate comprehensive report for additional OpenRouter models"""
    print(f"\n{'='*80}")
    print("📊 ADDITIONAL OPENROUTER MODELS TEST REPORT")
    print(f"{'='*80}")

    total_models = len(successful_models) + len(failed_models)
    success_rate = (len(successful_models) / total_models * 100) if total_models > 0 else 0

    print(f"✅ Successful models: {len(successful_models)}/{total_models} ({success_rate:.1f}%)")
    print(f"❌ Failed models: {len(failed_models)}/{total_models}")

    if successful_models:
        print(f"\n🏆 TOP PERFORMING MODELS (by Detection Rate)")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model Name':<45} {'Speed':<8} {'Detection':<10} {'Category':<15}")
        print("-" * 80)

        # Sort by detection rate, then by speed
        successful_sorted = sorted(
            successful_models,
            key=lambda x: (
                results[x['model_id']]['test_result'].get('detection_rate', 0),
                -results[x['model_id']]['test_result'].get('response_time', float('inf'))
            ),
            reverse=True
        )

        for i, model in enumerate(successful_sorted[:10], 1):  # Top 10
            result = results[model['model_id']]['test_result']
            model_name = model['model_name'][:40]
            speed = f"{result.get('response_time', 0):.1f}s"
            detection = f"{result.get('metabolites_count', 0)}/9"
            category = model['category'][:12]

            print(f"{i:<4} {model_name:<45} {speed:<8} {detection:<10} {category:<15}")

        print(f"\n⚡ FASTEST MODELS")
        print("-" * 60)
        print(f"{'Rank':<4} {'Model Name':<45} {'Speed':<10}")
        print("-" * 60)

        # Sort by speed
        speed_sorted = sorted(
            successful_models,
            key=lambda x: results[x['model_id']]['test_result'].get('response_time', float('inf'))
        )

        for i, model in enumerate(speed_sorted[:5], 1):  # Top 5 fastest
            result = results[model['model_id']]['test_result']
            model_name = model['model_name'][:40]
            speed = f"{result.get('response_time', 0):.2f}s"

            print(f"{i:<4} {model_name:<45} {speed:<10}")

        print(f"\n📊 PERFORMANCE BY CATEGORY")
        print("-" * 70)

        for category, category_models in categories.items():
            category_successful = [m for m in category_models if m in successful_models]
            category_total = len(category_models)
            category_success_rate = (len(category_successful) / category_total * 100) if category_total > 0 else 0

            print(f"{category:<20}: {len(category_successful)}/{category_total} ({category_success_rate:.1f}%)")

            if category_successful:
                # Best model in category
                best_model = max(
                    category_successful,
                    key=lambda x: results[x['model_id']]['test_result'].get('detection_rate', 0)
                )
                best_result = results[best_model['model_id']]['test_result']
                print(f"   Best: {best_model['model_name'][:35]} - {best_result.get('metabolites_count', 0)}/9 metabolites, {best_result.get('response_time', 0):.1f}s")

    if failed_models:
        print(f"\n❌ FAILED MODELS BY CATEGORY")
        print("-" * 60)

        failed_by_category = {}
        for model in failed_models:
            category = model['category']
            if category not in failed_by_category:
                failed_by_category[category] = []
            failed_by_category[category].append(model)

        for category, category_failed in failed_by_category.items():
            print(f"\n{category}:")
            for model in category_failed:
                result = results[model['model_id']]['test_result']
                error = result.get('error', 'Unknown error')[:50]
                print(f"   • {model['model_name']}: {error}")

    # Save detailed results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"additional_openrouter_testing_results_{timestamp}.json"

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

        detection_rates = [
            results[model['model_id']]['test_result'].get('detection_rate', 0)
            for model in successful_models
        ]

        print(f"\n📈 PERFORMANCE STATISTICS")
        print("-" * 40)
        print(f"Average response time: {sum(response_times)/len(response_times):.2f}s")
        print(f"Fastest response: {min(response_times):.2f}s")
        print(f"Slowest response: {max(response_times):.2f}s")
        print(f"Average detection rate: {sum(detection_rates)/len(detection_rates)*100:.1f}%")
        print(f"Best detection rate: {max(detection_rates)*100:.1f}%")

def main():
    """Main testing function"""
    print("🧪 ADDITIONAL OPENROUTER MODELS COMPREHENSIVE TEST")
    print("=" * 70)

    try:
        results = test_all_additional_models()

        print(f"\n🎉 COMPREHENSIVE TESTING COMPLETED!")
        print(f"📊 Check additional_openrouter_testing_results_*.json for detailed results")
        print(f"📋 Check additional_openrouter_testing.log for complete logs")

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

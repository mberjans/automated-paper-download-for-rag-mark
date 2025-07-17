#!/usr/bin/env python3
"""
Test New Groq Models for Metabolite Extraction
Comprehensive testing of new Groq models for FOODB pipeline compatibility
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add FOODB pipeline to path
sys.path.append('FOODB_LLM_pipeline')

def setup_logging():
    """Setup logging for model testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('groq_model_testing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_api_key():
    """Load Groq API key from environment"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return api_key

def test_groq_models():
    """Test new Groq models for metabolite extraction"""
    logger = setup_logging()
    
    # New models to test
    models_to_test = [
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-maverick-17b-128e-instruct", 
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "mistral-saba-24b",
        "moonshotai/kimi-k2-instruct",
        "qwen/qwen3-32b"
    ]
    
    # Test text chunk (from wine biomarkers paper)
    test_chunk = """
    Wine consumption has been associated with various health benefits, particularly cardiovascular protection. 
    The bioactive compounds in wine include resveratrol, quercetin, catechin, epicatechin, and anthocyanins. 
    These polyphenolic compounds are metabolized in the human body to produce various metabolites including 
    3-hydroxyphenylacetic acid, vanillic acid, and hippuric acid. Urinary biomarkers of wine consumption 
    include tartaric acid, gallic acid, and various flavonoid metabolites. The gut microbiota plays a 
    crucial role in metabolizing these compounds, producing metabolites such as equol, urolithins, and 
    phenylacetic acid derivatives.
    """
    
    # Document-only extraction prompt
    document_only_prompt = f"""
    Based ONLY on the text provided below, extract any metabolites, biomarkers, or chemical compounds mentioned. 
    List each compound on a separate line. Do not include any compounds not explicitly mentioned in the text.
    
    Text: {test_chunk}
    
    Compounds found:
    """
    
    logger.info("🧪 Starting Groq model testing for metabolite extraction")
    logger.info(f"📋 Testing {len(models_to_test)} models")
    
    results = {}
    api_key = load_api_key()
    
    for i, model in enumerate(models_to_test, 1):
        logger.info(f"\n🔬 Testing model {i}/{len(models_to_test)}: {model}")
        
        try:
            result = test_single_model(model, document_only_prompt, api_key, logger)
            results[model] = result
            
            if result['success']:
                logger.info(f"✅ {model}: SUCCESS - {len(result['compounds'])} compounds extracted")
                logger.info(f"   Response time: {result['response_time']:.2f}s")
                logger.info(f"   Sample compounds: {result['compounds'][:3]}")
            else:
                logger.error(f"❌ {model}: FAILED - {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ {model}: EXCEPTION - {str(e)}")
            results[model] = {
                'success': False,
                'error': str(e),
                'response_time': 0,
                'compounds': [],
                'raw_response': ''
            }
        
        # Add delay between requests to avoid rate limiting
        time.sleep(2)
    
    # Generate comprehensive report
    generate_model_report(results, logger)
    
    return results

def test_single_model(model: str, prompt: str, api_key: str, logger) -> Dict[str, Any]:
    """Test a single Groq model"""
    start_time = time.time()
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': 200,
        'temperature': 0.1
    }
    
    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            response_data = response.json()
            raw_response = response_data['choices'][0]['message']['content']
            
            # Parse compounds from response
            compounds = parse_compounds_from_response(raw_response)
            
            return {
                'success': True,
                'response_time': response_time,
                'compounds': compounds,
                'raw_response': raw_response,
                'error': None
            }
        else:
            return {
                'success': False,
                'response_time': response_time,
                'compounds': [],
                'raw_response': '',
                'error': f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            'success': False,
            'response_time': response_time,
            'compounds': [],
            'raw_response': '',
            'error': str(e)
        }

def parse_compounds_from_response(response: str) -> List[str]:
    """Parse compounds from model response"""
    if not response or response.lower().strip() in ['no compounds found', 'no specific compounds mentioned']:
        return []
    
    compounds = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering and bullet points
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '•', '*')):
            line = line[2:].strip()
        
        # Skip common non-compound responses
        skip_phrases = [
            'based on', 'however', 'note that', 'unfortunately', 'please note',
            'i can provide', 'the text', 'no specific', 'not mentioned',
            'compounds found:', 'metabolites:', 'biomarkers:'
        ]
        
        if any(skip in line.lower() for skip in skip_phrases):
            continue
        
        if line and len(line) > 2:
            compounds.append(line)
    
    return compounds

def generate_model_report(results: Dict[str, Any], logger):
    """Generate comprehensive model testing report"""
    logger.info("\n📊 GROQ MODEL TESTING REPORT")
    logger.info("=" * 50)
    
    successful_models = [model for model, result in results.items() if result['success']]
    failed_models = [model for model, result in results.items() if not result['success']]
    
    logger.info(f"✅ Successful models: {len(successful_models)}/{len(results)}")
    logger.info(f"❌ Failed models: {len(failed_models)}/{len(results)}")
    
    if successful_models:
        logger.info(f"\n🎯 SUCCESSFUL MODELS:")
        for model in successful_models:
            result = results[model]
            logger.info(f"   • {model}")
            logger.info(f"     Response time: {result['response_time']:.2f}s")
            logger.info(f"     Compounds extracted: {len(result['compounds'])}")
            logger.info(f"     Sample: {result['compounds'][:2]}")
    
    if failed_models:
        logger.info(f"\n❌ FAILED MODELS:")
        for model in failed_models:
            result = results[model]
            logger.info(f"   • {model}: {result['error']}")
    
    # Performance comparison
    if successful_models:
        logger.info(f"\n⚡ PERFORMANCE COMPARISON:")
        performance_data = []
        for model in successful_models:
            result = results[model]
            performance_data.append({
                'model': model,
                'response_time': result['response_time'],
                'compounds_count': len(result['compounds'])
            })
        
        # Sort by response time
        performance_data.sort(key=lambda x: x['response_time'])
        
        logger.info("   Ranked by speed (fastest first):")
        for i, data in enumerate(performance_data, 1):
            logger.info(f"   {i}. {data['model']}: {data['response_time']:.2f}s ({data['compounds_count']} compounds)")
    
    # Save detailed results to JSON
    save_results_to_file(results, logger)

def save_results_to_file(results: Dict[str, Any], logger):
    """Save detailed results to JSON file"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"groq_model_testing_results_{timestamp}.json"
    
    # Prepare results for JSON serialization
    json_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_models_tested': len(results),
        'successful_models': len([r for r in results.values() if r['success']]),
        'failed_models': len([r for r in results.values() if not r['success']]),
        'model_results': results
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"\n💾 Detailed results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

def update_llm_wrapper_with_new_models(successful_models: List[str], logger):
    """Update LLM wrapper to include new successful models"""
    logger.info(f"\n🔧 Updating LLM wrapper with {len(successful_models)} new models")
    
    # This would update the llm_wrapper_enhanced.py file
    # For now, just log the models that should be added
    logger.info("Models to add to LLM wrapper:")
    for model in successful_models:
        logger.info(f"   • {model}")
    
    # TODO: Implement actual file update
    logger.info("💡 Manual update required: Add these models to llm_wrapper_enhanced.py")

def main():
    """Main testing function"""
    print("🧪 GROQ MODEL TESTING FOR FOODB PIPELINE")
    print("=" * 50)
    
    try:
        # Test all models
        results = test_groq_models()
        
        # Get successful models
        successful_models = [model for model, result in results.items() if result['success']]
        
        if successful_models:
            print(f"\n🎉 Testing completed successfully!")
            print(f"✅ {len(successful_models)} models are compatible with FOODB pipeline")
            print(f"📊 Check groq_model_testing_results_*.json for detailed results")
            print(f"📋 Check groq_model_testing.log for complete logs")
            
            # Suggest next steps
            print(f"\n🔧 NEXT STEPS:")
            print(f"1. Review detailed results in JSON file")
            print(f"2. Update llm_wrapper_enhanced.py with successful models")
            print(f"3. Test pipeline with new models using --primary-provider")
            print(f"4. Update documentation with new model options")
            
        else:
            print(f"\n❌ No models passed compatibility testing")
            print(f"📋 Check groq_model_testing.log for error details")
            
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

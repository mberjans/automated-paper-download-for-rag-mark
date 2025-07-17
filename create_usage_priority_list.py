#!/usr/bin/env python3
"""
Create LLM usage priority list ordered by:
1. Cerebras models (by F1/recall score)
2. Groq models (by F1/recall score) 
3. OpenRouter models (by F1/recall score)
"""

import json
from datetime import datetime

def load_v4_ranking():
    """Load V4 ranking file"""
    try:
        with open('free_models_reasoning_ranked_v4.json', 'r') as f:
            data = json.load(f)
        return data['models']
    except Exception as e:
        print(f"Error loading V4 file: {e}")
        return []

def get_performance_score(model):
    """Extract F1 score or recall for ranking"""
    model_id = model.get('model_id', '')

    # Enhanced OpenRouter F1 scores from comprehensive evaluation
    openrouter_f1_scores = {
        'mistralai/mistral-nemo:free': 0.5772,
        'tngtech/deepseek-r1t-chimera:free': 0.4372,
        'google/gemini-2.0-flash-exp:free': 0.4065,
        'mistralai/mistral-small-3.1-24b-instruct:free': 0.3619,
        'mistralai/mistral-small-3.2-24b-instruct:free': 0.3421,
        'google/gemma-3-27b-it:free': 0.3333,
        'nousresearch/deephermes-3-llama-3-8b-preview:free': 0.3178,
        'moonshotai/kimi-vl-a3b-thinking:free': 0.2797,
        'moonshotai/kimi-k2:free': 0.2549,
        'deepseek/deepseek-chat:free': 0.2330,
        'meta-llama/llama-3.3-70b-instruct:free': 0.2198,
        'qwen/qwen3-235b-a22b:free': 0.1143,
        'meta-llama/llama-3.2-11b-vision-instruct:free': 0.1128,
        'deepseek/deepseek-v3-base:free': 0.0635,
        'google/gemma-3-12b-it:free': 0.0290
    }

    # Enhanced OpenRouter recall scores from comprehensive evaluation
    openrouter_recall_scores = {
        'mistralai/mistral-nemo:free': 0.7288,
        'tngtech/deepseek-r1t-chimera:free': 0.6780,
        'mistralai/mistral-small-3.2-24b-instruct:free': 0.6610,
        'mistralai/mistral-small-3.1-24b-instruct:free': 0.6441,
        'nousresearch/deephermes-3-llama-3-8b-preview:free': 0.5763,
        'google/gemini-2.0-flash-exp:free': 0.4237,
        'google/gemma-3-27b-it:free': 0.3559,
        'moonshotai/kimi-vl-a3b-thinking:free': 0.3390,
        'moonshotai/kimi-k2:free': 0.2203,
        'deepseek/deepseek-chat:free': 0.2034,
        'meta-llama/llama-3.2-11b-vision-instruct:free': 0.1864,
        'meta-llama/llama-3.3-70b-instruct:free': 0.1695,
        'qwen/qwen3-235b-a22b:free': 0.0678,
        'deepseek/deepseek-v3-base:free': 0.0339,
        'google/gemma-3-12b-it:free': 0.0169
    }

    # Check for FOODB performance (Groq models)
    if 'foodb_performance' in model:
        foodb = model['foodb_performance']
        if 'f1_score' in foodb:
            return foodb['f1_score'], foodb.get('recall', 0), 'f1_score'

    # Check for enhanced OpenRouter F1 scores
    if model_id in openrouter_f1_scores:
        f1_score = openrouter_f1_scores[model_id]
        recall = openrouter_recall_scores.get(model_id, 0)
        return f1_score, recall, 'f1_score'

    # Check for basic OpenRouter performance
    if 'openrouter_performance' in model:
        perf = model['openrouter_performance']
        if 'detection_rate' in perf:
            return perf['detection_rate'], perf['detection_rate'], 'detection_rate'

    # Fallback to reasoning score for Cerebras models (no F1 data available)
    return model.get('reasoning_score', 0) / 10.0, 0, 'reasoning_score'

def create_usage_priority_list():
    """Create prioritized usage list by provider and performance"""
    
    models = load_v4_ranking()
    if not models:
        print("Failed to load models")
        return
    
    # Separate models by provider
    cerebras_models = []
    groq_models = []
    openrouter_models = []
    
    for model in models:
        provider = model.get('provider', '')
        score, recall, score_type = get_performance_score(model)
        
        model_info = {
            'rank': model.get('rank'),
            'model_name': model.get('model_name'),
            'model_id': model.get('model_id'),
            'provider': provider,
            'reasoning_score': model.get('reasoning_score'),
            'performance_score': score,
            'recall': recall,
            'score_type': score_type,
            'speed': model.get('average_response_time'),
            'context_tokens': model.get('context_tokens'),
            'api_url': model.get('api_url'),
            'api_key_env': model.get('api_key_env')
        }
        
        if provider == 'Cerebras':
            cerebras_models.append(model_info)
        elif provider == 'Groq':
            groq_models.append(model_info)
        elif provider == 'OpenRouter':
            openrouter_models.append(model_info)
    
    # Sort each provider group by performance score (descending)
    cerebras_models.sort(key=lambda x: x['performance_score'], reverse=True)
    groq_models.sort(key=lambda x: x['performance_score'], reverse=True)
    openrouter_models.sort(key=lambda x: x['performance_score'], reverse=True)
    
    # Create final prioritized list
    priority_list = []
    priority_rank = 1
    
    # Add Cerebras models first
    for model in cerebras_models:
        model['priority_rank'] = priority_rank
        priority_list.append(model)
        priority_rank += 1
    
    # Add Groq models second
    for model in groq_models:
        model['priority_rank'] = priority_rank
        priority_list.append(model)
        priority_rank += 1
    
    # Add OpenRouter models last
    for model in openrouter_models:
        model['priority_rank'] = priority_rank
        priority_list.append(model)
        priority_rank += 1
    
    # Create output data
    output_data = {
        "metadata": {
            "title": "LLM Usage Priority List",
            "version": "1.0",
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Prioritized LLM usage order: Cerebras → Groq → OpenRouter, ranked by F1/recall scores",
            "total_models": len(priority_list),
            "provider_order": ["Cerebras", "Groq", "OpenRouter"],
            "ranking_criteria": "F1 score > Recall > Detection Rate > Reasoning Score",
            "providers": {
                "Cerebras": len(cerebras_models),
                "Groq": len(groq_models), 
                "OpenRouter": len(openrouter_models)
            }
        },
        "priority_list": priority_list
    }
    
    # Save to file
    output_file = 'llm_usage_priority_list.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Successfully created {output_file}")
        print(f"📊 Total models: {len(priority_list)}")
        print(f"\n🏆 TOP 10 PRIORITY MODELS:")
        print("-" * 80)
        
        for i, model in enumerate(priority_list[:10], 1):
            score_display = f"{model['performance_score']:.4f}" if model['performance_score'] < 1 else f"{model['performance_score']:.3f}"
            print(f"{i:2d}. [{model['provider']:9}] {model['model_name'][:45]:<45}")
            print(f"    Score: {score_display} ({model['score_type']}) | Speed: {model['speed']:.2f}s | Reasoning: {model['reasoning_score']}")
            print()
        
        print(f"\n📋 Provider breakdown:")
        for provider, count in output_data['metadata']['providers'].items():
            percentage = (count / len(priority_list)) * 100
            print(f"   {provider}: {count} models ({percentage:.1f}%)")
        
        return output_file
        
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return None

if __name__ == "__main__":
    create_usage_priority_list()

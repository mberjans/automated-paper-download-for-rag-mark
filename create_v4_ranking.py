#!/usr/bin/env python3
"""
Create free_models_reasoning_ranked_v4.json by combining:
- OpenRouter models from V3 file
- Groq and Cerebras models from V2 file
"""

import json
from datetime import datetime

def load_json_file(filename):
    """Load JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def create_v4_ranking():
    """Create comprehensive V4 ranking combining V2 and V3 models"""
    
    # Load V2 file (contains Groq and Cerebras models)
    v2_models = load_json_file('free_models_reasoning_ranked_v2.json')
    
    # Load V3 file (contains OpenRouter models)
    v3_models = load_json_file('free_models_reasoning_ranked_v3.json')
    
    if not v2_models or not v3_models:
        print("Failed to load source files")
        return
    
    # Extract Groq and Cerebras models from V2
    groq_cerebras_models = []
    for model in v2_models:
        if model.get('provider') in ['Groq', 'Cerebras']:
            groq_cerebras_models.append(model)
    
    print(f"Found {len(groq_cerebras_models)} Groq/Cerebras models in V2")
    print(f"Found {len(v3_models)} OpenRouter models in V3")
    
    # Combine all models
    all_models = groq_cerebras_models + v3_models
    
    # Sort by reasoning score (descending)
    all_models.sort(key=lambda x: x.get('reasoning_score', 0), reverse=True)
    
    # Re-rank all models
    for i, model in enumerate(all_models, 1):
        model['rank'] = i
    
    # Add metadata
    v4_data = {
        "metadata": {
            "version": "4.0",
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Comprehensive ranking combining Groq, Cerebras, and OpenRouter models",
            "total_models": len(all_models),
            "providers": {
                "Groq": len([m for m in all_models if m.get('provider') == 'Groq']),
                "Cerebras": len([m for m in all_models if m.get('provider') == 'Cerebras']),
                "OpenRouter": len([m for m in all_models if m.get('provider') == 'OpenRouter'])
            },
            "source_files": [
                "free_models_reasoning_ranked_v2.json (Groq/Cerebras)",
                "free_models_reasoning_ranked_v3.json (OpenRouter)"
            ],
            "ranking_criteria": "reasoning_score (descending)",
            "score_range": f"{min(m.get('reasoning_score', 0) for m in all_models):.1f} - {max(m.get('reasoning_score', 0) for m in all_models):.1f}"
        },
        "models": all_models
    }
    
    # Save V4 file
    output_file = 'free_models_reasoning_ranked_v4.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(v4_data, f, indent=2)
        
        print(f"\n✅ Successfully created {output_file}")
        print(f"📊 Total models: {len(all_models)}")
        print(f"🏆 Top model: {all_models[0]['model_name']} ({all_models[0]['provider']}) - Score: {all_models[0]['reasoning_score']}")
        print(f"📈 Score range: {v4_data['metadata']['score_range']}")
        
        # Show provider breakdown
        print(f"\n📋 Provider breakdown:")
        for provider, count in v4_data['metadata']['providers'].items():
            percentage = (count / len(all_models)) * 100
            print(f"   {provider}: {count} models ({percentage:.1f}%)")
        
        return output_file
        
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return None

if __name__ == "__main__":
    create_v4_ranking()

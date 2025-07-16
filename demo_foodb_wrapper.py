#!/usr/bin/env python3
"""
Demo: FOODB LLM Pipeline Wrapper Usage
This script demonstrates how to use the API-based wrapper for FOODB pipeline testing
"""

import sys
import os
import json
import time

# Add the FOODB_LLM_pipeline directory to the path
sys.path.append('FOODB_LLM_pipeline')

def demo_simple_sentence_generation():
    """Demo: Simple sentence generation"""
    print("🔬 Demo 1: Simple Sentence Generation")
    print("=" * 50)
    
    from llm_wrapper import LLMWrapper
    
    # Create wrapper and set model
    wrapper = LLMWrapper()
    
    # Scientific text sample
    scientific_text = """Quercetin is a flavonoid compound found abundantly in onions, apples, and berries. 
    Research has demonstrated that quercetin exhibits potent anti-inflammatory and antioxidant properties. 
    Clinical studies indicate that quercetin supplementation may help reduce blood pressure and improve 
    cardiovascular health in hypertensive patients. The compound works by modulating various cellular 
    pathways involved in oxidative stress and inflammation."""
    
    print(f"📝 Input text:")
    print(f"{scientific_text}")
    
    # Create prompt for simple sentence generation
    prompt = f"""Convert the given scientific text into a list of simple, clear sentences. Each sentence should:
1. Express a single fact or relationship
2. Be grammatically complete and independent
3. Use clear, direct language
4. Preserve all important information from the original text

Text: {scientific_text}

Simple sentences:"""
    
    print(f"\n⚡ Generating simple sentences...")
    start_time = time.time()
    
    response = wrapper.generate_single(prompt, max_tokens=300)
    
    end_time = time.time()
    
    print(f"\n✅ Generated simple sentences:")
    print(response)
    print(f"\n⏱️ Processing time: {end_time - start_time:.2f} seconds")


def demo_entity_extraction():
    """Demo: Entity extraction"""
    print("\n🧬 Demo 2: Entity Extraction")
    print("=" * 50)
    
    from llm_wrapper import LLMWrapper
    
    wrapper = LLMWrapper()
    
    sentences = [
        "Resveratrol activates SIRT1 protein pathways.",
        "Green tea contains EGCG which has antioxidant properties.",
        "Curcumin reduces inflammation in arthritis patients.",
        "Omega-3 fatty acids support brain health and cognitive function."
    ]
    
    entity_prompt = """Extract food entities and bioactive compounds from the given sentence. 
    Focus on:
    - Specific foods (e.g., "green tea", "turmeric")
    - Food compounds (e.g., "resveratrol", "EGCG", "curcumin")
    - Biological targets (e.g., "SIRT1", "inflammation")
    - Health effects (e.g., "antioxidant", "anti-inflammatory")
    
    Return the entities as a simple list.
    
    Sentence: {sentence}
    
    Entities:"""
    
    print("📋 Extracting entities from sentences:")
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. Sentence: {sentence}")
        
        prompt = entity_prompt.format(sentence=sentence)
        entities = wrapper.generate_single(prompt, max_tokens=100)
        
        print(f"   Entities: {entities.strip()}")


def demo_triple_extraction():
    """Demo: Triple extraction"""
    print("\n🔗 Demo 3: Triple Extraction")
    print("=" * 50)
    
    from llm_wrapper import LLMWrapper
    
    wrapper = LLMWrapper()
    
    sentences = [
        "Resveratrol activates SIRT1 protein.",
        "Curcumin reduces inflammation.",
        "Green tea contains EGCG.",
        "Omega-3 fatty acids improve cognitive function."
    ]
    
    triple_prompt = """Extract subject-predicate-object triples from the given sentence.
    Focus on relationships between food compounds, biological targets, and health effects.
    Return each triple in the format: [subject, predicate, object]
    
    Sentence: {sentence}
    
    Triples:"""
    
    print("🔗 Extracting triples from sentences:")
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. Sentence: {sentence}")
        
        prompt = triple_prompt.format(sentence=sentence)
        triples = wrapper.generate_single(prompt, max_tokens=100)
        
        print(f"   Triples: {triples.strip()}")


def demo_batch_processing():
    """Demo: Batch processing multiple texts"""
    print("\n📦 Demo 4: Batch Processing")
    print("=" * 50)
    
    from llm_wrapper import LLMWrapper
    
    wrapper = LLMWrapper()
    
    # Multiple scientific texts
    texts = [
        "Lycopene gives tomatoes their red color and may reduce prostate cancer risk.",
        "Anthocyanins in berries provide antioxidant benefits and improve memory.",
        "Beta-carotene is converted to vitamin A and supports eye health.",
        "Polyphenols in dark chocolate may improve cardiovascular function."
    ]
    
    # Create prompts for simple sentence generation
    prompts = []
    for text in texts:
        prompt = f"Convert to simple sentences: {text}\n\nSimple sentences:"
        prompts.append(prompt)
    
    print(f"📋 Processing {len(texts)} texts in batch...")
    
    start_time = time.time()
    responses = wrapper.generate_batch(prompts, max_tokens=150, max_concurrent=4)
    end_time = time.time()
    
    print(f"\n✅ Batch processing results:")
    for i, (text, response) in enumerate(zip(texts, responses), 1):
        print(f"\n{i}. Original: {text}")
        print(f"   Simple: {response.strip()}")
    
    total_time = end_time - start_time
    print(f"\n⏱️ Total time: {total_time:.2f}s | Average: {total_time/len(texts):.2f}s per text")


def demo_model_comparison():
    """Demo: Compare different models"""
    print("\n🏁 Demo 5: Model Performance Comparison")
    print("=" * 50)
    
    from llm_wrapper import LLMWrapper
    
    wrapper = LLMWrapper()
    
    # Get available models
    models = wrapper.list_available_models()
    test_text = "Resveratrol activates SIRT1 and reduces inflammation."
    
    print(f"🧪 Testing {min(3, len(models))} models with: {test_text}")
    
    results = []
    
    for i, model in enumerate(models[:3]):  # Test first 3 models
        print(f"\n{i+1}. Testing {model.get('model_name')} ({model.get('provider')})")
        
        try:
            # Set model
            if wrapper.set_model(model):
                # Test performance
                start_time = time.time()
                response = wrapper.generate_single(
                    f"Convert to simple sentences: {test_text}",
                    max_tokens=100
                )
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                results.append({
                    'model': model.get('model_name'),
                    'provider': model.get('provider'),
                    'time': processing_time,
                    'response': response.strip()
                })
                
                print(f"   ⏱️ Time: {processing_time:.2f}s")
                print(f"   📄 Response: {response.strip()[:100]}...")
            else:
                print(f"   ❌ Failed to initialize")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n📊 Performance Summary:")
        results.sort(key=lambda x: x['time'])  # Sort by speed
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['model']} ({result['provider']}) - {result['time']:.2f}s")


def demo_scientific_pipeline():
    """Demo: Complete scientific text processing pipeline"""
    print("\n🧬 Demo 6: Complete Scientific Pipeline")
    print("=" * 50)
    
    from llm_wrapper import LLMWrapper
    
    wrapper = LLMWrapper()
    
    # Sample scientific abstract
    scientific_abstract = """Polyphenolic compounds, particularly flavonoids such as quercetin and catechins, 
    have been extensively studied for their potential health benefits. These bioactive molecules, found in 
    various plant-based foods including tea, berries, and onions, exhibit significant antioxidant and 
    anti-inflammatory properties. Recent clinical trials have demonstrated that regular consumption of 
    polyphenol-rich foods may contribute to reduced risk of cardiovascular disease, improved cognitive 
    function, and enhanced metabolic health. The mechanisms underlying these effects involve modulation 
    of cellular signaling pathways, including the activation of antioxidant enzymes and inhibition of 
    pro-inflammatory mediators."""
    
    print(f"📄 Processing scientific abstract:")
    print(f"{scientific_abstract[:150]}...")
    
    # Step 1: Simple sentence generation
    print(f"\n🔄 Step 1: Generating simple sentences...")
    simple_prompt = f"""Convert this scientific text into simple, clear sentences:

{scientific_abstract}

Simple sentences:"""
    
    simple_sentences = wrapper.generate_single(simple_prompt, max_tokens=400)
    print(f"✅ Generated {len(simple_sentences.split('.'))} simple sentences")
    
    # Step 2: Entity extraction from first few sentences
    print(f"\n🔄 Step 2: Extracting entities...")
    sentences = [s.strip() for s in simple_sentences.split('.') if s.strip()][:3]
    
    for i, sentence in enumerate(sentences, 1):
        if sentence:
            entity_prompt = f"Extract food compounds and health effects from: {sentence}\n\nEntities:"
            entities = wrapper.generate_single(entity_prompt, max_tokens=100)
            print(f"  {i}. {sentence[:50]}... → {entities.strip()}")
    
    # Step 3: Triple extraction
    print(f"\n🔄 Step 3: Extracting relationships...")
    for i, sentence in enumerate(sentences[:2], 1):
        if sentence:
            triple_prompt = f"Extract [subject, predicate, object] from: {sentence}\n\nTriple:"
            triple = wrapper.generate_single(triple_prompt, max_tokens=50)
            print(f"  {i}. {sentence[:50]}... → {triple.strip()}")
    
    print(f"\n✅ Pipeline processing complete!")


def main():
    """Run all demos"""
    print("🧬 FOODB LLM Pipeline Wrapper - Comprehensive Demo")
    print("=" * 70)
    
    try:
        # Check if wrapper is working
        from llm_wrapper import LLMWrapper
        wrapper = LLMWrapper()
        
        if not wrapper.list_available_models():
            print("❌ No models available. Please check your configuration.")
            return
        
        print(f"✅ Wrapper initialized with {len(wrapper.list_available_models())} models")
        print(f"🎯 Using model: {wrapper.current_model.get('model_name', 'Unknown')}")
        
        # Run demos
        demo_simple_sentence_generation()
        demo_entity_extraction()
        demo_triple_extraction()
        demo_batch_processing()
        demo_model_comparison()
        demo_scientific_pipeline()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"  1. Integrate the wrapper into your FOODB pipeline scripts")
        print(f"  2. Process your own scientific texts")
        print(f"  3. Experiment with different models and parameters")
        print(f"  4. Use batch processing for large datasets")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"Make sure you're in the correct directory and have dependencies installed")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

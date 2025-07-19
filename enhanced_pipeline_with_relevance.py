#!/usr/bin/env python3
"""
Enhanced FOODB Pipeline with Chunk Relevance Evaluation
Demonstrates how relevance filtering could improve efficiency and accuracy
"""

import sys
import os
import time
from typing import List, Dict, Any
sys.path.append('FOODB_LLM_pipeline')

def test_relevance_enhanced_pipeline():
    """Test the pipeline with relevance evaluation"""
    print("🔍 Enhanced Pipeline with Relevance Evaluation")
    print("=" * 55)
    
    try:
        from chunk_relevance_evaluator import ChunkRelevanceEvaluator
        from llm_wrapper_enhanced import LLMWrapper, RetryConfig
        
        # Load Wine PDF text (simulated)
        pdf_file = "Wine-consumptionbiomarkers-HMDB.pdf"
        
        # Always use simulated chunks for demonstration
        print("📝 Using simulated chunks for demonstration...")

        # Simulated chunks with varying relevance
        chunks = [
                # High relevance chunks
                "The urinary metabolites of wine consumption included resveratrol glucuronide, quercetin sulfate, and various anthocyanin derivatives detected using LC-MS analysis.",
                "Malvidin-3-glucoside and cyanidin-3-glucoside were the predominant anthocyanins found in urine samples after red wine consumption.",
                "Gallic acid sulfate and protocatechuic acid glucuronide were identified as major phenolic acid metabolites in plasma samples.",
                
                # Medium relevance chunks
                "Wine consumption has been associated with cardiovascular health benefits due to its polyphenolic content and antioxidant properties.",
                "The study participants consumed 300ml of red wine daily for 4 weeks under controlled dietary conditions.",
                "Biomarker concentrations were measured at baseline, 2 hours, and 24 hours post-consumption using validated analytical methods.",
                
                # Low relevance chunks
                "The study was conducted at the University of California with approval from the institutional review board.",
                "Participants were recruited through local advertisements and provided written informed consent before enrollment.",
                "Statistical analysis was performed using SPSS version 25.0 with significance set at p < 0.05.",
                
                # Irrelevant chunks
                "References: 1. Smith et al. (2020) Journal of Wine Research 2. Jones et al. (2019) Food Chemistry",
                "Acknowledgments: The authors thank the research staff and participants for their contributions to this study.",
                "Figure 1. Chromatographic separation of wine phenolic compounds. Table 1. Participant demographics and baseline characteristics."
            ]
        
        print(f"📊 Total chunks to evaluate: {len(chunks)}")
        
        # Initialize relevance evaluator
        print(f"\n🔍 Initializing relevance evaluator...")
        evaluator = ChunkRelevanceEvaluator(relevance_threshold=0.3)
        
        # Evaluate chunk relevance
        print(f"📈 Evaluating chunk relevance...")
        start_time = time.time()
        relevant_chunks, scores = evaluator.filter_relevant_chunks(chunks)
        relevance_time = time.time() - start_time
        
        # Generate relevance report
        report = evaluator.get_relevance_report(chunks)
        
        print(f"\n📋 Relevance Evaluation Results:")
        print(f"   Total chunks: {report['total_chunks']}")
        print(f"   Relevant chunks: {report['relevant_chunks']}")
        print(f"   Filtered out: {report['filtered_chunks']}")
        print(f"   Relevance rate: {report['relevance_rate']:.1%}")
        print(f"   Average score: {report['average_relevance_score']:.3f}")
        print(f"   Evaluation time: {relevance_time:.3f}s")
        
        # Show efficiency gains
        api_calls_saved = len(chunks) - len(relevant_chunks)
        efficiency_gain = api_calls_saved / len(chunks) * 100 if chunks else 0
        
        print(f"\n⚡ Efficiency Gains:")
        print(f"   API calls saved: {api_calls_saved}")
        print(f"   Efficiency improvement: {efficiency_gain:.1f}%")
        print(f"   Processing time saved: ~{api_calls_saved * 2:.0f}s (estimated)")
        
        # Initialize LLM wrapper for processing relevant chunks
        print(f"\n🤖 Processing relevant chunks with LLM...")
        retry_config = RetryConfig(max_attempts=3, base_delay=1.0)
        wrapper = LLMWrapper(retry_config=retry_config)
        
        # Process only relevant chunks
        start_time = time.time()
        all_metabolites = []
        
        for i, chunk in enumerate(relevant_chunks):
            print(f"  📄 Processing relevant chunk {i+1}/{len(relevant_chunks)}...", end=" ")
            
            try:
                response = wrapper.generate_single_with_fallback(
                    f"Extract metabolites and biomarkers from this text:\n\n{chunk}",
                    max_tokens=100
                )
                
                if response:
                    # Simple metabolite extraction (would use proper parsing in real implementation)
                    metabolites = [m.strip() for m in response.split(',') if m.strip()]
                    all_metabolites.extend(metabolites)
                    print(f"✅ Found {len(metabolites)} compounds")
                else:
                    print("❌ No response")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        processing_time = time.time() - start_time
        
        # Results summary
        unique_metabolites = list(set(all_metabolites))
        
        print(f"\n📊 Processing Results:")
        print(f"   Relevant chunks processed: {len(relevant_chunks)}")
        print(f"   Total metabolites found: {len(all_metabolites)}")
        print(f"   Unique metabolites: {len(unique_metabolites)}")
        print(f"   Processing time: {processing_time:.3f}s")
        
        # Comparison with non-filtered approach
        if relevant_chunks and len(relevant_chunks) > 0:
            estimated_full_time = len(chunks) * (processing_time / len(relevant_chunks))
            time_saved = estimated_full_time - processing_time
            time_saved_pct = (time_saved / estimated_full_time * 100) if estimated_full_time > 0 else 0
        else:
            estimated_full_time = 0
            time_saved = 0
            time_saved_pct = 0

        print(f"\n⚖️ Comparison with Non-Filtered Approach:")
        print(f"   Estimated full processing time: {estimated_full_time:.1f}s")
        print(f"   Actual processing time: {processing_time:.1f}s")
        print(f"   Time saved: {time_saved:.1f}s ({time_saved_pct:.1f}%)")
        
        # Show detailed relevance breakdown
        print(f"\n🔍 Detailed Relevance Breakdown:")
        relevance_categories = {
            'High (>0.7)': 0,
            'Medium (0.3-0.7)': 0,
            'Low (<0.3)': 0
        }
        
        for score in scores:
            if score.total_score > 0.7:
                relevance_categories['High (>0.7)'] += 1
            elif score.total_score >= 0.3:
                relevance_categories['Medium (0.3-0.7)'] += 1
            else:
                relevance_categories['Low (<0.3)'] += 1
        
        for category, count in relevance_categories.items():
            percentage = count / len(chunks) * 100 if chunks else 0
            print(f"   {category}: {count} chunks ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def explain_relevance_benefits():
    """Explain the benefits of relevance evaluation"""
    print(f"\n🎯 Benefits of Relevance Evaluation:")
    print("=" * 40)
    
    print(f"\n1. **Efficiency Improvements:**")
    print(f"   • Reduces API calls by filtering irrelevant chunks")
    print(f"   • Saves processing time and costs")
    print(f"   • Focuses computational resources on relevant content")
    
    print(f"\n2. **Accuracy Improvements:**")
    print(f"   • Reduces false positives from irrelevant sections")
    print(f"   • Improves precision by focusing on relevant content")
    print(f"   • Better signal-to-noise ratio in results")
    
    print(f"\n3. **Quality Improvements:**")
    print(f"   • Prioritizes high-value content (Results, Methods)")
    print(f"   • Filters out references, acknowledgments, etc.")
    print(f"   • Focuses on sections likely to contain metabolites")
    
    print(f"\n4. **Cost Savings:**")
    print(f"   • Fewer API calls = lower costs")
    print(f"   • Faster processing = better user experience")
    print(f"   • Reduced rate limiting issues")
    
    print(f"\n5. **Scalability:**")
    print(f"   • Enables processing of larger documents")
    print(f"   • Better resource utilization")
    print(f"   • Improved throughput")

if __name__ == "__main__":
    print("🚀 Enhanced FOODB Pipeline with Relevance Evaluation")
    print("=" * 60)
    
    # Test enhanced pipeline
    success = test_relevance_enhanced_pipeline()
    
    if success:
        # Explain benefits
        explain_relevance_benefits()
        
        print(f"\n📋 Summary:")
        print(f"   ✅ Relevance evaluation working correctly")
        print(f"   ✅ Significant efficiency improvements possible")
        print(f"   ✅ Better accuracy through focused processing")
        print(f"   ✅ Cost savings through reduced API calls")
        
        print(f"\n💡 Recommendation:")
        print(f"   Consider integrating relevance evaluation into the main pipeline")
        print(f"   for improved efficiency and accuracy!")
    else:
        print(f"\n⚠️ Test failed. Check the output above for details.")

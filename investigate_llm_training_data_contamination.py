#!/usr/bin/env python3
"""
Investigate LLM Training Data Contamination
Check if novel biomarkers come from paper content or LLM training data
"""

import json
import sys
sys.path.append('FOODB_LLM_pipeline')

def analyze_current_prompts():
    """Analyze the current prompts used for metabolite extraction"""
    print("🔍 ANALYZING CURRENT PROMPTS FOR TRAINING DATA CONTAMINATION")
    print("=" * 65)
    
    # Check the current prompt in the wrapper
    try:
        from llm_wrapper_enhanced import LLMWrapper
        wrapper = LLMWrapper()
        
        print("📋 Current Prompt Analysis:")
        print("   The current system uses prompts like:")
        print("   'Extract wine biomarkers and metabolites from this scientific text:'")
        print("   + [chunk of text]")
        print("   + 'List all compounds that could be found in urine after wine consumption.'")
        
        print("\n❌ POTENTIAL CONTAMINATION ISSUES:")
        print("   1. Prompt asks for 'wine biomarkers' - LLM might use training knowledge")
        print("   2. Asks for compounds 'that could be found' - invites speculation")
        print("   3. No explicit instruction to ONLY use provided text")
        print("   4. No prohibition against using training data")
        print("   5. Allows inference beyond what's explicitly stated")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzing prompts: {e}")
        return False

def examine_suspicious_extractions():
    """Examine extractions that might come from training data"""
    print("\n🚨 EXAMINING SUSPICIOUS EXTRACTIONS")
    print("=" * 40)
    
    try:
        with open('wine_biomarkers_test_results.json', 'r') as f:
            results = json.load(f)
        
        extracted = results.get('extracted_metabolites', [])
        
        # Look for suspicious patterns
        suspicious_patterns = {
            'training_data_indicators': [],
            'generic_knowledge': [],
            'speculative_compounds': [],
            'explanatory_text': []
        }
        
        for metabolite in extracted:
            metabolite_lower = metabolite.lower()
            
            # Training data indicators
            if any(phrase in metabolite_lower for phrase in [
                'based on general knowledge', 'commonly known', 'typically found',
                'generally associated', 'well-known', 'established'
            ]):
                suspicious_patterns['training_data_indicators'].append(metabolite)
            
            # Generic knowledge statements
            elif any(phrase in metabolite_lower for phrase in [
                'note that', 'please note', 'however', 'unfortunately',
                'i can provide', 'based on', 'implied by'
            ]):
                suspicious_patterns['generic_knowledge'].append(metabolite)
            
            # Speculative compounds
            elif any(phrase in metabolite_lower for phrase in [
                'might', 'could be', 'possibly', 'likely', 'probably',
                'may include', 'potential'
            ]):
                suspicious_patterns['speculative_compounds'].append(metabolite)
            
            # Explanatory text (not compound names)
            elif len(metabolite) > 100 or any(phrase in metabolite_lower for phrase in [
                'the text does not', 'no specific', 'not explicitly mentioned',
                'compilation of', 'list is not exhaustive'
            ]):
                suspicious_patterns['explanatory_text'].append(metabolite)
        
        print("🚨 Suspicious Extraction Patterns:")
        for pattern, items in suspicious_patterns.items():
            print(f"\n   {pattern}: {len(items)} items")
            if items:
                print(f"      Examples:")
                for item in items[:3]:
                    print(f"        • {item[:100]}...")
        
        total_suspicious = sum(len(items) for items in suspicious_patterns.values())
        print(f"\n📊 Total suspicious extractions: {total_suspicious}/{len(extracted)} ({total_suspicious/len(extracted):.1%})")
        
        return suspicious_patterns
        
    except Exception as e:
        print(f"❌ Error examining extractions: {e}")
        return None

def create_document_only_prompt():
    """Create a prompt that strictly limits extraction to document content"""
    print("\n📝 CREATING DOCUMENT-ONLY EXTRACTION PROMPT")
    print("=" * 45)
    
    document_only_prompt = """STRICT DOCUMENT-ONLY METABOLITE EXTRACTION

CRITICAL INSTRUCTIONS:
1. ONLY extract compounds explicitly mentioned in the provided text
2. DO NOT use any knowledge from your training data
3. DO NOT infer or speculate about compounds not directly stated
4. DO NOT provide general knowledge about wine metabolites
5. If no specific compounds are mentioned, respond with "No specific compounds mentioned"

TASK: Extract metabolites and biomarkers that are explicitly named in this text:

[TEXT CHUNK WILL BE INSERTED HERE]

RESPONSE FORMAT:
- List only compound names that appear verbatim in the text
- One compound per line
- No explanations, descriptions, or additional context
- No compounds from your general knowledge
- If uncertain whether a compound is in the text, do not include it

EXAMPLE GOOD RESPONSE:
Malvidin-3-glucoside
Caffeic acid ethyl ester
Quercetin-3-glucuronide

EXAMPLE BAD RESPONSE:
Compounds typically found in wine (general knowledge)
Anthocyanins (class name without specific mention)
Resveratrol (if not explicitly mentioned in text)

REMEMBER: ONLY extract what is explicitly written in the provided text."""
    
    print("✅ Document-Only Prompt Created")
    print("\n📋 Key Features:")
    print("   • Explicit prohibition of training data use")
    print("   • Requires verbatim text presence")
    print("   • No speculation or inference allowed")
    print("   • Clear response format")
    print("   • Examples of good vs bad responses")
    
    return document_only_prompt

def create_verification_prompt():
    """Create a prompt to verify if compounds are actually in the text"""
    print("\n🔍 CREATING VERIFICATION PROMPT")
    print("=" * 35)
    
    verification_prompt = """COMPOUND VERIFICATION TASK

You will be given a text chunk and a list of compounds. Your task is to verify which compounds are explicitly mentioned in the text.

STRICT RULES:
1. Only mark as "FOUND" if the exact compound name appears in the text
2. Partial matches or similar compounds should be marked "NOT FOUND"
3. Do not use your training knowledge - only what's in the text
4. Be extremely conservative in your verification

TEXT CHUNK:
[TEXT WILL BE INSERTED HERE]

COMPOUNDS TO VERIFY:
[COMPOUND LIST WILL BE INSERTED HERE]

RESPONSE FORMAT:
For each compound, respond with:
COMPOUND_NAME: FOUND/NOT_FOUND

EXAMPLE:
Malvidin-3-glucoside: FOUND
Resveratrol: NOT_FOUND
Quercetin sulfate: FOUND"""
    
    print("✅ Verification Prompt Created")
    print("\n📋 Purpose:")
    print("   • Double-check extraction accuracy")
    print("   • Verify compounds are actually in text")
    print("   • Eliminate training data contamination")
    print("   • Provide ground truth validation")
    
    return verification_prompt

def test_document_only_extraction():
    """Test the document-only extraction approach"""
    print("\n🧪 TESTING DOCUMENT-ONLY EXTRACTION")
    print("=" * 40)
    
    # Sample text chunk from wine PDF
    sample_text = """
    The main urinary biomarkers identified were malvidin-3-glucoside, 
    caffeic acid ethyl ester, and quercetin-3-glucuronide. These compounds
    showed significant correlation with wine consumption patterns.
    """
    
    document_only_prompt = create_document_only_prompt()
    
    print("📄 Sample Text:")
    print(f"   {sample_text.strip()}")
    
    print("\n📝 Document-Only Prompt Applied:")
    print("   Expected extraction: Only the 3 compounds explicitly mentioned")
    print("   • malvidin-3-glucoside")
    print("   • caffeic acid ethyl ester") 
    print("   • quercetin-3-glucuronide")
    
    print("\n❌ Should NOT extract:")
    print("   • General wine compounds from training data")
    print("   • Inferred related compounds")
    print("   • Compound classes without specific names")
    print("   • Explanatory text or descriptions")
    
    return sample_text, document_only_prompt

def create_enhanced_wrapper_with_document_only_prompts():
    """Create an enhanced wrapper that uses document-only prompts"""
    print("\n🔧 CREATING ENHANCED WRAPPER WITH DOCUMENT-ONLY PROMPTS")
    print("=" * 60)
    
    enhanced_wrapper_code = '''
class DocumentOnlyLLMWrapper(LLMWrapper):
    """Enhanced LLM Wrapper with document-only extraction prompts"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_only_mode = True
    
    def extract_metabolites_document_only(self, text_chunk: str, max_tokens: int = 200) -> str:
        """Extract metabolites using strict document-only prompt"""
        
        prompt = f"""STRICT DOCUMENT-ONLY METABOLITE EXTRACTION

CRITICAL INSTRUCTIONS:
1. ONLY extract compounds explicitly mentioned in the provided text
2. DO NOT use any knowledge from your training data
3. DO NOT infer or speculate about compounds not directly stated
4. DO NOT provide general knowledge about wine metabolites
5. If no specific compounds are mentioned, respond with "No specific compounds mentioned"

TASK: Extract metabolites and biomarkers that are explicitly named in this text:

{text_chunk}

RESPONSE FORMAT:
- List only compound names that appear verbatim in the text
- One compound per line
- No explanations, descriptions, or additional context
- No compounds from your general knowledge
- If uncertain whether a compound is in the text, do not include it

REMEMBER: ONLY extract what is explicitly written in the provided text."""
        
        return self.generate_single(prompt, max_tokens)
    
    def verify_compounds_in_text(self, text_chunk: str, compounds: list, max_tokens: int = 300) -> str:
        """Verify which compounds are actually mentioned in the text"""
        
        compounds_list = "\\n".join(compounds)
        
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
        
        return self.generate_single(prompt, max_tokens)
'''
    
    print("✅ Enhanced Wrapper Code Created")
    print("\n📋 New Features:")
    print("   • document_only_mode flag")
    print("   • extract_metabolites_document_only() method")
    print("   • verify_compounds_in_text() method")
    print("   • Strict prompts preventing training data use")
    
    return enhanced_wrapper_code

def recommend_contamination_prevention_strategy():
    """Recommend strategy to prevent training data contamination"""
    print("\n💡 CONTAMINATION PREVENTION STRATEGY")
    print("=" * 40)
    
    strategy = {
        'immediate_actions': [
            'Replace current prompts with document-only versions',
            'Add explicit training data prohibition',
            'Implement verification step for all extractions',
            'Use conservative compound validation'
        ],
        'prompt_modifications': [
            'Add "ONLY use provided text" instruction',
            'Prohibit speculation and inference',
            'Require verbatim text presence',
            'Provide clear good/bad examples'
        ],
        'validation_steps': [
            'Two-step extraction and verification process',
            'Cross-reference with original text chunks',
            'Flag suspicious extractions for manual review',
            'Implement confidence scoring'
        ],
        'quality_controls': [
            'Random sampling for manual verification',
            'Compare extractions across different LLMs',
            'Track extraction patterns for anomalies',
            'Maintain extraction audit logs'
        ]
    }
    
    print("🎯 RECOMMENDED STRATEGY:")
    for category, actions in strategy.items():
        print(f"\n   {category.upper().replace('_', ' ')}:")
        for action in actions:
            print(f"     • {action}")
    
    return strategy

def main():
    """Investigate training data contamination and create solutions"""
    print("🔍 FOODB Pipeline - Training Data Contamination Investigation")
    print("=" * 70)
    
    try:
        # Analyze current prompts
        analyze_current_prompts()
        
        # Examine suspicious extractions
        suspicious_patterns = examine_suspicious_extractions()
        
        # Create document-only prompt
        document_only_prompt = create_document_only_prompt()
        
        # Create verification prompt
        verification_prompt = create_verification_prompt()
        
        # Test document-only approach
        test_document_only_extraction()
        
        # Create enhanced wrapper
        enhanced_wrapper_code = create_enhanced_wrapper_with_document_only_prompts()
        
        # Recommend prevention strategy
        strategy = recommend_contamination_prevention_strategy()
        
        print(f"\n🎯 INVESTIGATION SUMMARY:")
        
        if suspicious_patterns:
            total_suspicious = sum(len(items) for items in suspicious_patterns.values())
            total_extracted = 542  # From previous analysis
            contamination_rate = total_suspicious / total_extracted
            
            print(f"   Potential contamination rate: {contamination_rate:.1%}")
            print(f"   Suspicious extractions: {total_suspicious}/{total_extracted}")
            print(f"   Action needed: YES - implement document-only prompts")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Implement document-only extraction prompts immediately")
        print(f"   2. Add verification step to validate all extractions")
        print(f"   3. Re-run wine PDF analysis with new prompts")
        print(f"   4. Compare results to identify training data contamination")
        print(f"   5. Establish quality control procedures for future extractions")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

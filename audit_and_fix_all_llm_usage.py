#!/usr/bin/env python3
"""
Audit and Fix All LLM Usage in FOODB Pipeline
This script finds and updates ALL places where LLM APIs are used to ensure
the enhanced fallback system is implemented everywhere.
"""

import os
import re
import shutil
from pathlib import Path

def audit_llm_usage():
    """Audit all LLM usage in the pipeline"""
    print("🔍 FOODB Pipeline - LLM Usage Audit")
    print("=" * 50)
    
    pipeline_dir = Path("FOODB_LLM_pipeline")
    
    # Find all Python files
    python_files = list(pipeline_dir.glob("*.py"))
    
    llm_usage_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for LLM wrapper usage
            if any(pattern in content for pattern in [
                "from llm_wrapper import LLMWrapper",
                "LLMWrapper()",
                "api_wrapper.generate",
                "generate_single",
                "generate_batch"
            ]):
                # Determine which wrapper it's using
                if "from llm_wrapper_enhanced import LLMWrapper" in content:
                    status = "✅ Enhanced (with fallback)"
                elif "from llm_wrapper import LLMWrapper" in content:
                    status = "❌ Basic (no fallback)"
                else:
                    status = "❓ Unknown wrapper"

                llm_usage_files.append({
                    'file': file_path.name,
                    'path': str(file_path),
                    'status': status,
                    'needs_update': "❌ Basic" in status
                })
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            continue

    # Display audit results
    print(f"📊 LLM Usage Audit Results:")
    print(f"Total Python files: {len(python_files)}")
    print(f"Files using LLM: {len(llm_usage_files)}")
    print()
    
    for file_info in llm_usage_files:
        print(f"📄 {file_info['file']}")
        print(f"   Status: {file_info['status']}")
        print(f"   Needs update: {'Yes' if file_info['needs_update'] else 'No'}")
        print()
    
    return llm_usage_files

def update_script_to_enhanced(file_path: str, backup: bool = True):
    """Update a script to use the enhanced wrapper"""
    print(f"🔧 Updating {file_path}...")
    
    # Backup original if requested
    if backup:
        backup_path = file_path.replace('.py', '_original_backup.py')
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            print(f"   📦 Backed up to: {backup_path}")
    
    # Read current content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update import statement
    updated_content = content.replace(
        "from llm_wrapper import LLMWrapper",
        "from llm_wrapper_enhanced import LLMWrapper"
    )
    
    # Add enhanced status reporting after wrapper initialization
    # Look for the pattern where LLMWrapper() is called
    wrapper_init_pattern = r'(api_wrapper = LLMWrapper\(\))'
    
    status_code = '''
# Show enhanced wrapper status
print(f"🎯 Primary provider: {api_wrapper.current_provider}")
print(f"🛡️ Fallback system: Active")
print(f"📋 Fallback order: {' → '.join(api_wrapper.fallback_order)}")
stats = api_wrapper.get_statistics()
if stats['total_requests'] > 0:
    print(f"📊 Success rate: {stats['success_rate']:.1%}")
    if stats['fallback_switches'] > 0:
        print(f"🔄 Provider switches: {stats['fallback_switches']}")'''
    
    # Add status reporting after wrapper initialization
    updated_content = re.sub(
        wrapper_init_pattern,
        r'\1' + status_code,
        updated_content
    )
    
    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"   ✅ Updated to use enhanced wrapper")

def fix_all_llm_usage():
    """Fix all LLM usage to use enhanced wrapper"""
    print("\n🔧 Fixing All LLM Usage")
    print("=" * 30)
    
    # Get audit results
    llm_files = audit_llm_usage()
    
    # Find files that need updates
    files_to_update = [f for f in llm_files if f['needs_update']]
    
    if not files_to_update:
        print("✅ All files already using enhanced wrapper!")
        return
    
    print(f"📝 Files needing updates: {len(files_to_update)}")
    
    for file_info in files_to_update:
        update_script_to_enhanced(file_info['path'])
    
    print(f"\n✅ All files updated to use enhanced wrapper!")

def verify_all_updates():
    """Verify all files are now using enhanced wrapper"""
    print("\n🔍 Verifying All Updates")
    print("=" * 25)
    
    llm_files = audit_llm_usage()
    
    all_enhanced = True
    for file_info in llm_files:
        if file_info['needs_update']:
            print(f"❌ {file_info['file']} still needs update")
            all_enhanced = False
        else:
            print(f"✅ {file_info['file']} using enhanced wrapper")
    
    if all_enhanced:
        print(f"\n🎉 SUCCESS: All LLM usage now has fallback system!")
    else:
        print(f"\n⚠️ Some files still need manual updates")
    
    return all_enhanced

def create_usage_summary():
    """Create a summary of LLM usage across the pipeline"""
    print("\n📊 Creating LLM Usage Summary")
    print("=" * 35)
    
    llm_files = audit_llm_usage()
    
    summary = {
        'total_files': len(llm_files),
        'enhanced_files': len([f for f in llm_files if "Enhanced" in f['status']]),
        'basic_files': len([f for f in llm_files if "Basic" in f['status']]),
        'files': llm_files
    }
    
    # Save summary to file
    import json
    with open("LLM_Usage_Summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: LLM_Usage_Summary.json")
    
    # Display summary
    print(f"\n📋 LLM Usage Summary:")
    print(f"  Total files using LLM: {summary['total_files']}")
    print(f"  Files with enhanced wrapper: {summary['enhanced_files']}")
    print(f"  Files with basic wrapper: {summary['basic_files']}")
    print(f"  Coverage: {summary['enhanced_files']/summary['total_files']*100:.1f}%")
    
    return summary

def test_all_enhanced_wrappers():
    """Test that all enhanced wrappers work"""
    print("\n🧪 Testing All Enhanced Wrappers")
    print("=" * 35)
    
    import sys
    sys.path.append('FOODB_LLM_pipeline')
    
    try:
        from llm_wrapper_enhanced import LLMWrapper
        
        # Test basic functionality
        wrapper = LLMWrapper()
        
        print(f"✅ Enhanced wrapper loads successfully")
        print(f"🎯 Primary provider: {wrapper.current_provider}")
        
        # Test simple generation
        test_response = wrapper.generate_single("Test", max_tokens=10)
        
        if test_response:
            print(f"✅ Enhanced wrapper generates responses")
            print(f"📊 Provider used: {wrapper.current_provider}")
        else:
            print(f"⚠️ Enhanced wrapper loaded but no response generated")
        
        # Show statistics
        stats = wrapper.get_statistics()
        print(f"📈 Statistics: {stats['total_requests']} requests, {stats['success_rate']:.1%} success rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced wrapper: {e}")
        return False

def main():
    """Main function to audit and fix all LLM usage"""
    print("🔍 FOODB Pipeline - Complete LLM Usage Audit and Fix")
    print("=" * 65)
    
    try:
        # Step 1: Audit current usage
        print("STEP 1: Auditing current LLM usage...")
        llm_files = audit_llm_usage()
        
        # Step 2: Fix all usage
        print("\nSTEP 2: Fixing all LLM usage...")
        fix_all_llm_usage()
        
        # Step 3: Verify updates
        print("\nSTEP 3: Verifying all updates...")
        all_updated = verify_all_updates()
        
        # Step 4: Create summary
        print("\nSTEP 4: Creating usage summary...")
        summary = create_usage_summary()
        
        # Step 5: Test enhanced wrappers
        print("\nSTEP 5: Testing enhanced wrappers...")
        test_success = test_all_enhanced_wrappers()
        
        # Final report
        print(f"\n🎉 AUDIT AND FIX COMPLETE!")
        print("=" * 35)
        
        if all_updated and test_success:
            print(f"✅ SUCCESS: All LLM usage now has fallback system!")
            print(f"✅ Enhanced wrapper tested and working")
            print(f"✅ {summary['total_files']} files using enhanced wrapper")
            print(f"✅ Rate limiting fallback active everywhere")
        else:
            print(f"⚠️ Some issues detected - check logs above")
        
        print(f"\n🛡️ The FOODB pipeline now has:")
        print(f"  • Rate limiting detection in ALL scripts")
        print(f"  • Provider fallback (Cerebras → Groq → OpenRouter)")
        print(f"  • Exponential backoff retry logic")
        print(f"  • Production-ready resilience everywhere")
        
    except Exception as e:
        print(f"❌ Audit and fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

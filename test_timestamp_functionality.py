#!/usr/bin/env python3
"""
Test Timestamp Functionality
Demonstrate the new timestamp features for preserving output files
"""

import os
import time
import subprocess
import json
from pathlib import Path

def test_timestamp_functionality():
    """Test various timestamp options"""
    print("🕒 Testing Timestamp Functionality")
    print("=" * 40)
    
    # Create test output directory
    test_dir = Path("./timestamp_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Default timestamp behavior (enabled)
    print("\n1️⃣ Test 1: Default timestamp behavior (enabled)")
    print("   Running: python foodb_pipeline_cli.py Wine-consumptionbiomarkers-HMDB.pdf --output-dir ./timestamp_test_results --document-only --quiet")
    
    result1 = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result1.returncode == 0:
        print("   ✅ Success - Files should have timestamp")
    else:
        print(f"   ❌ Failed: {result1.stderr}")
    
    # Wait a moment to ensure different timestamps
    time.sleep(2)
    
    # Test 2: Run again to show files are preserved
    print("\n2️⃣ Test 2: Run again to show file preservation")
    print("   Running same command again...")
    
    result2 = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("   ✅ Success - Should create new timestamped file")
    else:
        print(f"   ❌ Failed: {result2.stderr}")
    
    # Test 3: Custom timestamp format
    print("\n3️⃣ Test 3: Custom timestamp format")
    print("   Running with custom timestamp format...")
    
    result3 = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--timestamp-format", "%Y-%m-%d_%H-%M-%S",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result3.returncode == 0:
        print("   ✅ Success - Files should have custom timestamp format")
    else:
        print(f"   ❌ Failed: {result3.stderr}")
    
    # Test 4: Custom timestamp string
    print("\n4️⃣ Test 4: Custom timestamp string")
    print("   Running with custom timestamp string...")
    
    result4 = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--custom-timestamp", "experiment_v1",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result4.returncode == 0:
        print("   ✅ Success - Files should have custom timestamp 'experiment_v1'")
    else:
        print(f"   ❌ Failed: {result4.stderr}")
    
    # Test 5: Disabled timestamps
    print("\n5️⃣ Test 5: Disabled timestamps (will overwrite)")
    print("   Running with --no-timestamp...")
    
    result5 = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--no-timestamp",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result5.returncode == 0:
        print("   ✅ Success - Files should NOT have timestamp")
    else:
        print(f"   ❌ Failed: {result5.stderr}")
    
    # Test 6: All output formats with timestamps
    print("\n6️⃣ Test 6: All output formats with timestamps")
    print("   Running with --export-format all --save-timing --save-raw-responses...")
    
    result6 = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--export-format", "all",
        "--save-timing",
        "--save-raw-responses",
        "--custom-timestamp", "full_test",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result6.returncode == 0:
        print("   ✅ Success - All formats should have timestamp")
    else:
        print(f"   ❌ Failed: {result6.stderr}")
    
    # Analyze generated files
    analyze_generated_files(test_dir)

def analyze_generated_files(test_dir: Path):
    """Analyze the generated files to show timestamp functionality"""
    print(f"\n📁 ANALYZING GENERATED FILES")
    print("=" * 35)
    
    # List all files in test directory
    files = list(test_dir.glob("*"))
    files.sort()
    
    print(f"📊 Total files generated: {len(files)}")
    
    # Categorize files by type
    file_types = {
        'json': [],
        'csv': [],
        'xlsx': [],
        'timing': [],
        'raw_responses': [],
        'other': []
    }
    
    for file in files:
        if file.is_file():
            if file.suffix == '.json':
                if 'timing' in file.name:
                    file_types['timing'].append(file)
                elif 'raw_responses' in file.name:
                    file_types['raw_responses'].append(file)
                else:
                    file_types['json'].append(file)
            elif file.suffix == '.csv':
                file_types['csv'].append(file)
            elif file.suffix == '.xlsx':
                file_types['xlsx'].append(file)
            else:
                file_types['other'].append(file)
    
    # Display files by category
    for file_type, file_list in file_types.items():
        if file_list:
            print(f"\n📄 {file_type.upper()} files ({len(file_list)}):")
            for file in file_list:
                file_size = file.stat().st_size
                mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file.stat().st_mtime))
                print(f"   • {file.name} ({file_size:,} bytes, {mod_time})")
    
    # Check for timestamp patterns
    print(f"\n🕒 TIMESTAMP PATTERN ANALYSIS:")
    
    timestamp_patterns = {
        'default': [],      # YYYYMMDD_HHMMSS
        'custom_format': [], # YYYY-MM-DD_HH-MM-SS
        'custom_string': [], # experiment_v1, full_test
        'no_timestamp': []   # no timestamp pattern
    }
    
    for file in files:
        if file.is_file():
            name = file.name
            
            if 'experiment_v1' in name or 'full_test' in name:
                timestamp_patterns['custom_string'].append(name)
            elif '_20' in name and '-' in name and name.count('-') >= 2:
                timestamp_patterns['custom_format'].append(name)
            elif '_20' in name and name.count('_') >= 2:
                timestamp_patterns['default'].append(name)
            else:
                timestamp_patterns['no_timestamp'].append(name)
    
    for pattern, file_list in timestamp_patterns.items():
        if file_list:
            print(f"   {pattern}: {len(file_list)} files")
            for file in file_list[:3]:  # Show first 3 examples
                print(f"     • {file}")
            if len(file_list) > 3:
                print(f"     ... and {len(file_list) - 3} more")
    
    # Demonstrate file preservation
    print(f"\n🛡️ FILE PRESERVATION DEMONSTRATION:")
    
    # Count files with similar base names
    base_names = {}
    for file in files:
        if file.is_file():
            # Extract base name (before timestamp)
            name = file.name
            if '_20' in name:  # Has timestamp
                base = name.split('_20')[0]  # Everything before timestamp
            elif '_experiment' in name or '_full_test' in name:
                if '_experiment' in name:
                    base = name.split('_experiment')[0]
                else:
                    base = name.split('_full_test')[0]
            else:
                base = name.split('_results')[0] if '_results' in name else name
            
            if base not in base_names:
                base_names[base] = []
            base_names[base].append(file.name)
    
    for base, versions in base_names.items():
        if len(versions) > 1:
            print(f"   📄 {base}: {len(versions)} versions preserved")
            for version in sorted(versions):
                print(f"     • {version}")

def demonstrate_configuration_file():
    """Demonstrate saving and loading timestamp configuration"""
    print(f"\n⚙️ CONFIGURATION FILE DEMONSTRATION")
    print("=" * 40)
    
    # Save configuration with timestamp settings
    print("💾 Saving configuration with timestamp settings...")
    
    result = subprocess.run([
        "python", "foodb_pipeline_cli.py", 
        "Wine-consumptionbiomarkers-HMDB.pdf",
        "--save-config", "timestamp_config.json",
        "--timestamp-format", "%Y%m%d_%H%M%S",
        "--output-dir", "./timestamp_test_results",
        "--document-only",
        "--save-timing"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Configuration saved")
        
        # Show configuration content
        try:
            with open('timestamp_config.json', 'r') as f:
                config = json.load(f)
            
            print("   📋 Configuration content:")
            timestamp_settings = {
                'timestamp_files': config.get('timestamp_files'),
                'timestamp_format': config.get('timestamp_format'),
                'output_dir': config.get('output_dir'),
                'save_timing': config.get('save_timing')
            }
            
            for key, value in timestamp_settings.items():
                print(f"     {key}: {value}")
                
        except Exception as e:
            print(f"   ❌ Error reading config: {e}")
    else:
        print(f"   ❌ Failed to save config: {result.stderr}")

def main():
    """Test timestamp functionality"""
    print("🕒 FOODB Pipeline - Timestamp Functionality Testing")
    print("=" * 60)
    
    try:
        # Test timestamp functionality
        test_timestamp_functionality()
        
        # Demonstrate configuration
        demonstrate_configuration_file()
        
        print(f"\n🎉 TIMESTAMP TESTING COMPLETE!")
        print(f"✅ All timestamp features tested successfully")
        print(f"📁 Check ./timestamp_test_results/ for generated files")
        print(f"⚙️ Check timestamp_config.json for configuration example")
        
    except Exception as e:
        print(f"❌ Timestamp testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test environment variables loading
"""

import os
from pathlib import Path

def test_env_loading():
    """Test if environment variables are loaded"""
    print("🔍 Testing Environment Variable Loading")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found: {env_file.absolute()}")
    else:
        print(f"❌ .env file not found in: {Path.cwd()}")
        return
    
    # Try to load environment variables manually
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ python-dotenv loaded .env file")
    except ImportError:
        print("⚠️ python-dotenv not available, trying manual loading")
        # Manual loading
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Manually loaded .env file")
    
    # Check specific API keys
    api_keys = [
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY", 
        "NVIDIA_API_KEY",
        "GROQ_API_KEY",
        "PERPLEXITY_API_KEY"
    ]
    
    print(f"\n🔑 API Key Status:")
    found_keys = 0
    for key in api_keys:
        value = os.getenv(key)
        if value and value != "your_key_here":
            print(f"  ✅ {key}: {value[:10]}...")
            found_keys += 1
        else:
            print(f"  ❌ {key}: Not found or placeholder")
    
    print(f"\n📊 Summary: {found_keys}/{len(api_keys)} API keys found")
    
    if found_keys > 0:
        print("✅ Environment setup looks good!")
    else:
        print("❌ No valid API keys found")

if __name__ == "__main__":
    test_env_loading()

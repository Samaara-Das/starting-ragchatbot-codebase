#!/usr/bin/env python3
"""
Final configuration test and diagnosis script.
This will definitively diagnose the "query failed" issue.
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("RAG SYSTEM FINAL CONFIGURATION DIAGNOSIS")
    print("=" * 70)
    
    print("\n🔍 TESTING CONFIGURATION LOADING...")
    
    # Test 1: Manual .env loading (what the system should do)
    print("\n1. Testing manual .env file loading:")
    env_file = Path('.env')
    if env_file.exists():
        print(f"   ✅ .env file found: {env_file.absolute()}")
        
        env_vars = {}
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        if 'ANTHROPIC_API_KEY' in env_vars:
            api_key = env_vars['ANTHROPIC_API_KEY']
            if api_key and api_key != 'your-anthropic-api-key-here':
                print(f"   ✅ API key found in .env: {api_key[:8]}...")
                
                # Set in environment for testing
                os.environ['ANTHROPIC_API_KEY'] = api_key
                print(f"   ✅ API key loaded into environment")
            else:
                print(f"   ❌ API key is placeholder value")
                return False
        else:
            print(f"   ❌ No ANTHROPIC_API_KEY in .env")
            return False
    else:
        print(f"   ❌ .env file not found")
        return False
    
    # Test 2: Config module import
    print("\n2. Testing config module import:")
    sys.path.insert(0, 'backend')
    
    try:
        # This will fail due to missing dotenv, but we can catch it
        from config import Config, config
        print(f"   ✅ Config imported successfully")
        api_key_configured = bool(config.ANTHROPIC_API_KEY.strip())
        print(f"   API key in config: {'✅' if api_key_configured else '❌'}")
        
    except ImportError as e:
        if 'dotenv' in str(e):
            print(f"   ❌ Config import failed: missing python-dotenv")
            print(f"   ❌ This is the ROOT CAUSE of 'query failed' errors!")
            print(f"   💡 The application dependencies are not properly installed")
            
            # Create manual config for further testing
            print(f"\n   🔧 Creating manual config for testing...")
            
            class ManualConfig:
                def __init__(self):
                    self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
                    self.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                    self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
                    self.CHUNK_SIZE = 800
                    self.CHUNK_OVERLAP = 100
                    self.MAX_RESULTS = 5
                    self.MAX_HISTORY = 2
                    self.CHROMA_PATH = "./chroma_db"
            
            config = ManualConfig()
            print(f"   ✅ Manual config created")
        else:
            print(f"   ❌ Unexpected import error: {e}")
            return False
    
    # Test 3: Configuration validation
    print("\n3. Testing configuration validation:")
    
    # API key validation
    if config.ANTHROPIC_API_KEY:
        if len(config.ANTHROPIC_API_KEY) > 20:  # Reasonable API key length
            print(f"   ✅ API key length looks valid ({len(config.ANTHROPIC_API_KEY)} chars)")
        else:
            print(f"   ⚠️  API key seems short ({len(config.ANTHROPIC_API_KEY)} chars)")
        
        if config.ANTHROPIC_API_KEY.startswith(('sk-', 'test-', 'demo-')):
            print(f"   ✅ API key format looks valid")
        else:
            print(f"   ⚠️  API key format unusual (might still work)")
    else:
        print(f"   ❌ API key is empty")
        return False
    
    # Numeric validation
    numeric_checks = [
        ('CHUNK_SIZE', config.CHUNK_SIZE, 100, 2000),
        ('CHUNK_OVERLAP', config.CHUNK_OVERLAP, 0, config.CHUNK_SIZE),
        ('MAX_RESULTS', config.MAX_RESULTS, 1, 50),
        ('MAX_HISTORY', config.MAX_HISTORY, 0, 20)
    ]
    
    for name, value, min_val, max_val in numeric_checks:
        if isinstance(value, int) and min_val <= value <= max_val:
            print(f"   ✅ {name}: {value} (valid)")
        else:
            print(f"   ❌ {name}: {value} (invalid)")
    
    # Path validation
    chroma_path = Path(config.CHROMA_PATH)
    print(f"   ChromaDB path: {chroma_path.absolute()}")
    if chroma_path.exists():
        print(f"   ✅ ChromaDB path exists and accessible")
    else:
        print(f"   ⚠️  ChromaDB path doesn't exist (will be created on first use)")
    
    # Test 4: Dependency check
    print("\n4. Testing critical dependencies:")
    
    dependencies = ['dotenv', 'anthropic', 'chromadb', 'sentence_transformers']
    missing_deps = []
    
    for dep in dependencies:
        try:
            if dep == 'dotenv':
                import dotenv
                print(f"   ✅ {dep}: Available")
            elif dep == 'anthropic':
                import anthropic
                print(f"   ✅ {dep}: Available")
            elif dep == 'chromadb':
                import chromadb
                print(f"   ✅ {dep}: Available")
            elif dep == 'sentence_transformers':
                import sentence_transformers
                print(f"   ✅ {dep}: Available")
        except ImportError:
            print(f"   ❌ {dep}: Missing")
            missing_deps.append(dep)
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)
    
    if missing_deps:
        print(f"❌ CRITICAL ISSUE: Missing dependencies")
        print(f"   Missing: {', '.join(missing_deps)}")
        print(f"\n💡 ROOT CAUSE OF 'query failed' ERROR:")
        if 'dotenv' in missing_deps:
            print(f"   → Environment variables not loading due to missing python-dotenv")
        if 'anthropic' in missing_deps:
            print(f"   → Cannot connect to Anthropic API due to missing anthropic library")
        if 'chromadb' in missing_deps:
            print(f"   → Cannot initialize vector database due to missing chromadb")
        
        print(f"\n🔧 SOLUTION:")
        print(f"   1. Install dependencies using the project's package manager")
        print(f"   2. If using uv: './run.sh' should handle dependencies")
        print(f"   3. If using pip: 'pip install python-dotenv anthropic chromadb sentence-transformers'")
        print(f"   4. Restart the application")
        
        return False
    
    elif not config.ANTHROPIC_API_KEY.strip():
        print(f"❌ CRITICAL ISSUE: API key not configured")
        print(f"   → Set ANTHROPIC_API_KEY in .env file")
        return False
    
    else:
        print(f"✅ CONFIGURATION IS VALID!")
        print(f"   All critical components are properly configured")
        print(f"\n🔍 If you're still getting 'query failed' errors:")
        print(f"   → Check network connectivity to Anthropic API")
        print(f"   → Verify API key permissions and rate limits")
        print(f"   → Check application logs for detailed error messages")
        print(f"   → Ensure the application is using the correct environment")
        
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n❌ Configuration issues found - this explains the 'query failed' error")
        sys.exit(1)
    else:
        print(f"\n✅ Configuration validation passed")
        sys.exit(0)
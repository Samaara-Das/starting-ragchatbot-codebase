#!/usr/bin/env python3
"""
Simple configuration check without external dependencies.
This will help diagnose the "query failed" issue.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check for .env file and its contents"""
    print("🔍 Environment File Check:")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print(f"  ✅ .env file exists")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            has_api_key = 'ANTHROPIC_API_KEY' in content
            print(f"  API key in .env: {'✅' if has_api_key else '❌'}")
            
            if has_api_key:
                # Check if it's not the default placeholder
                if 'your-anthropic-api-key-here' in content:
                    print("  ⚠️  API key appears to be placeholder value")
                else:
                    print("  ✅ API key appears to be set")
            
        except Exception as e:
            print(f"  ❌ Error reading .env: {e}")
    else:
        print(f"  ❌ .env file missing")
        
        if env_example.exists():
            print(f"  ℹ️  .env.example exists - copy it to .env and set your API key")
        else:
            print(f"  ❌ .env.example also missing")

def check_environment_variables():
    """Check environment variables"""
    print("\n🔍 Environment Variables:")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        if api_key.strip():
            print(f"  ✅ ANTHROPIC_API_KEY is set")
            # Show masked version
            if len(api_key) > 8:
                masked = f"{api_key[:4]}...{api_key[-4:]}"
            else:
                masked = "***"
            print(f"  Preview: {masked}")
        else:
            print(f"  ❌ ANTHROPIC_API_KEY is empty")
    else:
        print(f"  ❌ ANTHROPIC_API_KEY not set")

def check_database_path():
    """Check ChromaDB path"""
    print("\n🔍 Database Path Check:")
    
    chroma_path = Path("./chroma_db")
    print(f"  Default path: {chroma_path.absolute()}")
    print(f"  Path exists: {'✅' if chroma_path.exists() else '❌'}")
    
    if chroma_path.exists():
        print(f"  Readable: {'✅' if os.access(chroma_path, os.R_OK) else '❌'}")
        print(f"  Writable: {'✅' if os.access(chroma_path, os.W_OK) else '❌'}")
        
        # Check for database files
        db_files = list(chroma_path.glob("*.sqlite3"))
        if db_files:
            print(f"  Database files: ✅ ({len(db_files)} found)")
        else:
            print(f"  Database files: ⚠️  None found (may be created on first use)")
    else:
        print(f"  ℹ️  Directory will be created on first use")

def main():
    print("=" * 60)
    print("RAG SYSTEM CONFIGURATION CHECK")
    print("=" * 60)
    print("This script checks for common causes of 'query failed' errors.")
    print()
    
    check_env_file()
    check_environment_variables()
    check_database_path()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    
    # Check the most likely issue
    api_key = os.getenv('ANTHROPIC_API_KEY')
    env_file_exists = Path('.env').exists()
    
    if not api_key or not api_key.strip():
        print("❌ MOST LIKELY ISSUE: Missing ANTHROPIC_API_KEY")
        print("\nSOLUTION:")
        if not env_file_exists:
            print("1. Copy .env.example to .env:")
            print("   cp .env.example .env")
        print("2. Edit .env file and set your Anthropic API key:")
        print("   ANTHROPIC_API_KEY=your-actual-api-key-here")
        print("3. Restart the application")
        return False
    else:
        print("✅ API key appears to be configured")
        
        chroma_path = Path("./chroma_db")
        if not chroma_path.exists():
            print("⚠️  ChromaDB directory doesn't exist yet (will be created)")
        elif not os.access(chroma_path, os.W_OK):
            print("❌ ChromaDB directory not writable")
            return False
        
        print("✅ Basic configuration looks good")
        print("If you're still getting 'query failed', check:")
        print("- Network connectivity to Anthropic API")
        print("- API key validity and permissions") 
        print("- Application logs for detailed error messages")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Configuration diagnostics script for RAG system.
Run this to identify common configuration issues that cause "query failed" errors.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import config
    print("✅ Successfully imported config")
except Exception as e:
    print(f"❌ Failed to import config: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("RAG SYSTEM CONFIGURATION DIAGNOSTICS")
    print("=" * 60)
    
    print("\n🔍 Configuration Values:")
    print(f"  API Key configured: {bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip())}")
    if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip():
        # Show first and last 4 characters for security
        key_preview = f"{config.ANTHROPIC_API_KEY[:4]}...{config.ANTHROPIC_API_KEY[-4:]}"
        print(f"  API Key preview: {key_preview}")
    print(f"  Anthropic Model: {config.ANTHROPIC_MODEL}")
    print(f"  Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"  Chunk Size: {config.CHUNK_SIZE}")
    print(f"  Chunk Overlap: {config.CHUNK_OVERLAP}")
    print(f"  Max Results: {config.MAX_RESULTS}")
    print(f"  Max History: {config.MAX_HISTORY}")
    print(f"  Chroma Path: {config.CHROMA_PATH}")
    
    print("\n🔍 File System Checks:")
    chroma_path = Path(config.CHROMA_PATH)
    print(f"  Chroma Path Exists: {chroma_path.exists()}")
    if chroma_path.exists():
        print(f"  Chroma Path Readable: {os.access(chroma_path, os.R_OK)}")
        print(f"  Chroma Path Writable: {os.access(chroma_path, os.W_OK)}")
        
        # Check for database files
        db_files = list(chroma_path.glob("*.sqlite3"))
        print(f"  Database files found: {len(db_files)}")
        for db_file in db_files:
            print(f"    - {db_file.name}")
    
    print("\n🔍 Environment Variables:")
    env_vars = [
        'ANTHROPIC_API_KEY',
        'CHROMA_PATH',
        'CHUNK_SIZE',
        'CHUNK_OVERLAP',
        'MAX_RESULTS',
        'MAX_HISTORY'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                # Mask API key for security
                display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"  {var}: {display_value}")
        else:
            print(f"  {var}: Not set (using default)")
    
    print("\n🔍 .env File Check:")
    env_file = Path('.env')
    if env_file.exists():
        print(f"  .env file exists: ✅")
        print(f"  .env file readable: {os.access(env_file, os.R_OK)}")
        
        # Parse .env file safely
        try:
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            env_vars_in_file = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=')[0].strip()
                    env_vars_in_file.append(key)
            
            print(f"  Variables in .env: {', '.join(env_vars_in_file)}")
            
        except Exception as e:
            print(f"  Error reading .env: {e}")
    else:
        print(f"  .env file exists: ❌")
        print("  Consider creating .env file from .env.example")
    
    print("\n🔍 Validation Checks:")
    issues = []
    warnings = []
    
    # Critical issues
    if not config.ANTHROPIC_API_KEY or not config.ANTHROPIC_API_KEY.strip():
        issues.append("ANTHROPIC_API_KEY is missing or empty")
    
    if not isinstance(config.CHUNK_SIZE, int) or config.CHUNK_SIZE <= 0:
        issues.append(f"Invalid CHUNK_SIZE: {config.CHUNK_SIZE}")
    
    if not isinstance(config.CHUNK_OVERLAP, int) or config.CHUNK_OVERLAP < 0:
        issues.append(f"Invalid CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")
    
    if config.CHUNK_OVERLAP >= config.CHUNK_SIZE:
        issues.append(f"CHUNK_OVERLAP ({config.CHUNK_OVERLAP}) >= CHUNK_SIZE ({config.CHUNK_SIZE})")
    
    if not isinstance(config.MAX_RESULTS, int) or config.MAX_RESULTS <= 0:
        issues.append(f"Invalid MAX_RESULTS: {config.MAX_RESULTS}")
    
    if not isinstance(config.MAX_HISTORY, int) or config.MAX_HISTORY < 0:
        issues.append(f"Invalid MAX_HISTORY: {config.MAX_HISTORY}")
    
    # Warnings
    if not chroma_path.exists():
        warnings.append(f"ChromaDB path does not exist: {config.CHROMA_PATH}")
    
    if config.CHUNK_SIZE > 2000:
        warnings.append(f"Large CHUNK_SIZE ({config.CHUNK_SIZE}) may impact performance")
    
    if config.MAX_RESULTS > 20:
        warnings.append(f"High MAX_RESULTS ({config.MAX_RESULTS}) may impact performance")
    
    # Report results
    if issues:
        print("  ❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ No critical issues found")
    
    if warnings:
        print("  ⚠️  WARNINGS:")
        for warning in warnings:
            print(f"    - {warning}")
    
    print("\n" + "=" * 60)
    
    if issues:
        print("❌ CONFIGURATION ISSUES DETECTED")
        print("\nMost likely cause of 'query failed' error:")
        if "ANTHROPIC_API_KEY is missing or empty" in issues:
            print("  → Missing or invalid ANTHROPIC_API_KEY")
            print("  → Solution: Set ANTHROPIC_API_KEY in .env file")
        print("\nPlease fix the issues above and try again.")
        return False
    else:
        print("✅ CONFIGURATION APPEARS VALID")
        if warnings:
            print("  (Some warnings found, but shouldn't cause failures)")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
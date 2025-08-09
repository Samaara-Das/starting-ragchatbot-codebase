#!/usr/bin/env python3
"""
Simple wrapper to run the RAG system diagnostics from project root.

Usage:
    python diagnose.py [options]
    
Examples:
    python diagnose.py                    # Full diagnostic with human-readable output
    python diagnose.py --verbose          # Verbose output
    python diagnose.py --json-only        # JSON output only
    python diagnose.py --output report.json  # Save JSON to file
"""

import sys
import os
from pathlib import Path

# Add backend tests to path
backend_tests_path = Path(__file__).parent / "backend" / "tests"
sys.path.insert(0, str(backend_tests_path))

try:
    from run_diagnostics import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error importing diagnostic script: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error running diagnostics: {e}")
    sys.exit(1)
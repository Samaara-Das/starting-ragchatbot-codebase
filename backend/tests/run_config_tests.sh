#!/bin/bash
# Run configuration tests with proper environment setup

echo "=== RAG System Configuration Tests ==="
echo

# Change to project root
cd "$(dirname "$0")/../.."

echo "📍 Current directory: $(pwd)"
echo

# Check if we can use the uv environment
if [ -f "run.sh" ]; then
    echo "🔧 Running with uv environment..."
    ./run.sh python backend/tests/test_config.py
else
    echo "🔧 Running with system Python..."
    python3 backend/tests/test_config.py
fi

echo
echo "=== Test Summary ==="
echo "If you see 'python-dotenv' missing error:"
echo "  → This explains why 'query failed' occurs"
echo "  → The application needs proper dependency management"
echo "  → uv should handle this automatically when running via run.sh"
echo
echo "If API key is configured but queries still fail:"
echo "  → Check network connectivity to Anthropic API"
echo "  → Verify API key is valid and has correct permissions"
echo "  → Check application logs for detailed error messages"
# RAG System Diagnostic Tool

This comprehensive diagnostic tool helps identify and troubleshoot issues in your RAG chatbot system.

## Quick Start

From the project root directory, run:

```bash
python diagnose.py
```

Or directly from the tests directory:

```bash
cd backend/tests
python run_diagnostics.py
```

## What It Does

The diagnostic tool performs comprehensive checks on your RAG system:

### 1. Test Suite Execution
- Runs all pytest test suites programmatically
- Captures detailed output and error information
- Provides test-by-test failure analysis

### 2. Environment Validation
- Checks Python version compatibility
- Validates required environment variables (ANTHROPIC_API_KEY, etc.)
- Tests file system permissions
- Verifies module imports

### 3. Database Connectivity
- Tests ChromaDB connection and integrity
- Validates database schema and tables
- Checks for existing course data
- Tests vector store operations

### 4. API Connectivity
- Validates Anthropic API key configuration
- Tests actual API connectivity
- Verifies model responses
- Identifies rate limiting or authentication issues

### 5. Integration Testing
- Tests complete RAG system initialization
- Validates session management
- Tests end-to-end query processing
- Identifies specific failure points causing "query failed" errors

### 6. Data Integrity Checks
- Validates vector embeddings
- Checks course document availability
- Tests search functionality
- Verifies source attribution

## Command Line Options

```bash
python diagnose.py --help

Options:
  -v, --verbose         Verbose output during execution
  -j, --json-only      Output only JSON report (no human-readable format)
  -o, --output FILE    Save JSON report to specific file
  --skip-integration   Skip integration tests (faster execution)
```

## Output Formats

### Human-Readable Report
Default output format with:
- Executive summary
- Environment information
- Test results with failure details
- Diagnostic checks by category
- Actionable recommendations
- Remediation steps for each issue

### JSON Report
Structured data format containing:
- Complete test results with stack traces
- Detailed diagnostic information
- Environment metadata
- Programmatically parseable recommendations

## Common Issues and Solutions

### Import Failures
**Symptoms:** "CRITICAL - Import Failures" status
**Solutions:**
- Install dependencies: `pip install -r requirements.txt` or `uv sync`
- Check Python version compatibility (>=3.12 required)

### API Key Issues
**Symptoms:** Authentication errors, empty responses
**Solutions:**
- Set ANTHROPIC_API_KEY in .env file
- Verify API key is valid and has sufficient credits
- Check for rate limiting

### Database Issues
**Symptoms:** ChromaDB connection errors, no search results
**Solutions:**
- Run document processing to initialize database
- Check file permissions for database directory
- Verify course documents are in docs/ directory

### "Query Failed" Errors
**Symptoms:** End-to-end queries return "query failed"
**Diagnostic Focus:**
- Check "End-to-End Query Test" results
- Review API connectivity status  
- Verify database has indexed content
- Examine error details and stack traces

## Integration with Development Workflow

### During Development
```bash
# Quick health check
python diagnose.py --json-only | jq '.summary.overall_status'

# Full diagnostic with verbose output
python diagnose.py --verbose
```

### In CI/CD
```bash
# Generate JSON report for automated analysis
python diagnose.py --json-only --output diagnostic_report.json

# Exit codes:
# 0 = Healthy
# 1 = Critical failures
# 2 = Degraded (warnings/test failures)  
# 3 = Diagnostic tool failure
```

### Debugging Production Issues
```bash
# Focus on integration testing
python diagnose.py --verbose --output production_diagnostic.json
```

## Understanding the Output

### Status Levels
- ✅ **PASSED**: Component working correctly
- ⚠️ **WARNING**: Minor issues, system functional
- ❌ **FAILED**: Critical issues requiring attention

### Overall System Status
- **HEALTHY**: All systems operational
- **WARNING**: Minor issues present
- **DEGRADED**: Test failures but core functionality works
- **FAILED**: System issues preventing operation
- **CRITICAL**: Import failures or fundamental problems

### Remediation Priority
1. Fix CRITICAL and FAILED issues first
2. Address WARNING issues for optimal performance
3. Review test failures for development improvements

## Extending the Diagnostic Tool

The tool is designed to be extensible. To add new diagnostic checks:

1. Add methods to the `RAGDiagnostic` class
2. Return `DiagnosticResult` objects
3. Include remediation recommendations
4. Update the `run_complete_diagnostic()` method

## File Locations

- Main diagnostic script: `backend/tests/run_diagnostics.py`
- Wrapper script: `diagnose.py` (project root)
- JSON reports: `diagnostic_report.json` (default output)
- Test fixtures: `backend/tests/conftest.py`

## Dependencies

Required for full functionality:
- `pytest>=7.0.0`
- `pytest-json-report>=1.5.0`
- All RAG system dependencies (anthropic, chromadb, etc.)

The tool will gracefully degrade if optional dependencies are missing.
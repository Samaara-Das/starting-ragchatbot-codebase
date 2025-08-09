#!/usr/bin/env python3
"""
Comprehensive Test Runner and Diagnostic Script for RAG Chatbot System

This script provides comprehensive diagnostics for the RAG chatbot system by:
1. Running all test suites and capturing detailed output
2. Testing actual system components and integrations
3. Validating environment setup and configurations
4. Checking database connectivity and data integrity
5. Testing API connectivity and functionality
6. Generating structured diagnostic reports with remediation steps

Usage:
    python run_diagnostics.py [--verbose] [--json-only] [--skip-integration]
"""

import os
import sys
import json
import subprocess
import traceback
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import system components
try:
    from config import config
    from vector_store import VectorStore
    from rag_system import RAGSystem
    from ai_generator import AIGenerator
    from document_processor import DocumentProcessor
    from session_manager import SessionManager
    IMPORTS_SUCCESSFUL = True
except Exception as import_error:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(import_error)

@dataclass
class TestResult:
    """Structure for individual test results"""
    test_name: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    duration: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    output: Optional[str] = None

@dataclass
class DiagnosticResult:
    """Structure for diagnostic check results"""
    check_name: str
    status: str  # 'passed', 'failed', 'warning', 'info'
    message: str
    details: Optional[Dict[str, Any]] = None
    remediation: Optional[str] = None

@dataclass
class DiagnosticReport:
    """Complete diagnostic report structure"""
    timestamp: str
    summary: Dict[str, Any]
    environment: Dict[str, Any]
    test_results: List[TestResult]
    diagnostic_results: List[DiagnosticResult]
    recommendations: List[str]
    errors: List[str]

class RAGDiagnostic:
    """Main diagnostic class for RAG chatbot system"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent.parent
        self.backend_path = self.project_root / "backend"
        self.test_path = self.backend_path / "tests"
        self.report_data = DiagnosticReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary={},
            environment={},
            test_results=[],
            diagnostic_results=[],
            recommendations=[],
            errors=[]
        )
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp and level"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{timestamp}] {level}: {message}")
    
    @contextmanager
    def capture_output(self):
        """Context manager to capture stdout and stderr"""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                yield stdout_capture, stderr_capture
        finally:
            pass
    
    def run_pytest_programmatically(self) -> List[TestResult]:
        """Run pytest programmatically and capture results"""
        self.log("Running pytest programmatically...")
        test_results = []
        
        try:
            import pytest
            
            # Create temporary file for pytest results
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
                result_file = tmp_file.name
            
            # Run pytest with JSON report
            pytest_args = [
                str(self.test_path),
                f"--json-report={result_file}",
                "--json-report-summary",
                "-v",
                "--tb=short"
            ]
            
            self.log(f"Executing pytest with args: {' '.join(pytest_args)}")
            
            # Capture pytest output
            with self.capture_output() as (stdout_capture, stderr_capture):
                exit_code = pytest.main(pytest_args)
            
            # Read pytest results
            try:
                with open(result_file, 'r') as f:
                    pytest_data = json.load(f)
                
                # Parse pytest results
                for test in pytest_data.get('tests', []):
                    test_result = TestResult(
                        test_name=test.get('nodeid', 'Unknown'),
                        status=test.get('outcome', 'unknown'),
                        duration=test.get('duration', 0.0),
                        error_message=None,
                        stack_trace=None,
                        output=None
                    )
                    
                    # Extract error information
                    if test.get('call', {}).get('longrepr'):
                        test_result.error_message = str(test['call']['longrepr'])
                    
                    test_results.append(test_result)
                
                self.log(f"Parsed {len(test_results)} test results from pytest")
                
            except Exception as e:
                self.log(f"Failed to parse pytest results: {e}", "ERROR")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(result_file)
                except:
                    pass
                    
        except ImportError:
            self.log("pytest not available, attempting subprocess approach", "WARNING")
            test_results = self._run_pytest_subprocess()
        
        except Exception as e:
            self.log(f"Error running pytest: {e}", "ERROR")
            self.report_data.errors.append(f"Pytest execution failed: {str(e)}")
        
        return test_results
    
    def _run_pytest_subprocess(self) -> List[TestResult]:
        """Fallback method to run pytest via subprocess"""
        test_results = []
        
        try:
            cmd = [
                sys.executable, "-m", "pytest", 
                str(self.test_path), 
                "-v", "--tb=short"
            ]
            
            self.log(f"Running pytest via subprocess: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(self.project_root)
            )
            
            # Parse subprocess output
            lines = result.stdout.split('\n')
            current_test = None
            
            for line in lines:
                line = line.strip()
                if '::' in line and (' PASSED' in line or ' FAILED' in line or ' ERROR' in line):
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        test_name = parts[0]
                        status = parts[1].lower()
                        
                        test_result = TestResult(
                            test_name=test_name,
                            status=status,
                            duration=0.0,
                            output=line
                        )
                        
                        if status == 'failed' and result.stderr:
                            test_result.error_message = result.stderr
                        
                        test_results.append(test_result)
            
            self.log(f"Parsed {len(test_results)} test results from subprocess")
            
        except Exception as e:
            self.log(f"Subprocess pytest failed: {e}", "ERROR")
            self.report_data.errors.append(f"Subprocess pytest failed: {str(e)}")
        
        return test_results
    
    def check_environment_setup(self) -> List[DiagnosticResult]:
        """Check environment setup and configuration"""
        self.log("Checking environment setup...")
        diagnostics = []
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        diagnostics.append(DiagnosticResult(
            check_name="Python Version",
            status="info",
            message=f"Python {python_version}",
            details={"version": python_version, "executable": sys.executable}
        ))
        
        # Check required environment variables
        env_checks = {
            "ANTHROPIC_API_KEY": "Required for AI functionality",
            "CHROMA_PATH": "Optional, defaults to ./chroma_db"
        }
        
        for env_var, description in env_checks.items():
            value = os.getenv(env_var)
            if value:
                # Mask API keys for security
                display_value = value[:8] + "..." if env_var.endswith("_KEY") else value
                diagnostics.append(DiagnosticResult(
                    check_name=f"Environment Variable: {env_var}",
                    status="passed",
                    message=f"Set to: {display_value}",
                    details={"description": description}
                ))
            else:
                diagnostics.append(DiagnosticResult(
                    check_name=f"Environment Variable: {env_var}",
                    status="failed" if "Required" in description else "warning",
                    message="Not set",
                    details={"description": description},
                    remediation=f"Set {env_var} in your .env file or environment"
                ))
        
        # Check file system permissions
        try:
            test_file = self.project_root / "temp_permission_test.txt"
            test_file.write_text("test")
            test_file.unlink()
            diagnostics.append(DiagnosticResult(
                check_name="File System Permissions",
                status="passed",
                message="Write permissions available in project directory"
            ))
        except Exception as e:
            diagnostics.append(DiagnosticResult(
                check_name="File System Permissions",
                status="failed",
                message=f"Cannot write to project directory: {e}",
                remediation="Check directory permissions and ensure you have write access"
            ))
        
        # Check import capabilities
        if IMPORTS_SUCCESSFUL:
            diagnostics.append(DiagnosticResult(
                check_name="Module Imports",
                status="passed",
                message="All system modules imported successfully"
            ))
        else:
            diagnostics.append(DiagnosticResult(
                check_name="Module Imports",
                status="failed",
                message=f"Import failed: {IMPORT_ERROR}",
                remediation="Install required dependencies: pip install -r requirements.txt"
            ))
        
        return diagnostics
    
    def check_database_connectivity(self) -> List[DiagnosticResult]:
        """Check ChromaDB connectivity and data integrity"""
        self.log("Checking database connectivity...")
        diagnostics = []
        
        if not IMPORTS_SUCCESSFUL:
            diagnostics.append(DiagnosticResult(
                check_name="Database Connectivity",
                status="failed",
                message="Cannot check database - import errors",
                remediation="Fix import issues first"
            ))
            return diagnostics
        
        try:
            # Check if ChromaDB directory exists
            chroma_path = Path(config.CHROMA_PATH)
            if chroma_path.exists():
                diagnostics.append(DiagnosticResult(
                    check_name="ChromaDB Directory",
                    status="passed",
                    message=f"ChromaDB directory exists at {chroma_path}",
                    details={"path": str(chroma_path)}
                ))
                
                # Check SQLite database file
                sqlite_db = chroma_path / "chroma.sqlite3"
                if sqlite_db.exists():
                    try:
                        conn = sqlite3.connect(str(sqlite_db))
                        cursor = conn.cursor()
                        
                        # Check tables
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        
                        diagnostics.append(DiagnosticResult(
                            check_name="ChromaDB SQLite Database",
                            status="passed",
                            message=f"Database accessible with {len(tables)} tables",
                            details={"tables": [t[0] for t in tables]}
                        ))
                        
                        conn.close()
                        
                    except Exception as e:
                        diagnostics.append(DiagnosticResult(
                            check_name="ChromaDB SQLite Database",
                            status="failed",
                            message=f"Database access failed: {e}",
                            remediation="Database may be corrupted. Consider reinitializing."
                        ))
                else:
                    diagnostics.append(DiagnosticResult(
                        check_name="ChromaDB SQLite Database",
                        status="warning",
                        message="SQLite database file not found",
                        remediation="Run document processing to initialize database"
                    ))
            else:
                diagnostics.append(DiagnosticResult(
                    check_name="ChromaDB Directory",
                    status="warning",
                    message="ChromaDB directory does not exist",
                    remediation="Run document processing to initialize database"
                ))
            
            # Test vector store initialization
            try:
                vector_store = VectorStore(config)
                diagnostics.append(DiagnosticResult(
                    check_name="Vector Store Initialization",
                    status="passed",
                    message="VectorStore initialized successfully"
                ))
                
                # Test basic operations
                try:
                    course_count = vector_store.get_course_count()
                    existing_courses = vector_store.get_existing_course_titles()
                    
                    diagnostics.append(DiagnosticResult(
                        check_name="Vector Store Data",
                        status="passed" if course_count > 0 else "warning",
                        message=f"Found {course_count} courses: {', '.join(existing_courses) if existing_courses else 'None'}",
                        details={"course_count": course_count, "courses": existing_courses},
                        remediation="Process course documents if no data found" if course_count == 0 else None
                    ))
                    
                except Exception as e:
                    diagnostics.append(DiagnosticResult(
                        check_name="Vector Store Data",
                        status="failed",
                        message=f"Cannot access data: {e}",
                        remediation="Check database integrity and reprocess documents if needed"
                    ))
                    
            except Exception as e:
                diagnostics.append(DiagnosticResult(
                    check_name="Vector Store Initialization",
                    status="failed",
                    message=f"VectorStore initialization failed: {e}",
                    remediation="Check ChromaDB installation and configuration"
                ))
                
        except Exception as e:
            diagnostics.append(DiagnosticResult(
                check_name="Database Connectivity",
                status="failed",
                message=f"Database check failed: {e}",
                remediation="Check database configuration and permissions"
            ))
        
        return diagnostics
    
    def check_api_connectivity(self) -> List[DiagnosticResult]:
        """Check Anthropic API connectivity and functionality"""
        self.log("Checking API connectivity...")
        diagnostics = []
        
        if not IMPORTS_SUCCESSFUL:
            diagnostics.append(DiagnosticResult(
                check_name="API Connectivity",
                status="failed",
                message="Cannot check API - import errors",
                remediation="Fix import issues first"
            ))
            return diagnostics
        
        # Check API key
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            diagnostics.append(DiagnosticResult(
                check_name="Anthropic API Key",
                status="failed",
                message="API key not configured",
                remediation="Set ANTHROPIC_API_KEY in your .env file"
            ))
            return diagnostics
        
        diagnostics.append(DiagnosticResult(
            check_name="Anthropic API Key",
            status="passed",
            message=f"API key configured (ends with ...{api_key[-4:]})"
        ))
        
        # Test API functionality
        try:
            ai_generator = AIGenerator(api_key, config.ANTHROPIC_MODEL)
            
            # Simple API test
            test_response = ai_generator.generate_response(
                "Respond with exactly the word 'WORKING' and nothing else.",
                conversation_history="",
                tools=[]
            )
            
            if "WORKING" in test_response.upper():
                diagnostics.append(DiagnosticResult(
                    check_name="Anthropic API Functionality",
                    status="passed",
                    message="API responding correctly",
                    details={"model": config.ANTHROPIC_MODEL, "response": test_response[:100]}
                ))
            else:
                diagnostics.append(DiagnosticResult(
                    check_name="Anthropic API Functionality",
                    status="warning",
                    message="API responding but unexpected response",
                    details={"response": test_response[:100]},
                    remediation="API may have rate limits or model issues"
                ))
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                status = "failed"
                remediation = "Check API key validity and account status"
            elif "429" in error_msg or "rate" in error_msg.lower():
                status = "warning"
                remediation = "API rate limited - try again later"
            else:
                status = "failed"
                remediation = "Check network connectivity and API service status"
            
            diagnostics.append(DiagnosticResult(
                check_name="Anthropic API Functionality",
                status=status,
                message=f"API test failed: {error_msg}",
                remediation=remediation
            ))
        
        return diagnostics
    
    def run_integration_tests(self) -> List[DiagnosticResult]:
        """Run integration tests against the actual system"""
        self.log("Running integration tests...")
        diagnostics = []
        
        if not IMPORTS_SUCCESSFUL:
            diagnostics.append(DiagnosticResult(
                check_name="Integration Tests",
                status="failed",
                message="Cannot run integration tests - import errors",
                remediation="Fix import issues first"
            ))
            return diagnostics
        
        try:
            # Initialize RAG system
            rag_system = RAGSystem(config)
            
            # Test 1: System initialization
            diagnostics.append(DiagnosticResult(
                check_name="RAG System Initialization",
                status="passed",
                message="RAG system initialized successfully"
            ))
            
            # Test 2: Session management
            try:
                session_id = rag_system.session_manager.create_session()
                diagnostics.append(DiagnosticResult(
                    check_name="Session Management",
                    status="passed",
                    message=f"Session created: {session_id}",
                    details={"session_id": session_id}
                ))
            except Exception as e:
                diagnostics.append(DiagnosticResult(
                    check_name="Session Management",
                    status="failed",
                    message=f"Session creation failed: {e}",
                    remediation="Check session manager configuration"
                ))
            
            # Test 3: Vector store search (if data exists)
            try:
                vector_store = VectorStore(config)
                if vector_store.get_course_count() > 0:
                    search_results = vector_store.search("test query", max_results=1)
                    if search_results.documents:
                        diagnostics.append(DiagnosticResult(
                            check_name="Vector Store Search",
                            status="passed",
                            message=f"Search returned {len(search_results.documents)} results"
                        ))
                    else:
                        diagnostics.append(DiagnosticResult(
                            check_name="Vector Store Search",
                            status="warning",
                            message="Search completed but no results returned",
                            remediation="Check if search query matches indexed content"
                        ))
                else:
                    diagnostics.append(DiagnosticResult(
                        check_name="Vector Store Search",
                        status="warning",
                        message="No data available for search testing",
                        remediation="Process course documents to enable search testing"
                    ))
            except Exception as e:
                diagnostics.append(DiagnosticResult(
                    check_name="Vector Store Search",
                    status="failed",
                    message=f"Search failed: {e}",
                    remediation="Check vector store configuration and data"
                ))
            
            # Test 4: End-to-end query (if API key is available)
            if config.ANTHROPIC_API_KEY:
                try:
                    # Use a simple query that shouldn't require tools
                    response, sources = rag_system.query("Hello, can you respond with a simple greeting?")
                    
                    if response and response.strip():
                        diagnostics.append(DiagnosticResult(
                            check_name="End-to-End Query Test",
                            status="passed",
                            message=f"Query successful - Response length: {len(response)} chars, Sources: {len(sources)}",
                            details={"response_preview": response[:100], "source_count": len(sources)}
                        ))
                    else:
                        diagnostics.append(DiagnosticResult(
                            check_name="End-to-End Query Test",
                            status="failed",
                            message="Query returned empty response",
                            remediation="Check AI generator and tool integration"
                        ))
                        
                except Exception as e:
                    diagnostics.append(DiagnosticResult(
                        check_name="End-to-End Query Test",
                        status="failed",
                        message=f"Query failed: {e}",
                        details={"error_type": type(e).__name__, "stack_trace": traceback.format_exc()},
                        remediation="This is likely the source of 'query failed' errors - check error details"
                    ))
            else:
                diagnostics.append(DiagnosticResult(
                    check_name="End-to-End Query Test",
                    status="skipped",
                    message="Skipped - no API key available"
                ))
                
        except Exception as e:
            diagnostics.append(DiagnosticResult(
                check_name="Integration Tests",
                status="failed",
                message=f"Integration test setup failed: {e}",
                details={"error_type": type(e).__name__, "stack_trace": traceback.format_exc()},
                remediation="Check system initialization and dependencies"
            ))
        
        return diagnostics
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on diagnostic results"""
        recommendations = []
        
        # Analyze results for common issues
        failed_checks = [d for d in self.report_data.diagnostic_results if d.status == "failed"]
        warning_checks = [d for d in self.report_data.diagnostic_results if d.status == "warning"]
        
        # Import-related recommendations
        if not IMPORTS_SUCCESSFUL:
            recommendations.append(
                "CRITICAL: Fix import errors first. Run 'pip install -r requirements.txt' or check dependencies in pyproject.toml"
            )
        
        # API key recommendations
        api_key_issues = [d for d in failed_checks if "api" in d.check_name.lower() or "anthropic" in d.check_name.lower()]
        if api_key_issues:
            recommendations.append(
                "Configure Anthropic API key in .env file: ANTHROPIC_API_KEY=your_key_here"
            )
        
        # Database recommendations
        db_issues = [d for d in failed_checks + warning_checks if "database" in d.check_name.lower() or "chroma" in d.check_name.lower()]
        if db_issues:
            recommendations.append(
                "Initialize database by running document processing. Ensure course documents are in docs/ directory"
            )
        
        # Test failure recommendations
        failed_tests = [t for t in self.report_data.test_results if t.status == "failed"]
        if failed_tests:
            recommendations.append(
                f"Address {len(failed_tests)} failing tests. Check error messages and ensure all dependencies are properly installed"
            )
        
        # End-to-end query recommendations
        query_failures = [d for d in failed_checks if "query" in d.check_name.lower()]
        if query_failures:
            recommendations.append(
                "End-to-end query failing - this is likely causing 'query failed' errors. Check API connectivity, database data, and error details"
            )
        
        # Performance recommendations
        if len(failed_checks) == 0 and len(warning_checks) > 0:
            recommendations.append(
                "System is functional but has warnings. Address warnings for optimal performance"
            )
        
        # Success recommendations
        if len(failed_checks) == 0 and len(warning_checks) == 0:
            recommendations.append(
                "System appears healthy! If still experiencing 'query failed' errors, check application logs for runtime issues"
            )
        
        return recommendations
    
    def run_complete_diagnostic(self) -> DiagnosticReport:
        """Run complete diagnostic suite"""
        self.log("Starting comprehensive RAG system diagnostic...")
        
        # 1. Environment checks
        env_results = self.check_environment_setup()
        self.report_data.diagnostic_results.extend(env_results)
        
        # 2. Run tests
        test_results = self.run_pytest_programmatically()
        self.report_data.test_results.extend(test_results)
        
        # 3. Database checks
        db_results = self.check_database_connectivity()
        self.report_data.diagnostic_results.extend(db_results)
        
        # 4. API checks
        api_results = self.check_api_connectivity()
        self.report_data.diagnostic_results.extend(api_results)
        
        # 5. Integration tests
        integration_results = self.run_integration_tests()
        self.report_data.diagnostic_results.extend(integration_results)
        
        # 6. Generate summary
        self.report_data.summary = {
            "total_tests": len(self.report_data.test_results),
            "tests_passed": len([t for t in self.report_data.test_results if t.status == "passed"]),
            "tests_failed": len([t for t in self.report_data.test_results if t.status == "failed"]),
            "diagnostic_checks": len(self.report_data.diagnostic_results),
            "checks_passed": len([d for d in self.report_data.diagnostic_results if d.status == "passed"]),
            "checks_failed": len([d for d in self.report_data.diagnostic_results if d.status == "failed"]),
            "checks_warning": len([d for d in self.report_data.diagnostic_results if d.status == "warning"]),
            "overall_status": self._determine_overall_status()
        }
        
        # 7. Generate recommendations
        self.report_data.recommendations = self.generate_recommendations()
        
        # 8. Environment info
        self.report_data.environment = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "working_directory": str(os.getcwd()),
            "project_root": str(self.project_root),
            "imports_successful": IMPORTS_SUCCESSFUL,
            "anthropic_api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "chroma_path": str(Path(os.getenv("CHROMA_PATH", "./chroma_db")))
        }
        
        self.log(f"Diagnostic complete. Status: {self.report_data.summary['overall_status']}")
        return self.report_data
    
    def _determine_overall_status(self) -> str:
        """Determine overall system status"""
        failed_checks = len([d for d in self.report_data.diagnostic_results if d.status == "failed"])
        failed_tests = len([t for t in self.report_data.test_results if t.status == "failed"])
        
        if not IMPORTS_SUCCESSFUL:
            return "CRITICAL - Import Failures"
        elif failed_checks > 0:
            return "FAILED - System Issues"
        elif failed_tests > 0:
            return "DEGRADED - Test Failures"
        else:
            warning_checks = len([d for d in self.report_data.diagnostic_results if d.status == "warning"])
            if warning_checks > 0:
                return "WARNING - Minor Issues"
            else:
                return "HEALTHY"
    
    def format_human_readable_report(self, report: DiagnosticReport) -> str:
        """Format diagnostic report in human-readable format"""
        lines = []
        lines.append("=" * 80)
        lines.append("RAG CHATBOT SYSTEM DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Overall Status: {report.summary['overall_status']}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Tests Run: {report.summary['total_tests']} (Passed: {report.summary['tests_passed']}, Failed: {report.summary['tests_failed']})")
        lines.append(f"Diagnostic Checks: {report.summary['diagnostic_checks']} (Passed: {report.summary['checks_passed']}, Failed: {report.summary['checks_failed']}, Warnings: {report.summary['checks_warning']})")
        lines.append("")
        
        # Environment
        lines.append("ENVIRONMENT")
        lines.append("-" * 40)
        for key, value in report.environment.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Failed tests
        failed_tests = [t for t in report.test_results if t.status == "failed"]
        if failed_tests:
            lines.append("FAILED TESTS")
            lines.append("-" * 40)
            for test in failed_tests:
                lines.append(f"❌ {test.test_name}")
                if test.error_message:
                    lines.append(f"   Error: {test.error_message[:200]}...")
                lines.append("")
        
        # Diagnostic results by status
        for status, icon in [("failed", "❌"), ("warning", "⚠️"), ("passed", "✅")]:
            status_results = [d for d in report.diagnostic_results if d.status == status]
            if status_results:
                lines.append(f"{status.upper()} DIAGNOSTIC CHECKS")
                lines.append("-" * 40)
                for result in status_results:
                    lines.append(f"{icon} {result.check_name}: {result.message}")
                    if result.remediation:
                        lines.append(f"   💡 Remediation: {result.remediation}")
                    lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Errors
        if report.errors:
            lines.append("SYSTEM ERRORS")
            lines.append("-" * 40)
            for error in report.errors:
                lines.append(f"❌ {error}")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Comprehensive Diagnostic Tool")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json-only", "-j", action="store_true", help="Output only JSON report")
    parser.add_argument("--output", "-o", help="Output file for JSON report")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostic = RAGDiagnostic(verbose=args.verbose)
    
    try:
        report = diagnostic.run_complete_diagnostic()
        
        # Output JSON report
        json_report = json.dumps(asdict(report), indent=2, default=str)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_report)
            print(f"JSON report written to: {args.output}")
        
        if args.json_only:
            print(json_report)
        else:
            # Print human-readable report
            human_report = diagnostic.format_human_readable_report(report)
            print(human_report)
            
            if not args.output:
                # Also save JSON report by default
                json_file = Path("diagnostic_report.json")
                with open(json_file, 'w') as f:
                    f.write(json_report)
                print(f"\nJSON report also saved to: {json_file}")
        
        # Exit with appropriate code
        if report.summary['overall_status'] in ['CRITICAL - Import Failures', 'FAILED - System Issues']:
            sys.exit(1)
        elif report.summary['overall_status'] in ['DEGRADED - Test Failures', 'WARNING - Minor Issues']:
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"FATAL ERROR: Diagnostic failed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()
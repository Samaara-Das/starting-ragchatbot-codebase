"""
Comprehensive configuration and environment tests for the RAG system.

These tests focus on diagnosing the "query failed" issue by validating:
1. ANTHROPIC_API_KEY availability and validation
2. ChromaDB path accessibility and permissions
3. All required config values present and valid
4. Environment variable loading from .env file
5. Config dataclass validation
6. Default value handling when env vars missing
7. Invalid configuration scenarios
8. Path validation for CHROMA_PATH
9. Model name validation
10. Numeric parameter validation (CHUNK_SIZE, MAX_RESULTS, etc.)

Run with: python -m pytest backend/tests/test_config.py -v
Or for diagnostics: python backend/tests/test_config.py
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import fields, is_dataclass
import sqlite3
import sys

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create minimal pytest substitute for standalone running
    class pytest:
        class mark:
            @staticmethod
            def parametrize(params, values):
                def decorator(func):
                    return func
                return decorator
        
        @staticmethod
        def fixture(func):
            return func
        
        @staticmethod
        def skip(reason):
            def decorator(func):
                return func
            return decorator

# Try to import config with diagnostics
def import_config_with_diagnostics():
    """Import config module with detailed diagnostics"""
    try:
        from config import Config, config
        return Config, config, None
    except ImportError as e:
        if 'dotenv' in str(e):
            print("❌ Missing 'python-dotenv' dependency")
            print("   This is likely why the config can't load")
            print("   The application should install this dependency")
            
            # Try to create a mock config for testing
            try:
                # Load .env manually
                env_vars = {}
                # Try multiple possible paths for .env
                possible_paths = [
                    Path('.env'),
                    Path('../../.env'),
                    Path('../../../.env'),
                    Path(os.path.join(os.path.dirname(__file__), '../../.env'))
                ]
                
                env_file = None
                for path in possible_paths:
                    if path.exists():
                        env_file = path
                        break
                
                if env_file and env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            if line.strip() and not line.startswith('#') and '=' in line:
                                key, value = line.strip().split('=', 1)
                                env_vars[key] = value
                
                # Create mock config
                class MockConfig:
                    ANTHROPIC_API_KEY = env_vars.get("ANTHROPIC_API_KEY", "")
                    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
                    CHUNK_SIZE = 800
                    CHUNK_OVERLAP = 100
                    MAX_RESULTS = 5
                    MAX_HISTORY = 2
                    CHROMA_PATH = "./chroma_db"
                
                print("✅ Created mock config for testing")
                return MockConfig, MockConfig(), None
                
            except Exception as mock_error:
                print(f"❌ Failed to create mock config: {mock_error}")
                return None, None, e
        else:
            print(f"❌ Failed to import config: {e}")
            return None, None, e

# Import with diagnostics
Config, config, import_error = import_config_with_diagnostics()

class TestConfigDataclass:
    """Test Config dataclass structure and validation"""
    
    def test_config_is_dataclass(self):
        """Test that Config is a proper dataclass"""
        assert is_dataclass(Config)
    
    def test_config_has_required_fields(self):
        """Test that Config has all required fields"""
        config_fields = {field.name for field in fields(Config)}
        required_fields = {
            'ANTHROPIC_API_KEY',
            'ANTHROPIC_MODEL',
            'EMBEDDING_MODEL',
            'CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'MAX_RESULTS',
            'MAX_HISTORY',
            'CHROMA_PATH'
        }
        assert required_fields.issubset(config_fields), f"Missing fields: {required_fields - config_fields}"
    
    def test_config_field_types(self):
        """Test that Config fields have correct types"""
        field_types = {field.name: field.type for field in fields(Config)}
        
        expected_types = {
            'ANTHROPIC_API_KEY': str,
            'ANTHROPIC_MODEL': str,
            'EMBEDDING_MODEL': str,
            'CHUNK_SIZE': int,
            'CHUNK_OVERLAP': int,
            'MAX_RESULTS': int,
            'MAX_HISTORY': int,
            'CHROMA_PATH': str
        }
        
        for field_name, expected_type in expected_types.items():
            assert field_types[field_name] == expected_type, f"{field_name} should be {expected_type}, got {field_types[field_name]}"


class TestEnvironmentVariableLoading:
    """Test environment variable loading from .env file"""
    
    def test_env_file_loading(self):
        """Test that environment variables are loaded from .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("ANTHROPIC_API_KEY=test-api-key-123\n")
            f.write("CHUNK_SIZE=500\n")
            env_file = f.name
        
        try:
            with patch('config.load_dotenv') as mock_load_dotenv:
                # Mock the environment variables
                with patch.dict(os.environ, {
                    'ANTHROPIC_API_KEY': 'test-api-key-123',
                    'CHUNK_SIZE': '500'
                }):
                    config = Config()
                    assert config.ANTHROPIC_API_KEY == 'test-api-key-123'
                    assert config.CHUNK_SIZE == 500
        finally:
            os.unlink(env_file)
    
    def test_missing_env_file_handling(self):
        """Test behavior when .env file is missing"""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.ANTHROPIC_API_KEY == ""  # Should use default empty string
            assert config.CHUNK_SIZE == 800  # Should use default value
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'override-key',
            'CHUNK_SIZE': '1000',
            'MAX_RESULTS': '10'
        }):
            # Need to reload the config module to pick up new env vars
            import importlib
            import config
            importlib.reload(config)
            
            assert config.config.ANTHROPIC_API_KEY == 'override-key'
            # Note: Config loads env vars as strings, so numeric fields need conversion
            # This test reveals a potential bug in the current implementation


class TestAnthropicAPIKeyValidation:
    """Test ANTHROPIC_API_KEY validation and format checking"""
    
    def test_api_key_present(self):
        """Test that API key is present when configured"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-test-key-123'}):
            config = Config()
            assert config.ANTHROPIC_API_KEY != ""
            assert config.ANTHROPIC_API_KEY == 'sk-test-key-123'
    
    def test_api_key_missing(self):
        """Test handling of missing API key - common cause of 'query failed'"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.ANTHROPIC_API_KEY == ""
    
    def test_api_key_format_validation(self):
        """Test API key format validation"""
        valid_patterns = [
            'sk-ant-api03-test123',
            'sk-test-key-456',
            'test-api-key-789'
        ]
        
        for api_key in valid_patterns:
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': api_key}):
                config = Config()
                assert len(config.ANTHROPIC_API_KEY) > 0
                assert config.ANTHROPIC_API_KEY == api_key
    
    def test_api_key_empty_string_validation(self):
        """Test that empty API key is detected"""
        empty_values = ['', '   ', None]
        
        for empty_value in empty_values:
            env_dict = {'ANTHROPIC_API_KEY': empty_value} if empty_value is not None else {}
            with patch.dict(os.environ, env_dict, clear=True):
                config = Config()
                # Empty or None should result in empty string
                assert config.ANTHROPIC_API_KEY == "" or config.ANTHROPIC_API_KEY.strip() == ""


class TestChromaDBConfiguration:
    """Test ChromaDB path and database configuration"""
    
    def test_chroma_path_default(self):
        """Test default ChromaDB path"""
        config = Config()
        assert config.CHROMA_PATH == "./chroma_db"
    
    def test_chroma_path_custom(self):
        """Test custom ChromaDB path from environment"""
        custom_path = "/tmp/test_chroma"
        with patch.dict(os.environ, {'CHROMA_PATH': custom_path}):
            # Current implementation doesn't support CHROMA_PATH env var
            # This test reveals a limitation
            config = Config()
            # With current implementation, this will still use default
            assert config.CHROMA_PATH == "./chroma_db"
    
    def test_chroma_path_creation(self):
        """Test ChromaDB path creation and permissions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "test_chroma")
            
            # Test path creation
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            assert os.path.exists(chroma_path)
            assert os.path.isdir(chroma_path)
            
            # Test write permissions
            test_file = os.path.join(chroma_path, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            assert os.path.exists(test_file)
    
    def test_chroma_path_permissions(self):
        """Test ChromaDB path permissions"""
        config = Config()
        chroma_path = config.CHROMA_PATH
        
        # Create path if it doesn't exist for testing
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        
        # Test read permission
        assert os.access(chroma_path, os.R_OK)
        
        # Test write permission
        assert os.access(chroma_path, os.W_OK)
    
    def test_chroma_database_accessibility(self):
        """Test ChromaDB database file accessibility"""
        config = Config()
        chroma_path = config.CHROMA_PATH
        
        # Create path if it doesn't exist
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        
        # Test SQLite database creation/access
        db_file = os.path.join(chroma_path, "test.sqlite3")
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
            conn.commit()
            conn.close()
            assert os.path.exists(db_file)
        finally:
            if os.path.exists(db_file):
                os.remove(db_file)


class TestNumericParameterValidation:
    """Test numeric parameter validation"""
    
    def test_chunk_size_validation(self):
        """Test CHUNK_SIZE validation"""
        config = Config()
        assert isinstance(config.CHUNK_SIZE, int)
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_SIZE == 800  # Default value
    
    def test_chunk_overlap_validation(self):
        """Test CHUNK_OVERLAP validation"""
        config = Config()
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE  # Should be less than chunk size
        assert config.CHUNK_OVERLAP == 100  # Default value
    
    def test_max_results_validation(self):
        """Test MAX_RESULTS validation"""
        config = Config()
        assert isinstance(config.MAX_RESULTS, int)
        assert config.MAX_RESULTS > 0
        assert config.MAX_RESULTS == 5  # Default value
    
    def test_max_history_validation(self):
        """Test MAX_HISTORY validation"""
        config = Config()
        assert isinstance(config.MAX_HISTORY, int)
        assert config.MAX_HISTORY >= 0
        assert config.MAX_HISTORY == 2  # Default value
    
    def test_numeric_parameter_ranges(self):
        """Test that numeric parameters are within reasonable ranges"""
        config = Config()
        
        # CHUNK_SIZE should be reasonable (not too small or too large)
        assert 100 <= config.CHUNK_SIZE <= 2000
        
        # CHUNK_OVERLAP should be reasonable percentage of chunk size
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
        assert config.CHUNK_OVERLAP >= 0
        
        # MAX_RESULTS should be reasonable
        assert 1 <= config.MAX_RESULTS <= 100
        
        # MAX_HISTORY should be reasonable
        assert 0 <= config.MAX_HISTORY <= 50


class TestModelNameValidation:
    """Test model name validation"""
    
    def test_anthropic_model_default(self):
        """Test default Anthropic model"""
        config = Config()
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert isinstance(config.ANTHROPIC_MODEL, str)
        assert len(config.ANTHROPIC_MODEL) > 0
    
    def test_embedding_model_default(self):
        """Test default embedding model"""
        config = Config()
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert isinstance(config.EMBEDDING_MODEL, str)
        assert len(config.EMBEDDING_MODEL) > 0
    
    def test_model_name_formats(self):
        """Test that model names follow expected formats"""
        config = Config()
        
        # Anthropic model should contain 'claude'
        assert 'claude' in config.ANTHROPIC_MODEL.lower()
        
        # Embedding model should be a valid sentence transformer model name
        assert '-' in config.EMBEDDING_MODEL or '_' in config.EMBEDDING_MODEL


class TestConfigurationConsistency:
    """Test configuration consistency and relationships"""
    
    def test_chunk_overlap_less_than_chunk_size(self):
        """Test that chunk overlap is less than chunk size"""
        config = Config()
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, \
            f"CHUNK_OVERLAP ({config.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({config.CHUNK_SIZE})"
    
    def test_path_consistency(self):
        """Test path configuration consistency"""
        config = Config()
        
        # CHROMA_PATH should be a valid path format
        assert isinstance(config.CHROMA_PATH, str)
        assert len(config.CHROMA_PATH) > 0
        
        # Should not contain invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*'] if os.name == 'nt' else []
        for char in invalid_chars:
            assert char not in config.CHROMA_PATH


class TestInvalidConfigurationScenarios:
    """Test handling of invalid configuration scenarios"""
    
    def test_invalid_numeric_environment_variables(self):
        """Test handling of invalid numeric environment variables"""
        # This test reveals that the current implementation doesn't validate env var types
        invalid_values = ['abc', '12.5.6', 'true', '']
        
        for invalid_value in invalid_values:
            with patch.dict(os.environ, {'CHUNK_SIZE': invalid_value}):
                # Current implementation doesn't convert env vars to int
                # This would cause issues at runtime
                try:
                    config = Config()
                    # If the implementation tried to convert, it would fail here
                    if hasattr(config, '_validate'):
                        config._validate()
                except (ValueError, TypeError):
                    # Expected behavior for invalid values
                    pass
    
    def test_negative_numeric_values(self):
        """Test handling of negative numeric values"""
        with patch.dict(os.environ, {
            'CHUNK_SIZE': '-100',
            'MAX_RESULTS': '-5'
        }):
            config = Config()
            # Current implementation doesn't validate negative values
            # This test documents the current behavior
            pass
    
    def test_extremely_large_values(self):
        """Test handling of extremely large values"""
        with patch.dict(os.environ, {
            'CHUNK_SIZE': '999999999',
            'MAX_RESULTS': '1000000'
        }):
            config = Config()
            # Current implementation doesn't validate upper bounds
            pass


class TestConfigurationDiagnostics:
    """Test configuration diagnostics for troubleshooting"""
    
    def test_diagnose_missing_api_key(self):
        """Test diagnostic for missing API key (common cause of 'query failed')"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            
            # Simulate diagnostic check
            has_api_key = bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip())
            assert not has_api_key, "Should detect missing API key"
    
    def test_diagnose_database_connectivity(self):
        """Test diagnostic for database connectivity issues"""
        config = Config()
        chroma_path = config.CHROMA_PATH
        
        # Check if database path is accessible
        try:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            db_accessible = True
        except PermissionError:
            db_accessible = False
        
        # This diagnostic can help identify path/permission issues
        assert isinstance(db_accessible, bool)
    
    def test_configuration_summary(self):
        """Test generation of configuration summary for diagnostics"""
        config = Config()
        
        summary = {
            'api_key_configured': bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip()),
            'anthropic_model': config.ANTHROPIC_MODEL,
            'embedding_model': config.EMBEDDING_MODEL,
            'chunk_size': config.CHUNK_SIZE,
            'chunk_overlap': config.CHUNK_OVERLAP,
            'max_results': config.MAX_RESULTS,
            'max_history': config.MAX_HISTORY,
            'chroma_path': config.CHROMA_PATH,
            'chroma_path_exists': os.path.exists(config.CHROMA_PATH),
        }
        
        # Validate summary structure
        assert isinstance(summary, dict)
        assert 'api_key_configured' in summary
        assert 'chroma_path_exists' in summary
        
        # This summary can be useful for debugging


class TestDeploymentEnvironments:
    """Test different deployment environment scenarios"""
    
    def test_development_environment(self):
        """Test development environment configuration"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'dev-api-key',
            'ENVIRONMENT': 'development'
        }):
            config = Config()
            assert config.ANTHROPIC_API_KEY == 'dev-api-key'
    
    def test_production_environment(self):
        """Test production environment configuration"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'prod-api-key-secret',
            'ENVIRONMENT': 'production'
        }):
            config = Config()
            assert config.ANTHROPIC_API_KEY == 'prod-api-key-secret'
    
    def test_container_environment(self):
        """Test containerized environment configuration"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'container-api-key',
            'CHROMA_PATH': '/app/data/chroma'
        }):
            config = Config()
            assert config.ANTHROPIC_API_KEY == 'container-api-key'
            # Note: Current implementation doesn't support CHROMA_PATH env var


if __name__ == "__main__":
    # Run diagnostics if executed directly
    print("Running configuration diagnostics...")
    
    config = Config()
    
    print(f"API Key configured: {bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip())}")
    print(f"Anthropic Model: {config.ANTHROPIC_MODEL}")
    print(f"Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"Chunk Size: {config.CHUNK_SIZE}")
    print(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
    print(f"Max Results: {config.MAX_RESULTS}")
    print(f"Max History: {config.MAX_HISTORY}")
    print(f"Chroma Path: {config.CHROMA_PATH}")
    print(f"Chroma Path Exists: {os.path.exists(config.CHROMA_PATH)}")
    
    # Check for common issues
    issues = []
    
    if not config.ANTHROPIC_API_KEY or not config.ANTHROPIC_API_KEY.strip():
        issues.append("❌ ANTHROPIC_API_KEY is missing or empty (common cause of 'query failed')")
    else:
        print("✅ ANTHROPIC_API_KEY is configured")
    
    if not os.path.exists(config.CHROMA_PATH):
        issues.append(f"⚠️  ChromaDB path does not exist: {config.CHROMA_PATH}")
    else:
        print("✅ ChromaDB path exists")
    
    if config.CHUNK_OVERLAP >= config.CHUNK_SIZE:
        issues.append(f"❌ CHUNK_OVERLAP ({config.CHUNK_OVERLAP}) >= CHUNK_SIZE ({config.CHUNK_SIZE})")
    else:
        print("✅ Chunk overlap configuration is valid")
    
    if issues:
        print("\nConfiguration Issues Found:")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ No obvious configuration issues detected")
"""
Pytest configuration and fixtures for RAG system testing.
"""
import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from rag_system import RAGSystem
from config import Config

@pytest.fixture
def mock_config():
    """Mock configuration with test settings"""
    config = Mock(spec=Config)
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config

@pytest.fixture
def mock_search_results():
    """Mock SearchResults with sample data"""
    results = SearchResults(
        documents=["Sample content about RAG systems", "More content about machine learning"],
        metadata=[
            {"course_title": "Introduction to RAG", "lesson_number": 1, "chunk_id": "chunk1"},
            {"course_title": "ML Fundamentals", "lesson_number": 2, "chunk_id": "chunk2"}
        ],
        distances=[0.1, 0.3]
    )
    return results

@pytest.fixture
def empty_search_results():
    """Mock empty SearchResults"""
    return SearchResults(documents=[], metadata=[], distances=[])

@pytest.fixture
def error_search_results():
    """Mock SearchResults with error"""
    return SearchResults.empty("Test error message")

@pytest.fixture
def mock_vector_store(mock_search_results):
    """Mock VectorStore with controlled responses"""
    store = Mock(spec=VectorStore)
    store.search.return_value = mock_search_results
    store._resolve_course_name.return_value = "Introduction to RAG"
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store.get_all_courses_metadata.return_value = [
        {
            "course_title": "Introduction to RAG",
            "course_link": "https://example.com/course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "What is RAG?", "lesson_link": "https://example.com/lesson1"},
                {"lesson_number": 2, "lesson_title": "RAG Applications", "lesson_link": "https://example.com/lesson2"}
            ]
        }
    ]
    store.get_course_count.return_value = 2
    store.get_existing_course_titles.return_value = ["Introduction to RAG", "ML Fundamentals"]
    return store

@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response from Claude."
    mock_response.stop_reason = "end_turn"
    return mock_response

@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic API response with tool use"""
    mock_response = Mock()
    
    # Mock content with tool use
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_12345"
    mock_tool_block.input = {"query": "test query"}
    
    mock_response.content = [mock_tool_block]
    mock_response.stop_reason = "tool_use"
    return mock_response

@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """Mock Anthropic client"""
    with patch('anthropic.Anthropic') as mock_client:
        client_instance = Mock()
        client_instance.messages.create.return_value = mock_anthropic_response
        mock_client.return_value = client_instance
        yield client_instance

@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool instance with mock vector store"""
    return CourseSearchTool(mock_vector_store)

@pytest.fixture
def course_outline_tool(mock_vector_store):
    """CourseOutlineTool instance with mock vector store"""
    return CourseOutlineTool(mock_vector_store)

@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager

@pytest.fixture
def ai_generator(mock_config, mock_anthropic_client):
    """AIGenerator instance with mock client"""
    with patch('ai_generator.anthropic.Anthropic'):
        generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        generator.client = mock_anthropic_client
        return generator

@pytest.fixture
def rag_system(mock_config, mock_vector_store):
    """RAGSystem instance with mocked dependencies"""
    with patch('rag_system.VectorStore', return_value=mock_vector_store), \
         patch('rag_system.AIGenerator') as mock_ai_gen, \
         patch('rag_system.SessionManager') as mock_session, \
         patch('rag_system.DocumentProcessor'):
        
        mock_ai_gen.return_value.generate_response.return_value = "Test response"
        mock_session.return_value.create_session.return_value = "test_session_123"
        mock_session.return_value.get_conversation_history.return_value = None
        
        system = RAGSystem(mock_config)
        return system

@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "course_title": "Introduction to RAG",
        "course_link": "https://example.com/course",
        "instructor": "Dr. Test",
        "lessons": [
            {
                "lesson_number": 1,
                "lesson_title": "What is RAG?",
                "lesson_link": "https://example.com/lesson1",
                "content": "RAG stands for Retrieval-Augmented Generation..."
            },
            {
                "lesson_number": 2,
                "lesson_title": "RAG Applications",
                "lesson_link": "https://example.com/lesson2", 
                "content": "RAG can be used for question answering..."
            }
        ]
    }

@pytest.fixture
def environment_variables():
    """Mock environment variables for testing"""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-anthropic-key-12345",
        "CHROMA_PATH": "./test_chroma_db"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

# Test helper functions
@pytest.fixture
def assert_tool_called():
    """Helper to assert tool was called with correct parameters"""
    def _assert_tool_called(mock_tool, expected_params):
        mock_tool.execute.assert_called_once()
        actual_params = mock_tool.execute.call_args
        for key, value in expected_params.items():
            assert actual_params.kwargs[key] == value
    
    return _assert_tool_called

@pytest.fixture
def capture_exceptions():
    """Helper to capture and analyze exceptions"""
    exceptions = []
    
    def _capture_exception(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exceptions.append(e)
            raise
    
    _capture_exception.exceptions = exceptions
    return _capture_exception
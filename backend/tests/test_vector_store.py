"""
Comprehensive unit tests for the VectorStore class.

Tests cover:
1. VectorStore initialization and ChromaDB connection
2. The search() method with various filters (course_name, lesson_number)
3. The _resolve_course_name() method for fuzzy course matching
4. Course metadata retrieval methods (get_all_courses_metadata, get_course_link)
5. Error handling for ChromaDB failures, corrupted data, missing collections
6. SearchResults object creation and validation
7. Filter building with _build_filter() method
8. Edge cases like empty databases, malformed queries, etc.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Any, Optional

# Add parent directory to path to import backend modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, CourseChunk, Lesson


class TestSearchResults:
    """Test SearchResults dataclass and its methods"""

    def test_search_results_init(self):
        """Test SearchResults initialization with all fields"""
        docs = ["doc1", "doc2"]
        metadata = [{"key": "value1"}, {"key": "value2"}]
        distances = [0.1, 0.2]
        error = "test error"
        
        results = SearchResults(docs, metadata, distances, error)
        
        assert results.documents == docs
        assert results.metadata == metadata
        assert results.distances == distances
        assert results.error == error

    def test_search_results_default_error(self):
        """Test SearchResults initialization with default error=None"""
        docs = ["doc1"]
        metadata = [{"key": "value1"}]
        distances = [0.1]
        
        results = SearchResults(docs, metadata, distances)
        
        assert results.error is None

    def test_from_chroma_with_results(self):
        """Test SearchResults.from_chroma with valid ChromaDB results"""
        chroma_results = {
            'documents': [["doc1", "doc2"]],
            'metadatas': [[{"key": "value1"}, {"key": "value2"}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"key": "value1"}, {"key": "value2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test SearchResults.from_chroma with empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_from_chroma_none_results(self):
        """Test SearchResults.from_chroma with None values in ChromaDB results"""
        chroma_results = {
            'documents': None,
            'metadatas': None,
            'distances': None
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_class_method(self):
        """Test SearchResults.empty class method"""
        error_msg = "No results found"
        
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults([], [], [])
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(["doc1"], [{"key": "value"}], [0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Test VectorStore class initialization and core functionality"""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init_success(self, mock_embedding_fn, mock_client_class):
        """Test successful VectorStore initialization"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_fn.return_value = mock_embedding
        
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Initialize VectorStore
        chroma_path = "/test/path"
        embedding_model = "test-model"
        max_results = 10
        
        store = VectorStore(chroma_path, embedding_model, max_results)
        
        # Verify initialization
        assert store.max_results == max_results
        mock_client_class.assert_called_once()
        mock_embedding_fn.assert_called_once_with(model_name=embedding_model)
        
        # Verify collections were created
        assert mock_client.get_or_create_collection.call_count == 2
        expected_calls = [
            call(name="course_catalog", embedding_function=mock_embedding),
            call(name="course_content", embedding_function=mock_embedding)
        ]
        mock_client.get_or_create_collection.assert_has_calls(expected_calls)
        
        assert store.course_catalog == mock_collection
        assert store.course_content == mock_collection

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init_default_max_results(self, mock_embedding_fn, mock_client_class):
        """Test VectorStore initialization with default max_results"""
        mock_client_class.return_value = Mock()
        mock_embedding_fn.return_value = Mock()
        mock_client_class.return_value.get_or_create_collection.return_value = Mock()
        
        store = VectorStore("/test/path", "test-model")
        assert store.max_results == 5  # Default value

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init_chromadb_connection_error(self, mock_embedding_fn, mock_client_class):
        """Test VectorStore initialization when ChromaDB connection fails"""
        mock_client_class.side_effect = Exception("ChromaDB connection failed")
        
        with pytest.raises(Exception, match="ChromaDB connection failed"):
            VectorStore("/test/path", "test-model")


class TestVectorStoreSearch:
    """Test VectorStore search functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            store.course_catalog = Mock()
            store.course_content = Mock()
            return store

    def test_search_success_no_filters(self, mock_vector_store):
        """Test successful search without any filters"""
        # Mock the course_content.query response
        mock_chroma_results = {
            'documents': [["Found content about machine learning"]],
            'metadatas': [[{"course_title": "ML Course", "lesson_number": 1}]],
            'distances': [[0.2]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("machine learning")
        
        assert not results.is_empty()
        assert results.documents == ["Found content about machine learning"]
        assert results.metadata == [{"course_title": "ML Course", "lesson_number": 1}]
        assert results.distances == [0.2]
        assert results.error is None
        
        # Verify query was called correctly
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["machine learning"],
            n_results=5,  # default max_results
            where=None
        )

    def test_search_with_course_name_resolved(self, mock_vector_store):
        """Test search with course name that gets resolved"""
        # Mock course name resolution
        mock_vector_store._resolve_course_name = Mock(return_value="Deep Learning Fundamentals")
        
        # Mock course content search
        mock_chroma_results = {
            'documents': [["Content about neural networks"]],
            'metadatas': [[{"course_title": "Deep Learning Fundamentals", "lesson_number": 2}]],
            'distances': [[0.1]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("neural networks", course_name="deep learning")
        
        assert not results.is_empty()
        assert results.documents == ["Content about neural networks"]
        
        # Verify course name resolution was called
        mock_vector_store._resolve_course_name.assert_called_once_with("deep learning")
        
        # Verify search was called with resolved course title filter
        expected_filter = {"course_title": "Deep Learning Fundamentals"}
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["neural networks"],
            n_results=5,
            where=expected_filter
        )

    def test_search_course_name_not_resolved(self, mock_vector_store):
        """Test search when course name cannot be resolved"""
        # Mock course name resolution to return None
        mock_vector_store._resolve_course_name = Mock(return_value=None)
        
        results = mock_vector_store.search("test query", course_name="nonexistent course")
        
        assert results.is_empty()
        assert results.error == "No course found matching 'nonexistent course'"
        
        # Verify course content search was not called
        mock_vector_store.course_content.query.assert_not_called()

    def test_search_with_lesson_number_filter(self, mock_vector_store):
        """Test search with lesson number filter"""
        mock_chroma_results = {
            'documents': [["Lesson 3 content"]],
            'metadatas': [[{"course_title": "Test Course", "lesson_number": 3}]],
            'distances': [[0.15]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("test query", lesson_number=3)
        
        assert not results.is_empty()
        
        # Verify search was called with lesson number filter
        expected_filter = {"lesson_number": 3}
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )

    def test_search_with_both_filters(self, mock_vector_store):
        """Test search with both course name and lesson number filters"""
        # Mock course name resolution
        mock_vector_store._resolve_course_name = Mock(return_value="Advanced Python")
        
        mock_chroma_results = {
            'documents': [["Specific lesson content"]],
            'metadatas': [[{"course_title": "Advanced Python", "lesson_number": 5}]],
            'distances': [[0.05]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search(
            "decorators", 
            course_name="python", 
            lesson_number=5
        )
        
        assert not results.is_empty()
        
        # Verify search was called with combined filter
        expected_filter = {
            "$and": [
                {"course_title": "Advanced Python"},
                {"lesson_number": 5}
            ]
        }
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["decorators"],
            n_results=5,
            where=expected_filter
        )

    def test_search_with_custom_limit(self, mock_vector_store):
        """Test search with custom result limit"""
        mock_chroma_results = {
            'documents': [["Result 1", "Result 2", "Result 3"]],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("test query", limit=3)
        
        assert not results.is_empty()
        
        # Verify search was called with custom limit
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where=None
        )

    def test_search_chromadb_error(self, mock_vector_store):
        """Test search when ChromaDB query fails"""
        # Mock ChromaDB query to raise exception
        mock_vector_store.course_content.query.side_effect = Exception("Database connection error")
        
        results = mock_vector_store.search("test query")
        
        assert results.is_empty()
        assert results.error == "Search error: Database connection error"

    def test_search_empty_results(self, mock_vector_store):
        """Test search that returns no results"""
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("nonexistent query")
        
        assert results.is_empty()
        assert results.error is None


class TestVectorStoreCourseNameResolution:
    """Test VectorStore course name resolution functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            store.course_catalog = Mock()
            store.course_content = Mock()
            return store

    def test_resolve_course_name_exact_match(self, mock_vector_store):
        """Test course name resolution with exact match"""
        mock_chroma_results = {
            'documents': [["Introduction to Machine Learning"]],
            'metadatas': [[{"title": "Introduction to Machine Learning", "instructor": "Dr. Smith"}]]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name("Introduction to Machine Learning")
        
        assert result == "Introduction to Machine Learning"
        mock_vector_store.course_catalog.query.assert_called_once_with(
            query_texts=["Introduction to Machine Learning"],
            n_results=1
        )

    def test_resolve_course_name_fuzzy_match(self, mock_vector_store):
        """Test course name resolution with fuzzy matching"""
        mock_chroma_results = {
            'documents': [["Advanced Deep Learning with PyTorch"]],
            'metadatas': [[{"title": "Advanced Deep Learning with PyTorch", "instructor": "Prof. Johnson"}]]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name("deep learning pytorch")
        
        assert result == "Advanced Deep Learning with PyTorch"

    def test_resolve_course_name_no_results(self, mock_vector_store):
        """Test course name resolution when no courses are found"""
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name("nonexistent course")
        
        assert result is None

    def test_resolve_course_name_empty_documents(self, mock_vector_store):
        """Test course name resolution with empty documents"""
        mock_chroma_results = {
            'documents': [None],
            'metadatas': [None]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name("test course")
        
        assert result is None

    def test_resolve_course_name_chromadb_error(self, mock_vector_store):
        """Test course name resolution when ChromaDB query fails"""
        mock_vector_store.course_catalog.query.side_effect = Exception("Database error")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store._resolve_course_name("test course")
            
            assert result is None
            mock_print.assert_called_once_with("Error resolving course name: Database error")

    def test_resolve_course_name_malformed_metadata(self, mock_vector_store):
        """Test course name resolution with malformed metadata"""
        mock_chroma_results = {
            'documents': [["Some course"]],
            'metadatas': [[{"invalid_key": "no title field"}]]  # Missing 'title' key
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        # This should raise a KeyError which gets caught and returns None
        with patch('builtins.print') as mock_print:
            result = mock_vector_store._resolve_course_name("test course")
            
            assert result is None
            # Verify error was printed
            assert mock_print.called


class TestVectorStoreBuildFilter:
    """Test VectorStore filter building functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            return store

    def test_build_filter_no_filters(self, mock_vector_store):
        """Test building filter with no parameters"""
        result = mock_vector_store._build_filter(None, None)
        assert result is None

    def test_build_filter_course_title_only(self, mock_vector_store):
        """Test building filter with course title only"""
        result = mock_vector_store._build_filter("Test Course", None)
        expected = {"course_title": "Test Course"}
        assert result == expected

    def test_build_filter_lesson_number_only(self, mock_vector_store):
        """Test building filter with lesson number only"""
        result = mock_vector_store._build_filter(None, 5)
        expected = {"lesson_number": 5}
        assert result == expected

    def test_build_filter_both_parameters(self, mock_vector_store):
        """Test building filter with both course title and lesson number"""
        result = mock_vector_store._build_filter("Advanced Python", 3)
        expected = {
            "$and": [
                {"course_title": "Advanced Python"},
                {"lesson_number": 3}
            ]
        }
        assert result == expected

    def test_build_filter_lesson_number_zero(self, mock_vector_store):
        """Test building filter with lesson number 0 (edge case)"""
        result = mock_vector_store._build_filter(None, 0)
        expected = {"lesson_number": 0}
        assert result == expected

    def test_build_filter_empty_course_title(self, mock_vector_store):
        """Test building filter with empty course title"""
        result = mock_vector_store._build_filter("", 2)
        # Empty string should be treated as None
        expected = {"lesson_number": 2}
        assert result == expected

    def test_build_filter_both_with_empty_course_title(self, mock_vector_store):
        """Test building filter with empty course title and lesson number"""
        result = mock_vector_store._build_filter("", None)
        assert result is None


class TestVectorStoreMetadata:
    """Test VectorStore metadata retrieval methods"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            store.course_catalog = Mock()
            return store

    def test_get_all_courses_metadata_success(self, mock_vector_store):
        """Test successful retrieval of all courses metadata"""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "https://example.com/lesson2"}
        ]
        
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Course 1",
                    "instructor": "Dr. Smith",
                    "course_link": "https://example.com/course1",
                    "lessons_json": json.dumps(lessons_data),
                    "lesson_count": 2
                },
                {
                    "title": "Course 2", 
                    "instructor": "Prof. Johnson",
                    "course_link": "https://example.com/course2",
                    "lessons_json": json.dumps([]),
                    "lesson_count": 0
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_all_courses_metadata()
        
        assert len(result) == 2
        assert result[0]["title"] == "Course 1"
        assert result[0]["lessons"] == lessons_data
        assert "lessons_json" not in result[0]  # Should be removed
        
        assert result[1]["title"] == "Course 2"
        assert result[1]["lessons"] == []

    def test_get_all_courses_metadata_empty_database(self, mock_vector_store):
        """Test retrieving metadata from empty database"""
        mock_chroma_results = {'metadatas': []}
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_all_courses_metadata()
        
        assert result == []

    def test_get_all_courses_metadata_no_metadatas_key(self, mock_vector_store):
        """Test retrieving metadata when results don't have metadatas key"""
        mock_chroma_results = {'ids': ['course1', 'course2']}  # Missing metadatas
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_all_courses_metadata()
        
        assert result == []

    def test_get_all_courses_metadata_malformed_json(self, mock_vector_store):
        """Test retrieving metadata with malformed lessons JSON"""
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Course 1",
                    "lessons_json": "invalid json{{{",  # Malformed JSON
                    "instructor": "Dr. Smith"
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_all_courses_metadata()
            
            assert result == []
            assert mock_print.called  # Error should be printed

    def test_get_all_courses_metadata_chromadb_error(self, mock_vector_store):
        """Test retrieving metadata when ChromaDB fails"""
        mock_vector_store.course_catalog.get.side_effect = Exception("Database connection error")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_all_courses_metadata()
            
            assert result == []
            mock_print.assert_called_once_with("Error getting courses metadata: Database connection error")

    def test_get_course_link_success(self, mock_vector_store):
        """Test successful course link retrieval"""
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Test Course",
                    "course_link": "https://example.com/test-course",
                    "instructor": "Dr. Test"
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_course_link("Test Course")
        
        assert result == "https://example.com/test-course"
        mock_vector_store.course_catalog.get.assert_called_once_with(ids=["Test Course"])

    def test_get_course_link_not_found(self, mock_vector_store):
        """Test course link retrieval for non-existent course"""
        mock_chroma_results = {'metadatas': []}
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_course_link("Nonexistent Course")
        
        assert result is None

    def test_get_course_link_no_link_metadata(self, mock_vector_store):
        """Test course link retrieval when course has no link"""
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Test Course",
                    "instructor": "Dr. Test"
                    # Missing course_link
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_course_link("Test Course")
        
        assert result is None

    def test_get_course_link_chromadb_error(self, mock_vector_store):
        """Test course link retrieval when ChromaDB fails"""
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_course_link("Test Course")
            
            assert result is None
            mock_print.assert_called_once_with("Error getting course link: Database error")

    def test_get_lesson_link_success(self, mock_vector_store):
        """Test successful lesson link retrieval"""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "https://example.com/lesson2"}
        ]
        
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Test Course",
                    "lessons_json": json.dumps(lessons_data)
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_lesson_link("Test Course", 2)
        
        assert result == "https://example.com/lesson2"

    def test_get_lesson_link_lesson_not_found(self, mock_vector_store):
        """Test lesson link retrieval for non-existent lesson"""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"}
        ]
        
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Test Course",
                    "lessons_json": json.dumps(lessons_data)
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_lesson_link("Test Course", 999)
        
        assert result is None

    def test_get_lesson_link_no_lessons_json(self, mock_vector_store):
        """Test lesson link retrieval when course has no lessons_json"""
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Test Course",
                    "instructor": "Dr. Test"
                    # Missing lessons_json
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_lesson_link("Test Course", 1)
        
        assert result is None

    def test_get_lesson_link_malformed_json(self, mock_vector_store):
        """Test lesson link retrieval with malformed lessons JSON"""
        mock_chroma_results = {
            'metadatas': [
                {
                    "title": "Test Course",
                    "lessons_json": "invalid json{{"
                }
            ]
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_lesson_link("Test Course", 1)
            
            assert result is None
            assert mock_print.called

    def test_get_lesson_link_course_not_found(self, mock_vector_store):
        """Test lesson link retrieval for non-existent course"""
        mock_chroma_results = {'metadatas': []}
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_lesson_link("Nonexistent Course", 1)
        
        assert result is None

    def test_get_lesson_link_chromadb_error(self, mock_vector_store):
        """Test lesson link retrieval when ChromaDB fails"""
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_lesson_link("Test Course", 1)
            
            assert result is None
            mock_print.assert_called_once_with("Error getting lesson link: Database error")


class TestVectorStoreUtilityMethods:
    """Test VectorStore utility and management methods"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            store.course_catalog = Mock()
            store.course_content = Mock()
            store.client = Mock()
            return store

    def test_get_existing_course_titles_success(self, mock_vector_store):
        """Test successful retrieval of existing course titles"""
        mock_chroma_results = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_existing_course_titles()
        
        assert result == ['Course 1', 'Course 2', 'Course 3']

    def test_get_existing_course_titles_empty(self, mock_vector_store):
        """Test retrieval of course titles from empty database"""
        mock_chroma_results = {'ids': []}
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_existing_course_titles()
        
        assert result == []

    def test_get_existing_course_titles_no_ids_key(self, mock_vector_store):
        """Test retrieval when results don't have ids key"""
        mock_chroma_results = {'metadatas': ['some', 'data']}  # Missing ids key
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_existing_course_titles()
        
        assert result == []

    def test_get_existing_course_titles_chromadb_error(self, mock_vector_store):
        """Test course titles retrieval when ChromaDB fails"""
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_existing_course_titles()
            
            assert result == []
            mock_print.assert_called_once_with("Error getting existing course titles: Database error")

    def test_get_course_count_success(self, mock_vector_store):
        """Test successful course count retrieval"""
        mock_chroma_results = {
            'ids': ['Course 1', 'Course 2', 'Course 3', 'Course 4']
        }
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_course_count()
        
        assert result == 4

    def test_get_course_count_empty_database(self, mock_vector_store):
        """Test course count for empty database"""
        mock_chroma_results = {'ids': []}
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_course_count()
        
        assert result == 0

    def test_get_course_count_no_ids_key(self, mock_vector_store):
        """Test course count when results don't have ids key"""
        mock_chroma_results = {}
        mock_vector_store.course_catalog.get.return_value = mock_chroma_results
        
        result = mock_vector_store.get_course_count()
        
        assert result == 0

    def test_get_course_count_chromadb_error(self, mock_vector_store):
        """Test course count when ChromaDB fails"""
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store.get_course_count()
            
            assert result == 0
            mock_print.assert_called_once_with("Error getting course count: Database error")

    def test_clear_all_data_success(self, mock_vector_store):
        """Test successful clearing of all data"""
        # Mock _create_collection to return new mock collections
        new_catalog_mock = Mock()
        new_content_mock = Mock()
        mock_vector_store._create_collection = Mock(side_effect=[new_catalog_mock, new_content_mock])
        
        mock_vector_store.clear_all_data()
        
        # Verify collections were deleted
        expected_calls = [
            call("course_catalog"),
            call("course_content")
        ]
        mock_vector_store.client.delete_collection.assert_has_calls(expected_calls)
        
        # Verify collections were recreated
        mock_vector_store._create_collection.assert_has_calls([
            call("course_catalog"),
            call("course_content")
        ])
        
        # Verify new collections were assigned
        assert mock_vector_store.course_catalog == new_catalog_mock
        assert mock_vector_store.course_content == new_content_mock

    def test_clear_all_data_chromadb_error(self, mock_vector_store):
        """Test clearing data when ChromaDB fails"""
        mock_vector_store.client.delete_collection.side_effect = Exception("Delete failed")
        
        with patch('builtins.print') as mock_print:
            mock_vector_store.clear_all_data()
            
            mock_print.assert_called_once_with("Error clearing data: Delete failed")


class TestVectorStoreDataManagement:
    """Test VectorStore data addition methods"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            store.course_catalog = Mock()
            store.course_content = Mock()
            return store

    @pytest.fixture
    def sample_course(self):
        """Create a sample Course object for testing"""
        lessons = [
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
        ]
        return Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Dr. Test",
            lessons=lessons
        )

    @pytest.fixture  
    def sample_course_chunks(self):
        """Create sample CourseChunk objects for testing"""
        return [
            CourseChunk(
                content="This is the first chunk of content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="This is the second chunk of content", 
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            ),
            CourseChunk(
                content="Advanced content from lesson 2",
                course_title="Test Course", 
                lesson_number=2,
                chunk_index=2
            )
        ]

    def test_add_course_metadata_success(self, mock_vector_store, sample_course):
        """Test successful addition of course metadata"""
        mock_vector_store.add_course_metadata(sample_course)
        
        # Verify course_catalog.add was called with correct parameters
        mock_vector_store.course_catalog.add.assert_called_once()
        
        call_args = mock_vector_store.course_catalog.add.call_args
        
        # Check documents
        assert call_args.kwargs['documents'] == ["Test Course"]
        
        # Check IDs
        assert call_args.kwargs['ids'] == ["Test Course"]
        
        # Check metadata structure
        metadata = call_args.kwargs['metadatas'][0]
        assert metadata['title'] == "Test Course"
        assert metadata['instructor'] == "Dr. Test"
        assert metadata['course_link'] == "https://example.com/course"
        assert metadata['lesson_count'] == 2
        
        # Check lessons JSON
        lessons_data = json.loads(metadata['lessons_json'])
        assert len(lessons_data) == 2
        assert lessons_data[0]['lesson_number'] == 1
        assert lessons_data[0]['lesson_title'] == "Introduction"
        assert lessons_data[0]['lesson_link'] == "https://example.com/lesson1"

    def test_add_course_metadata_no_lessons(self, mock_vector_store):
        """Test adding course metadata with no lessons"""
        course = Course(title="Empty Course", instructor="Dr. Empty")
        
        mock_vector_store.add_course_metadata(course)
        
        call_args = mock_vector_store.course_catalog.add.call_args
        metadata = call_args.kwargs['metadatas'][0]
        
        assert metadata['lesson_count'] == 0
        lessons_data = json.loads(metadata['lessons_json'])
        assert lessons_data == []

    def test_add_course_content_success(self, mock_vector_store, sample_course_chunks):
        """Test successful addition of course content chunks"""
        mock_vector_store.add_course_content(sample_course_chunks)
        
        # Verify course_content.add was called with correct parameters
        mock_vector_store.course_content.add.assert_called_once()
        
        call_args = mock_vector_store.course_content.add.call_args
        
        # Check documents
        expected_docs = [
            "This is the first chunk of content",
            "This is the second chunk of content", 
            "Advanced content from lesson 2"
        ]
        assert call_args.kwargs['documents'] == expected_docs
        
        # Check metadata
        expected_metadata = [
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 1},
            {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 2}
        ]
        assert call_args.kwargs['metadatas'] == expected_metadata
        
        # Check IDs
        expected_ids = [
            "Test_Course_0",
            "Test_Course_1", 
            "Test_Course_2"
        ]
        assert call_args.kwargs['ids'] == expected_ids

    def test_add_course_content_empty_chunks(self, mock_vector_store):
        """Test adding empty list of course content chunks"""
        mock_vector_store.add_course_content([])
        
        # Verify course_content.add was not called
        mock_vector_store.course_content.add.assert_not_called()

    def test_add_course_content_with_spaces_in_title(self, mock_vector_store):
        """Test adding course content with spaces in course title"""
        chunks = [
            CourseChunk(
                content="Test content",
                course_title="Course With Spaces In Title",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        mock_vector_store.add_course_content(chunks)
        
        call_args = mock_vector_store.course_content.add.call_args
        # Spaces should be replaced with underscores in IDs
        assert call_args.kwargs['ids'] == ["Course_With_Spaces_In_Title_0"]


class TestVectorStoreEdgeCases:
    """Test VectorStore edge cases and error scenarios"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB components"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            store = VectorStore("/test/path", "test-model")
            store.course_catalog = Mock()
            store.course_content = Mock()
            return store

    def test_search_with_none_query(self, mock_vector_store):
        """Test search with None as query"""
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        # This should still work but return empty results
        results = mock_vector_store.search(None)
        
        assert results.is_empty()
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=[None],
            n_results=5,
            where=None
        )

    def test_search_with_empty_string_query(self, mock_vector_store):
        """Test search with empty string query"""
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("")
        
        assert results.is_empty()
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=[""],
            n_results=5,
            where=None
        )

    def test_search_with_very_large_limit(self, mock_vector_store):
        """Test search with extremely large limit"""
        mock_chroma_results = {
            'documents': [["result"]],
            'metadatas': [[{"key": "value"}]],
            'distances': [[0.1]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("test", limit=999999)
        
        assert not results.is_empty()
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test"],
            n_results=999999,
            where=None
        )

    def test_search_with_negative_lesson_number(self, mock_vector_store):
        """Test search with negative lesson number"""
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        mock_vector_store.course_content.query.return_value = mock_chroma_results
        
        results = mock_vector_store.search("test", lesson_number=-1)
        
        # Should still create filter with negative number
        expected_filter = {"lesson_number": -1}
        mock_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test"],
            n_results=5,
            where=expected_filter
        )

    def test_resolve_course_name_with_special_characters(self, mock_vector_store):
        """Test course name resolution with special characters"""
        mock_chroma_results = {
            'documents': [["C++ Programming & Advanced Topics"]],
            'metadatas': [[{"title": "C++ Programming & Advanced Topics"}]]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name("C++ & Advanced")
        
        assert result == "C++ Programming & Advanced Topics"

    def test_malformed_chroma_response_missing_keys(self, mock_vector_store):
        """Test handling of malformed ChromaDB response missing required keys"""
        # ChromaDB response missing 'documents' key
        malformed_results = {
            'metadatas': [[{"key": "value"}]],
            'distances': [[0.1]]
            # Missing 'documents' key
        }
        mock_vector_store.course_content.query.return_value = malformed_results
        
        # This should raise a KeyError which gets caught and returns error result
        results = mock_vector_store.search("test query")
        
        assert results.is_empty()
        assert results.error is not None
        assert "Search error:" in results.error

    def test_chroma_response_with_none_values(self, mock_vector_store):
        """Test handling of ChromaDB response with None values"""
        chroma_results = {
            'documents': None,
            'metadatas': None,
            'distances': None
        }
        mock_vector_store.course_content.query.return_value = chroma_results
        
        results = mock_vector_store.search("test query")
        
        # SearchResults.from_chroma should handle None values gracefully
        assert results.is_empty()
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_course_catalog_query_timeout(self, mock_vector_store):
        """Test course catalog query with timeout error"""
        mock_vector_store.course_catalog.query.side_effect = TimeoutError("Query timeout")
        
        with patch('builtins.print') as mock_print:
            result = mock_vector_store._resolve_course_name("test course")
            
            assert result is None
            mock_print.assert_called_once_with("Error resolving course name: Query timeout")

    def test_course_content_collection_missing(self, mock_vector_store):
        """Test search when course_content collection is missing/corrupted"""
        mock_vector_store.course_content.query.side_effect = Exception("Collection 'course_content' does not exist")
        
        results = mock_vector_store.search("test query")
        
        assert results.is_empty()
        assert "Collection 'course_content' does not exist" in results.error

    def test_concurrent_access_simulation(self, mock_vector_store):
        """Test handling of concurrent access errors"""
        mock_vector_store.course_content.query.side_effect = Exception("Database is locked")
        
        results = mock_vector_store.search("test query")
        
        assert results.is_empty()
        assert "Database is locked" in results.error

    def test_very_long_course_name_resolution(self, mock_vector_store):
        """Test course name resolution with very long course name"""
        very_long_name = "A" * 1000  # 1000 character course name
        mock_chroma_results = {
            'documents': [[very_long_name]],
            'metadatas': [[{"title": very_long_name}]]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name(very_long_name)
        
        assert result == very_long_name

    def test_unicode_course_names(self, mock_vector_store):
        """Test handling of Unicode characters in course names"""
        unicode_course = "机器学习基础 - Machine Learning Fundamentals 🤖"
        mock_chroma_results = {
            'documents': [[unicode_course]],
            'metadatas': [[{"title": unicode_course}]]
        }
        mock_vector_store.course_catalog.query.return_value = mock_chroma_results
        
        result = mock_vector_store._resolve_course_name("机器学习")
        
        assert result == unicode_course


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
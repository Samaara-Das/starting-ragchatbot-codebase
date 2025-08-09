"""
Unit tests for CourseSearchTool to diagnose "query failed" issues.
"""
import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool functionality"""
    
    def test_tool_definition(self, course_search_tool):
        """Test tool definition is correct for Anthropic API"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert definition["description"]
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        
        # Check required parameters
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        
        # Check required fields
        required = definition["input_schema"]["required"]
        assert "query" in required
        assert len(required) == 1  # Only query is required
    
    def test_execute_successful_search(self, course_search_tool, mock_search_results):
        """Test successful search execution with results"""
        # Setup
        course_search_tool.store.search.return_value = mock_search_results
        
        # Execute
        result = course_search_tool.execute("What is RAG?")
        
        # Verify
        assert result
        assert "Introduction to RAG" in result
        assert "ML Fundamentals" in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result
        
        # Verify store was called correctly
        course_search_tool.store.search.assert_called_once_with(
            query="What is RAG?",
            course_name=None,
            lesson_number=None
        )
        
        # Verify sources were tracked
        assert len(course_search_tool.last_sources) == 2
        assert "Introduction to RAG - Lesson 1" in course_search_tool.last_sources[0]
        assert "ML Fundamentals - Lesson 2" in course_search_tool.last_sources[1]
    
    def test_execute_with_course_name_filter(self, course_search_tool, mock_search_results):
        """Test search with course name filtering"""
        # Setup
        course_search_tool.store.search.return_value = mock_search_results
        
        # Execute
        result = course_search_tool.execute("What is RAG?", course_name="MCP")
        
        # Verify store called with course filter
        course_search_tool.store.search.assert_called_once_with(
            query="What is RAG?",
            course_name="MCP",
            lesson_number=None
        )
        
        assert result
    
    def test_execute_with_lesson_number_filter(self, course_search_tool, mock_search_results):
        """Test search with lesson number filtering"""
        # Setup
        course_search_tool.store.search.return_value = mock_search_results
        
        # Execute
        result = course_search_tool.execute("What is RAG?", lesson_number=1)
        
        # Verify store called with lesson filter
        course_search_tool.store.search.assert_called_once_with(
            query="What is RAG?",
            course_name=None,
            lesson_number=1
        )
        
        assert result
    
    def test_execute_with_both_filters(self, course_search_tool, mock_search_results):
        """Test search with both course name and lesson number filters"""
        # Setup
        course_search_tool.store.search.return_value = mock_search_results
        
        # Execute
        result = course_search_tool.execute("What is RAG?", course_name="MCP", lesson_number=1)
        
        # Verify store called with both filters
        course_search_tool.store.search.assert_called_once_with(
            query="What is RAG?",
            course_name="MCP",
            lesson_number=1
        )
        
        assert result
    
    def test_execute_with_search_error(self, course_search_tool, error_search_results):
        """Test handling when vector store search returns error"""
        # Setup
        course_search_tool.store.search.return_value = error_search_results
        
        # Execute
        result = course_search_tool.execute("What is RAG?")
        
        # Verify error is returned
        assert result == "Test error message"
        assert len(course_search_tool.last_sources) == 0
    
    def test_execute_with_empty_results(self, course_search_tool, empty_search_results):
        """Test handling when search returns no results"""
        # Setup
        course_search_tool.store.search.return_value = empty_search_results
        
        # Execute
        result = course_search_tool.execute("Non-existent query")
        
        # Verify empty results message
        assert result == "No relevant content found."
        assert len(course_search_tool.last_sources) == 0
    
    def test_execute_with_empty_results_and_filters(self, course_search_tool, empty_search_results):
        """Test empty results message includes filter information"""
        # Setup
        course_search_tool.store.search.return_value = empty_search_results
        
        # Execute with course filter
        result = course_search_tool.execute("query", course_name="Non-existent Course")
        assert "in course 'Non-existent Course'" in result
        
        # Execute with lesson filter
        result = course_search_tool.execute("query", lesson_number=99)
        assert "in lesson 99" in result
        
        # Execute with both filters
        result = course_search_tool.execute("query", course_name="Test", lesson_number=5)
        assert "in course 'Test'" in result
        assert "in lesson 5" in result
    
    def test_format_results_with_lesson_links(self, course_search_tool):
        """Test results formatting includes lesson links when available"""
        # Setup results with metadata
        results = SearchResults(
            documents=["Content with lesson link"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1, "chunk_id": "chunk1"}],
            distances=[0.1]
        )
        
        # Mock get_lesson_link to return a link
        course_search_tool.store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Execute
        formatted = course_search_tool._format_results(results)
        
        # Verify format
        assert "[Test Course - Lesson 1]" in formatted
        assert "Content with lesson link" in formatted
        
        # Verify source includes link
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0] == "Test Course - Lesson 1|https://example.com/lesson1"
    
    def test_format_results_without_lesson_links(self, course_search_tool):
        """Test results formatting when no lesson links available"""
        # Setup results without lesson numbers
        results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "Test Course", "chunk_id": "chunk1"}],
            distances=[0.1]
        )
        
        # Execute
        formatted = course_search_tool._format_results(results)
        
        # Verify format
        assert "[Test Course]" in formatted
        assert "Content without lesson" in formatted
        
        # Verify source without link
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0] == "Test Course"
    
    def test_format_results_with_missing_metadata(self, course_search_tool):
        """Test results formatting handles missing metadata gracefully"""
        # Setup results with incomplete metadata
        results = SearchResults(
            documents=["Content with incomplete metadata"],
            metadata=[{"chunk_id": "chunk1"}],  # Missing course_title
            distances=[0.1]
        )
        
        # Execute
        formatted = course_search_tool._format_results(results)
        
        # Verify fallback values
        assert "[unknown]" in formatted
        assert "Content with incomplete metadata" in formatted
        
        # Verify source uses fallback
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0] == "unknown"
    
    def test_multiple_results_formatting(self, course_search_tool):
        """Test formatting multiple search results"""
        # Setup multiple results
        results = SearchResults(
            documents=["First result content", "Second result content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        course_search_tool.store.get_lesson_link.side_effect = [
            "https://example.com/a1",
            "https://example.com/b2"
        ]
        
        # Execute
        formatted = course_search_tool._format_results(results)
        
        # Verify both results included
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted
        assert "First result content" in formatted
        assert "Second result content" in formatted
        assert "\n\n" in formatted  # Results separated by double newlines
        
        # Verify both sources tracked
        assert len(course_search_tool.last_sources) == 2
        assert "Course A - Lesson 1|https://example.com/a1" in course_search_tool.last_sources
        assert "Course B - Lesson 2|https://example.com/b2" in course_search_tool.last_sources
    
    def test_source_tracking_reset_on_new_search(self, course_search_tool, mock_search_results):
        """Test that sources are properly tracked and reset between searches"""
        # First search
        course_search_tool.store.search.return_value = mock_search_results
        course_search_tool.execute("First query")
        
        first_sources = course_search_tool.last_sources.copy()
        assert len(first_sources) > 0
        
        # Second search with different results
        new_results = SearchResults(
            documents=["New content"],
            metadata=[{"course_title": "New Course", "lesson_number": 3}],
            distances=[0.1]
        )
        course_search_tool.store.search.return_value = new_results
        course_search_tool.execute("Second query")
        
        # Verify sources were updated, not appended
        assert course_search_tool.last_sources != first_sources
        assert len(course_search_tool.last_sources) == 1
        assert "New Course - Lesson 3" in course_search_tool.last_sources[0]

    def test_vector_store_exception_handling(self, course_search_tool):
        """Test handling when vector store raises exceptions"""
        # Setup vector store to raise exception
        course_search_tool.store.search.side_effect = Exception("ChromaDB connection failed")
        
        # Execute - should not crash
        result = course_search_tool.execute("test query")
        
        # Should return error message from SearchResults.empty()
        # Note: This tests the vector_store.search() method's exception handling
        course_search_tool.store.search.assert_called_once()
    
    def test_edge_case_empty_query(self, course_search_tool, empty_search_results):
        """Test behavior with empty query string"""
        course_search_tool.store.search.return_value = empty_search_results
        
        result = course_search_tool.execute("")
        
        # Should still attempt search
        course_search_tool.store.search.assert_called_once_with(
            query="",
            course_name=None, 
            lesson_number=None
        )
        assert "No relevant content found" in result
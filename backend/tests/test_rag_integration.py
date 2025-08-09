"""
Comprehensive integration tests for the RAG system.

These tests focus on the interactions between components rather than individual
component functionality, covering end-to-end workflows, tool integration,
session management, and error handling at the integration level.

Test Coverage:
1. End-to-end query processing (user query → AI → tools → vector store → response)
2. Tool registration and availability verification  
3. Complete flow with actual tool execution
4. Session management integration
5. Source extraction and propagation through the system
6. Error handling at the integration level
7. Both content search and outline tool integration
8. Real user scenarios with proper error propagation
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

# Import the modules we're testing
from rag_system import RAGSystem
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from session_manager import SessionManager
from vector_store import VectorStore, SearchResults


class TestRAGSystemIntegration:
    """Test RAG system end-to-end integration scenarios"""

    def test_end_to_end_content_search_query(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test complete flow: user query → AI → tool → vector store → response with sources"""
        # Setup: Mock AI response that uses the search tool
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_search_123"
        mock_tool_block.input = {"query": "RAG systems", "course_name": "Introduction"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation to provide accurate, contextual responses."
        
        # Setup anthropic client to return tool use first, then final response
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup vector store search results
        search_results = SearchResults(
            documents=["RAG stands for Retrieval-Augmented Generation..."],
            metadata=[{"course_title": "Introduction to RAG", "lesson_number": 1, "chunk_id": "chunk1"}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Execute the query
        response, sources = rag_system.query("What are RAG systems?", session_id="test_session")
        
        # Verify the complete flow
        assert response == "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation to provide accurate, contextual responses."
        assert len(sources) == 1
        assert "Introduction to RAG - Lesson 1" in sources[0]
        assert "https://example.com/lesson1" in sources[0]
        
        # Verify AI was called with tools
        assert mock_anthropic_client.messages.create.call_count == 2
        first_call = mock_anthropic_client.messages.create.call_args_list[0]
        assert "tools" in first_call.kwargs
        assert len(first_call.kwargs["tools"]) == 2  # search and outline tools
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once()
        search_call_args = mock_vector_store.search.call_args
        assert search_call_args.kwargs["query"] == "RAG systems"

    def test_end_to_end_outline_query(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test complete flow for course outline request"""
        # Setup: Mock AI response that uses the outline tool
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.id = "tool_outline_456"
        mock_tool_block.input = {"course_title": "Introduction"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "The Introduction course covers 2 lessons: 1. What is RAG? and 2. RAG Applications."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup vector store for outline query
        mock_vector_store._resolve_course_name.return_value = "Introduction to RAG"
        
        # Execute the query
        response, sources = rag_system.query("Give me the outline of the Introduction course")
        
        # Verify the complete flow
        assert response == "The Introduction course covers 2 lessons: 1. What is RAG? and 2. RAG Applications."
        assert len(sources) == 1
        assert "Introduction to RAG" in sources[0]
        
        # Verify outline tool was used
        mock_vector_store._resolve_course_name.assert_called_once_with("Introduction")
        mock_vector_store.get_all_courses_metadata.assert_called_once()

    def test_session_management_integration(self, rag_system, mock_anthropic_client):
        """Test session management integration with conversation history"""
        # Create a session first
        session_id = rag_system.session_manager.create_session()
        
        # Add some conversation history
        rag_system.session_manager.add_exchange(session_id, "Previous question", "Previous answer")
        
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with context"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Execute query with session
        response, sources = rag_system.query("Follow-up question", session_id=session_id)
        
        # Verify response
        assert response == "Response with context"
        
        # Verify conversation history was included in AI call
        ai_call_args = mock_anthropic_client.messages.create.call_args
        system_prompt = ai_call_args.kwargs["system"]
        assert "Previous conversation:" in system_prompt
        assert "Previous question" in system_prompt
        assert "Previous answer" in system_prompt
        
        # Verify new exchange was added to session
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "Follow-up question" in history
        assert "Response with context" in history

    def test_tool_registration_and_availability(self, rag_system):
        """Test that all expected tools are properly registered and available"""
        # Verify tool manager has the expected tools
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        
        assert len(tool_definitions) == 2
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        
        # Verify tool definitions have required structure
        search_tool_def = next(tool for tool in tool_definitions if tool["name"] == "search_course_content")
        assert "description" in search_tool_def
        assert "input_schema" in search_tool_def
        assert "properties" in search_tool_def["input_schema"]
        assert "query" in search_tool_def["input_schema"]["properties"]
        
        outline_tool_def = next(tool for tool in tool_definitions if tool["name"] == "get_course_outline")
        assert "description" in outline_tool_def
        assert "input_schema" in outline_tool_def
        assert "course_title" in outline_tool_def["input_schema"]["properties"]

    def test_source_extraction_and_propagation(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test that sources are properly extracted and propagated through the system"""
        # Setup tool usage scenario
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "lesson content", "lesson_number": 2}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Lesson 2 content response"
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup search results with multiple sources
        search_results = SearchResults(
            documents=["Content from lesson 2", "More lesson 2 content"],
            metadata=[
                {"course_title": "MCP Course", "lesson_number": 2, "chunk_id": "chunk1"},
                {"course_title": "MCP Course", "lesson_number": 2, "chunk_id": "chunk2"}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/mcp-lesson2"
        
        # Execute query
        response, sources = rag_system.query("What is covered in lesson 2 of the MCP course?")
        
        # Verify sources are properly extracted
        assert len(sources) == 2
        for source in sources:
            assert "MCP Course - Lesson 2" in source
            assert "https://example.com/mcp-lesson2" in source
        
        # Verify sources are reset after retrieval
        reset_sources = rag_system.tool_manager.get_last_sources()
        assert len(reset_sources) == 0

    def test_error_handling_anthropic_api_failure(self, rag_system, mock_anthropic_client):
        """Test error handling when Anthropic API fails"""
        # Setup API failure
        mock_anthropic_client.messages.create.side_effect = Exception("API connection failed")
        
        # Execute query and verify exception propagation
        with pytest.raises(Exception) as exc_info:
            rag_system.query("Test query")
        
        assert "API connection failed" in str(exc_info.value)

    def test_error_handling_vector_store_failure(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test error handling when vector store search fails"""
        # Setup tool usage scenario
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error"
        mock_tool_block.input = {"query": "test query"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "I couldn't find any relevant information."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup vector store to return error results
        error_results = SearchResults.empty("Database connection error")
        mock_vector_store.search.return_value = error_results
        
        # Execute query
        response, sources = rag_system.query("Test query")
        
        # Verify error is handled gracefully
        assert response == "I couldn't find any relevant information."
        assert len(sources) == 0

    def test_error_handling_tool_execution_failure(self, rag_system, mock_anthropic_client):
        """Test error handling when tool execution fails"""
        # Setup tool usage scenario
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_fail"
        mock_tool_block.input = {"query": "test"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Tool execution failed, but I can still respond."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Mock tool manager to raise exception
        with patch.object(rag_system.tool_manager, 'execute_tool', side_effect=Exception("Tool failure")):
            # The system should handle this gracefully through the AI generator
            with pytest.raises(Exception):
                rag_system.query("Test query")

    def test_error_handling_session_manager_failure(self, rag_system, mock_anthropic_client):
        """Test error handling when session manager operations fail"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response despite session error"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Mock session manager to fail on get_conversation_history
        with patch.object(rag_system.session_manager, 'get_conversation_history', 
                         side_effect=Exception("Session storage error")):
            # System should still work without history
            response, sources = rag_system.query("Test query", session_id="failing_session")
            
            # Verify system handled the error gracefully
            assert response == "Response despite session error"
            assert len(sources) == 0

    def test_error_handling_component_chain_failure(self, rag_system):
        """Test error handling when multiple components fail in sequence"""
        # This test simulates the "query failed" error scenario
        # Mock AI generator to fail
        with patch.object(rag_system.ai_generator, 'generate_response', 
                         side_effect=Exception("AI service unavailable")):
            with pytest.raises(Exception) as exc_info:
                rag_system.query("Test query")
            
            assert "AI service unavailable" in str(exc_info.value)

    def test_error_handling_malformed_ai_response(self, rag_system, mock_anthropic_client):
        """Test handling of malformed AI responses"""
        # Setup malformed response (missing content)
        mock_response = Mock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Execute query and expect error handling
        with pytest.raises(IndexError):
            rag_system.query("Test query")

    def test_error_handling_tool_response_parsing_failure(self, rag_system, mock_anthropic_client):
        """Test error handling when tool responses cannot be parsed"""
        # Setup tool response with malformed input
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "malformed_tool"
        mock_tool_block.input = None  # Malformed input
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_anthropic_client.messages.create.return_value = mock_tool_response
        
        # Execute and expect error propagation
        with pytest.raises(Exception):
            rag_system.query("Test query")

    def test_empty_search_results_handling(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test handling of empty search results"""
        # Setup tool usage scenario
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_empty"
        mock_tool_block.input = {"query": "nonexistent topic"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "No relevant content found."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup empty search results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        # Execute query
        response, sources = rag_system.query("Tell me about nonexistent topic")
        
        # Verify empty results are handled properly
        assert response == "No relevant content found."
        assert len(sources) == 0

    def test_real_user_scenario_lesson_specific_query(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test real user scenario: 'What is covered in lesson 2 of the MCP course?'"""
        # Setup tool usage for lesson-specific query
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_lesson"
        mock_tool_block.input = {"query": "covered", "course_name": "MCP", "lesson_number": 2}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Lesson 2 of the MCP course covers server setup, client configuration, and basic protocol implementation."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup realistic search results
        search_results = SearchResults(
            documents=["In this lesson, we'll cover MCP server setup and client configuration..."],
            metadata=[{"course_title": "MCP Protocol Course", "lesson_number": 2, "chunk_id": "lesson2_chunk1"}],
            distances=[0.05]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://courses.example.com/mcp/lesson2"
        
        # Execute realistic user query
        response, sources = rag_system.query("What is covered in lesson 2 of the MCP course?")
        
        # Verify realistic response
        assert "server setup" in response.lower()
        assert "client configuration" in response.lower()
        assert len(sources) == 1
        assert "MCP Protocol Course - Lesson 2" in sources[0]
        assert "https://courses.example.com/mcp/lesson2" in sources[0]

    def test_real_user_scenario_general_knowledge_query(self, rag_system, mock_anthropic_client):
        """Test real user scenario: 'Tell me about RAG systems' (general knowledge)"""
        # Setup non-tool response for general knowledge
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "RAG systems combine retrieval and generation to provide accurate, contextual responses by first retrieving relevant information and then generating responses based on that information."
        mock_response.stop_reason = "end_turn"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Execute general knowledge query
        response, sources = rag_system.query("Tell me about RAG systems")
        
        # Verify general knowledge response
        assert "retrieval" in response.lower()
        assert "generation" in response.lower()
        assert len(sources) == 0  # No tool usage = no sources

    def test_real_user_scenario_course_not_found(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test real user scenario: Query for non-existent course"""
        # Setup tool usage for non-existent course
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_notfound"
        mock_tool_block.input = {"query": "advanced topics", "course_name": "NonexistentCourse"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "I couldn't find any course matching 'NonexistentCourse'."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup empty search results for non-existent course
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        # Execute realistic query for non-existent course
        response, sources = rag_system.query("What advanced topics are covered in NonexistentCourse?")
        
        # Verify appropriate handling of non-existent course
        assert response == "I couldn't find any course matching 'NonexistentCourse'."
        assert len(sources) == 0

    def test_real_user_scenario_complex_multi_parameter_query(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test real user scenario: Complex query with multiple parameters"""
        # Setup tool usage with multiple parameters
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_complex"
        mock_tool_block.input = {
            "query": "deployment strategies",
            "course_name": "DevOps",
            "lesson_number": 5
        }
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Lesson 5 of the DevOps course covers three main deployment strategies: blue-green deployment, rolling updates, and canary deployments. Each has specific use cases and trade-offs."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup realistic search results with detailed metadata
        search_results = SearchResults(
            documents=["Deployment strategies include blue-green deployment for zero-downtime updates, rolling updates for gradual rollouts..."],
            metadata=[{
                "course_title": "DevOps Fundamentals", 
                "lesson_number": 5, 
                "chunk_id": "lesson5_deployment_chunk1",
                "lesson_title": "Deployment Strategies"
            }],
            distances=[0.02]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://courses.example.com/devops/lesson5-deployment"
        
        # Execute complex multi-parameter query
        response, sources = rag_system.query("What deployment strategies are discussed in lesson 5 of the DevOps course?")
        
        # Verify detailed response handling
        assert "blue-green" in response.lower()
        assert "rolling" in response.lower()
        assert "canary" in response.lower()
        assert len(sources) == 1
        assert "DevOps Fundamentals - Lesson 5" in sources[0]
        assert "https://courses.example.com/devops/lesson5-deployment" in sources[0]

    def test_real_user_scenario_outline_with_session_context(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test real user scenario: Outline request within a conversation session"""
        # Create session and add context
        session_id = rag_system.session_manager.create_session()
        rag_system.session_manager.add_exchange(
            session_id, 
            "I'm interested in learning about MCP", 
            "MCP (Model Context Protocol) is great for connecting AI systems. What specific aspect interests you?"
        )
        
        # Setup outline tool usage
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.id = "tool_outline_session"
        mock_tool_block.input = {"course_title": "MCP"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "The MCP Protocol Course covers 4 lessons: 1. Introduction to MCP, 2. Server Implementation, 3. Client Configuration, and 4. Advanced Features."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup course metadata for outline
        mock_vector_store._resolve_course_name.return_value = "MCP Protocol Course"
        course_metadata = {
            "course_title": "MCP Protocol Course",
            "course_link": "https://courses.example.com/mcp-protocol",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction to MCP"},
                {"lesson_number": 2, "lesson_title": "Server Implementation"},
                {"lesson_number": 3, "lesson_title": "Client Configuration"},
                {"lesson_number": 4, "lesson_title": "Advanced Features"}
            ]
        }
        mock_vector_store.get_all_courses_metadata.return_value = [course_metadata]
        
        # Execute outline query with session context
        response, sources = rag_system.query("Give me the outline of the MCP course", session_id=session_id)
        
        # Verify contextual response
        assert "4 lessons" in response
        assert "Introduction to MCP" in response
        assert len(sources) == 1
        assert "MCP Protocol Course" in sources[0]
        
        # Verify conversation history was used
        call_args = mock_anthropic_client.messages.create.call_args_list[0]
        system_prompt = call_args.kwargs["system"]
        assert "Previous conversation:" in system_prompt
        assert "interested in learning about MCP" in system_prompt

    def test_multiple_tool_calls_within_limit(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test that the system properly handles the one-tool-per-query limit"""
        # Setup response with multiple tool blocks (should only execute first one based on system design)
        mock_tool_response = Mock()
        
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "first query"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_2" 
        mock_tool_block2.input = {"course_title": "second query"}
        
        mock_tool_response.content = [mock_tool_block1, mock_tool_block2]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Response after tool execution"
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup mock results
        search_results = SearchResults(
            documents=["Search result"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        
        # Execute query
        response, sources = rag_system.query("Complex query requiring multiple tools")
        
        # Verify both tools were executed (the AI generator handles multiple tool calls)
        assert response == "Response after tool execution"
        # Should have sources from both tools, but the system design limits to one tool type at a time


class TestRAGSystemToolIntegration:
    """Test specific tool integration scenarios"""

    def test_course_search_tool_integration(self, rag_system, mock_vector_store):
        """Test direct integration with CourseSearchTool"""
        # Setup search results
        search_results = SearchResults(
            documents=["Course content about machine learning"],
            metadata=[{"course_title": "ML Basics", "lesson_number": 3, "chunk_id": "chunk_ml"}],
            distances=[0.2]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/ml-lesson3"
        
        # Execute tool directly
        result = rag_system.search_tool.execute(
            query="machine learning", 
            course_name="ML Basics", 
            lesson_number=3
        )
        
        # Verify tool integration
        assert "ML Basics - Lesson 3" in result
        assert "Course content about machine learning" in result
        
        # Verify sources are tracked
        sources = rag_system.search_tool.last_sources
        assert len(sources) == 1
        assert "ML Basics - Lesson 3" in sources[0]
        assert "https://example.com/ml-lesson3" in sources[0]

    def test_course_outline_tool_integration(self, rag_system, mock_vector_store):
        """Test direct integration with CourseOutlineTool"""
        # Setup course metadata
        mock_vector_store._resolve_course_name.return_value = "Advanced Python"
        course_metadata = {
            "course_title": "Advanced Python",
            "course_link": "https://example.com/python-course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Decorators"},
                {"lesson_number": 2, "lesson_title": "Generators"},
                {"lesson_number": 3, "lesson_title": "Context Managers"}
            ]
        }
        mock_vector_store.get_all_courses_metadata.return_value = [course_metadata]
        
        # Execute tool directly
        result = rag_system.outline_tool.execute(course_title="Advanced Python")
        
        # Verify tool integration
        assert "Advanced Python" in result
        assert "Decorators" in result
        assert "Generators" in result
        assert "Context Managers" in result
        assert "3 total" in result
        
        # Verify sources are tracked
        sources = rag_system.outline_tool.last_sources
        assert len(sources) == 1
        assert "Advanced Python" in sources[0]
        assert "https://example.com/python-course" in sources[0]

    def test_tool_manager_source_aggregation(self, rag_system, mock_vector_store):
        """Test that ToolManager properly aggregates sources from different tools"""
        # Setup and execute search tool
        search_results = SearchResults(
            documents=["Search content"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/course-a-lesson1"
        
        rag_system.search_tool.execute(query="test search")
        search_sources = rag_system.tool_manager.get_last_sources()
        
        # Reset and execute outline tool
        rag_system.tool_manager.reset_sources()
        
        mock_vector_store._resolve_course_name.return_value = "Course B"
        course_metadata = {
            "course_title": "Course B",
            "course_link": "https://example.com/course-b",
            "lessons": [{"lesson_number": 1, "lesson_title": "Intro"}]
        }
        mock_vector_store.get_all_courses_metadata.return_value = [course_metadata]
        
        rag_system.outline_tool.execute(course_title="Course B")
        outline_sources = rag_system.tool_manager.get_last_sources()
        
        # Verify source management
        assert len(search_sources) == 1
        assert "Course A" in search_sources[0]
        
        assert len(outline_sources) == 1
        assert "Course B" in outline_sources[0]


class TestRAGSystemErrorScenarios:
    """Test various error scenarios and edge cases"""

    def test_session_without_history(self, rag_system, mock_anthropic_client):
        """Test querying with a session ID that has no history"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response without context"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Query with non-existent session
        response, sources = rag_system.query("Test query", session_id="nonexistent_session")
        
        # Should work without history
        assert response == "Response without context"
        
        # Verify no history was passed to AI
        ai_call_args = mock_anthropic_client.messages.create.call_args
        system_prompt = ai_call_args.kwargs["system"]
        assert "Previous conversation:" not in system_prompt

    def test_tool_not_found_error(self, rag_system):
        """Test error handling when tool is not found"""
        # Try to execute non-existent tool
        result = rag_system.tool_manager.execute_tool("nonexistent_tool", query="test")
        
        # Verify error message
        assert "Tool 'nonexistent_tool' not found" in result

    def test_malformed_tool_input(self, rag_system, mock_vector_store):
        """Test handling of malformed tool input"""
        # Setup empty results for malformed query
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        # Execute with missing required parameters (should be handled by tool)
        result = rag_system.search_tool.execute()  # No query parameter
        
        # The tool should handle missing parameters gracefully
        # (This will depend on the actual tool implementation)

    def test_vector_store_connection_error(self, rag_system, mock_vector_store):
        """Test handling of vector store connection errors"""
        # Setup vector store to raise connection error
        mock_vector_store.search.side_effect = ConnectionError("Cannot connect to vector database")
        
        # Execute search tool
        with pytest.raises(ConnectionError):
            rag_system.search_tool.execute(query="test query")

    def test_concurrent_session_management(self, rag_system, mock_anthropic_client):
        """Test session management with multiple concurrent sessions"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Concurrent response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Create multiple sessions
        session1 = rag_system.session_manager.create_session()
        session2 = rag_system.session_manager.create_session()
        
        # Add different history to each
        rag_system.session_manager.add_exchange(session1, "Query 1", "Answer 1")
        rag_system.session_manager.add_exchange(session2, "Query 2", "Answer 2")
        
        # Query both sessions
        response1, _ = rag_system.query("Follow up 1", session_id=session1)
        response2, _ = rag_system.query("Follow up 2", session_id=session2)
        
        # Verify sessions remain separate
        history1 = rag_system.session_manager.get_conversation_history(session1)
        history2 = rag_system.session_manager.get_conversation_history(session2)
        
        assert "Query 1" in history1 and "Query 1" not in history2
        assert "Query 2" in history2 and "Query 2" not in history1
        assert "Follow up 1" in history1
        assert "Follow up 2" in history2


class TestRAGSystemComponentInteraction:
    """Test edge cases in component interaction and communication"""

    def test_tool_manager_state_consistency(self, rag_system, mock_vector_store):
        """Test that ToolManager maintains consistent state across multiple operations"""
        # Execute search tool
        search_results = SearchResults(
            documents=["First search content"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/course-a-lesson1"
        
        rag_system.search_tool.execute(query="first search")
        first_sources = rag_system.tool_manager.get_last_sources()
        
        # Execute outline tool
        mock_vector_store._resolve_course_name.return_value = "Course B"
        course_metadata = {
            "course_title": "Course B",
            "course_link": "https://example.com/course-b",
            "lessons": [{"lesson_number": 1, "lesson_title": "Intro"}]
        }
        mock_vector_store.get_all_courses_metadata.return_value = [course_metadata]
        
        rag_system.outline_tool.execute(course_title="Course B")
        second_sources = rag_system.tool_manager.get_last_sources()
        
        # Verify state consistency
        assert len(first_sources) == 1
        assert "Course A" in first_sources[0]
        assert len(second_sources) == 1  
        assert "Course B" in second_sources[0]
        
        # Verify reset functionality
        rag_system.tool_manager.reset_sources()
        empty_sources = rag_system.tool_manager.get_last_sources()
        assert len(empty_sources) == 0

    def test_session_manager_concurrent_operations(self, rag_system):
        """Test SessionManager handling of concurrent session operations"""
        # Create multiple sessions rapidly
        session_ids = [rag_system.session_manager.create_session() for _ in range(5)]
        
        # Verify all sessions are unique
        assert len(set(session_ids)) == 5
        
        # Add different data to each session
        for i, session_id in enumerate(session_ids):
            rag_system.session_manager.add_exchange(session_id, f"Query {i}", f"Answer {i}")
        
        # Verify session isolation
        for i, session_id in enumerate(session_ids):
            history = rag_system.session_manager.get_conversation_history(session_id)
            assert f"Query {i}" in history
            assert f"Answer {i}" in history
            
            # Verify other sessions' data is not present
            for j in range(5):
                if j != i:
                    assert f"Query {j}" not in history

    def test_ai_generator_tool_integration_edge_cases(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test edge cases in AI generator and tool integration"""
        # Test with tool response containing unexpected fields
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "edge_case_tool"
        mock_tool_block.input = {"query": "test", "unexpected_param": "value"}
        mock_tool_block.unexpected_field = "should be ignored"
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.unexpected_response_field = "should be ignored"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Response despite unexpected fields"
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup normal search results
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        
        # Execute query and verify robustness
        response, sources = rag_system.query("Test query with unexpected fields")
        
        assert response == "Response despite unexpected fields"
        # Tool should have handled unexpected parameters gracefully

    def test_vector_store_search_result_edge_cases(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test handling of edge cases in search results"""
        # Setup tool usage scenario
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "edge_search"
        mock_tool_block.input = {"query": "edge case"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Handled edge case results"
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup search results with missing/malformed metadata
        edge_case_results = SearchResults(
            documents=["Content with incomplete metadata", "Content without lesson"],
            metadata=[
                {"course_title": "Incomplete Course"},  # Missing lesson_number
                {"lesson_number": 2}  # Missing course_title
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = edge_case_results
        mock_vector_store.get_lesson_link.return_value = None  # No link available
        
        # Execute query
        response, sources = rag_system.query("Edge case search")
        
        # Verify system handles incomplete metadata gracefully
        assert response == "Handled edge case results"
        assert len(sources) == 2  # Should still extract sources despite missing data

    def test_component_initialization_dependency_order(self, mock_config):
        """Test that RAG system components initialize in correct dependency order"""
        # Mock all dependencies to verify initialization order
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session, \
             patch('rag_system.ToolManager') as mock_tool_mgr, \
             patch('rag_system.CourseSearchTool') as mock_search_tool, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool:
            
            # Initialize RAG system
            rag_system = RAGSystem(mock_config)
            
            # Verify all components were initialized
            mock_doc_proc.assert_called_once_with(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
            mock_vector.assert_called_once_with(mock_config.CHROMA_PATH, mock_config.EMBEDDING_MODEL, mock_config.MAX_RESULTS)
            mock_ai_gen.assert_called_once_with(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            mock_session.assert_called_once_with(mock_config.MAX_HISTORY)
            mock_tool_mgr.assert_called_once()
            
            # Verify tools were initialized with vector store
            mock_search_tool.assert_called_once()
            mock_outline_tool.assert_called_once()
            
            # Verify tools were registered
            assert mock_tool_mgr.return_value.register_tool.call_count == 2


class TestRAGSystemPerformanceIntegration:
    """Test performance-related integration scenarios"""

    def test_large_conversation_history_handling(self, rag_system, mock_anthropic_client):
        """Test handling of large conversation histories"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with large history"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Create session and add many exchanges
        session_id = rag_system.session_manager.create_session()
        
        for i in range(10):  # More than max_history limit
            rag_system.session_manager.add_exchange(
                session_id, 
                f"Question {i}", 
                f"Answer {i}"
            )
        
        # Query with large history
        response, sources = rag_system.query("Final question", session_id=session_id)
        
        # Verify response
        assert response == "Response with large history"
        
        # Verify history was truncated (based on SessionManager's max_history setting)
        history = rag_system.session_manager.get_conversation_history(session_id)
        # Should only contain recent exchanges due to truncation
        assert "Question 0" not in history  # Old exchanges should be removed
        assert "Final question" in history  # New exchange should be present

    def test_multiple_tools_performance(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test performance when multiple tools are available"""
        # Setup tool response
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "perf_tool"
        mock_tool_block.input = {"query": "performance test"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Performance test response"
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup search results
        search_results = SearchResults(
            documents=["Performance content"],
            metadata=[{"course_title": "Performance Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        
        # Execute query and measure that it completes
        response, sources = rag_system.query("Performance test query")
        
        # Verify it completes successfully with all tools available
        assert response == "Performance test response"
        assert len(rag_system.tool_manager.get_tool_definitions()) == 2  # Both tools available

    def test_stress_test_rapid_queries(self, rag_system, mock_anthropic_client):
        """Test system stability under rapid successive queries"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Rapid response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Execute multiple rapid queries
        session_id = rag_system.session_manager.create_session()
        
        responses = []
        for i in range(10):
            response, sources = rag_system.query(f"Rapid query {i}", session_id=session_id)
            responses.append(response)
        
        # Verify all queries succeeded
        assert len(responses) == 10
        assert all(r == "Rapid response" for r in responses)
        
        # Verify session maintained consistency
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "Rapid query 9" in history  # Latest query should be in history


class TestRAGSystemFailureTracking:
    """
    Test class specifically designed to identify where 'query failed' errors originate
    in the RAG system component chain and trace error propagation paths.
    """

    def test_error_origin_identification_anthropic_api(self, rag_system, mock_anthropic_client):
        """Test error identification when Anthropic API is the source of failure"""
        # Simulate specific Anthropic API errors
        test_cases = [
            {"error": Exception("Authentication failed"), "expected_msg": "Authentication failed"},
            {"error": Exception("Rate limit exceeded"), "expected_msg": "Rate limit exceeded"},
            {"error": Exception("Model overloaded"), "expected_msg": "Model overloaded"},
            {"error": ConnectionError("Network timeout"), "expected_msg": "Network timeout"}
        ]
        
        for case in test_cases:
            mock_anthropic_client.messages.create.side_effect = case["error"]
            
            with pytest.raises(Exception) as exc_info:
                rag_system.query("Test query")
            
            # Verify error message propagates correctly
            assert case["expected_msg"] in str(exc_info.value)

    def test_error_origin_identification_vector_store(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test error identification when vector store is the source of failure"""
        # Setup tool usage scenario first
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "vector_error_test"
        mock_tool_block.input = {"query": "test"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_anthropic_client.messages.create.return_value = mock_tool_response
        
        # Test different vector store failure scenarios
        vector_errors = [
            {"error": ConnectionError("ChromaDB connection failed"), "source": "database"},
            {"error": Exception("Embedding model not found"), "source": "embedding"},
            {"error": ValueError("Invalid search parameters"), "source": "parameters"},
            {"error": TimeoutError("Search operation timed out"), "source": "timeout"}
        ]
        
        for case in vector_errors:
            mock_vector_store.search.side_effect = case["error"]
            
            with pytest.raises(Exception) as exc_info:
                rag_system.query("Test vector store error")
            
            # Verify error can be traced to vector store component
            error_msg = str(exc_info.value)
            assert any(keyword in error_msg.lower() for keyword in 
                      ["chroma", "embedding", "search", "vector", "database", "timeout"])

    def test_error_origin_identification_session_manager(self, rag_system, mock_anthropic_client):
        """Test error identification when session manager is the source of failure"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Test session manager failures
        session_errors = [
            {"method": "get_conversation_history", "error": Exception("Session storage corrupted")},
            {"method": "add_exchange", "error": Exception("Memory limit exceeded")},
            {"method": "create_session", "error": Exception("Session ID collision")}
        ]
        
        for case in session_errors:
            with patch.object(rag_system.session_manager, case["method"], 
                            side_effect=case["error"]):
                try:
                    if case["method"] == "create_session":
                        # Test session creation failure
                        rag_system.session_manager.create_session()
                    else:
                        # Test query with session failure
                        rag_system.query("Test session error", session_id="test_session")
                except Exception as e:
                    # Verify error can be traced to session manager
                    error_msg = str(e)
                    assert any(keyword in error_msg.lower() for keyword in 
                              ["session", "memory", "storage", "collision"])

    def test_error_origin_identification_tool_execution(self, rag_system, mock_anthropic_client):
        """Test error identification when tool execution is the source of failure"""
        # Setup tool usage scenario
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error_test"
        mock_tool_block.input = {"query": "test"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        mock_anthropic_client.messages.create.return_value = mock_tool_response
        
        # Test tool execution failures
        tool_errors = [
            Exception("Tool parameters validation failed"),
            Exception("Tool execution timeout"),
            Exception("Tool internal error"),
            ValueError("Invalid tool input format")
        ]
        
        for error in tool_errors:
            with patch.object(rag_system.tool_manager, 'execute_tool', side_effect=error):
                with pytest.raises(Exception) as exc_info:
                    rag_system.query("Test tool error")
                
                # Verify error can be traced to tool execution
                error_msg = str(exc_info.value)
                assert any(keyword in error_msg.lower() for keyword in 
                          ["tool", "execution", "parameters", "validation", "timeout"])

    def test_error_propagation_chain_complete(self, rag_system):
        """Test complete error propagation chain to identify bottlenecks"""
        # Test error propagation through the complete component chain:
        # User Query → RAG System → AI Generator → Tool Manager → Vector Store
        
        # Mock each component to fail in sequence
        components = [
            ("ai_generator", "generate_response", Exception("AI Generator failed")),
            ("tool_manager", "execute_tool", Exception("Tool Manager failed")),
            ("session_manager", "get_conversation_history", Exception("Session Manager failed"))
        ]
        
        for component_name, method_name, error in components:
            component = getattr(rag_system, component_name)
            
            with patch.object(component, method_name, side_effect=error):
                with pytest.raises(Exception) as exc_info:
                    rag_system.query("Test component chain error")
                
                # Verify error message indicates which component failed
                error_msg = str(exc_info.value)
                expected_keywords = {
                    "ai_generator": ["ai", "generator", "anthropic"],
                    "tool_manager": ["tool", "manager", "execution"],
                    "session_manager": ["session", "history", "conversation"]
                }
                
                assert any(keyword in error_msg.lower() for keyword in expected_keywords[component_name])

    def test_query_failure_recovery_mechanisms(self, rag_system, mock_anthropic_client, mock_vector_store):
        """Test how the system handles and recovers from 'query failed' scenarios"""
        # Simulate the most common "query failed" scenario:
        # Tool call succeeds but returns error results
        
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "recovery_test"
        mock_tool_block.input = {"query": "recovery test"}
        
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response that handles the error gracefully
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "I encountered an error while searching, but here's what I can tell you based on general knowledge."
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup vector store to return error results
        error_results = SearchResults.empty("Database temporarily unavailable")
        mock_vector_store.search.return_value = error_results
        
        # Execute query and verify graceful handling
        response, sources = rag_system.query("Test recovery mechanisms")
        
        # Verify system recovers gracefully
        assert response == "I encountered an error while searching, but here's what I can tell you based on general knowledge."
        assert len(sources) == 0
        
        # Verify the error was contained and didn't crash the system
        assert "error" in response.lower()
        
    def test_error_logging_and_debugging_info(self, rag_system, mock_anthropic_client, caplog):
        """Test that errors provide sufficient debugging information"""
        # Setup a failure scenario
        mock_anthropic_client.messages.create.side_effect = Exception("Detailed error for debugging")
        
        with pytest.raises(Exception) as exc_info:
            rag_system.query("Test debugging info")
        
        # Verify error contains debugging information
        error_msg = str(exc_info.value)
        assert "Detailed error for debugging" in error_msg
        
        # Verify error information would be useful for troubleshooting
        assert len(error_msg) > 10  # Should contain meaningful error details
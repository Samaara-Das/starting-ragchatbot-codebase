"""
Unit tests for AIGenerator to diagnose tool calling and API issues.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator
import anthropic


class TestAIGenerator:
    """Test cases for AIGenerator functionality"""
    
    def test_initialization_with_valid_config(self, mock_config):
        """Test AIGenerator initializes correctly with valid configuration"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Verify Anthropic client was created
            mock_anthropic.assert_called_once_with(api_key=mock_config.ANTHROPIC_API_KEY)
            
            # Verify properties set
            assert generator.model == mock_config.ANTHROPIC_MODEL
            assert generator.base_params["model"] == mock_config.ANTHROPIC_MODEL
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_initialization_with_empty_api_key(self):
        """Test AIGenerator behavior with empty API key"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("", "claude-3-5-sonnet-20241022")
            
            # Should still create client (Anthropic SDK will handle invalid key)
            mock_anthropic.assert_called_once_with(api_key="")
    
    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test basic response generation without tools"""
        # Setup
        mock_anthropic_client.messages.create.return_value.content[0].text = "Test response"
        mock_anthropic_client.messages.create.return_value.stop_reason = "end_turn"
        
        # Execute
        response = ai_generator.generate_response("What is RAG?")
        
        # Verify
        assert response == "Test response"
        
        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == ai_generator.model
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is RAG?"}]
        assert ai_generator.SYSTEM_PROMPT in call_args["system"]
        assert "tools" not in call_args
    
    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test response generation includes conversation history in system prompt"""
        # Setup
        mock_anthropic_client.messages.create.return_value.content[0].text = "Response with context"
        mock_anthropic_client.messages.create.return_value.stop_reason = "end_turn"
        
        history = "User: Previous question\\nAssistant: Previous answer"
        
        # Execute
        response = ai_generator.generate_response("Follow-up question", conversation_history=history)
        
        # Verify
        assert response == "Response with context"
        
        # Verify history included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert history in call_args["system"]
        assert "Previous conversation:" in call_args["system"]
    
    def test_generate_response_with_tools_but_no_tool_use(self, ai_generator, mock_anthropic_client, tool_manager):
        """Test response with tools provided but AI doesn't use them"""
        # Setup
        mock_anthropic_client.messages.create.return_value.content[0].text = "Direct response"
        mock_anthropic_client.messages.create.return_value.stop_reason = "end_turn"
        
        tool_definitions = tool_manager.get_tool_definitions()
        
        # Execute
        response = ai_generator.generate_response(
            "General knowledge question",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify
        assert response == "Direct response"
        
        # Verify tools were provided to API
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tool_definitions
        assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client, tool_manager, mock_anthropic_tool_response):
        """Test response generation with tool calling"""
        # Setup tool execution
        tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Setup initial response with tool use
        mock_anthropic_client.messages.create.side_effect = [
            mock_anthropic_tool_response,  # First call returns tool use
            Mock(content=[Mock(text="Final response with tool results")], stop_reason="end_turn")  # Second call returns final answer
        ]
        
        tool_definitions = tool_manager.get_tool_definitions()
        
        # Execute
        response = ai_generator.generate_response(
            "What is in the MCP course?",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify final response
        assert response == "Final response with tool results"
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test query")
        
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify second call includes tool results
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert len(second_call_args["messages"]) == 3  # user, assistant (tool use), user (tool results)
        
        # Check tool result message format
        tool_result_message = second_call_args["messages"][2]
        assert tool_result_message["role"] == "user"
        assert len(tool_result_message["content"]) == 1
        assert tool_result_message["content"][0]["type"] == "tool_result"
        assert tool_result_message["content"][0]["content"] == "Tool execution result"
    
    def test_api_key_authentication_error(self, ai_generator, mock_anthropic_client):
        """Test handling of authentication errors"""
        # Setup API to raise authentication error
        mock_anthropic_client.messages.create.side_effect = anthropic.AuthenticationError("Invalid API key")
        
        # Execute and verify exception propagates
        with pytest.raises(anthropic.AuthenticationError):
            ai_generator.generate_response("Test query")
    
    def test_api_rate_limit_error(self, ai_generator, mock_anthropic_client):
        """Test handling of rate limit errors"""
        # Setup API to raise rate limit error
        mock_anthropic_client.messages.create.side_effect = anthropic.RateLimitError("Rate limit exceeded")
        
        # Execute and verify exception propagates
        with pytest.raises(anthropic.RateLimitError):
            ai_generator.generate_response("Test query")
    
    def test_api_server_error(self, ai_generator, mock_anthropic_client):
        """Test handling of server errors"""
        # Setup API to raise server error
        mock_anthropic_client.messages.create.side_effect = anthropic.InternalServerError("Server error")
        
        # Execute and verify exception propagates
        with pytest.raises(anthropic.InternalServerError):
            ai_generator.generate_response("Test query")
    
    def test_tool_execution_error(self, ai_generator, mock_anthropic_client, tool_manager, mock_anthropic_tool_response):
        """Test handling when tool execution fails"""
        # Setup tool manager to raise exception
        tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Setup tool use response
        mock_anthropic_client.messages.create.side_effect = [
            mock_anthropic_tool_response,
            Mock(content=[Mock(text="Response despite tool error")], stop_reason="end_turn")
        ]
        
        tool_definitions = tool_manager.get_tool_definitions()
        
        # Execute - should not crash
        response = ai_generator.generate_response(
            "Test query",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify it handled the error gracefully
        assert response == "Response despite tool error"
        
        # Verify tool execution was attempted
        tool_manager.execute_tool.assert_called_once()
    
    def test_multiple_tool_calls_in_response(self, ai_generator, mock_anthropic_client, tool_manager):
        """Test handling multiple tool calls in single response"""
        # Setup multiple tool calls
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "search_course_content"
        mock_tool_block_1.id = "tool_1"
        mock_tool_block_1.input = {"query": "first query"}
        
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"  
        mock_tool_block_2.name = "get_course_outline"
        mock_tool_block_2.id = "tool_2"
        mock_tool_block_2.input = {"course_title": "test course"}
        
        mock_response = Mock()
        mock_response.content = [mock_tool_block_1, mock_tool_block_2]
        mock_response.stop_reason = "tool_use"
        
        # Setup tool manager responses
        tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        mock_anthropic_client.messages.create.side_effect = [
            mock_response,
            Mock(content=[Mock(text="Final response with both results")], stop_reason="end_turn")
        ]
        
        tool_definitions = tool_manager.get_tool_definitions()
        
        # Execute
        response = ai_generator.generate_response(
            "Complex query",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify
        assert response == "Final response with both results"
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="first query")
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="test course")
        
        # Verify tool results were included in second API call
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        tool_result_message = second_call_args["messages"][2]
        assert len(tool_result_message["content"]) == 2  # Two tool results
    
    def test_tool_use_without_tool_manager(self, ai_generator, mock_anthropic_client, tool_manager, mock_anthropic_tool_response):
        """Test handling when tools provided but no tool_manager"""
        # Setup
        mock_anthropic_client.messages.create.return_value = mock_anthropic_tool_response
        tool_definitions = tool_manager.get_tool_definitions()
        
        # Execute without tool_manager
        response = ai_generator.generate_response(
            "Test query",
            tools=tool_definitions,
            tool_manager=None  # No tool manager provided
        )
        
        # Should return None or handle gracefully
        # The current implementation would return None for tool_use without manager
        assert response is None or isinstance(response, str)
    
    def test_system_prompt_content(self, ai_generator):
        """Test system prompt contains expected instructions"""
        prompt = ai_generator.SYSTEM_PROMPT
        
        # Verify key instructions are present
        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt
        assert "Maximum one tool call per query" in prompt
        assert "Course content questions" in prompt
        assert "Course outline questions" in prompt
    
    def test_handle_tool_execution_with_mixed_content(self, ai_generator, mock_anthropic_client, tool_manager):
        """Test handling response with both text and tool use content"""
        # Setup mixed content response
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Let me search for that information."
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test query"}
        
        mixed_response = Mock()
        mixed_response.content = [text_block, tool_block]
        mixed_response.stop_reason = "tool_use"
        
        # Setup responses
        tool_manager.execute_tool.return_value = "Search results"
        mock_anthropic_client.messages.create.side_effect = [
            mixed_response,
            Mock(content=[Mock(text="Here's what I found")], stop_reason="end_turn")
        ]
        
        tool_definitions = tool_manager.get_tool_definitions()
        
        # Execute
        response = ai_generator.generate_response(
            "Test query",
            tools=tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify
        assert response == "Here's what I found"
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test query")
        
        # Verify assistant message includes both text and tool use
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        assistant_message = second_call_args["messages"][1]
        assert assistant_message["role"] == "assistant"
        assert len(assistant_message["content"]) == 2
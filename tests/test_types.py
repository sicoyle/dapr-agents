"""
Unit tests for core types used in dapr-agents.
"""
import pytest
from datetime import datetime
from typing import Dict, Any, List

from dapr_agents.types import (
    MessageContent,
    ChatCompletion,
    ToolMessage,
    AgentError,
    AgentStatus,
    MessagePlaceHolder,
    Choice,
    ToolCall,
    FunctionCall
)


class TestMessageContent:
    """Test cases for MessageContent type."""

    def test_message_content_initialization(self):
        """Test MessageContent initialization with basic fields."""
        message = MessageContent(
            role="user",
            content="Hello, how are you?"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        # name is not a field in MessageContent, it's in BaseMessage
        assert not hasattr(message, 'name')

    def test_message_content_with_name(self):
        """Test MessageContent initialization with name."""
        message = MessageContent(
            role="assistant",
            content="I'm doing well, thank you!",
            name="TestAssistant"
        )
        
        assert message.role == "assistant"
        assert message.content == "I'm doing well, thank you!"
        assert message.name == "TestAssistant"

    def test_message_content_with_tool_calls(self):
        """Test MessageContent initialization with tool calls."""
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="test_function",
                    arguments='{"arg": "value"}'
                )
            )
        ]
        
        message = MessageContent(
            role="assistant",
            content="I'll use a tool to help you",
            tool_calls=tool_calls
        )
        
        assert message.role == "assistant"
        assert message.content == "I'll use a tool to help you"
        assert message.tool_calls == tool_calls

    def test_message_content_validation(self):
        """Test MessageContent validation."""
        # Should not raise an error for valid data
        message = MessageContent(
            role="user",
            content="Valid message"
        )
        
        assert message.role == "user"
        assert message.content == "Valid message"

    def test_message_content_from_dict(self):
        """Test creating MessageContent from dictionary."""
        data = {
            "role": "user",
            "content": "Test message",
            "name": "TestUser"
        }
        
        message = MessageContent(**data)
        
        assert message.role == "user"
        assert message.content == "Test message"
        assert message.name == "TestUser"


class TestChatCompletion:
    """Test cases for ChatCompletion type."""

    def test_chat_completion_initialization(self):
        """Test ChatCompletion initialization."""
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=MessageContent(
                role="assistant",
                content="Test response"
            ),
            logprobs=None
        )
        
        completion = ChatCompletion(
            choices=[choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        
        assert len(completion.choices) == 1
        assert completion.choices[0].finish_reason == "stop"
        assert completion.choices[0].message.content == "Test response"

    def test_chat_completion_with_tool_calls(self):
        """Test ChatCompletion with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "New York"}'
            )
        )
        
        choice = Choice(
            finish_reason="tool_calls",
            index=0,
            message=MessageContent(
                role="assistant",
                tool_calls=[tool_call]
            ),
            logprobs=None
        )
        
        completion = ChatCompletion(
            choices=[choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        
        assert completion.get_tool_calls() == [tool_call]

    def test_chat_completion_model_dump(self):
        """Test ChatCompletion model_dump method."""
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=MessageContent(
                role="assistant",
                content="Test response"
            ),
            logprobs=None
        )
        
        completion = ChatCompletion(
            choices=[choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        
        data = completion.model_dump()
        assert "choices" in data
        assert "created" in data
        assert "model" in data
        assert "usage" in data


class TestToolMessage:
    """Test cases for ToolMessage type."""

    def test_tool_message_initialization(self):
        """Test ToolMessage initialization."""
        tool_message = ToolMessage(
            tool_call_id="call_1",
            content="Function result"
        )
        
        assert tool_message.tool_call_id == "call_1"
        assert tool_message.content == "Function result"
        assert tool_message.role == "tool"  # Default role

    def test_tool_message_validation(self):
        """Test ToolMessage validation."""
        # Should not raise an error for valid data
        tool_message = ToolMessage(
            tool_call_id="call_1",
            content="Valid result"
        )
        
        assert tool_message.tool_call_id == "call_1"
        assert tool_message.content == "Valid result"


class TestAgentError:
    """Test cases for AgentError type."""

    def test_agent_error_initialization(self):
        """Test AgentError initialization."""
        error = AgentError("Test error message")
        
        assert str(error) == "Test error message"

    def test_agent_error_inheritance(self):
        """Test that AgentError inherits from Exception."""
        error = AgentError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, AgentError)


class TestAgentStatus:
    """Test cases for AgentStatus enum."""

    def test_agent_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.ACTIVE == "active"
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.PAUSED == "paused"
        assert AgentStatus.COMPLETE == "complete"
        assert AgentStatus.ERROR == "error"

    def test_agent_status_enumeration(self):
        """Test that AgentStatus is a proper enum."""
        statuses = list(AgentStatus)
        
        assert len(statuses) == 5
        assert "active" in [s.value for s in statuses]
        assert "idle" in [s.value for s in statuses]
        assert "paused" in [s.value for s in statuses]
        assert "complete" in [s.value for s in statuses]
        assert "error" in [s.value for s in statuses]


class TestMessagePlaceHolder:
    """Test cases for MessagePlaceHolder type."""

    def test_message_placeholder_initialization(self):
        """Test MessagePlaceHolder initialization."""
        placeholder = MessagePlaceHolder(variable_name="chat_history")
        
        assert placeholder.variable_name == "chat_history"

    def test_message_placeholder_repr(self):
        """Test MessagePlaceHolder string representation."""
        placeholder = MessagePlaceHolder(variable_name="messages")
        expected_repr = "MessagePlaceHolder(variable_name='messages')"
        assert repr(placeholder) == expected_repr


class TestTypesIntegration:
    """Integration tests for types."""

    def test_message_content_in_chat_completion(self):
        """Test MessageContent integration with ChatCompletion."""
        message = MessageContent(
            role="assistant",
            content="This is a test response"
        )
        
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=message,
            logprobs=None
        )
        
        completion = ChatCompletion(
            choices=[choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        
        assert completion.get_content() == "This is a test response"

    def test_tool_message_in_agent_error(self):
        """Test using ToolMessage in AgentError context."""
        tool_message = ToolMessage(
            tool_call_id="call_1",
            content="Function failed"
        )
        
        error = AgentError("Tool execution failed")
        
        assert "Tool execution failed" in str(error)

    def test_agent_status_workflow(self):
        """Test AgentStatus in a workflow context."""
        # Simulate agent status transitions
        statuses = [AgentStatus.IDLE, AgentStatus.ACTIVE, AgentStatus.COMPLETE]
        
        assert statuses[0] == AgentStatus.IDLE
        assert statuses[1] == AgentStatus.ACTIVE
        assert statuses[2] == AgentStatus.COMPLETE

    def test_chat_completion_serialization(self):
        """Test ChatCompletion serialization."""
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=MessageContent(
                role="assistant",
                content="Serializable content"
            ),
            logprobs=None
        )
        
        completion = ChatCompletion(
            choices=[choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        
        # Test JSON serialization
        json_data = completion.model_dump_json()
        assert "choices" in json_data
        assert "Serializable content" in json_data 
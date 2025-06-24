"""
Pytest configuration and common fixtures for dapr-agents tests.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

from dapr_agents.llm import OpenAIChatClient
from dapr_agents.memory import ConversationListMemory
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import ChatCompletion, MessageContent, Choice, ToolCall, FunctionCall


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_client = Mock(spec=OpenAIChatClient)
    mock_client.generate = AsyncMock()
    mock_client.prompt_template = None
    return mock_client


@pytest.fixture
def mock_memory():
    """Mock memory for testing."""
    mock_mem = Mock(spec=ConversationListMemory)
    mock_mem.add_message = Mock()
    mock_mem.get_messages = Mock(return_value=[])
    mock_mem.clear = Mock()
    mock_mem.reset_memory = Mock()
    return mock_mem


@pytest.fixture
def sample_tool():
    """Sample tool for testing."""
    def sample_function(text: str) -> str:
        """Sample tool function."""
        return f"Processed: {text}"
    
    tool = AgentTool(
        name="sample_tool",
        description="A sample tool for testing",
        function=sample_function
    )
    return tool


@pytest.fixture
def sample_chat_completion():
    """Sample chat completion response."""
    return ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                index=0,
                message=MessageContent(
                    role="assistant",
                    content="This is a test response"
                ),
                finish_reason="stop",
                logprobs=None
            )
        ],
        model="gpt-3.5-turbo",
        created=1234567890,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )


@pytest.fixture
def sample_message():
    """Sample message for testing."""
    return MessageContent(
        role="user",
        content="Hello, how are you?"
    )


@pytest.fixture
def basic_agent_config():
    """Basic configuration for agent testing."""
    return {
        "name": "TestAgent",
        "role": "Test Assistant",
        "goal": "Help with testing",
        "instructions": ["Be helpful", "Be accurate"],
        "max_iterations": 5
    }


@pytest.fixture
def durable_agent_config():
    """Configuration for durable agent testing."""
    return {
        "name": "TestDurableAgent",
        "role": "Test Durable Assistant", 
        "goal": "Help with testing",
        "instructions": ["Be helpful", "Be accurate"],
        "message_bus_name": "test-pubsub",
        "state_store_name": "test-state",
        "agents_registry_store_name": "test-registry",
        "max_iterations": 5
    }


@pytest.fixture
def sample_tool_call():
    """Sample tool call for testing."""
    return ToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(
            name="sample_tool",
            arguments='{"text": "test input"}'
        )
    )


@pytest.fixture
def sample_choice():
    """Sample Choice object for testing."""
    return Choice(
        finish_reason="stop",
        index=0,
        message=MessageContent(
            role="assistant",
            content="Hello! How can I help you today?"
        ),
        logprobs=None
    ) 
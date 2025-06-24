"""
Basic tests for dapr-agents that should work without complex dependencies.
"""
import pytest
from unittest.mock import Mock, AsyncMock

from dapr_agents.types import AgentStatus, AgentError, MessageContent
from dapr_agents.agent.base import AgentBase


class SimpleTestAgent(AgentBase):
    """Simple test implementation of AgentBase."""
    
    async def run(self, input_data):
        """Simple run implementation."""
        return f"Processed: {input_data}"


class TestBasicFunctionality:
    """Basic functionality tests that should work."""

    def test_agent_status_enum(self):
        """Test that AgentStatus enum works correctly."""
        assert AgentStatus.ACTIVE == "active"
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.PAUSED == "paused"
        assert AgentStatus.COMPLETE == "complete"
        assert AgentStatus.ERROR == "error"

    def test_agent_error(self):
        """Test that AgentError works correctly."""
        error = AgentError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_message_content(self):
        """Test that MessageContent works correctly."""
        message = MessageContent(
            role="user",
            content="Hello, world!"
        )
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_simple_agent_initialization(self):
        """Test basic agent initialization."""
        agent = SimpleTestAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    def test_agent_with_role_as_name(self):
        """Test that name defaults to role when not provided."""
        agent = SimpleTestAgent(role="Weather Expert")
        assert agent.name == "Weather Expert"

    @pytest.mark.asyncio
    async def test_simple_agent_run(self):
        """Test simple agent run method."""
        agent = SimpleTestAgent(name="TestAgent")
        result = await agent.run("test input")
        assert result == "Processed: test input"

    def test_agent_system_prompt_construction(self):
        """Test system prompt construction."""
        agent = SimpleTestAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        prompt = agent.construct_system_prompt()
        
        # Should contain actual values
        assert "TestAgent" in prompt
        assert "Test Assistant" in prompt
        assert "Help with testing" in prompt

    def test_agent_message_construction(self):
        """Test message construction."""
        agent = SimpleTestAgent(name="TestAgent")
        
        messages = agent.construct_messages("Hello")
        
        # Should have at least one message
        assert len(messages) >= 1
        
        # Should have a user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assert len(user_messages) >= 1

    def test_agent_tool_executor(self):
        """Test that tool executor is available."""
        agent = SimpleTestAgent(name="TestAgent")
        
        # Tool executor should be available even without tools
        assert hasattr(agent, 'tool_executor')
        assert agent.tool_executor is not None

    def test_agent_text_formatter(self):
        """Test that text formatter is available."""
        agent = SimpleTestAgent(name="TestAgent")
        
        # Text formatter should be available
        assert hasattr(agent, 'text_formatter')
        assert agent.text_formatter is not None

    def test_agent_memory(self):
        """Test that memory is available."""
        agent = SimpleTestAgent(name="TestAgent")
        
        # Memory should be available
        assert hasattr(agent, 'memory')
        assert agent.memory is not None

    def test_agent_llm(self):
        """Test that LLM is available."""
        agent = SimpleTestAgent(name="TestAgent")
        
        # LLM should be available
        assert hasattr(agent, 'llm')
        assert agent.llm is not None

    def test_agent_max_iterations(self):
        """Test max iterations setting."""
        agent = SimpleTestAgent(
            name="TestAgent",
            max_iterations=10
        )
        
        assert agent.max_iterations == 10

    def test_agent_instructions(self):
        """Test instructions setting."""
        instructions = ["Be helpful", "Be accurate"]
        agent = SimpleTestAgent(
            name="TestAgent",
            instructions=instructions
        )
        
        assert agent.instructions == instructions

    def test_agent_system_prompt(self):
        """Test custom system prompt."""
        system_prompt = "You are a helpful test assistant."
        agent = SimpleTestAgent(
            name="TestAgent",
            system_prompt=system_prompt
        )
        
        assert agent.system_prompt == system_prompt 
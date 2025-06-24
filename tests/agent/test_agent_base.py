"""
Unit tests for AgentBase class.
"""
import pytest
from unittest.mock import Mock, patch
from typing import List

from dapr_agents.agent.base import AgentBase
from dapr_agents.types import MessageContent
from dapr_agents.memory import ConversationListMemory


class MockAgent(AgentBase):
    """Mock agent for testing AgentBase functionality."""
    
    def run(self, input_data):
        return "test response"


class TestAgentBase:
    """Test cases for AgentBase class."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    def test_agent_initialization_with_instructions(self):
        """Test agent initialization with instructions."""
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            instructions=["Be helpful", "Be accurate"]
        )
        
        assert agent.instructions == ["Be helpful", "Be accurate"]

    def test_agent_name_defaults_to_role(self):
        """Test that agent name defaults to role if not provided."""
        agent = MockAgent(role="Test Assistant")
        
        assert agent.name == "Test Assistant"

    def test_agent_default_values(self):
        """Test agent default values."""
        agent = MockAgent()
        
        assert agent.role == "Assistant"
        assert agent.goal == "Help humans"
        assert agent.max_iterations == 10
        assert agent.template_format == "jinja2"

    def test_construct_system_prompt(self):
        """Test system prompt construction."""
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            instructions=["Be helpful", "Be accurate"]
        )
        
        prompt = agent.construct_system_prompt()
        
        # The prompt should contain template variables, not actual values
        assert "{{name}}" in prompt
        assert "{{role}}" in prompt
        assert "{{goal}}" in prompt
        assert "{{instructions}}" in prompt

    def test_construct_messages_with_string_input(self):
        """Test message construction with string input."""
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        messages = agent.construct_messages("Hello")
        
        # Should have system message and user message
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_construct_messages_with_dict_input(self):
        """Test message construction with dictionary input."""
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        # Provide valid input variables that match the prompt template
        input_data = {
            "name": "TestAgent",
            "role": "Test Assistant", 
            "goal": "Help with testing"
        }
        
        messages = agent.construct_messages(input_data)
        
        assert len(messages) >= 1
        # Should return formatted messages without adding a user message
        assert all(isinstance(msg, dict) for msg in messages)

    def test_construct_messages_with_chat_history(self, mock_memory):
        """Test message construction with chat history."""
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            memory=mock_memory
        )
        
        # Add some messages to memory
        mock_memory.add_message(MessageContent(role="user", content="Previous message"))
        mock_memory.add_message(MessageContent(role="assistant", content="Previous response"))
        
        messages = agent.construct_messages("New message")
        
        # Should include chat history
        assert len(messages) >= 3  # system + chat history + new user message

    def test_reset_memory(self, mock_memory):
        """Test memory reset functionality."""
        agent = MockAgent(memory=mock_memory)
        
        # Add some messages
        mock_memory.add_message(MessageContent(role="user", content="Test"))
        
        # Reset memory
        agent.reset_memory()
        
        # Verify reset was called
        mock_memory.reset_memory.assert_called_once()

    def test_get_last_message(self, mock_memory):
        """Test getting last message from memory."""
        agent = MockAgent(memory=mock_memory)
        
        last_message = MessageContent(role="user", content="Last message")
        mock_memory.get_messages.return_value = [last_message]
        
        result = agent.get_last_message()
        
        assert result == last_message

    def test_get_last_user_message(self):
        """Test getting last user message from message list."""
        agent = MockAgent()
        
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "User message 2"}
        ]
        
        result = agent.get_last_user_message(messages)
        
        assert result["role"] == "user"
        assert result["content"] == "User message 2"

    def test_agent_with_custom_prompt_template(self):
        """Test agent with custom prompt template."""
        from dapr_agents.prompt.chat import ChatPromptTemplate
        
        custom_template = ChatPromptTemplate.from_messages([
            ("system", "Custom system prompt"),
            ("user", "{{user_input}}")
        ])
        
        agent = MockAgent(
            name="TestAgent",
            role="Test Assistant",
            prompt_template=custom_template
        )
        
        assert agent.prompt_template == custom_template

    def test_agent_conflicting_prompt_templates(self):
        """Test agent with conflicting prompt templates."""
        from dapr_agents.prompt.chat import ChatPromptTemplate
        from dapr_agents.llm.openai import OpenAIChatClient
        
        custom_template = ChatPromptTemplate.from_messages([
            ("system", "Custom system prompt")
        ])
        
        llm_with_template = OpenAIChatClient()
        llm_with_template.prompt_template = custom_template
        
        # This should raise an error due to conflicting templates
        with pytest.raises(ValueError, match="Conflicting prompt templates"):
            MockAgent(
                name="TestAgent",
                role="Test Assistant",
                prompt_template=custom_template,
                llm=llm_with_template
            ) 
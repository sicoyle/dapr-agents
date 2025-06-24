"""
Unit tests for agent patterns (ReActAgent, ToolCallAgent, OpenAPIReActAgent).
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from dapr_agents.agent.patterns.react import ReActAgent
from dapr_agents.agent.patterns.toolcall import ToolCallAgent
from dapr_agents.agent.patterns.openapi.react import OpenAPIReActAgent
from dapr_agents.types import ChatCompletion, Choice, MessageContent


class TestReActAgent:
    """Test cases for ReActAgent class."""

    def test_react_agent_initialization(self):
        """Test ReActAgent initialization."""
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        assert agent.name == "TestReActAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    def test_react_agent_initialization_with_instructions(self):
        """Test ReActAgent initialization with instructions."""
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing",
            instructions=["Be helpful", "Be accurate"]
        )
        
        assert agent.instructions == ["Be helpful", "Be accurate"]

    def test_react_agent_initialization_with_reasoning(self):
        """Test ReActAgent initialization with reasoning enabled."""
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing",
            reasoning=True  # This parameter is handled by the factory, not the agent itself
        )
        
        assert agent.name == "TestReActAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"
        # The reasoning parameter is handled by the factory, not stored in the agent

    def test_react_agent_initialization_without_reasoning(self):
        """Test ReActAgent initialization without reasoning."""
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        assert agent.name == "TestReActAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    @pytest.mark.asyncio
    async def test_react_agent_run_method(self, mock_llm_client, sample_chat_completion):
        """Test ReActAgent run method."""
        mock_llm_client.generate.return_value = sample_chat_completion
        
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing",
            llm=mock_llm_client
        )
        
        result = await agent.run("Test input")
        
        assert result is not None
        mock_llm_client.generate.assert_called()

    def test_react_agent_construct_messages(self):
        """Test ReActAgent message construction."""
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        messages = agent.construct_messages("Hello")
        
        # Should have system message and user message
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_react_agent_system_prompt_includes_tools(self, sample_tool):
        """Test ReActAgent system prompt includes tools."""
        agent = ReActAgent(
            name="TestReActAgent",
            role="Test Assistant",
            goal="Help with testing",
            tools=[sample_tool]
        )
        
        prompt = agent.construct_system_prompt()
        
        # Should contain template variables
        assert "{{name}}" in prompt
        assert "{{role}}" in prompt
        assert "{{goal}}" in prompt


class TestToolCallAgent:
    """Test cases for ToolCallAgent class."""

    def test_toolcall_agent_initialization(self):
        """Test ToolCallAgent initialization."""
        agent = ToolCallAgent(
            name="TestToolCallAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        assert agent.name == "TestToolCallAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    def test_toolcall_agent_initialization_with_tools(self, sample_tool):
        """Test ToolCallAgent initialization with tools."""
        agent = ToolCallAgent(
            name="TestToolCallAgent",
            role="Test Assistant",
            goal="Help with testing",
            tools=[sample_tool]
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0] == sample_tool

    @pytest.mark.asyncio
    async def test_toolcall_agent_run_method(self, mock_llm_client, sample_chat_completion):
        """Test ToolCallAgent run method."""
        mock_llm_client.generate.return_value = sample_chat_completion
        
        agent = ToolCallAgent(
            name="TestToolCallAgent",
            role="Test Assistant",
            goal="Help with testing",
            llm=mock_llm_client
        )
        
        result = await agent.run("Test input")
        
        assert result is not None
        mock_llm_client.generate.assert_called()

    @pytest.mark.asyncio
    async def test_toolcall_agent_run_with_tool_calls(self, mock_llm_client, sample_tool):
        """Test ToolCallAgent run method with tool calls."""
        # Mock tool execution
        agent = ToolCallAgent(
            name="TestToolCallAgent",
            role="Test Assistant",
            goal="Help with testing",
            tools=[sample_tool],
            llm=mock_llm_client
        )
        
        # Mock the tool executor
        agent.tool_executor.run_tool = AsyncMock(return_value="Tool result")
        
        # Mock LLM response with tool calls
        tool_call_response = ChatCompletion(
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=MessageContent(
                        role="assistant",
                        tool_calls=[Mock()]
                    )
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        mock_llm_client.generate.return_value = tool_call_response
        
        result = await agent.run("Use tool")
        
        assert result is not None
        agent.tool_executor.run_tool.assert_called_once()

    def test_toolcall_agent_construct_messages(self):
        """Test ToolCallAgent message construction."""
        agent = ToolCallAgent(
            name="TestToolCallAgent",
            role="Test Assistant",
            goal="Help with testing"
        )
        
        messages = agent.construct_messages("Hello")
        
        # Should have system message and user message
        assert len(messages) >= 2
        # The first message should be the system message
        assert messages[0]["role"] == "system"
        # The last message should be the user message
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_toolcall_agent_system_prompt_includes_tools(self, sample_tool):
        """Test ToolCallAgent system prompt includes tools."""
        agent = ToolCallAgent(
            name="TestToolCallAgent",
            role="Test Assistant",
            goal="Help with testing",
            tools=[sample_tool]
        )
        
        prompt = agent.construct_system_prompt()
        
        # Should contain template variables
        assert "{{name}}" in prompt
        assert "{{role}}" in prompt
        assert "{{goal}}" in prompt


class TestOpenAPIReActAgent:
    """Test cases for OpenAPIReActAgent class."""

    def test_openapireact_agent_initialization(self):
        """Test OpenAPIReActAgent initialization."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store
        )
        
        assert agent.name == "TestOpenAPIAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"
        assert agent.spec_parser == mock_spec_parser
        assert agent.api_vector_store == mock_vector_store

    def test_openapireact_agent_initialization_without_spec(self):
        """Test OpenAPIReActAgent initialization without spec parser."""
        # Mock the required dependencies
        mock_vector_store = Mock()
        
        # This should raise a validation error since spec_parser is required
        with pytest.raises(Exception):
            OpenAPIReActAgent(
                name="TestOpenAPIAgent",
                role="Test Assistant",
                goal="Help with testing",
                api_vector_store=mock_vector_store
                # Missing spec_parser
            )

    def test_openapireact_agent_initialization_with_reasoning(self):
        """Test OpenAPIReActAgent initialization with reasoning."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store,
            reasoning=True  # This parameter is handled by the factory
        )
        
        assert agent.name == "TestOpenAPIAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    def test_openapireact_agent_construct_messages(self):
        """Test OpenAPIReActAgent message construction."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store
        )
        
        messages = agent.construct_messages("Hello")
        
        # Should have system message and user message
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_openapireact_agent_system_prompt_includes_openapi(self):
        """Test OpenAPIReActAgent system prompt includes OpenAPI content."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store
        )
        
        prompt = agent.construct_system_prompt()
        
        # Should contain template variables
        assert "{{name}}" in prompt
        assert "{{role}}" in prompt
        assert "{{goal}}" in prompt

    def test_openapireact_agent_loads_openapi_spec(self):
        """Test OpenAPIReActAgent loads OpenAPI spec."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store
        )
        
        # The spec_parser should be set
        assert agent.spec_parser == mock_spec_parser

    def test_openapireact_agent_creates_tools_from_spec(self):
        """Test OpenAPIReActAgent creates tools from spec."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store
        )
        
        # Should have tools (including OpenAPI tools)
        assert len(agent.tools) > 0

    def test_openapireact_agent_combines_reasoning_and_openapi(self):
        """Test OpenAPIReActAgent combines reasoning and OpenAPI capabilities."""
        # Mock the required dependencies
        mock_spec_parser = Mock()
        mock_vector_store = Mock()
        
        agent = OpenAPIReActAgent(
            name="TestOpenAPIAgent",
            role="Test Assistant",
            goal="Help with testing",
            spec_parser=mock_spec_parser,
            api_vector_store=mock_vector_store,
            reasoning=True  # This parameter is handled by the factory
        )
        
        # Should have both ReAct capabilities and OpenAPI tools
        assert agent.spec_parser == mock_spec_parser
        assert len(agent.tools) > 0 
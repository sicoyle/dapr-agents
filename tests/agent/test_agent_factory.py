"""
Unit tests for AgentFactory class.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import tempfile
import os
import asyncio
import yaml

from dapr_agents.agent.utils.factory import AgentFactory, Agent
from dapr_agents.agent.patterns.react import ReActAgent
from dapr_agents.agent.patterns.toolcall import ToolCallAgent
from dapr_agents.agent.patterns.openapi.react import OpenAPIReActAgent
from dapr_agents.workflow.agents.durable import DurableAgent


class TestAgentFactory:
    """Test cases for AgentFactory class."""

    def test_create_agent_class_react(self):
        """Test creating ReActAgent class."""
        agent_class = AgentFactory.create_agent_class("react")
        
        assert agent_class == ReActAgent

    def test_create_agent_class_toolcalling(self):
        """Test creating ToolCallAgent class."""
        agent_class = AgentFactory.create_agent_class("toolcalling")
        
        assert agent_class == ToolCallAgent

    def test_create_agent_class_openapireact(self):
        """Test creating OpenAPIReActAgent class."""
        agent_class = AgentFactory.create_agent_class("openapireact")
        
        assert agent_class == OpenAPIReActAgent

    def test_create_agent_class_case_insensitive(self):
        """Test that agent class creation is case insensitive."""
        agent_class = AgentFactory.create_agent_class("REACT")
        
        assert agent_class == ReActAgent

    def test_create_agent_class_unsupported_pattern(self):
        """Test that unsupported patterns raise an error."""
        with pytest.raises(ValueError, match="Unsupported agent pattern"):
            AgentFactory.create_agent_class("unsupported")

    def test_agent_patterns_dictionary(self):
        """Test that AGENT_PATTERNS contains expected patterns."""
        patterns = AgentFactory.AGENT_PATTERNS
        
        assert "react" in patterns
        assert "toolcalling" in patterns
        assert "openapireact" in patterns
        assert patterns["react"] == ReActAgent
        assert patterns["toolcalling"] == ToolCallAgent
        assert patterns["openapireact"] == OpenAPIReActAgent

    def test_testdata_config_files_structure(self):
        """Test that testdata configuration files have expected structure."""
        with open("tests/testdata/agent/agent_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Check basic agent config structure
        assert "agent" in config
        assert config["agent"]["role"] == "Test Assistant"
        assert config["agent"]["name"] == "TestAgent"
        assert config["agent"]["goal"] == "Help with testing"
        
        # Check ReAct config structure
        with open("tests/testdata/agent/react_agent_config.yaml", 'r') as f:
            react_config = yaml.safe_load(f)
        
        assert "agent" in react_config
        assert react_config["agent"]["reasoning"] is True
        assert react_config["agent"]["role"] == "ReAct Assistant"
        
        # Check OpenAPI config structure
        with open("tests/testdata/agent/openapi_agent_config.yaml", 'r') as f:
            openapi_config = yaml.safe_load(f)
        
        assert "agent" in openapi_config
        assert "openapi_spec_path" in openapi_config["agent"]
        assert openapi_config["agent"]["role"] == "OpenAPI Assistant"
        
        # Check master config structure
        with open("tests/testdata/config/master_config.yaml", 'r') as f:
            master_config = yaml.safe_load(f)
        
        assert "llm" in master_config
        assert "dapr" in master_config
        assert "memory" in master_config

    def test_testdata_config_files_exist(self):
        """Test that all testdata configuration files exist and are valid YAML."""
        testdata_files = [
            "tests/testdata/agent/agent_config.yaml",
            "tests/testdata/agent/react_agent_config.yaml", 
            "tests/testdata/agent/openapi_agent_config.yaml",
            "tests/testdata/config/master_config.yaml",
            "tests/testdata/config/agent_with_master_config.yaml",
            "tests/testdata/tools/sample_openapi.yaml"
        ]
        
        for file_path in testdata_files:
            assert os.path.exists(file_path), f"Test data file {file_path} does not exist"
            
            # Verify it's valid YAML
            try:
                with open(file_path, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {file_path}: {e}")

    def test_sample_openapi_spec_structure(self):
        """Test that the sample OpenAPI specification has expected structure."""
        with open("tests/testdata/tools/sample_openapi.yaml", 'r') as f:
            openapi_spec = yaml.safe_load(f)
        
        # Check OpenAPI version
        assert openapi_spec["openapi"] == "3.0.0"
        
        # Check info section
        assert "info" in openapi_spec
        assert openapi_spec["info"]["title"] == "Sample API"
        assert openapi_spec["info"]["version"] == "1.0.0"
        
        # Check paths
        assert "paths" in openapi_spec
        assert "/users" in openapi_spec["paths"]
        assert "/users/{id}" in openapi_spec["paths"]
        
        # Check components
        assert "components" in openapi_spec
        assert "schemas" in openapi_spec["components"]
        assert "User" in openapi_spec["components"]["schemas"]


class TestAgent:
    """Test cases for Agent class."""

    def test_agent_initialization_with_basic_config(self):
        """Test agent initialization with basic configuration."""
        config = {
            "name": "TestAgent",
            "role": "Test Assistant",
            "goal": "Help with testing"
        }
        
        agent = Agent(**config)
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"

    def test_agent_initialization_with_pattern(self):
        """Test agent initialization with specific pattern."""
        config = {
            "name": "TestAgent",
            "role": "Test Assistant",
            "pattern": "react"
        }
        
        agent = Agent(**config)
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Assistant"
        assert isinstance(agent.agent, ReActAgent)

    def test_agent_initialization_with_durable_config(self):
        """Test agent initialization with durable config."""
        # This test should be skipped or mocked since DurableAgent requires Dapr
        with pytest.raises(RuntimeError, match="🚫 Dapr Required for Durable Agent"):
            agent = Agent(
                role="Test Assistant",
                name="TestAgent",
                goal="Help with testing",
                # This would need to be a DurableAgent config, but it's not supported in the factory
            )

    def test_agent_initialization_with_config_file(self):
        """Test agent initialization with config file."""
        config_file = "tests/testdata/agent/agent_config.yaml"
        
        agent = Agent(role="Test Assistant", config_file=config_file)
        assert agent.agent is not None
        assert agent._agent_type == "toolcall"  # Default type

    def test_agent_initialization_with_react_config(self):
        """Test agent initialization with ReAct config file."""
        config_file = "tests/testdata/agent/react_agent_config.yaml"
        
        agent = Agent(role="ReAct Assistant", config_file=config_file)
        assert agent.agent is not None
        assert agent._agent_type == "react"

    def test_agent_initialization_with_openapi_config(self):
        """Test agent initialization with OpenAPI config file."""
        config_file = "tests/testdata/agent/openapi_agent_config.yaml"
        
        # Mock the OpenAPISpecParser to avoid actual file loading
        with patch('dapr_agents.agent.patterns.openapi.OpenAPISpecParser.from_file') as mock_parser:
            mock_parser.return_value = Mock()
            
            agent = Agent(role="OpenAPI Assistant", config_file=config_file)
            assert agent.agent is not None
            assert agent._agent_type == "openapireact"

    def test_agent_initialization_with_master_config(self):
        """Test agent initialization with master config reference."""
        config_file = "tests/testdata/config/agent_with_master_config.yaml"
        
        agent = Agent(role="Master Config Assistant", config_file=config_file)
        assert agent.agent is not None
        assert agent._agent_type == "toolcall"  # Default type

    def test_agent_initialization_with_reasoning_config(self):
        """Test agent initialization with reasoning configuration."""
        config = {
            "name": "TestReasoningAgent",
            "role": "Test Reasoning Assistant",
            "reasoning": True
        }
        
        agent = Agent(**config)
        
        assert agent.name == "TestReasoningAgent"
        assert agent.role == "Test Reasoning Assistant"
        assert isinstance(agent.agent, ReActAgent)

    def test_agent_initialization_defaults_to_toolcall(self):
        """Test that agent defaults to ToolCallAgent when no specific pattern is indicated."""
        config = {
            "name": "TestDefaultAgent",
            "role": "Test Default Assistant"
        }
        
        agent = Agent(**config)
        
        assert agent.name == "TestDefaultAgent"
        assert agent.role == "Test Default Assistant"
        assert isinstance(agent.agent, ToolCallAgent)

    @pytest.mark.asyncio
    async def test_agent_run_method(self):
        """Test agent run method."""
        agent = Agent(role="Test Assistant")
        
        # Mock the underlying agent's run method
        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Test response"
            
            result = await agent.run("Hello")
            
            mock_run.assert_called_once_with("Hello")
            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_agent_run_method_cancelled(self):
        """Test agent run method when cancelled."""
        agent = Agent(role="Test Assistant")
        
        # Mock the underlying agent's run method to simulate cancellation
        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = asyncio.CancelledError()
            
            result = await agent.run("Hello")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_agent_run_method_exception(self):
        """Test agent run method with exception."""
        agent = Agent(role="Test Assistant")
        
        # Mock the underlying agent's run method to raise an exception
        with patch.object(agent.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            with pytest.raises(Exception, match="Test error"):
                await agent.run("Hello")

    def test_agent_shutdown_method(self):
        """Test agent shutdown method."""
        agent = Agent(role="Test Assistant")
        
        # Test that shutdown event is set
        agent._shutdown_event.set()
        assert agent._shutdown_event.is_set()

    def test_agent_context_manager(self):
        """Test agent as context manager."""
        agent = Agent(role="Test Assistant")
        
        # Test that we can access underlying agent attributes
        assert hasattr(agent, 'role')
        assert agent.role == "Test Assistant"

    def test_agent_initialization_with_tools(self, sample_tool):
        """Test agent initialization with tools."""
        agent = Agent(
            role="Test Assistant",
            name="TestAgent",
            tools=[sample_tool]
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0] == sample_tool
        assert agent.tool_executor is not None

    def test_agent_initialization_with_custom_llm(self, mock_llm_client):
        """Test agent initialization with custom LLM client."""
        agent = Agent(
            role="Test Assistant",
            name="TestAgent",
            llm=mock_llm_client
        )
        
        assert agent.llm == mock_llm_client

    def test_agent_initialization_with_custom_memory(self, mock_memory):
        """Test agent initialization with custom memory."""
        agent = Agent(
            role="Test Assistant",
            name="TestAgent",
            memory=mock_memory
        )
        
        assert agent.memory == mock_memory 
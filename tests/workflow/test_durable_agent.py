"""
Unit tests for DurableAgent class.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from dapr_agents.workflow.agents.durable.agent import DurableAgent
from dapr_agents.types import ChatCompletion, Choice, MessageContent


class TestDurableAgent:
    """Test cases for DurableAgent class."""

    def test_durable_agent_initialization(self):
        """Test DurableAgent initialization."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            assert agent.name == "TestDurableAgent"
            assert agent.role == "Test Assistant"
            assert agent.goal == "Help with testing"

    def test_durable_agent_initialization_with_config_file(self):
        """Test DurableAgent initialization with config file."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            # This should work with a valid config file
            with patch('dapr_agents.config.Config') as mock_config:
                mock_config_instance = Mock()
                mock_config_instance.load_config_with_global.return_value = {
                    "dapr": {
                        "message_bus_name": "config-pubsub",
                        "state_store_name": "config-state",
                        "agents_registry_store_name": "config-registry"
                    }
                }
                mock_config.return_value = mock_config_instance
                
                agent = DurableAgent(
                    name="TestDurableAgent",
                    role="Test Assistant",
                    goal="Help with testing",
                    config_file="tests/testdata/config/master_config.yaml"
                )
                
                assert agent.name == "TestDurableAgent"
                assert agent.role == "Test Assistant"
                assert agent.goal == "Help with testing"

    def test_durable_agent_tool_choice_auto_with_tools(self, sample_tool):
        """Test DurableAgent tool choice with tools."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                tools=[sample_tool],
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Should default to "auto" when tools are provided
            assert agent.tool_choice == "auto"

    def test_durable_agent_tool_choice_none_without_tools(self):
        """Test DurableAgent tool choice without tools."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Should default to None when no tools are provided
            assert agent.tool_choice is None

    def test_durable_agent_tool_choice_custom(self, sample_tool):
        """Test DurableAgent with custom tool choice."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                tools=[sample_tool],
                tool_choice="required",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            assert agent.tool_choice == "required"

    def test_get_response_message(self, mock_llm_client, sample_chat_completion):
        """Test getting response message from chat completion."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                llm=mock_llm_client,
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            message = agent.get_response_message(sample_chat_completion.model_dump())
            assert message is not None

    def test_get_finish_reason(self, mock_llm_client, sample_chat_completion):
        """Test getting finish reason from chat completion."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                llm=mock_llm_client,
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            reason = agent.get_finish_reason(sample_chat_completion.model_dump())
            assert reason == "stop"

    def test_get_tool_calls(self, mock_llm_client):
        """Test getting tool calls from chat completion."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                llm=mock_llm_client,
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Create a chat completion with tool calls
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
            
            tool_calls = agent.get_tool_calls(tool_call_response.model_dump())
            assert tool_calls is not None

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, sample_tool):
        """Test successful tool execution."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                tools=[sample_tool],
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Mock tool execution
            agent.tool_executor.run_tool = AsyncMock(return_value="Tool result")
            
            tool_call = {
                "id": "call_1",
                "function": {
                    "name": "sample_tool",
                    "arguments": '{"text": "test"}'
                }
            }
            
            await agent.execute_tool("test-instance", tool_call)
            
            # Verify tool was executed
            agent.tool_executor.run_tool.assert_called_once_with("sample_tool", text="test")

    @pytest.mark.asyncio
    async def test_execute_tool_missing_function_name(self):
        """Test tool execution with missing function name."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            tool_call = {
                "id": "call_1",
                "function": {
                    "arguments": '{"text": "test"}'
                }
            }
            
            with pytest.raises(Exception):
                await agent.execute_tool("test-instance", tool_call)

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_json(self):
        """Test tool execution with invalid JSON arguments."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            tool_call = {
                "id": "call_1",
                "function": {
                    "name": "test_function",
                    "arguments": "invalid json"
                }
            }
            
            with pytest.raises(Exception):
                await agent.execute_tool("test-instance", tool_call)

    @pytest.mark.asyncio
    async def test_execute_tool_function_not_found(self):
        """Test tool execution with non-existent function."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            tool_call = {
                "id": "call_1",
                "function": {
                    "name": "nonexistent",
                    "arguments": '{"text": "test"}'
                }
            }
            
            with pytest.raises(Exception):
                await agent.execute_tool("test-instance", tool_call)

    @pytest.mark.asyncio
    async def test_broadcast_message_to_agents(self):
        """Test broadcasting message to agents."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            message = {"content": "test message"}
            await agent.broadcast_message_to_agents(message)
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_send_response_back(self):
        """Test sending response back."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            response = {"content": "test response"}
            await agent.send_response_back(response, "target_agent", "target_instance")
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_finish_workflow(self):
        """Test finishing workflow."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            message = {"content": "final result"}
            await agent.finish_workflow("test-instance", message)
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_update_workflow_state_with_message(self):
        """Test updating workflow state with message."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            message = {"role": "user", "content": "test message"}
            await agent.update_workflow_state("test-instance", message=message)
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_update_workflow_state_with_tool_message(self):
        """Test updating workflow state with tool message."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            tool_message = {"role": "tool", "content": "tool result"}
            await agent.update_workflow_state("test-instance", tool_message=tool_message)
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_update_workflow_state_with_final_output(self):
        """Test updating workflow state with final output."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            final_output = "This is the final answer"
            await agent.update_workflow_state("test-instance", final_output=final_output)
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_update_workflow_state_invalid_instance(self):
        """Test updating workflow state with invalid instance."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Initialize state with no instances
            agent.state.instances = {}
            
            with pytest.raises(ValueError, match="No workflow entry found"):
                await agent.update_workflow_state("invalid-instance", message={"content": "test"})

    @pytest.mark.asyncio
    async def test_process_broadcast_message_from_self(self):
        """Test processing broadcast message from self."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            message = {"sender": "TestDurableAgent", "content": "test"}
            await agent.process_broadcast_message(message)
            
            # This is a task method, so we just verify it doesn't raise an error

    @pytest.mark.asyncio
    async def test_process_broadcast_message_from_other(self):
        """Test processing broadcast message from other agent."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            message = {"sender": "OtherAgent", "content": "test"}
            await agent.process_broadcast_message(message)
            
            # This is a task method, so we just verify it doesn't raise an error

    def test_durable_agent_state_initialization(self):
        """Test DurableAgent state initialization."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Check that state is initialized
            assert hasattr(agent, 'state')
            assert agent.state is not None

    def test_durable_agent_workflow_name(self):
        """Test DurableAgent workflow name."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            # Check that workflow name is set
            assert hasattr(agent, '_workflow_name')
            assert agent._workflow_name == "ToolCallingWorkflow"

    @pytest.mark.asyncio
    async def test_generate_response(self, mock_llm_client, sample_chat_completion):
        """Test generating response."""
        # Mock Dapr dependencies
        with patch('dapr_agents.workflow.agents.durable.agent.DaprWorkflowContext') as mock_dapr_context:
            mock_dapr_context.return_value = Mock()
            
            agent = DurableAgent(
                name="TestDurableAgent",
                role="Test Assistant",
                goal="Help with testing",
                llm=mock_llm_client,
                message_bus_name="test-pubsub",
                state_store_name="test-state",
                agents_registry_store_name="test-registry"
            )
            
            mock_llm_client.generate.return_value = sample_chat_completion
            
            response = await agent.generate_response("test-instance", "test input")
            
            assert response is not None
            mock_llm_client.generate.assert_called_once() 
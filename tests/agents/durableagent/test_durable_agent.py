# TODO(@Sicoyle): this test file is a bit of a mess, and needs to be refactored when we clean the remaining classes up.
# Right now we have to do a bunch of patching at the class-level instead of patching at the instance-level.
# In future, we should do dependency injection instead of patching at the class-level to make it easier to test.
# This applies to all areas in this file where we have with patch.object()...
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from dapr.ext.workflow import DaprWorkflowContext

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentStateConfig,
    AgentRegistryConfig,
    AgentMemoryConfig,
    AgentExecutionConfig,
)
from dapr_agents.agents.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    AgentWorkflowMessage,
    AgentWorkflowEntry,
    AgentWorkflowState,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import (
    AssistantMessage,
    LLMChatCandidate,
    LLMChatResponse,
    ToolExecutionRecord,
    ToolMessage,
)


# We need this otherwise these tests all fail since they require Dapr to be available.
@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    """Mock Dapr dependencies to prevent requiring a running Dapr instance."""
    from unittest.mock import Mock
    import dapr.ext.workflow as wf

    # Mock WorkflowRuntime to prevent Dapr checks
    mock_runtime = Mock(spec=wf.WorkflowRuntime)
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

    # Return the mock runtime for tests that need it
    yield mock_runtime


class MockDaprClient:
    """Mock DaprClient that supports context manager protocol"""

    def __init__(self):
        self.get_state = MagicMock(return_value=Mock(data=None, json=lambda: {}))
        self.save_state = MagicMock()
        self.delete_state = MagicMock()
        self.query_state = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, *args, **kwargs):
        return self


class TestDurableAgent:
    """Test cases for the DurableAgent class."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock DaprClient to use our context manager supporting mock
        mock_client = MockDaprClient()
        mock_client.get_state.return_value = Mock(data=None)  # Default empty state

        # Patch both the client import locations
        monkeypatch.setattr("dapr.clients.DaprClient", lambda: mock_client)
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        mock = Mock(spec=OpenAIChatClient)
        mock.generate = AsyncMock()
        mock.prompt_template = None
        # Set the class name to avoid OpenAI validation
        mock.__class__.__name__ = "MockLLMClient"
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock(spec=AgentTool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.run = AsyncMock(return_value="test_result")
        tool._is_async = True
        return tool

    @pytest.fixture
    def mock_workflow_context(self):
        """Create a mock Dapr workflow context."""
        context = DaprWorkflowContext()
        context.instance_id = "test-instance-123"
        context.is_replaying = False
        context.call_activity = AsyncMock()
        context.wait_for_external_event = AsyncMock()
        context.current_utc_datetime = Mock()
        context.current_utc_datetime.isoformat = Mock(
            return_value="2024-01-01T00:00:00.000000"
        )
        return context

    @pytest.fixture
    def basic_durable_agent(self, mock_llm):
        """Create a basic durable agent instance for testing."""
        return DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            goal="Help with testing",
            instructions=["Be helpful", "Test things"],
            llm=mock_llm,
            pubsub_config=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="TestDurableAgent",
            ),
            state_config=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry_config=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
            memory_config=AgentMemoryConfig(
                store=ConversationDaprStateMemory(
                    store_name="teststatestore", session_id="test_session"
                )
            ),
            execution_config=AgentExecutionConfig(max_iterations=5),
        )

    @pytest.fixture
    def durable_agent_with_tools(self, mock_llm, mock_tool):
        """Create a durable agent with tools for testing."""
        return DurableAgent(
            name="ToolDurableAgent",
            role="Tool Durable Assistant",
            goal="Execute tools",
            instructions=["Use tools when needed"],
            llm=mock_llm,
            pubsub_config=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="ToolDurableAgent",
            ),
            state_config=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry_config=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
            memory_config=AgentMemoryConfig(
                store=ConversationDaprStateMemory(
                    store_name="teststatestore", session_id="test_session"
                )
            ),
            tools=[mock_tool],
            execution_config=AgentExecutionConfig(max_iterations=5),
        )

    def test_durable_agent_initialization(self, mock_llm):
        """Test durable agent initialization with basic parameters."""
        agent = DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            goal="Help with testing",
            instructions=["Be helpful"],
            llm=mock_llm,
            pubsub_config=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="TestDurableAgent",
            ),
            state_config=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry_config=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
        )

        assert agent.name == "TestDurableAgent"
        assert agent.prompting_helper.role == "Test Durable Assistant"
        assert agent.prompting_helper.goal == "Help with testing"
        assert agent.prompting_helper.instructions == ["Be helpful"]
        assert agent.execution_config.max_iterations == 10  # default value
        assert agent.tool_history == []
        assert agent.pubsub_config.pubsub_name == "testpubsub"
        assert agent.pubsub_config.agent_topic == "TestDurableAgent"

    def test_durable_agent_initialization_with_custom_topic(self, mock_llm):
        """Test durable agent initialization with custom topic name."""
        agent = DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            goal="Help with testing",
            llm=mock_llm,
            pubsub_config=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="custom-topic",
            ),
            state_config=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry_config=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
        )

        assert agent.pubsub_config.agent_topic == "custom-topic"

    def test_durable_agent_initialization_name_from_role(self, mock_llm):
        """Test durable agent initialization with name derived from role."""
        agent = DurableAgent(
            name="Test Durable Assistant",
            role="Test Durable Assistant",
            goal="Help with testing",
            llm=mock_llm,
            pubsub_config=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="Test Durable Assistant",
            ),
            state_config=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry_config=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
        )

        assert agent.name == "Test Durable Assistant"
        assert agent.pubsub_config.agent_topic == "Test Durable Assistant"

    def test_durable_agent_metadata(self, basic_durable_agent):
        """Test durable agent metadata creation."""
        metadata = basic_durable_agent.agent_metadata

        assert metadata is not None
        assert metadata["name"] == "TestDurableAgent"
        assert metadata["role"] == "Test Durable Assistant"
        assert metadata["goal"] == "Help with testing"
        assert metadata["topic_name"] == "TestDurableAgent"
        assert metadata["pubsub_name"] == "testpubsub"
        assert metadata["orchestrator"] is False

    @pytest.fixture
    def mock_wf_client(self):
        client = Mock()
        client.wait_for_workflow_completion.return_value.serialized_output = {
            "output": "test"
        }
        return client

    @pytest.mark.skip(reason="DurableAgent doesn't have a run() method - uses workflow invocation")
    @pytest.mark.asyncio
    async def test_run_method(self, basic_durable_agent, mock_wf_client):
        """Test the run method returns the workflow result from the injected mock client."""
        basic_durable_agent.wf_client = mock_wf_client
        result = await basic_durable_agent.run("test input")
        assert result == {"output": "test"}

    @pytest.mark.asyncio
    async def test_tool_calling_workflow_initialization(
        self, basic_durable_agent, mock_workflow_context
    ):
        """Test workflow initialization on first iteration."""
        message = {
            "task": "Test task",
            "iteration": 0,
            "workflow_instance_id": "parent-instance-123",
        }

        mock_workflow_context.instance_id = "test-instance-123"
        mock_workflow_context.call_activity.side_effect = [
            {"content": "Test response"},
            {"message": "Test response"},
            "stop",
        ]

        basic_durable_agent.state["instances"]["test-instance-123"] = {
            "input": "Test task",
            "source": None,
            "triggering_workflow_instance_id": "parent-instance-123",
            "workflow_instance_id": "test-instance-123",
            "workflow_name": "AgenticWorkflow",
            "status": "RUNNING",
            "messages": [],
            "tool_history": [],
            "end_time": None,
            "trace_context": None,
        }

        workflow_gen = basic_durable_agent.agent_workflow(
            mock_workflow_context, message
        )
        try:
            await workflow_gen.__anext__()
        except StopAsyncIteration:
            pass

        assert "test-instance-123" in basic_durable_agent.state["instances"]
        instance_data = basic_durable_agent.state["instances"]["test-instance-123"]
        assert instance_data["input"] == "Test task"
        assert instance_data["source"] is None
        assert instance_data["triggering_workflow_instance_id"] == "parent-instance-123"

    @pytest.mark.asyncio
    async def test_call_llm_activity(self, basic_durable_agent):
        """Test that call_llm unwraps an LLMChatResponse properly."""

        # create a fake LLMChatResponse with one choice
        fake_response = LLMChatResponse(
            results=[
                LLMChatCandidate(
                    message=AssistantMessage(content="Test response", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            metadata={},
        )
        basic_durable_agent.llm.generate = Mock(return_value=fake_response)

        instance_id = "test-instance-123"
        # set up a minimal instance record
        basic_durable_agent.state["instances"] = {
            instance_id: {
                "input": "Test task",
                "source": "test_source",
                "triggering_workflow_instance_id": None,
                "workflow_instance_id": instance_id,
                "workflow_name": "AgenticWorkflow",
                "status": "RUNNING",
                "messages": [],
                "tool_history": [],
                "end_time": None,
                "trace_context": None,
            }
        }

        from datetime import datetime

        test_time = datetime.fromisoformat(
            "2024-01-01T00:00:00Z".replace("Z", "+00:00")
        )
        
        # Mock the activity context
        mock_ctx = Mock()
        
        assistant_dict = basic_durable_agent.call_llm(
            mock_ctx,
            {
                "instance_id": instance_id,
                "time": test_time.isoformat(),
                "task": "Test task"
            }
        )
        # The dict dumped from AssistantMessage should have our content
        assert assistant_dict["content"] == "Test response"
        assert assistant_dict["tool_calls"] == []
        basic_durable_agent.llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_message_to_agents_activity(self, basic_durable_agent):
        """Test broadcasting message to agents activity."""
        message = {
            "type": "broadcast",
            "content": "Test broadcast message",
            "sender": "TestDurableAgent",
        }

        # Mock the activity context
        mock_ctx = Mock()
        
        # The basic_durable_agent fixture doesn't have a broadcast_topic configured,
        # so this should execute without error but skip the actual broadcast
        basic_durable_agent.broadcast_message_to_agents(
            mock_ctx,
            {"message": message}
        )
        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_send_response_back_activity(self, basic_durable_agent):
        """Test sending response back to target agent activity."""
        response = {"content": "Test response"}
        target_agent = "TargetAgent"
        target_instance_id = "target-instance-123"

        # Mock the activity context and _run_asyncio_task
        mock_ctx = Mock()
        
        with patch.object(basic_durable_agent, '_run_asyncio_task') as mock_run_task:
            basic_durable_agent.send_response_back(
                mock_ctx,
                {
                    "response": response,
                    "target_agent": target_agent,
                    "target_instance_id": target_instance_id
                }
            )
            # Verify the async task was called
            mock_run_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_finish_workflow_activity(self, basic_durable_agent):
        """Test finishing workflow activity."""
        from datetime import datetime, timezone
        
        instance_id = "test-instance-123"
        final_output = "Final response"
        # Set up state in the state model using AgentWorkflowEntry
        if not hasattr(basic_durable_agent._state_model, 'instances'):
            basic_durable_agent._state_model.instances = {}
        
        basic_durable_agent._state_model.instances[instance_id] = AgentWorkflowEntry(
            input_value="Test task",
            source="test_source",
            triggering_workflow_instance_id=None,
            workflow_instance_id=instance_id,
            workflow_name="AgenticWorkflow",
            status="RUNNING",
            messages=[],
            tool_history=[],
            end_time=None,
            start_time=datetime.now(timezone.utc),
        )

        # Mock the activity context and save_state
        mock_ctx = Mock()
        
        with patch.object(basic_durable_agent, 'save_state'):
            basic_durable_agent.finalize_workflow(
                mock_ctx,
                {
                    "instance_id": instance_id,
                    "final_output": final_output,
                    "end_time": "2024-01-01T00:00:00Z",
                    "triggering_workflow_instance_id": None
                }
            )
        entry = basic_durable_agent._state_model.instances[instance_id]
        assert entry.output == final_output
        assert entry.end_time is not None

    @pytest.mark.asyncio
    async def test_run_tool(self, basic_durable_agent, mock_tool):
        """Test that run_tool atomically executes and persists tool results."""
        from datetime import datetime, timezone
        
        instance_id = "test-instance-123"
        tool_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"arg1": "value1"}'},
        }

        # Mock the tool executor
        with patch.object(
            type(basic_durable_agent.tool_executor), "run_tool", new_callable=AsyncMock
        ) as mock_run_tool:
            mock_run_tool.return_value = "tool_result"

            # Set up state in the state model using AgentWorkflowEntry
            if not hasattr(basic_durable_agent._state_model, 'instances'):
                basic_durable_agent._state_model.instances = {}
            
            basic_durable_agent._state_model.instances[instance_id] = AgentWorkflowEntry(
                input_value="Test task",
                source="test_source",
                triggering_workflow_instance_id=None,
                workflow_instance_id=instance_id,
                workflow_name="AgenticWorkflow",
                status="RUNNING",
                messages=[],
                tool_history=[],
                end_time=None,
                start_time=datetime.now(timezone.utc),
            )

            test_time = datetime.fromisoformat(
                "2024-01-01T00:00:00Z".replace("Z", "+00:00")
            )
            
            # Mock the activity context and save_state
            mock_ctx = Mock()
            
            with patch.object(basic_durable_agent, 'save_state'):
                result = await basic_durable_agent.run_tool(
                    mock_ctx,
                    {
                        "tool_call": tool_call,
                        "instance_id": instance_id,
                        "time": test_time.isoformat(),
                        "order": 1
                    }
                )

            # Verify tool was executed and result was returned
            assert result["tool_call_id"] == "call_123"
            assert result["tool_name"] == "test_tool"
            assert result["execution_result"] == "tool_result"

            # Verify state was updated atomically
            entry = basic_durable_agent._state_model.instances[instance_id]
            assert len(entry.messages) == 1  # Tool message added
            assert len(entry.tool_history) == 1  # Tool execution record added

            # Verify tool execution record in tool_history
            tool_history_entry = entry.tool_history[0]
            assert tool_history_entry.tool_call_id == "call_123"
            assert tool_history_entry.tool_name == "test_tool"
            assert tool_history_entry.execution_result == "tool_result"

            # Verify agent-level tool_history was also updated
            assert len(basic_durable_agent.tool_history) == 1

    @pytest.mark.skip(reason="get_source_or_default() method removed in refactored architecture")
    def test_get_source_or_default(self, basic_durable_agent):
        """Test get_source_or_default helper method."""
        # Test with valid source
        assert basic_durable_agent.get_source_or_default("test_source") == "test_source"

        # Test with None source
        assert basic_durable_agent.get_source_or_default(None) == "direct"

        # Test with empty string
        assert basic_durable_agent.get_source_or_default("") == "direct"

    def test_record_initial_entry(self, basic_durable_agent):
        """Test record_initial_entry helper method."""
        from datetime import datetime, timezone
        
        instance_id = "test-instance-123"
        input_data = "Test task"
        source = "test_source"
        triggering_workflow_instance_id = "parent-instance-123"
        start_time = "2024-01-01T00:00:00Z"

        # First, ensure instance exists with ensure_instance_exists
        basic_durable_agent.ensure_instance_exists(
            instance_id=instance_id,
            input_value=input_data,
            triggering_workflow_instance_id=None,
            time=datetime.now(timezone.utc)
        )

        # Mock the activity context
        mock_ctx = Mock()
        
        with patch.object(basic_durable_agent, 'save_state'):
            basic_durable_agent.record_initial_entry(
                mock_ctx,
                {
                    "instance_id": instance_id,
                    "input_value": input_data,
                    "source": source,
                    "triggering_workflow_instance_id": triggering_workflow_instance_id,
                    "start_time": start_time,
                    "trace_context": None
                }
            )

        # Verify instance was updated
        assert instance_id in basic_durable_agent._state_model.instances
        entry = basic_durable_agent._state_model.instances[instance_id]
        assert entry.input_value == input_data
        assert entry.source == source
        assert entry.triggering_workflow_instance_id == triggering_workflow_instance_id
        assert entry.status.lower() == "running"

    def test_ensure_instance_exists(self, basic_durable_agent):
        """Test _ensure_instance_exists helper method."""
        instance_id = "test-instance-123"
        triggering_workflow_instance_id = "parent-instance-123"
        time = "2024-01-01T00:00:00Z"

        # Test creating new instance
        from datetime import datetime

        test_time = datetime.fromisoformat(time.replace("Z", "+00:00"))
        basic_durable_agent._ensure_instance_exists(
            instance_id, "Test input", triggering_workflow_instance_id, test_time
        )

        assert instance_id in basic_durable_agent.state["instances"]
        instance_data = basic_durable_agent.state["instances"][instance_id]
        assert (
            instance_data["triggering_workflow_instance_id"]
            == triggering_workflow_instance_id
        )
        # start_time is stored as string in dict format
        assert instance_data["start_time"] == "2024-01-01T00:00:00+00:00"
        assert instance_data["workflow_name"] == "AgenticWorkflow"

        # Test that existing instance is not overwritten
        original_input = "Original input"
        basic_durable_agent.state["instances"][instance_id]["input"] = original_input

        basic_durable_agent._ensure_instance_exists(
            instance_id, "different-parent", "2024-01-02T00:00:00Z"
        )

        # Input should remain unchanged
        assert (
            basic_durable_agent.state["instances"][instance_id]["input"]
            == original_input
        )

    def test_process_user_message(self, basic_durable_agent):
        """Test _process_user_message helper method."""
        instance_id = "test-instance-123"
        task = "Hello, world!"
        user_message_copy = {"role": "user", "content": "Hello, world!"}

        # Set up instance
        basic_durable_agent.state["instances"][instance_id] = {
            "input": "Test task",
            "source": "test_source",
            "triggering_workflow_instance_id": None,
            "workflow_instance_id": instance_id,
            "workflow_name": "AgenticWorkflow",
            "status": "RUNNING",
            "messages": [],
            "tool_history": [],
            "end_time": None,
            "trace_context": None,
        }

        # Mock memory.add_message
        with patch.object(type(basic_durable_agent.memory), "add_message"):
            basic_durable_agent._process_user_message(
                instance_id, task, user_message_copy
            )

        # Verify message was added to instance
        instance_data = basic_durable_agent.state["instances"][instance_id]
        assert len(instance_data["messages"]) == 1
        assert instance_data["messages"][0]["role"] == "user"
        assert instance_data["messages"][0]["content"] == "Hello, world!"
        assert instance_data["last_message"]["role"] == "user"

    def test_save_assistant_message(self, basic_durable_agent):
        """Test _save_assistant_message helper method."""
        instance_id = "test-instance-123"
        assistant_message = {"role": "assistant", "content": "Hello back!"}

        # Set up instance
        basic_durable_agent.state["instances"][instance_id] = {
            "input": "Test task",
            "source": "test_source",
            "triggering_workflow_instance_id": None,
            "workflow_instance_id": instance_id,
            "workflow_name": "AgenticWorkflow",
            "status": "RUNNING",
            "messages": [],
            "tool_history": [],
            "end_time": None,
            "trace_context": None,
        }

        # Mock memory.add_message
        with patch.object(type(basic_durable_agent.memory), "add_message"):
            basic_durable_agent._save_assistant_message(instance_id, assistant_message)

        # Verify message was added to instance
        instance_data = basic_durable_agent.state["instances"][instance_id]
        assert len(instance_data["messages"]) == 1
        assert instance_data["messages"][0]["role"] == "assistant"
        assert instance_data["messages"][0]["content"] == "Hello back!"
        assert instance_data["last_message"]["role"] == "assistant"

    def test_get_last_message_from_state(self, basic_durable_agent):
        """Test _get_last_message_from_state helper method."""
        instance_id = "test-instance-123"

        # Set up instance with last_message
        basic_durable_agent.state["instances"][instance_id] = {
            "input": "Test task",
            "source": "test_source",
            "triggering_workflow_instance_id": None,
            "workflow_instance_id": instance_id,
            "workflow_name": "AgenticWorkflow",
            "status": "RUNNING",
            "messages": [],
            "tool_history": [],
            "end_time": None,
            "trace_context": None,
            "last_message": AgentWorkflowMessage(
                role="assistant", content="Last message"
            ).model_dump(mode="json"),
        }

        result = basic_durable_agent._get_last_message_from_state(instance_id)
        assert result["role"] == "assistant"
        assert result["content"] == "Last message"

        # Test with non-existent instance
        result = basic_durable_agent._get_last_message_from_state("non-existent")
        assert result is None

    def test_create_tool_message_objects(self, basic_durable_agent):
        """Test _create_tool_message_objects helper method."""
        tool_result = {
            "tool_call_id": "call_123",
            "tool_name": "test_tool",
            "tool_args": {"arg1": "value1"},
            "execution_result": "tool_result",
        }

        (
            tool_msg,
            agent_msg,
            tool_history_entry,
        ) = basic_durable_agent._create_tool_message_objects(tool_result)

        # Verify tool message
        assert tool_msg.tool_call_id == "call_123"
        assert tool_msg.name == "test_tool"
        assert tool_msg.content == "tool_result"

        # Verify agent message (AgentWorkflowMessage)
        assert agent_msg.role == "tool"
        assert agent_msg.tool_call_id == "call_123"
        assert agent_msg.content == "tool_result"

        # Verify tool history entry (ToolExecutionRecord)
        assert tool_history_entry.tool_call_id == "call_123"
        assert tool_history_entry.tool_name == "test_tool"
        assert tool_history_entry.tool_args == {"arg1": "value1"}
        assert tool_history_entry.execution_result == "tool_result"

    def test_append_tool_message_to_instance(self, basic_durable_agent):
        """Test _append_tool_message_to_instance helper method."""
        instance_id = "test-instance-123"

        # Set up instance
        basic_durable_agent.state["instances"][instance_id] = {
            "input": "Test task",
            "source": "test_source",
            "triggering_workflow_instance_id": None,
            "workflow_instance_id": instance_id,
            "workflow_name": "AgenticWorkflow",
            "status": "RUNNING",
            "messages": [],
            "tool_history": [],
            "end_time": None,
            "trace_context": None,
        }

        # Create mock objects

        agent_msg = AgentWorkflowMessage(role="assistant", content="Tool result")
        tool_history_entry = ToolExecutionRecord(
            tool_call_id="call_123",
            tool_name="test_tool",
            execution_result="tool_result",
        )

        basic_durable_agent._append_tool_message_to_instance(
            instance_id, agent_msg, tool_history_entry
        )

        # Verify instance was updated
        instance_data = basic_durable_agent.state["instances"][instance_id]
        assert len(instance_data["messages"]) == 1
        assert instance_data["messages"][0]["role"] == "assistant"
        assert len(instance_data["tool_history"]) == 1
        assert instance_data["tool_history"][0]["tool_call_id"] == "call_123"

    def test_update_agent_memory_and_history(self, basic_durable_agent):
        """Test _update_agent_memory_and_history helper method."""

        tool_msg = ToolMessage(
            tool_call_id="call_123", name="test_tool", content="Tool result"
        )
        tool_history_entry = ToolExecutionRecord(
            tool_call_id="call_123",
            tool_name="test_tool",
            execution_result="tool_result",
        )

        # Mock the memory add_message method
        with patch.object(
            type(basic_durable_agent.memory), "add_message"
        ) as mock_add_message:
            basic_durable_agent._update_agent_memory_and_history(
                tool_msg, tool_history_entry
            )

            # Verify memory was updated
            mock_add_message.assert_called_once_with(tool_msg)

        # Verify agent-level tool_history was updated
        assert len(basic_durable_agent.tool_history) == 1
        assert basic_durable_agent.tool_history[0].tool_call_id == "call_123"

    def test_construct_messages_with_instance_history(self, basic_durable_agent):
        """Test _construct_messages_with_instance_history helper method."""
        instance_id = "test-instance-123"
        input_data = "Test input"

        # Set up instance with messages
        basic_durable_agent.state["instances"][instance_id] = {
            "input": "Test task",
            "source": "test_source",
            "triggering_workflow_instance_id": None,
            "workflow_instance_id": instance_id,
            "workflow_name": "AgenticWorkflow",
            "status": "RUNNING",
            "messages": [
                AgentWorkflowMessage(role="user", content="Hello").model_dump(
                    mode="json"
                ),
                AgentWorkflowMessage(role="assistant", content="Hi there!").model_dump(
                    mode="json"
                ),
            ],
            "tool_history": [],
            "end_time": None,
            "trace_context": None,
        }

        # Mock prompt template
        basic_durable_agent.prompt_template = Mock()
        basic_durable_agent.prompt_template.format_prompt.return_value = [
            {"role": "system", "content": "System prompt"}
        ]

        messages = basic_durable_agent._construct_messages_with_instance_history(
            instance_id, input_data
        )

        # Should include system message + user input
        assert len(messages) == 2  # system + user input
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test input"

    @pytest.mark.asyncio
    async def test_broadcast_message(self, basic_durable_agent):
        """Test broadcasting message."""
        broadcast_msg = BroadcastMessage(
            content="Test broadcast",
            role="assistant",
            type="broadcast",
            sender="TestDurableAgent",
        )

        # This needs refactoring / better implementation on this test since the actual implementation would depend on the pubsub msg broker.
        await basic_durable_agent.broadcast_message(broadcast_msg)

    @pytest.mark.asyncio
    async def test_send_message_to_agent(self, basic_durable_agent):
        """Test sending message to specific agent."""
        task_response = AgentTaskResponse(
            content="Test task",
            role="assistant",
            task="Test task",
            agent_name="TargetAgent",
            workflow_instance_id="target-instance-123",
        )

        # This needs refactoring / better implementation on this test since the actual implementation would depend on the pubsub msg broker.
        await basic_durable_agent.send_message_to_agent("TargetAgent", task_response)

    def test_register_agentic_system(self, basic_durable_agent):
        """Test registering agentic system."""
        # TODO(@Sicoyle): fix this to add assertions.
        basic_durable_agent.register_agentic_system()

    @pytest.mark.asyncio
    async def test_process_broadcast_message(self, basic_durable_agent):
        """Test processing broadcast message."""
        broadcast_msg = BroadcastMessage(
            content="Test broadcast",
            role="assistant",
            type="broadcast",
            sender="OtherAgent",
        )

        # This needs refactoring / better implementation on this test since the actual implementation would depend on the pubsub msg broker.
        await basic_durable_agent.process_broadcast_message(broadcast_msg)

    def test_durable_agent_properties(self, basic_durable_agent):
        """Test durable agent properties."""
        assert basic_durable_agent.tool_executor is not None
        assert basic_durable_agent.text_formatter is not None
        assert basic_durable_agent.state is not None

    def test_durable_agent_workflow_name(self, basic_durable_agent):
        """Test that the workflow name is set correctly."""
        assert basic_durable_agent._workflow_name == "AgenticWorkflow"

    def test_durable_agent_state_initialization(self, basic_durable_agent):
        """Test that the agent state is properly initialized."""
        validated_state = AgentWorkflowState.model_validate(basic_durable_agent.state)
        assert isinstance(validated_state, AgentWorkflowState)
        assert "instances" in basic_durable_agent.state
        assert basic_durable_agent.state["instances"] == {}

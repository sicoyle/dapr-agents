"""Tests for the RandomOrchestrator."""
import pytest
from unittest.mock import MagicMock, patch
from dapr_agents.agents.orchestrators import RandomOrchestrator
from dapr_agents.agents.configs import AgentPubSubConfig, AgentStateConfig, AgentRegistryConfig
from dapr_agents.storage.daprstores.stateservice import StateStoreService


@pytest.fixture
def orchestrator_config():
    """Fixture to provide common orchestrator configuration."""
    return {
        "name": "test_orchestrator",
        "pubsub_config": AgentPubSubConfig(pubsub_name="test-message-bus"),
        "state_config": AgentStateConfig(store=StateStoreService(store_name="test-state-store")),
        "registry_config": AgentRegistryConfig(store=StateStoreService(store_name="test-registry-store")),
    }


def test_random_orchestrator_initialization(orchestrator_config):
    """Test that RandomOrchestrator can be initialized."""
    with patch("dapr.ext.workflow.WorkflowRuntime") as mock_runtime:
        mock_runtime.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)
        assert orchestrator.name == "test_orchestrator"


@pytest.mark.asyncio
async def test_process_input(orchestrator_config):
    """Test the process_input task."""
    with patch("dapr.ext.workflow.WorkflowRuntime") as mock_runtime:
        mock_runtime.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)
        task = "test task"
        result = await orchestrator.process_input(task)

        assert result["role"] == "user"
        assert result["name"] == "test_orchestrator"
        assert result["content"] == task


def test_select_random_speaker(orchestrator_config):
    """Test the select_random_speaker task."""
    with patch("dapr.ext.workflow.WorkflowRuntime") as mock_runtime, patch.object(
        RandomOrchestrator,
        "get_agents_metadata",
        return_value={"agent1": {"name": "agent1"}, "agent2": {"name": "agent2"}},
    ):
        mock_runtime.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)

        speaker = orchestrator.select_random_speaker()
        assert speaker in ["agent1", "agent2"]
        assert orchestrator.current_speaker == speaker

import json
from pydantic import Field, BaseModel, PrivateAttr
from typing import Optional, Type, Dict, Any, List
from datetime import datetime
import uuid
from dapr_agents.types import MessageContent, ToolExecutionRecord
from dapr_agents.types.workflow import DaprWorkflowStatus


class Storage(BaseModel):
    """
    Unified storage interface for both Agent (in-memory) and DurableAgent (persistent).

    - If `name` is provided: Uses Dapr state store for persistence to provide persistent storage for the agent on workflow state (if DurableAgent), agent registration, and conversation history.
    - If `name` is None: Uses in-memory list storage to provide in-memory storage for the agent on conversation history.
    """

    name: Optional[str] = Field(
        default=None,
        description=(
            "Dapr state store for persistent data. "
            "If None, uses in-memory storage. "
            "If provided, uses persistent Dapr State Store for storage of workflow state (if DurableAgent), agent registration, and conversation history."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID to group related conversations and workflow state. "
            "If not provided, a unique ID will be generated. "
            "Can be overridden in agent.run()."
        ),
    )
    local_directory: Optional[str] = Field(
        default=None,
        description=(
            "Directory path where state files will be saved locally. "
            "If not set, no local saving occurs. "
            "Can be absolute or relative to workspace root. "
            "Files will be named '{agent_name}_state.json'."
        ),
    )
    _current_state: dict = PrivateAttr(default_factory=dict)
    _agent_name: str = PrivateAttr(
        default=None
    )  # Set by AgenticWorkflow -> TODO: SAM to double check this in adding regular agents registration capabilities!!
    _workflow_prefix: str = PrivateAttr(default="workflow")
    _key: str = PrivateAttr(default="workflow_state")
    _in_memory_messages: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def _set_key(self, agent_name: str) -> None:
        """Internal method to set the agent name and initialize storage."""
        self._agent_name = agent_name
        self._key = f"{agent_name}_workflow_state"

    def _get_instance_key(self, instance_id: str) -> str:
        """Get the state store key for a workflow instance."""
        return f"{self._agent_name}_{self._workflow_prefix}_{instance_id}"

    def _get_session_key(self, session_id: str) -> str:
        """Get the state store key for a specific session."""
        return f"{self._agent_name}_session_{session_id}"

    def _get_sessions_index_key(self) -> str:
        """Get the state store key for the sessions index."""
        return f"{self._agent_name}_sessions"

    def _get_session_id(self) -> str:
        """Get or generate a session ID."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self.session_id

    def _update_session_index(self, instance_id: str, state_store_client) -> None:
        """Update the session with a new workflow instance and update the sessions index."""
        session_id = self._get_session_id()
        session_key = self._get_session_key(session_id)
        sessions_index_key = self._get_sessions_index_key()
        
        # Get or create session data
        has_session, session_data = state_store_client.try_get_state(session_key)
        is_new_session = not has_session or not session_data
        if is_new_session:
            # Create new session
            session_data = {
                "session_id": session_id,
                "workflow_instances": [],
                "metadata": {
                    "agent_name": self._agent_name,
                    "created_at": datetime.now().isoformat(),
                },
            }
        else:
            # Deserialize if it's a JSON string
            if isinstance(session_data, str):
                session_data = json.loads(session_data)

        # Add instance to session if not already present
        if instance_id not in session_data.get("workflow_instances", []):
            session_data["workflow_instances"].append(instance_id)
        
        # Update last active timestamp
        session_data["last_active"] = datetime.now().isoformat()

        # Save the updated session data
        self._save_state_with_metadata(state_store_client, session_key, session_data)

        if is_new_session:
            has_index, index_data = state_store_client.try_get_state(sessions_index_key)
            
            if not has_index or not index_data:
                index_data = {"sessions": []}
            else:
                if isinstance(index_data, str):
                    index_data = json.loads(index_data)
            
            # Add session to index if not already present
            if session_id not in index_data.get("sessions", []):
                index_data["sessions"].append(session_id)
                index_data["last_updated"] = datetime.now().isoformat()
                self._save_state_with_metadata(state_store_client, sessions_index_key, index_data)

    def get_session_workflows(self, state_store_client) -> List[str]:
        """
        Get workflow instances for the current session.

        Args:
            state_store_client: The Dapr state store client

        Returns:
            List of workflow instance IDs (both active and completed)
        """
        session_id = self._get_session_id()
        session_key = self._get_session_key(session_id)
        has_session, session_data = state_store_client.try_get_state(session_key)

        if not has_session or not session_data:
            return []

        # Deserialize if it's a JSON string
        if isinstance(session_data, str):
            session_data = json.loads(session_data)

        return session_data.get("workflow_instances", [])

    def get_all_sessions(self, state_store_client) -> List[str]:
        """
        Get all session IDs for this agent.
        
        Args:
            state_store_client: The Dapr state store client
            
        Returns:
            List of session IDs
        """
        sessions_index_key = self._get_sessions_index_key()
        has_index, index_data = state_store_client.try_get_state(sessions_index_key)
        
        if not has_index or not index_data:
            return []
        
        # Deserialize if it's a JSON string
        if isinstance(index_data, str):
            index_data = json.loads(index_data)
        
        return index_data.get("sessions", [])

    def reset_session(self, state_store_client) -> None:
        """
        Reset all state for the current session, including workflow instances and session data.
        """
        session_id = self._get_session_id()
        session_key = self._get_session_key(session_id)
        sessions_index_key = self._get_sessions_index_key()

        # Get session data to find all workflow instances
        has_session, session_data = state_store_client.try_get_state(session_key)
        if has_session and session_data:
            # Deserialize if it's a JSON string
            if isinstance(session_data, str):
                session_data = json.loads(session_data)

            # Delete all workflow instances
            for instance_id in session_data.get("workflow_instances", []):
                instance_key = self._get_instance_key(instance_id)
                state_store_client.delete_state(instance_key)

            # Delete the session itself
            state_store_client.delete_state(session_key)

            # Remove session from the sessions index
            has_index, index_data = state_store_client.try_get_state(sessions_index_key)
            if has_index and index_data:
                if isinstance(index_data, str):
                    index_data = json.loads(index_data)
                
                if session_id in index_data.get("sessions", []):
                    index_data["sessions"].remove(session_id)
                    index_data["last_updated"] = datetime.now().isoformat()
                    self._save_state_with_metadata(state_store_client, sessions_index_key, index_data)

    def _save_state_with_metadata(
        self, state_store_client, key: str, data: Any
    ) -> None:
        """Save state with content type metadata."""
        # Serialize data to JSON string if it's not already
        if isinstance(data, dict):
            data_to_save = json.dumps(data)
        elif isinstance(data, str):
            data_to_save = data
        else:
            data_to_save = json.dumps(data)

        state_store_client.save_state(
            key, data_to_save, state_metadata={"contentType": "application/json"}
        )

    def is_persistent(self) -> bool:
        """Check if storage is persistent (has a state store name) or in-memory."""
        return self.name is not None

    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a single message to storage.
        Uses in-memory list if name is None.

        Args:
            message (Dict[str, Any]): The message to add
        """
        if not self.is_persistent():
            # In-memory mode for regular Agent
            self._in_memory_messages.append(message)

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add multiple messages to storage.

        Args:
            messages (List[Dict[str, Any]]): The messages to add
        """
        if not self.is_persistent():
            # In-memory mode for regular Agent
            self._in_memory_messages.extend(messages)

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages from storage.

        Returns:
            List[Dict[str, Any]]: All stored messages
        """
        if not self.is_persistent():
            # In-memory mode for regular Agent
            return self._in_memory_messages.copy()

    def reset_memory(self) -> None:
        """Clear all messages from storage."""
        if not self.is_persistent():
            # In-memory mode for regular Agent
            self._in_memory_messages.clear()


class DurableAgentMessage(MessageContent):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created",
    )


class DurableAgentWorkflowEntry(BaseModel):
    """Represents a workflow and its associated data, including metadata on the source of the task request."""

    input: str = Field(
        ..., description="The input or description of the Workflow to be performed"
    )
    output: Optional[str] = Field(
        default=None, description="The output or result of the Workflow, if completed"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the workflow was started",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the workflow was completed or failed",
    )
    messages: List[DurableAgentMessage] = Field(
        default_factory=list,
        description="Messages exchanged during the workflow (user, assistant, or tool messages).",
    )
    last_message: Optional[DurableAgentMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    tool_history: List[ToolExecutionRecord] = Field(
        default_factory=list, description="Tool message exchanged during the workflow"
    )
    source: Optional[str] = Field(None, description="Entity that initiated the task.")
    workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The agent's own workflow instance ID.",
    )
    triggering_workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The workflow instance ID of the entity that triggered this agent (for multi-agent communication).",
    )
    workflow_name: Optional[str] = Field(
        default=None,
        description="The name of the workflow.",
    )
    trace_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenTelemetry trace context for workflow resumption.",
    )
    status: str = Field(
        default=DaprWorkflowStatus.RUNNING.value,
        description="Current status of the workflow.",
    )


class DurableAgentWorkflowState(BaseModel):
    """Represents the state of multiple Agent workflows."""

    instances: Dict[str, DurableAgentWorkflowEntry] = Field(
        default_factory=dict,
        description="Workflow entries indexed by their instance_id.",
    )
    chat_history: List[DurableAgentMessage] = Field(
        default_factory=list,
        description="Chat history of messages exchanged during the workflow.",
    )

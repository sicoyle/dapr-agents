import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type, List

from cloudevents.http.conversion import from_http
from cloudevents.http.event import CloudEvent
from dapr.clients import DaprClient
from dapr.clients.grpc._request import (
    TransactionalStateOperation,
    TransactionOperationType,
)
from dapr.clients.grpc._response import StateResponse
from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, PrivateAttr

from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.memory import MemoryBase, ConversationVectorMemory
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.workflow.base import WorkflowApp
from dapr_agents.workflow.mixins import (
    MessagingMixin,
    PubSubMixin,
    ServiceMixin,
    StateManagementMixin,
)
from dapr_agents.agents.storage import Storage

logger = logging.getLogger(__name__)


class AgenticWorkflow(
    WorkflowApp,
    PubSubMixin,
    StateManagementMixin,
    ServiceMixin,
    MessagingMixin,
):
    """
    A class for managing agentic workflows, extending `WorkflowApp`.
    Handles agent interactions, workflow execution, messaging, and metadata management.
    """

    name: str = Field(..., description="The name of the agentic system.")
    message_bus_name: str = Field(
        ...,
        description="The name of the message bus component, defining the pub/sub base.",
    )
    broadcast_topic_name: Optional[str] = Field(
        default=None,
        description="Default topic for broadcasting messages. Set explicitly for multi-agent setups.",
    )

    storage: Storage = Field(
        ...,
        description="The durable storage for workflow state and agent registration.",
    )

    # TODO: test this is respected by runtime.
    max_iterations: int = Field(
        default=10, description="Maximum iterations for workflows.", ge=1
    )

    # Long term memory based on an execution run, so should be in the execution config class!
    memory: Optional[MemoryBase] = Field(
        default=None,
        description="Handles conversation history storage.",
    )

    client: Optional[DaprClient] = Field(
        default=None, init=False, description="Dapr client instance."
    )

    # Private internal attributes (not schema/validated)
    _state_store_client: Optional[DaprStateStore] = PrivateAttr(default=None)
    _text_formatter: ColorTextFormatter = PrivateAttr(default=ColorTextFormatter)
    _agent_metadata: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _workflow_name: str = PrivateAttr(default=None)
    _dapr_client: Optional[DaprClient] = PrivateAttr(default=None)
    _is_running: bool = PrivateAttr(default=False)
    _shutdown_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _http_server: Optional[Any] = PrivateAttr(default=None)
    _subscriptions: Dict[str, Callable] = PrivateAttr(default_factory=dict)
    _topic_handlers: Dict[
        Tuple[str, str], Dict[Type[BaseModel], Callable]
    ] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the AgenticWorkflow.

        This method initializes the workflow service, messaging, and metadata storage.
        It sets up the color formatter, state store client, loads or initializes the workflow state,
        and creates a Dapr client for service-to-service calls or state interactions.

        Args:
            __context (Any): The context passed by Pydantic's model initialization.

        Raises:
            RuntimeError: If Dapr is not available in the current environment.
        """
        self._dapr_client = DaprClient()
        self._text_formatter = ColorTextFormatter()

        # Set storage key based on agent name
        self.storage._set_key(self.name)

        self._state_store_client = DaprStateStore(store_name=self.storage.name)
        logger.info(f"State store '{self.storage.name}' initialized.")
        self.initialize_state()
        super().model_post_init(__context)

    def get_chat_history(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the chat history from memory as a list of dictionaries.

        Args:
            task (Optional[str]): The task or query provided by the user (used for vector search).

        Returns:
            List[Dict[str, Any]]: The chat history as dictionaries.
        """
        if isinstance(self.memory, ConversationVectorMemory) and task:
            if (
                hasattr(self.memory.vector_store, "embedding_function")
                and self.memory.vector_store.embedding_function
                and hasattr(
                    self.memory.vector_store.embedding_function, "embed_documents"
                )
            ):
                query_embeddings = self.memory.vector_store.embedding_function.embed(
                    task
                )
                vector_messages = self.memory.get_messages(
                    query_embeddings=query_embeddings
                )
                if vector_messages:
                    return vector_messages

        # Get messages from storage
        if self.storage._current_state is None:
            logger.warning("Agent state is None, initializing empty state")
            self.storage._current_state = {}

        # Get messages from all instances
        all_messages = []
        for instance in self.storage._current_state.get("instances", {}).values():
            messages = instance.get("messages", [])
            all_messages.extend(messages)

        # Get long-term memory from workflow state
        long_term_memory = self.storage._current_state.get("chat_history", [])
        all_messages.extend(long_term_memory)

        # If we have vector memory but no task, also include vector memory messages
        if isinstance(self.memory, ConversationVectorMemory):
            vector_messages = self.memory.get_messages()
            all_messages.extend(vector_messages)

        return all_messages

    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        """
        Returns the full chat history as a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: The chat history.
        """
        return self.get_chat_history()

    def get_data_from_store(self, store_name: str, key: str) -> Optional[dict]:
        """
        Retrieves data from the Dapr state store using the given key.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key to fetch data from.

        Returns:
            Optional[dict]: the retrieved dictionary or None if not found.
        """
        try:
            response: StateResponse = self._dapr_client.get_state(
                store_name=store_name, key=key
            )
            data = response.data
            return json.loads(data) if data else None
        except Exception:
            logger.warning(
                f"Error retrieving data for key '{key}' from store '{store_name}'"
            )
            return None

    def get_agents_metadata(
        self, exclude_self: bool = True, exclude_orchestrator: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for all registered agents.

        Args:
            exclude_self: If True, excludes the current agent from results.
            exclude_orchestrator: If True, excludes orchestrators from results.

        Returns:
            Dict[str, Any]: Mapping of agent names to their metadata.

        Raises:
            RuntimeError: If retrieval fails.
        """
        try:
            agents_metadata = (
                self.get_data_from_store(self.storage.name, "agent_registry") or {}
            )

            if agents_metadata:
                logger.info(
                    f"Agents found in '{self.storage.name}' for key 'agent_regisry'."
                )
                filtered = {
                    name: metadata
                    for name, metadata in agents_metadata.items()
                    if not (exclude_self and name == self.name)
                    and not (
                        exclude_orchestrator and metadata.get("orchestrator", False)
                    )
                }
                if not filtered:
                    logger.info("No other agents found after filtering.")
                return filtered

            logger.info(
                f"No agents found in '{self.storage.name}' for key 'agent_registry'."
            )
            return {}
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}", exc_info=True)
            raise RuntimeError(f"Error retrieving agents metadata: {str(e)}") from e

    def print_interaction(
        self, sender_agent_name: str, recipient_agent_name: str, message: str
    ) -> None:
        """
        Pretty-print an interaction between two agents.

        Args:
            sender_agent_name: The name of the agent sending the message.
            recipient_agent_name: The name of the agent receiving the message.
            message: The message content to display.
        """
        separator = "-" * 80
        interaction_text = [
            (sender_agent_name, "dapr_agents_mustard"),
            (" -> ", "dapr_agents_teal"),
            (f"{recipient_agent_name}\n\n", "dapr_agents_mustard"),
            (message + "\n\n", None),
            (separator + "\n", "dapr_agents_teal"),
        ]
        self._text_formatter.print_colored_text(interaction_text)

    async def run_workflow_from_request(self, request: Request) -> JSONResponse:
        """
        Run a workflow instance triggered by an HTTP POST request.

        Args:
            request: The incoming FastAPI request.

        Returns:
            JSONResponse: HTTP response with workflow instance ID or error.
        """
        try:
            workflow_name = request.query_params.get("name") or self._workflow_name
            if not workflow_name:
                return JSONResponse(
                    content={"error": "No workflow name specified."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            if workflow_name not in self.workflows:
                return JSONResponse(
                    content={
                        "error": f"Unknown workflow '{workflow_name}'. Available: {list(self.workflows.keys())}"
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            try:
                event: CloudEvent = from_http(
                    dict(request.headers), await request.body()
                )
                input_data = event.data
            except Exception:
                input_data = await request.json()

            logger.info(f"Starting workflow '{workflow_name}' with input: {input_data}")
            instance_id = await self.run_and_monitor_workflow_async(
                workflow=workflow_name, input=input_data
            )

            return JSONResponse(
                content={
                    "message": "Workflow initiated successfully.",
                    "workflow_instance_id": instance_id,
                },
                status_code=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            logger.error(f"Error starting workflow: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": "Failed to start workflow", "details": str(e)},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

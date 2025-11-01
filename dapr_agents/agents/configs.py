from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Type

from pydantic import BaseModel

from dapr_agents.agents.schemas import AgentWorkflowMessage, AgentWorkflowState
from dapr_agents.memory import ConversationListMemory, MemoryBase
from dapr_agents.storage.daprstores.stateservice import StateStoreService

_JINJA_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{\s*(\w+)\s*\}(?!\})")


def _ensure_jinja_placeholders(text: str) -> str:
    return _JINJA_PLACEHOLDER_PATTERN.sub(r"{{\1}}", text)


# Type hooks for state customization
EntryFactory = Callable[..., Any]
MessageCoercer = Callable[[Dict[str, Any]], Any]
EntryContainerGetter = Callable[[BaseModel], Optional[MutableMapping[str, Any]]]


@dataclass
class AgentStateConfig:
    """Configuration for agent state persistence and model customization.

    Attributes:
        store: Backing state store service.
        default_state: Default state payload or model. If a dict, validated by `state_model_cls`.
        state_key: Optional key override for the durable state entry.
        state_model_cls: Root Pydantic model class for the state (default: AgentWorkflowState).
        message_model_cls: Pydantic model class for workflow/system messages (default: AgentWorkflowMessage).
        entry_factory: Factory used by AgentComponents.ensure_instance_exists(...) to create an entry.
        message_coercer: Function to convert a raw dict into a message model instance.
        entry_container_getter: Function to extract the instance container (e.g., `model.instances`) from the root model.
    """

    store: "StateStoreService"
    default_state: Optional[Dict[str, Any] | BaseModel] = None
    state_key: Optional[str] = None

    state_model_cls: Type[BaseModel] = AgentWorkflowState
    message_model_cls: Type[BaseModel] = AgentWorkflowMessage
    entry_factory: Optional[EntryFactory] = None
    message_coercer: Optional[MessageCoercer] = None
    entry_container_getter: Optional[EntryContainerGetter] = None

    def __post_init__(self) -> None:
        # Defensive checks (optional but helpful during misconfigurations)
        if not issubclass(self.state_model_cls, BaseModel):
            raise TypeError("state_model_cls must be a subclass of pydantic.BaseModel")
        if not issubclass(self.message_model_cls, BaseModel):
            raise TypeError(
                "message_model_cls must be a subclass of pydantic.BaseModel"
            )

        # Normalize default_state against the selected state_model_cls
        Model = self.state_model_cls
        if self.default_state is None:
            self.default_state = Model().model_dump(mode="json")
        else:
            if isinstance(self.default_state, BaseModel):
                self.default_state = self.default_state.model_dump(mode="json")
            else:
                self.default_state = Model.model_validate(
                    self.default_state
                ).model_dump(mode="json")


@dataclass
class AgentRegistryConfig:
    """Configuration for agent registry storage."""

    store: StateStoreService
    team_name: Optional[str] = None


@dataclass
class AgentMemoryConfig:
    """Configuration wrapper for agent memory selection."""

    store: MemoryBase = field(default_factory=ConversationListMemory)


@dataclass
class AgentPubSubConfig:
    """Declarative pub/sub configuration for durable agents.

    Attributes:
        pubsub_name: Name of the Dapr pub/sub component to use for all agent traffic.
        agent_topic: Primary topic for direct messages to the agent. Defaults to ``name``.
        broadcast_topic: Optional topic shared by a team for broadcast messages.
    """

    pubsub_name: str
    agent_topic: Optional[str] = None
    broadcast_topic: Optional[str] = None


@dataclass
class PromptSection:
    """Reusable block for composing a structured system prompt."""

    title: str
    lines: List[str] = field(default_factory=list)

    def render(self, template_format: str) -> str:
        if not self.lines:
            return ""
        header = self.title.strip()
        body = "\n".join(f"- {line.strip()}" for line in self.lines if line.strip())
        section = f"{header}:\n{body}".strip()
        return (
            _ensure_jinja_placeholders(section)
            if template_format == "jinja2"
            else section
        )


@dataclass
class AgentProfileConfig:
    """
    High-level persona description for an agent.

    Mirrors common fields in OpenAI Agents SDK while remaining lightweight.
    """

    name: Optional[str] = None
    role: Optional[str] = None
    goal: Optional[str] = None
    instructions: List[str] = field(default_factory=list)
    style_guidelines: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    template_format: str = "jinja2"
    modules: Sequence[str] = field(default_factory=tuple)
    module_overrides: Dict[str, PromptSection] = field(default_factory=dict)


@dataclass
class AgentExecutionConfig:
    """
    Dials to configure the agent execution.
    """

    # TODO: add a forceFinalAnswer field in case maxIterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    # TODO: add stop_at_tokens
    max_iterations: int = 10
    tool_choice: Optional[str] = "auto"

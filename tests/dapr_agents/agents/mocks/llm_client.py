from dapr_agents.llm import LLMClientBase
from dapr_agents.prompt.base import PromptTemplateBase
from pydantic import Field
from typing import Optional, Dict, Any


class MockLLMClient(LLMClientBase):
    """Mock LLM client for testing."""

    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Mock prompt template for testing."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__class__.__name__ = "MockLLMClient"

    def get_client(self):
        """Mock implementation of get_client."""
        return None

    def get_config(self) -> Dict[str, Any]:
        """Mock implementation of get_config."""
        return {}

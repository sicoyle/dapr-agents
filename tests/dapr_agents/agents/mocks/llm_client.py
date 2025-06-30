from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.types import ChatCompletion
from pydantic import Field
from typing import Optional, Dict, Any, Union, Iterator, Type, Iterable
from pydantic import BaseModel
from pathlib import Path


class MockLLMClient(OpenAIChatClient):
    """Mock LLM client for testing."""

    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Mock prompt template for testing."
    )

    def __init__(self, **kwargs):
        # Set default values to avoid validation issues
        kwargs.setdefault("model", "gpt-4o")
        super().__init__(**kwargs)
        self.__class__.__name__ = "MockLLMClient"

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "MockLLMClient":
        """Mock implementation of from_prompty method."""
        return cls()

    def get_client(self):
        """Mock implementation of get_client."""
        return None

    def get_config(self) -> Dict[str, Any]:
        """Mock implementation of get_config."""
        return {}

    def generate(
        self,
        messages: Union[
            str,
            Dict[str, Any],
            Iterable[Union[Dict[str, Any]]],
        ] = None,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[list] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: str = "json",
        **kwargs,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """Mock implementation of generate method."""
        return {
            "choices": [
                {
                    "message": {
                        "content": "Mock response",
                        "role": "assistant"
                    }
                }
            ]
        }

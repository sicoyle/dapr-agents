from dapr_agents.types.llm import DaprInferenceClientConfig
from dapr_agents.llm.base import LLMClientBase
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput
from dapr.clients.grpc._response import ConversationResponse
from typing import Dict, Any, List, Iterator, Optional
from pydantic import model_validator

import logging

logger = logging.getLogger(__name__)


class DaprInferenceClient:
    def __init__(self):
        self.dapr_client = DaprClient()

    def translate_to_json(self, response: ConversationResponse) -> dict:
        response_dict = {"outputs": []}

        for output in response.outputs:
            output_dict = {
                "result": output.result,
            }

            # Add tool calls if present
            if hasattr(output, "tool_calls") and output.tool_calls:
                output_dict["tool_calls"] = []
                for tool_call in output.tool_calls:
                    tool_call_dict = {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    output_dict["tool_calls"].append(tool_call_dict)

            # Add finish reason if present
            if hasattr(output, "finish_reason") and output.finish_reason:
                output_dict["finish_reason"] = output.finish_reason

            response_dict["outputs"].append(output_dict)

        return response_dict

    def chat_completion(
        self,
        llm: str,
        conversation_inputs: List[ConversationInput],
        scrub_pii: bool | None = None,
        temperature: float | None = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        response = self.dapr_client.converse_alpha1(
            name=llm,
            inputs=conversation_inputs,
            scrub_pii=scrub_pii,
            temperature=temperature,
            parameters=parameters,
        )
        output = self.translate_to_json(response)

        return output

    def chat_completion_stream(
        self,
        llm: str,
        conversation_inputs: List[ConversationInput],
        context_id: str | None = None,
        scrub_pii: bool | None = None,
        temperature: float | None = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion responses using Dapr's converse_stream_alpha1 API.

        Args:
            llm: Name of the LLM component to use
            conversation_inputs: List of conversation inputs
            context_id: Optional context ID for continuing conversation
            scrub_pii: Optional flag to scrub PII from inputs and outputs
            temperature: Optional temperature setting for the LLM

        Yields:
            Dict[str, Any]: JSON-formatted streaming response chunks compatible with common LLM APIs
        """
        logger.info(f"Starting streaming conversation with LLM component: {llm}")

        try:
            # Use converse_stream_alpha1 and transform to JSON format
            for chunk in self.dapr_client.converse_stream_alpha1(
                name=llm,
                inputs=conversation_inputs,
                context_id=context_id,
                scrub_pii=scrub_pii,
                temperature=temperature,
                parameters=parameters,
            ):
                # Transform the chunk to JSON format compatible with common LLM APIs
                chunk_dict = {
                    "choices": [],
                    "context_id": None,
                    "usage": None,
                }

                # Handle streaming result chunks
                if hasattr(chunk, "result") and chunk.result:
                    choice_dict = {"delta": {}, "index": 0, "finish_reason": None}

                    # Handle content
                    if hasattr(chunk.result, "result") and chunk.result.result:
                        choice_dict["delta"]["content"] = chunk.result.result
                        choice_dict["delta"]["role"] = "assistant"

                    # Handle tool calls in streaming
                    if hasattr(chunk.result, "tool_calls") and chunk.result.tool_calls:
                        tool_calls = []
                        for tool_call in chunk.result.tool_calls:
                            tool_call_dict = {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            tool_calls.append(tool_call_dict)
                        choice_dict["delta"]["tool_calls"] = tool_calls

                    # Handle finish reason
                    if (
                        hasattr(chunk.result, "finish_reason")
                        and chunk.result.finish_reason
                    ):
                        choice_dict["finish_reason"] = chunk.result.finish_reason

                    # Only add choice if there's actual content
                    if choice_dict["delta"] or choice_dict["finish_reason"]:
                        chunk_dict["choices"] = [choice_dict]

                # Handle context ID
                if hasattr(chunk, "context_id") and chunk.context_id:
                    chunk_dict["context_id"] = chunk.context_id

                # Handle usage information (typically in the final chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    chunk_dict["usage"] = {
                        "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            chunk.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                    }

                yield chunk_dict

        except Exception as e:
            logger.error(f"Error during streaming conversation: {e}")
            raise


class DaprInferenceClientBase(LLMClientBase):
    """
    Base class for managing Dapr Inference API clients.
    Handles client initialization, configuration, and shared logic.
    """

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes after validation.
        """
        self._provider = "dapr"

        # Set up the private config and client attributes
        self._config = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> DaprInferenceClientConfig:
        """
        Returns the appropriate configuration for the Dapr Conversation API.
        """
        return DaprInferenceClientConfig()

    def get_client(self) -> DaprInferenceClient:
        """
        Initializes and returns the Dapr Inference client.
        """
        return DaprInferenceClient()

    @classmethod
    def from_config(
        cls, client_options: DaprInferenceClientConfig, timeout: float = 1500
    ):
        """
        Initializes the DaprInferenceClientBase using DaprInferenceClientConfig.

        Args:
            client_options: The configuration options for the client.
            timeout: Timeout for requests (default is 1500 seconds).

        Returns:
            DaprInferenceClientBase: The initialized client instance.
        """
        return cls()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def client(self) -> DaprInferenceClient:
        return self._client

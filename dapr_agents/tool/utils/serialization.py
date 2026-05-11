#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Utility functions for serializing tool execution results."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_text_from_content_block(item: Any) -> str:
    """Extract text from a single MCP content block.

    Handles all known formats:
      - protojson oneof:  ``{"text": {"text": "actual result"}}``
      - flat MCP spec:    ``{"type": "text", "text": "actual result"}``
      - plain string:     ``"actual result"``
    """
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    # protojson oneof: {"text": {"text": "..."}}
    text_field = item.get("text")
    if isinstance(text_field, dict):
        return text_field.get("text", "")
    # flat MCP: {"type": "text", "text": "..."}
    if isinstance(text_field, str):
        return text_field
    return ""


def _looks_like_mcp_response(d: dict) -> bool:
    """Check if a dict looks like a CallMCPToolResponse / CallToolResult.

    Handles both protojson (snake_case) and camelCase field names:
      - ``isError`` / ``is_error``
      - ``content``
    """
    has_content = "content" in d
    has_is_error = "isError" in d or "is_error" in d
    # Must have at least content OR isError to be an MCP response
    return has_content or has_is_error


def _get_is_error(d: dict) -> bool:
    """Get the isError value, handling both camelCase and snake_case."""
    return d.get("isError", d.get("is_error", False))


def _unwrap_mcp_call_tool_result(result: Any) -> Any:
    """Unwrap an MCP CallMCPToolResponse / CallToolResult envelope if present.

    The Dapr runtime's ``CallTool`` workflow returns a proto response.
    Protojson may serialize it in different formats::

        # camelCase (protojson default):
        {"isError": false, "content": [{"text": {"text": "result"}}]}

        # snake_case:
        {"is_error": false, "content": [{"text": {"text": "result"}}]}

        # flat MCP spec (older format):
        {"isError": false, "content": [{"type": "text", "text": "result"}]}

    The result may arrive as a JSON string or a pre-parsed dict.
    It may also be double-JSON-encoded (string within a string).

    Returns the unwrapped text, or *result* unchanged if it does not
    match any known MCP response shape.
    """
    # Try to parse JSON strings (possibly double-encoded).
    parsed = result
    for _ in range(2):  # at most 2 levels of JSON decoding
        if isinstance(parsed, str):
            try:
                decoded = json.loads(parsed)
                if isinstance(decoded, dict):
                    parsed = decoded
                else:
                    break
            except (json.JSONDecodeError, TypeError):
                break
        else:
            break

    if not isinstance(parsed, dict):
        return result

    if not _looks_like_mcp_response(parsed):
        return result

    content = parsed.get("content")
    if not isinstance(content, list):
        # No content list — might be an empty success response
        if _get_is_error(parsed):
            return "MCP tool call failed"
        return result

    if _get_is_error(parsed):
        parts = [
            _extract_text_from_content_block(item)
            for item in content
            if _extract_text_from_content_block(item)
        ]
        return "Error: " + " ".join(parts) if parts else "MCP tool call failed"

    parts = [
        _extract_text_from_content_block(item)
        for item in content
        if _extract_text_from_content_block(item)
    ]
    if parts:
        return "\n".join(parts)

    # Content blocks exist but no text extracted — return raw JSON
    return json.dumps(content)


def serialize_tool_result(result: Any) -> str:
    """
    Serialize a tool execution result to a JSON string.

    Handles various data types including:
    - MCP CallToolResult envelopes (unwrapped to text)
    - Strings (returned as-is)
    - Pydantic models (via model_dump)
    - Lists of Pydantic models
    - Objects with __dict__
    - JSON-serializable primitives

    Args:
        result: The tool execution result to serialize.

    Returns:
        str: JSON-serialized string representation of the result.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Flight(BaseModel):
        ...     airline: str
        ...     price: float
        >>>
        >>> flights = [Flight(airline="SkyHigh", price=450.0)]
        >>> serialize_tool_result(flights)
        '[{"airline": "SkyHigh", "price": 450.0}]'
    """
    # Unwrap MCP CallToolResult envelope if present.
    logger.debug(
        "serialize_tool_result input: type=%s, repr=%.500s",
        type(result).__name__,
        repr(result),
    )
    result = _unwrap_mcp_call_tool_result(result)
    logger.debug(
        "serialize_tool_result after unwrap: type=%s, repr=%.500s",
        type(result).__name__,
        repr(result),
    )

    # String results are already serialized
    if isinstance(result, str):
        return result

    try:
        # Handle lists of objects (most common case for collections)
        if isinstance(result, list):
            serialized_list = []
            for item in result:
                if hasattr(item, "model_dump"):
                    # Pydantic v2 models
                    serialized_list.append(item.model_dump())
                elif hasattr(item, "dict") and callable(item.dict):
                    # Pydantic v1 models (fallback)
                    serialized_list.append(item.dict())
                elif hasattr(item, "__dict__"):
                    # Regular objects with __dict__
                    serialized_list.append(item.__dict__)
                else:
                    # Primitive types or already serializable
                    serialized_list.append(item)
            return json.dumps(serialized_list)

        # Handle single Pydantic models
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump())

        # Fallback for Pydantic v1
        if hasattr(result, "dict") and callable(result.dict):
            return json.dumps(result.dict())

        # Handle objects with __dict__
        if hasattr(result, "__dict__"):
            return json.dumps(result.__dict__)

        # Try direct JSON serialization for primitives, dicts, lists, etc.
        return json.dumps(result)

    except (TypeError, ValueError):
        # Final fallback: convert to string
        # This handles non-JSON-serializable objects gracefully
        return str(result)

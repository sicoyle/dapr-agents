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

"""
MCPServer Middleware Workflows
==============================

Defines the Dapr workflows referenced by the ``middleware`` hooks in the
MCPServer resource YAMLs.  These are registered with the agent's Dapr
sidecar and are invoked automatically by the built-in MCP worker:

  - ``rate-limit-workflow`` (beforeCallTool): Enforces a per-tool call
    rate limit using a Dapr state store counter.
  - ``input-validation-workflow`` (beforeCallTool): Rejects tool arguments
    that contain disallowed patterns (e.g. PII, SQL injection attempts).
  - ``audit-log-workflow`` (afterCallTool): Logs every tool invocation and
    its result to a Dapr state store for audit trail.

Middleware hook contract
------------------------
- **beforeCallTool** receives::

      {"name": "<name>", "toolName": "<tool>", "arguments": {...}}

  Return ``None`` / empty to allow the call.  Raise an exception or return
  an error to abort the chain.

- **afterCallTool** receives::

      {"name": "<name>", "toolName": "<tool>", "arguments": {...}, "result": {...}}

  The return value is ignored.  Errors are logged but do not affect the
  result returned to the caller.
"""

import json
import logging
import time
from typing import Any

import dapr.ext.workflow as wf

logger = logging.getLogger("middleware-workflows")

# ---------------------------------------------------------------------------
# Rate limit workflow (beforeCallTool)
# ---------------------------------------------------------------------------

MAX_CALLS_PER_MINUTE = 30


def rate_limit_workflow(ctx: wf.DaprWorkflowContext, input: dict[str, Any]) -> None:
    """Enforce a per-tool rate limit using a state-store counter.

    If the tool has been called more than ``MAX_CALLS_PER_MINUTE`` times in
    the current minute window, raises an error to abort the call chain.
    """
    yield ctx.call_activity(
        rate_limit_check,
        input=input,
    )


def rate_limit_check(ctx: wf.WorkflowActivityContext, input: Any) -> None:
    """Activity: check rate limit counter."""
    from dapr.clients import DaprClient

    # input may arrive as a JSON string depending on SDK version.
    if isinstance(input, str):
        input = json.loads(input)

    mcp_server = input.get("name", "unknown")
    tool_name = input.get("toolName", "unknown")
    window_key = f"rate-limit:{mcp_server}:{tool_name}:{int(time.time()) // 60}"

    with DaprClient() as client:
        state = client.get_state(store_name="agentstatestore", key=window_key)
        # state = client.get_state(store_name="kvstore", key=window_key)

        count = 0
        if state.data and state.data.strip():
            try:
                count = int(state.data.decode().strip('"'))
            except (ValueError, UnicodeDecodeError):
                count = 0

        if count >= MAX_CALLS_PER_MINUTE:
            raise RuntimeError(
                f"Rate limit exceeded: {tool_name} on {mcp_server} "
                f"({count}/{MAX_CALLS_PER_MINUTE} calls this minute)"
            )

        client.save_state(
            # store_name="kvstore",
            store_name="agentstatestore",
            key=window_key,
            value=str(count + 1),
        )
        logger.info(
            "Rate limit check passed for %s.%s (%d/%d)",
            mcp_server,
            tool_name,
            count + 1,
            MAX_CALLS_PER_MINUTE,
        )


# ---------------------------------------------------------------------------
# Input validation workflow (beforeCallTool)
# ---------------------------------------------------------------------------

DISALLOWED_PATTERNS = [
    "DROP TABLE",
    "DELETE FROM",
    "'; --",
    "<script>",
    "javascript:",
]


def input_validation_workflow(
    ctx: wf.DaprWorkflowContext, input: dict[str, Any]
) -> None:
    """Reject tool arguments that contain disallowed patterns."""
    yield ctx.call_activity(
        input_validation_check,
        input=input,
    )


def input_validation_check(ctx: wf.WorkflowActivityContext, input: Any) -> None:
    """Activity: scan arguments for disallowed content."""
    if isinstance(input, str):
        input = json.loads(input)

    arguments = input.get("arguments", {})
    serialized = json.dumps(arguments).upper()

    for pattern in DISALLOWED_PATTERNS:
        if pattern.upper() in serialized:
            raise RuntimeError(
                f"Input validation failed: argument for "
                f"'{input.get('toolName', '?')}' contains disallowed "
                f"pattern '{pattern}'"
            )

    logger.info(
        "Input validation passed for %s.%s",
        input.get("name", "unknown"),
        input.get("toolName", "unknown"),
    )


# ---------------------------------------------------------------------------
# Audit log workflow (afterCallTool)
# ---------------------------------------------------------------------------


def audit_log_workflow(ctx: wf.DaprWorkflowContext, input: dict[str, Any]) -> None:
    """Write an audit record for every tool invocation."""
    yield ctx.call_activity(
        audit_log_write,
        input=input,
    )


def audit_log_write(ctx: wf.WorkflowActivityContext, input: Any) -> None:
    """Activity: persist an audit record to the state store."""
    from dapr.clients import DaprClient

    if isinstance(input, str):
        input = json.loads(input)

    mcp_server = input.get("name", "unknown")
    tool_name = input.get("toolName", "unknown")
    timestamp = int(time.time())
    audit_key = f"audit:{mcp_server}:{tool_name}:{timestamp}"

    record = {
        "name": mcp_server,
        "toolName": tool_name,
        "arguments": input.get("arguments"),
        "result": input.get("result"),
        "timestamp": timestamp,
    }

    with DaprClient() as client:
        client.save_state(
            store_name="agentstatestore",
            # store_name="kvstore",
            key=audit_key,
            value=json.dumps(record),
        )

    logger.info(
        "Audit record written: %s -> %s.%s",
        audit_key,
        mcp_server,
        tool_name,
    )

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
Utility to convert SDK :class:`~dapr.ext.workflow.mcp.MCPToolDef` instances
into dapr-agents :class:`~dapr_agents.tool.workflow.tool_context.WorkflowContextInjectedTool`
objects that can be used by :class:`~dapr_agents.agents.durable.DurableAgent`.
"""

import logging
from typing import Any, Optional, Type

from pydantic import BaseModel

from dapr.ext.workflow import MCPToolDef, create_pydantic_model_from_schema

from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool

logger = logging.getLogger(__name__)


def mcp_tool_def_to_workflow_tool(
    tool_def: MCPToolDef,
) -> WorkflowContextInjectedTool:
    """Convert an :class:`MCPToolDef` from the Dapr SDK into a
    :class:`WorkflowContextInjectedTool` for use with ``DurableAgent``.

    The returned tool schedules ``tool_def.call_tool_workflow`` as a child
    workflow when invoked from within a Dapr orchestrator context.

    Args:
        tool_def: A framework-agnostic MCP tool definition from
            :meth:`~dapr.ext.workflow.mcp.DaprMCPClient.get_all_tools`.

    Returns:
        A :class:`WorkflowContextInjectedTool` ready for registration
        on an :class:`~dapr_agents.tool.executor.AgentToolExecutor`.
    """
    args_model: Optional[Type[BaseModel]] = None
    if tool_def.input_schema:
        try:
            args_model = create_pydantic_model_from_schema(
                tool_def.input_schema, f"{tool_def.name}Args"
            )
        except Exception as exc:
            logger.warning(
                "Could not build args model for tool '%s': %s — "
                "tool will accept unvalidated **kwargs",
                tool_def.name,
                exc,
            )

    wf_name = tool_def.call_tool_workflow

    def _executor(
        ctx: Any,
        _source_agent: Optional[str] = None,
        _child_instance_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Schedule ``dapr.internal.mcp.<server>.CallTool.<tool>`` as a child workflow.

        The tool name is encoded in the workflow name suffix, so the payload
        only needs to carry the arguments.
        """
        payload = {"arguments": kwargs}
        if _child_instance_id:
            return ctx.call_child_workflow(
                workflow=wf_name,
                input=payload,
                instance_id=_child_instance_id,
            )
        return ctx.call_child_workflow(workflow=wf_name, input=payload)

    return WorkflowContextInjectedTool(
        name=tool_def.name,
        description=tool_def.description,
        func=_executor,
        args_model=args_model,
    )

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

import asyncio
import logging

import dapr.ext.workflow as wf

from dapr_agents import AgentRunner, DurableAgent

from middleware_workflows import (
    audit_log_workflow,
    audit_log_write,
    input_validation_check,
    input_validation_workflow,
    rate_limit_check,
    rate_limit_workflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcpserver-all-transports")

# MCPServer tools are auto-discovered from the sidecar metadata.
# To manually control which servers and tools are loaded:

#   from dapr.ext.workflow import DaprMCPClient
#   from dapr_agents.tool.mcp import mcp_tool_def_to_workflow_tool
#   client = DaprMCPClient(timeout_in_seconds=30)
#   client.connect("dapr-mcp")
#   client.connect("local-tools")
#   client.connect("remote-weather")
#   tools = [mcp_tool_def_to_workflow_tool(t) for t in client.get_all_tools()]
#   agent = DurableAgent(name="MultiTransportAgent", tools=tools, runtime=workflow_runtime, ...)


async def main() -> None:
    # ------------------------------------------------------------------
    # 0. Create a single WorkflowRuntime shared between middleware and
    #    the DurableAgent.  The sidecar only accepts one gRPC worker
    #    stream, so everything must be on the same runtime.
    # ------------------------------------------------------------------
    workflow_runtime = wf.WorkflowRuntime()

    # Register middleware workflows — the sidecar's MCP worker invokes
    # these by name when middleware hooks are configured on MCPServer
    # resources.
    workflow_runtime.register_workflow(rate_limit_workflow)
    workflow_runtime.register_activity(rate_limit_check)
    workflow_runtime.register_workflow(input_validation_workflow)
    workflow_runtime.register_activity(input_validation_check)
    workflow_runtime.register_workflow(audit_log_workflow)
    workflow_runtime.register_activity(audit_log_write)

    logger.info("Middleware workflows registered on shared WorkflowRuntime.")

    # ------------------------------------------------------------------
    # 1. Create agent — MCPServer tools are auto-discovered from the
    #    sidecar metadata.  The shared runtime carries middleware workflows.
    # ------------------------------------------------------------------
    agent = DurableAgent(
        name="MultiTransportAgent",
        role="Multi-tool assistant",
        goal=(
            "Answer user questions by leveraging tools from multiple MCP "
            "servers: Dapr infrastructure tools, local dev utilities, and "
            "a remote weather service."
        ),
        instructions=[
            "Use get_components to discover available Dapr components first.",
            "Use search_files and summarize_text for local file operations.",
            "Use get_weather and get_forecast for weather questions.",
            "Always explain which MCP server / tool you are using.",
        ],
        runtime=workflow_runtime,
    )

    # ------------------------------------------------------------------
    # 2. Run the agent.
    # ------------------------------------------------------------------
    try:
        async with AgentRunner() as runner:
            await runner.run(
                agent,
                payload={
                    "task": (
                        "First, list all available Dapr components. "
                        "Then tell me the weather in Tokyo. "
                        "Finally, search for Python files in /tmp and "
                        "summarize any README you find."
                    )
                },
                wait=True,
            )
    finally:
        workflow_runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

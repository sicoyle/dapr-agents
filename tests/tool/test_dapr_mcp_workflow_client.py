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

"""Unit tests for DaprMCPWorkflowClient.

All tests mock the DaprWorkflowClient so no running Dapr sidecar is required.
"""

import json
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool

_FAKE_TOOLS = [
    {
        "name": "greet",
        "description": "Return a greeting.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name to greet"}},
            "required": ["name"],
        },
    },
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
]

_LIST_TOOLS_RESULT = {"tools": _FAKE_TOOLS}


def _make_completed_state(output: dict) -> SimpleNamespace:
    """Return a fake WorkflowState-like object with COMPLETED status."""
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_output=json.dumps(output),
    )


def _make_failed_state(status_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name=status_name),
        serialized_output=None,
    )


def _make_client(wf_client_mock):
    """Create a DaprMCPWorkflowClient with the Dapr SDK mocked out."""
    with patch(
        "dapr_agents.tool.mcp.dapr_workflow_client.DaprMCPWorkflowClient.model_post_init"
    ):
        from dapr_agents.tool.mcp.dapr_workflow_client import DaprMCPWorkflowClient

        client = DaprMCPWorkflowClient()
        client._wf_client = wf_client_mock
        return client


class TestConnect:
    @pytest.mark.asyncio
    async def test_schedules_list_tools_workflow(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_completed_state(
            _LIST_TOOLS_RESULT
        )
        client = _make_client(wf)

        await client.connect("my-server")

        call_kwargs = wf.schedule_new_workflow.call_args.kwargs
        assert call_kwargs["workflow"].__name__ == "dapr.internal.mcp.my-server.ListTools"
        assert call_kwargs["input"] == {"name": "my-server"}
        assert "instance_id" in call_kwargs  # a UUID was generated

    @pytest.mark.asyncio
    async def test_caches_tools_after_connect(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_completed_state(
            _LIST_TOOLS_RESULT
        )
        client = _make_client(wf)

        await client.connect("my-server")

        tools = client.get_server_tools("my-server")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"greet", "add"}

    @pytest.mark.asyncio
    async def test_all_returned_tools_are_workflow_context_injected(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_completed_state(
            _LIST_TOOLS_RESULT
        )
        client = _make_client(wf)

        await client.connect("my-server")

        for tool in client.get_all_tools():
            assert isinstance(tool, WorkflowContextInjectedTool)

    @pytest.mark.asyncio
    async def test_tool_descriptions_preserved(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_completed_state(
            _LIST_TOOLS_RESULT
        )
        client = _make_client(wf)
        await client.connect("my-server")

        greet = next(t for t in client.get_all_tools() if t.name == "greet")
        assert greet.description == "Return a greeting."

    @pytest.mark.asyncio
    async def test_allowed_tools_filter(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_completed_state(
            _LIST_TOOLS_RESULT
        )

        with patch(
            "dapr_agents.tool.mcp.dapr_workflow_client.DaprMCPWorkflowClient.model_post_init"
        ):
            from dapr_agents.tool.mcp.dapr_workflow_client import (
                DaprMCPWorkflowClient,
            )

            client = DaprMCPWorkflowClient(allowed_tools={"greet"})
            client._wf_client = wf

        await client.connect("my-server")

        tools = client.get_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "greet"

    @pytest.mark.asyncio
    async def test_multiple_servers_accumulated(self):
        wf = MagicMock()

        def _state_for(server_name):
            tools = [{"name": f"{server_name}_tool", "description": ""}]
            return _make_completed_state({"tools": tools})

        wf.wait_for_workflow_completion.side_effect = lambda **kw: _state_for(
            wf.schedule_new_workflow.call_args.kwargs["input"]["name"]
        )

        client = _make_client(wf)
        await client.connect("server-a")
        await client.connect("server-b")

        assert set(client.get_connected_servers()) == {"server-a", "server-b"}
        assert len(client.get_all_tools()) == 2

    @pytest.mark.asyncio
    async def test_timeout_raises_runtime_error(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = None  # simulates timeout

        client = _make_client(wf)

        with pytest.raises(RuntimeError, match="timed out"):
            await client.connect("my-server")

    @pytest.mark.asyncio
    async def test_failed_workflow_raises_runtime_error(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_failed_state("FAILED")

        client = _make_client(wf)

        with pytest.raises(RuntimeError, match="FAILED"):
            await client.connect("my-server")

    @pytest.mark.asyncio
    async def test_empty_tools_list_is_handled(self):
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_completed_state(
            {"tools": []}
        )
        client = _make_client(wf)
        await client.connect("my-server")

        assert client.get_all_tools() == []


class TestToolExecution:
    def _connected_client(self):
        """Synchronously pre-populate a client's tool cache (no I/O needed)."""
        with patch(
            "dapr_agents.tool.mcp.dapr_workflow_client.DaprMCPWorkflowClient.model_post_init"
        ):
            from dapr_agents.tool.mcp.dapr_workflow_client import (
                DaprMCPWorkflowClient,
            )

            client = DaprMCPWorkflowClient()
            client._wf_client = MagicMock()
            # Pre-populate the cache directly.
            for tool_def in _FAKE_TOOLS:
                tool = client._make_tool("my-server", tool_def)
                client._server_tools.setdefault("my-server", []).append(tool)
        return client

    def test_call_tool_schedules_call_tool_workflow(self):
        client = self._connected_client()
        ctx = MagicMock()

        greet = next(t for t in client.get_all_tools() if t.name == "greet")
        greet(ctx=ctx, name="Dapr")

        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.mcp.my-server.CallTool",
            input={
                "name": "my-server",
                "tool": "greet",
                "arguments": {"name": "Dapr"},
            },
        )

    def test_call_tool_with_multiple_args(self):
        client = self._connected_client()
        ctx = MagicMock()

        add = next(t for t in client.get_all_tools() if t.name == "add")
        add(ctx=ctx, a=3, b=7)

        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.mcp.my-server.CallTool",
            input={
                "name": "my-server",
                "tool": "add",
                "arguments": {"a": 3, "b": 7},
            },
        )

    def test_child_instance_id_forwarded(self):
        client = self._connected_client()
        ctx = MagicMock()

        greet = next(t for t in client.get_all_tools() if t.name == "greet")
        greet(ctx=ctx, name="Dapr", _child_instance_id="my-id-123")

        ctx.call_child_workflow.assert_called_once_with(
            workflow="dapr.mcp.my-server.CallTool",
            input={
                "name": "my-server",
                "tool": "greet",
                "arguments": {"name": "Dapr"},
            },
            instance_id="my-id-123",
        )

    def test_ctx_not_in_llm_schema(self):
        client = self._connected_client()
        greet = next(t for t in client.get_all_tools() if t.name == "greet")
        schema = greet.to_function_call()
        props = schema["function"]["parameters"]["properties"]
        assert "ctx" not in props
        assert "name" in props

    def test_tool_missing_ctx_raises(self):
        from dapr_agents.types import ToolError

        client = self._connected_client()
        greet = next(t for t in client.get_all_tools() if t.name == "greet")

        with pytest.raises(ToolError, match="workflow context"):
            greet(name="Dapr")  # ctx omitted


class TestMCPClientDeprecation:
    def test_mcp_client_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from dapr_agents.tool.mcp.client import MCPClient

            MCPClient()

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "DaprMCPWorkflowClient" in str(dep_warnings[0].message)

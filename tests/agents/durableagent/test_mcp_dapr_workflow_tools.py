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

"""Tests for DurableAgent + DaprMCPWorkflowClient integration.

Covers:
- DurableAgent workflow/activity registration invariants when MCP tools are present
- Tool type guarantees (WorkflowContextInjectedTool) for MCP vs regular tools
- Mixed tool sets (MCP + plain AgentTool)
- get_server_tools / get_all_tools multi-server behaviour
- Failure scenarios not already covered in test_dapr_mcp_workflow_client.py

No live Dapr sidecar required; all Dapr SDK calls are mocked.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import AgentWorkflowEntry
from dapr_agents.tool.base import AgentTool
from dapr.ext.workflow import create_pydantic_model_from_schema
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.types import ToolError

_MCP_TOOLS = [
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
]

_WEATHER_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "inputSchema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
]

def _make_completed_state(output: dict) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name="COMPLETED"),
        serialized_output=json.dumps(output),
    )


def _make_failed_state(status_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_status=SimpleNamespace(name=status_name),
        serialized_output=None,
    )


def _make_mcp_client(server_name: str = "math-server", tool_defs: list = None):
    """Create a DaprMCPWorkflowClient with pre-populated tool cache (no I/O).

    Note: We do NOT patch model_post_init because doing so prevents Pydantic from
    initialising the _server_tools PrivateAttr.  The conftest already mocks
    dapr.ext.workflow.DaprWorkflowClient, so model_post_init runs safely and we
    simply replace _wf_client with a proper MagicMock afterward.
    """
    if tool_defs is None:
        tool_defs = _MCP_TOOLS
    from dapr_agents.tool.mcp.dapr_workflow_client import DaprMCPWorkflowClient

    client = DaprMCPWorkflowClient()
    client._wf_client = MagicMock()
    for td in tool_defs:
        tool = client._make_tool(server_name, td)
        client._server_tools.setdefault(server_name, []).append(tool)
    return client


def _make_regular_tool() -> AgentTool:
    """A plain AgentTool (not WorkflowContextInjectedTool)."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    args_model = create_pydantic_model_from_schema(schema, "EchoArgs")
    return AgentTool(
        name="echo",
        description="Echo the input text.",
        args_model=args_model,
        func=lambda text: text,
    )


def _make_agent(tools: list, name: str = "TestAgent") -> DurableAgent:
    from dapr_agents.agents.configs import AgentPubSubConfig, AgentStateConfig
    from dapr_agents.storage.daprstores.stateservice import StateStoreService

    return DurableAgent(
        name=name,
        role="Test Assistant",
        goal="Run tests",
        instructions=["Follow test instructions"],
        tools=tools,
        pubsub=AgentPubSubConfig(pubsub_name="testpubsub"),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
    )

# autouse fixture: suppress Dapr state-store I/O for all tests in this file
@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    monkeypatch.setattr(DurableAgent, "save_state", lambda self: None)

    import dapr_agents.storage.daprstores.statestore as statestore

    class MockDaprClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def save_state(self, *args, **kwargs):
            pass

        def get_state(self, *args, **kwargs):
            class R:
                data = "{}"
                etag = "etag"

            return R()

        def execute_state_transaction(self, *args, **kwargs):
            pass

        def get_metadata(self):
            response = MagicMock()
            response.registered_components = []
            response.application_id = "test-app-id"
            return response

    statestore.DaprClient = MockDaprClient
    monkeypatch.setattr(
        DurableAgent, "register_agentic_system", lambda self: None
    )
    yield

class TestWorkflowRegistration:
    """DurableAgent must register consistent workflow + activity sets."""

    def _registered_workflow_names(self, agent: DurableAgent) -> set:
        mock_runtime = MagicMock()
        agent.register_workflows(mock_runtime)
        return {
            call.args[0].__name__
            for call in mock_runtime.register_workflow.call_args_list
            if call.args
        }

    def _registered_activity_names(self, agent: DurableAgent) -> set:
        mock_runtime = MagicMock()
        agent.register_workflows(mock_runtime)
        return {
            call.args[0].__name__
            for call in mock_runtime.register_activity.call_args_list
            if call.args
        }

    def test_agent_workflow_registered_with_mcp_tools(self):
        client = _make_mcp_client()
        agent = _make_agent(client.get_all_tools())
        wf_names = self._registered_workflow_names(agent)
        assert agent.agent_workflow_name in wf_names

    def test_agent_workflow_registered_with_no_tools(self):
        agent = _make_agent([])
        wf_names = self._registered_workflow_names(agent)
        assert agent.agent_workflow_name in wf_names

    def test_agent_workflow_registered_with_mixed_tools(self):
        client = _make_mcp_client()
        agent = _make_agent(client.get_all_tools() + [_make_regular_tool()])
        wf_names = self._registered_workflow_names(agent)
        assert agent.agent_workflow_name in wf_names

    def test_run_tool_activity_always_registered(self):
        client = _make_mcp_client()
        agent = _make_agent(client.get_all_tools())
        activity_names = self._registered_activity_names(agent)
        assert "run_tool" in activity_names

    def test_call_llm_activity_always_registered(self):
        client = _make_mcp_client()
        agent = _make_agent(client.get_all_tools())
        activity_names = self._registered_activity_names(agent)
        assert "call_llm" in activity_names

    def test_all_core_activities_registered(self):
        client = _make_mcp_client()
        agent = _make_agent(client.get_all_tools())
        activity_names = self._registered_activity_names(agent)
        core = {
            "record_initial_entry",
            "call_llm",
            "run_tool",
            "save_tool_results",
            "finalize_workflow",
            "get_team_members",
            "load_tools",
        }
        missing = core - activity_names
        assert not missing, f"Missing activities: {missing}"

    def test_registration_count_same_with_or_without_mcp_tools(self):
        """MCP tools must not affect how many workflows/activities are registered."""
        mock_a = MagicMock()
        mock_b = MagicMock()

        agent_no_tools = _make_agent([])
        agent_no_tools.register_workflows(mock_a)

        client = _make_mcp_client()
        agent_mcp = _make_agent(client.get_all_tools())
        agent_mcp.register_workflows(mock_b)

        assert mock_a.register_workflow.call_count == mock_b.register_workflow.call_count
        assert mock_a.register_activity.call_count == mock_b.register_activity.call_count
        

class TestAllowedToolsFiltering:
    """allowed_tools restricts which tools are loaded during connect()."""

    def _connect_with_filter(self, allowed: set):
        """Run connect() with a mocked ListTools response containing add + multiply."""
        from dapr_agents.tool.mcp.dapr_workflow_client import DaprMCPWorkflowClient
        import asyncio

        client = DaprMCPWorkflowClient(allowed_tools=allowed)
        client._wf_client = MagicMock()

        completed_state = _make_completed_state({"tools": _MCP_TOOLS})

        with patch.object(client, "_run_list_tools", return_value=completed_state):
            asyncio.run(client.connect("math-server"))

        return client

    def test_allowed_tools_single_name_keeps_one_tool(self):
        client = self._connect_with_filter({"add"})
        names = {t.name for t in client.get_all_tools()}
        assert names == {"Add"}

    def test_allowed_tools_multiple_names_keeps_subset(self):
        """Allowing both names is identical to no filter for this tool set."""
        client = self._connect_with_filter({"add", "multiply"})
        names = {t.name for t in client.get_all_tools()}
        assert names == {"Add", "Multiply"}

    def test_allowed_tools_empty_set_keeps_no_tools(self):
        client = self._connect_with_filter(set())
        assert client.get_all_tools() == []

    def test_allowed_tools_none_keeps_all_tools(self):
        """None (default) means no filtering."""
        client = self._connect_with_filter(None)
        names = {t.name for t in client.get_all_tools()}
        assert names == {"Add", "Multiply"}

    def test_allowed_tools_unknown_name_keeps_no_tools(self):
        client = self._connect_with_filter({"does_not_exist"})
        assert client.get_all_tools() == []

    def test_allowed_tools_filter_uses_raw_mcp_name(self):
        """Filter is applied to the raw MCP name (snake_case), not the sanitized name."""
        # "add" is the raw name; "Add" is the sanitized name.
        # allowed_tools must use the raw name to match.
        client = self._connect_with_filter({"add"})
        assert len(client.get_all_tools()) == 1
        assert client.get_all_tools()[0].name == "Add"  # stored name is sanitized

class TestMCPToolTypes:
    """Tools from DaprMCPWorkflowClient must be WorkflowContextInjectedTool instances."""

    def test_mcp_tools_are_workflow_context_injected(self):
        client = _make_mcp_client()
        for tool in client.get_all_tools():
            assert isinstance(tool, WorkflowContextInjectedTool), (
                f"Expected WorkflowContextInjectedTool, got {type(tool)} for '{tool.name}'"
            )

    def test_regular_tool_is_not_workflow_context_injected(self):
        regular = _make_regular_tool()
        assert not isinstance(regular, WorkflowContextInjectedTool)

    def test_mcp_tool_schema_excludes_ctx(self):
        client = _make_mcp_client()
        for tool in client.get_all_tools():
            schema = tool.to_function_call()
            props = schema["function"]["parameters"]["properties"]
            assert "ctx" not in props, f"'ctx' must not appear in LLM schema for '{tool.name}'"

    def test_mcp_tool_schema_excludes_hidden_kwargs(self):
        client = _make_mcp_client()
        for tool in client.get_all_tools():
            schema = tool.to_function_call()
            props = schema["function"]["parameters"]["properties"]
            assert "_child_instance_id" not in props
            assert "_source_agent" not in props

    def test_mcp_tool_schema_contains_declared_params(self):
        client = _make_mcp_client()
        add = next(t for t in client.get_all_tools() if t.name == "Add")
        props = add.to_function_call()["function"]["parameters"]["properties"]
        assert "a" in props
        assert "b" in props

    def test_mcp_tool_description_preserved(self):
        client = _make_mcp_client()
        add = next(t for t in client.get_all_tools() if t.name == "Add")
        assert add.description == "Add two integers."

class TestMixedToolSet:
    """Agents must work correctly when tool list contains both tool types."""

    def test_agent_registers_both_tool_types_in_executor(self):
        client = _make_mcp_client()
        regular = _make_regular_tool()
        agent = _make_agent(client.get_all_tools() + [regular])

        tool_names = {t.name for t in agent.tool_executor.tools}
        assert "Add" in tool_names
        assert "Multiply" in tool_names
        assert "Echo" in tool_names

    def test_duplicate_tool_name_keeps_first_and_warns(self, caplog):
        """When two servers expose a tool with the same name, the first registration
        wins and a WARNING is emitted — no exception is raised."""
        import logging

        client1 = _make_mcp_client("server-a", _MCP_TOOLS)  # Add, Multiply
        client2 = _make_mcp_client("server-b", _MCP_TOOLS)  # Add, Multiply (duplicates)

        with caplog.at_level(logging.WARNING, logger="dapr_agents.tool.executor"):
            agent = _make_agent(client1.get_all_tools() + client2.get_all_tools())

        # Only 2 tools registered (first occurrence each)
        tool_names = {t.name for t in agent.tool_executor.tools}
        assert tool_names == {"Add", "Multiply"}

        # Warning logged for each duplicate
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Add" in w or "Duplicate" in w for w in warnings)

    def test_mcp_tools_remain_workflow_context_injected_in_agent(self):
        client = _make_mcp_client()
        regular = _make_regular_tool()
        agent = _make_agent(client.get_all_tools() + [regular])

        for tool in agent.tool_executor.tools:
            if tool.name in {"Add", "Multiply"}:
                assert isinstance(tool, WorkflowContextInjectedTool)
            elif tool.name == "Echo":
                assert not isinstance(tool, WorkflowContextInjectedTool)

    def test_run_tool_activity_executes_regular_tool(self):
        """run_tool activity can execute a regular (non-MCP) tool."""
        regular = _make_regular_tool()
        agent = _make_agent([regular])

        entry = AgentWorkflowEntry(
            source=None,
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        mock_ctx = Mock()
        with (
            patch.object(agent._infra, "load_state"),
            patch.object(agent, "save_state"),
        ):
            result = agent.run_tool(
                mock_ctx,
                {
                    "instance_id": "test-123",
                    "tool_call": {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"text": "hello"}',
                        },
                    },
                },
            )

        assert result["tool_call_id"] == "call_abc"
        assert result["name"] == "echo"


class TestGetServerTools:
    """get_server_tools returns only that server's tools; get_all_tools returns all."""

    def _two_server_client(self):
        client = _make_mcp_client("math-server", _MCP_TOOLS)
        # Add a second server's tools directly into the same client
        for td in _WEATHER_TOOLS:
            tool = client._make_tool("weather-server", td)
            client._server_tools.setdefault("weather-server", []).append(tool)
        return client

    def test_get_server_tools_returns_only_named_server(self):
        client = self._two_server_client()
        math_tools = client.get_server_tools("math-server")
        names = {t.name for t in math_tools}
        assert names == {"Add", "Multiply"}
        assert "GetWeather" not in names

    def test_get_server_tools_weather_server(self):
        client = self._two_server_client()
        weather_tools = client.get_server_tools("weather-server")
        names = {t.name for t in weather_tools}
        assert names == {"GetWeather"}

    def test_get_all_tools_returns_all_servers(self):
        client = self._two_server_client()
        all_names = {t.name for t in client.get_all_tools()}
        assert all_names == {"Add", "Multiply", "GetWeather"}

    def test_get_connected_servers_lists_both(self):
        client = self._two_server_client()
        assert set(client.get_connected_servers()) == {"math-server", "weather-server"}

    def test_get_server_tools_unknown_server_returns_empty(self):
        client = _make_mcp_client()
        assert client.get_server_tools("does-not-exist") == []


class TestFailureScenarios:
    """Additional failure paths for ListTools, CallTool, and agent run_tool."""

    # -- ListTools: non-FAILED terminal statuses --

    @pytest.mark.asyncio
    async def test_terminated_status_raises_runtime_error(self):
        from dapr_agents.tool.mcp.dapr_workflow_client import DaprMCPWorkflowClient

        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_failed_state("TERMINATED")
        client = DaprMCPWorkflowClient()
        client._wf_client = wf

        with pytest.raises(RuntimeError, match="TERMINATED"):
            await client.connect("math-server")

    @pytest.mark.asyncio
    async def test_pending_status_raises_runtime_error(self):
        from dapr_agents.tool.mcp.dapr_workflow_client import DaprMCPWorkflowClient

        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = _make_failed_state("PENDING")
        client = DaprMCPWorkflowClient()
        client._wf_client = wf

        with pytest.raises(RuntimeError, match="PENDING"):
            await client.connect("math-server")

    @pytest.mark.asyncio
    async def test_malformed_json_output_raises(self):
        """Non-JSON serialized_output should surface an error to the caller."""
        from dapr_agents.tool.mcp.dapr_workflow_client import DaprMCPWorkflowClient

        bad_state = SimpleNamespace(
            runtime_status=SimpleNamespace(name="COMPLETED"),
            serialized_output="THIS IS NOT JSON {{{",
        )
        wf = MagicMock()
        wf.wait_for_workflow_completion.return_value = bad_state
        client = DaprMCPWorkflowClient()
        client._wf_client = wf

        with pytest.raises(Exception):  # json.JSONDecodeError or RuntimeError
            await client.connect("math-server")

    # -- CallTool: invalid arguments caught before child workflow --

    def test_call_tool_invalid_args_raises_validation_error(self):
        """Passing args that don't match the tool schema must raise before child workflow."""
        client = _make_mcp_client()
        ctx = MagicMock()

        add = next(t for t in client.get_all_tools() if t.name == "Add")

        with pytest.raises((ValidationError, ToolError)):
            add(ctx=ctx, a="definitely-not-an-int", b="also-not")

        # The child workflow must NOT have been called since validation failed
        ctx.call_child_workflow.assert_not_called()

    # -- CallTool: exception from child workflow propagates --

    def test_call_tool_child_workflow_exception_propagates(self):
        client = _make_mcp_client()
        ctx = MagicMock()
        ctx.call_child_workflow.side_effect = RuntimeError("sidecar unavailable")

        add = next(t for t in client.get_all_tools() if t.name == "Add")

        with pytest.raises((RuntimeError, ToolError), match="sidecar unavailable"):
            add(ctx=ctx, a=1, b=2)

    # -- Agent-level: run_tool activity always returns a ToolMessage --

    def test_run_tool_returns_error_tool_message_on_tool_exception(self):
        """run_tool must return a ToolMessage dict even when the tool raises."""

        # Build an agent with a tool whose func raises
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        args_model = create_pydantic_model_from_schema(schema, "BoomArgs")

        def boom(x: int):
            raise ValueError("boom!")

        boom_tool = AgentTool(
            name="boom",
            description="Always fails.",
            args_model=args_model,
            func=boom,
        )
        agent = _make_agent([boom_tool])

        entry = AgentWorkflowEntry(
            source=None,
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        mock_ctx = Mock()
        with (
            patch.object(agent._infra, "load_state"),
            patch.object(agent, "save_state"),
        ):
            result = agent.run_tool(
                mock_ctx,
                {
                    "instance_id": "test-fail",
                    "tool_call": {
                        "id": "call_fail",
                        "type": "function",
                        "function": {"name": "boom", "arguments": '{"x": 1}'},
                    },
                },
            )

        # Must return a dict (ToolMessage) — never re-raise
        assert isinstance(result, dict)
        assert result["tool_call_id"] == "call_fail"
        assert result["name"] == "boom"
        # Content should contain some indication of the error
        assert result.get("content") is not None

    def test_run_tool_unknown_tool_name_returns_error_tool_message(self):
        """run_tool with an unknown tool name must not crash the orchestrator."""
        agent = _make_agent([])

        entry = AgentWorkflowEntry(
            source=None,
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        mock_ctx = Mock()
        with (
            patch.object(agent._infra, "load_state"),
            patch.object(agent, "save_state"),
        ):
            result = agent.run_tool(
                mock_ctx,
                {
                    "instance_id": "test-unknown",
                    "tool_call": {
                        "id": "call_unknown",
                        "type": "function",
                        "function": {
                            "name": "nonexistent_tool",
                            "arguments": "{}",
                        },
                    },
                },
            )

        assert isinstance(result, dict)
        assert result["tool_call_id"] == "call_unknown"


# ---------------------------------------------------------------------------
# TestCallToolResultSerialization
# Verifies that serialize_tool_result handles every realistic shape that the
# dapr.mcp.<name>.CallTool child workflow might return.
#
# Context: WorkflowContextInjectedTool results bypass run_tool entirely.
# The orchestrator receives the raw child-workflow return value and passes it
# directly to serialize_tool_result() before wrapping it in a ToolMessage.
# This is the only place that transformation is tested.
# ---------------------------------------------------------------------------


class TestCallToolResultSerialization:
    """serialize_tool_result handles every realistic CallTool return shape."""

    def _serialize(self, value):
        from dapr_agents.tool.utils.serialization import serialize_tool_result

        return serialize_tool_result(value)

    # -- Plain string (most common: sidecar returns text content directly) --

    def test_string_result_returned_as_is(self):
        assert self._serialize("sunny, 72°F") == "sunny, 72°F"

    def test_empty_string_result(self):
        assert self._serialize("") == ""

    # -- Dict result (structured JSON from CallTool workflow) --

    def test_dict_result_serializes_to_json(self):
        result = {"content": [{"type": "text", "text": "sunny, 72°F"}]}
        serialized = self._serialize(result)
        assert json.loads(serialized) == result

    def test_nested_dict_result_round_trips(self):
        result = {"result": 42, "unit": "°F", "location": "Seattle"}
        assert json.loads(self._serialize(result)) == result

    # -- None / null --

    def test_none_result_serializes_to_null(self):
        assert self._serialize(None) == "null"

    # -- Numeric primitives --

    def test_integer_result_serializes(self):
        assert self._serialize(5) == "5"

    def test_float_result_serializes(self):
        assert self._serialize(3.14) == "3.14"

    # -- List result (multi-content block shape) --

    def test_list_of_dicts_serializes(self):
        result = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert json.loads(self._serialize(result)) == result

    def test_empty_list_serializes(self):
        assert self._serialize([]) == "[]"

    # -- Ensures ToolMessage.content is always a non-None str --

    def test_serialized_result_is_always_str(self):
        for value in ("text", 42, None, {"k": "v"}, [1, 2], True):
            result = self._serialize(value)
            assert isinstance(result, str), f"Expected str for {value!r}, got {type(result)}"

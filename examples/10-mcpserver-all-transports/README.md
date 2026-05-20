<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# MCPServer All-Transports Example

A single `DurableAgent` that connects to **three** `MCPServer` resources — one
per transport type — showcasing how Dapr Agents integrates with MCP servers
through the sidecar's built-in workflow orchestrations.

## MCPServer Resources

| Resource | Transport | Scenario | Tools |
|---|---|---|---|
| `dapr-mcp` | `streamableHTTP` | [dapr-mcp-server](https://github.com/dapr/dapr-mcp-server) running in Kubernetes | `get_components`, `save_state`, `get_state`, `publish_event`, `invoke_service`, ... |
| `local-tools` | `stdio` | Local subprocess spawned by the sidecar | `search_files`, `summarize_text` |
| `remote-weather` | `sse` | External weather service (simulated locally) | `get_weather`, `get_forecast` |

## Middleware Hooks

Every MCPServer resource in this example defines middleware pipelines that the
sidecar executes around tool calls:

| Workflow | Hook Type | Purpose |
|---|---|---|
| `rate_limit_workflow` | `beforeCallTool` | Per-tool call rate limiting (30/min) using a state store counter |
| `input_validation_workflow` | `beforeCallTool` | Rejects arguments containing SQL injection or XSS patterns |
| `audit_log_workflow` | `beforeCallTool` / `afterCallTool` | Writes every tool invocation and result to the state store |

See [`middleware_workflows.py`](middleware_workflows.py) for the workflow
implementations and [`resources/`](resources/) for how they are wired into the
MCPServer specs.

## Prerequisites

- **Redis** running on `localhost:6379`
- **Dapr CLI** installed with `dapr init` completed
- **`OPENAI_API_KEY`** environment variable set
- **Python 3.11+** with `uv` (recommended) or `pip`

## Setup

```bash
# Install dependencies (from this directory)
uv sync

# Or with pip
pip install -e "../../[dev]"
```

## Run (Standalone / Local Dev)

### 1. Start the remote weather SSE server

This simulates an externally-hosted MCP server:

```bash
python weather_sse_server.py
# Listening on http://0.0.0.0:8081/sse
```

### 2. Launch the agent with Dapr

```bash
dapr run \
  --app-id mcp-agent \
  --resources-path ./resources \
  -- python agent.py
```

The sidecar will:
1. Load all three `MCPServer` resources from `resources/`
2. Spawn the `local-tools` stdio subprocess automatically
3. Register `dapr.internal.mcp.<server>.ListTools` and `dapr.internal.mcp.<server>.CallTool.<tool>` workflows for each
4. Execute middleware hooks on every tool call

## Kubernetes Deployment (dapr-mcp-server)

For the `dapr-mcp` resource to work, the
[dapr-mcp-server](https://github.com/dapr/dapr-mcp-server) must be running in
your cluster:

```bash
# Build and deploy (from the dapr-mcp-server repo)
docker build -t dapr-mcp-server:latest .
kubectl apply -f deploy/

# Verify it's running
kubectl get pods -l app=dapr-mcp-server
```

The `resources/dapr-mcp-server.yaml` points at the in-cluster Service URL:

```yaml
endpoint:
  streamableHTTP:
    url: http://dapr-mcp-server.default.svc.cluster.local:8080
```

Update the URL if deploying to a different namespace.

## Architecture

```
                        dapr run --app-id mcp-agent
                               |
              +----------------+----------------+
              |          Dapr sidecar           |
              |                                 |
              |  MCPServer "dapr-mcp"           |
              |    streamableHTTP -> K8s svc    |
              |    middleware: audit-log         |
              |                                 |
              |  MCPServer "local-tools"        |
              |    stdio -> python subprocess   |
              |    middleware: rate-limit        |
              |                                 |
              |  MCPServer "remote-weather"     |
              |    sse -> http://...:8081/sse   |
              |    middleware: rate-limit,       |
              |      input-validation, audit    |
              +----------------+----------------+
                               |
                    +----------+----------+
                    |   DurableAgent      |
                    |   (agent.py)        |
                    |                     |
                    |   DaprMCPWorkflow   |
                    |   Client.connect()  |
                    |   -> all 3 servers  |
                    +---------------------+
```

## File Overview

| File | Purpose |
|---|---|
| [`agent.py`](agent.py) | Main agent — connects to all MCPServers and runs a multi-tool task |
| [`weather_sse_server.py`](weather_sse_server.py) | SSE weather server simulating a remote third-party service |
| [`local_tools_server.py`](local_tools_server.py) | Stdio server for local dev tools (spawned by sidecar) |
| [`middleware_workflows.py`](middleware_workflows.py) | Rate limit, input validation, and audit log workflows |
| [`resources/dapr-mcp-server.yaml`](resources/dapr-mcp-server.yaml) | MCPServer: streamableHTTP (K8s dapr-mcp-server) |
| [`resources/local-tools.yaml`](resources/local-tools.yaml) | MCPServer: stdio (local dev tools) |
| [`resources/remote-weather.yaml`](resources/remote-weather.yaml) | MCPServer: SSE (remote weather service) |

## Key Concepts

- **MCPServer resource**: A first-class Dapr resource that declares MCP server
  connection details.  The sidecar handles transport, auth, and retries.
- **`DaprMCPClient`** (from `dapr.ext.workflow.aio`): Discovers tools via
  `dapr.internal.mcp.<name>.ListTools`. dapr-agents converts each
  `MCPToolDef` into a `WorkflowContextInjectedTool` via
  `mcp_tool_def_to_workflow_tool` for durable execution.
- **Middleware hooks**: `beforeCallTool` / `afterCallTool` workflow pipelines
  defined on the MCPServer spec.  "Before" hooks abort on error; "after" hooks
  log errors without affecting results.
- **Transport types**: `streamableHTTP` (production/K8s), `sse` (remote/legacy),
  `stdio` (local dev/testing).

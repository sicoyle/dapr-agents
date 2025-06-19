# PoC New DX Agent Creation

This quickstart demonstrates the unified agent experience with automatic agent type selection and YAML configuration support.

## Overview

The unified agent interface provides a single entry point for all agent types with automatic selection based on configuration parameters:

- **ToolCallAgent** (default): Simple, stateless tool execution
- **ReActAgent**: Reasoning-action pattern with explicit thought processes
- **OpenAPIReActAgent**: API integration with vector store capabilities
- **AssistantAgent**: Durable, stateful workflow with Dapr integration

## Prerequisites

### Basic Requirements
- **Python 3.10+** with pip
- **OpenAI API Key** for LLM interactions
- **Dapr Agents package** installed

### For Durable Agents (AssistantAgent)
- **Dapr CLI** installed and initialized
- **Docker** (for Dapr components)

## Setup

### 1. Install Dapr Agents
From the project root directory:
```bash
pip-compile pyproject.toml
pip install -e .
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"
```

### 3. Install Dapr CLI (for durable agents only)

TODO: copy link in dapr docs

## Running the Quickstarts

Navigate to the quickstart directory:
```bash
cd quickstarts/poc-new-dx-agent-creation
```

### Running Without Dapr (Stateless Agents)

These examples work without Dapr and demonstrate the unified agent interface:

#### 1. Simple Tool Agent
```bash
python 01_simple_tool_agent.py
```
**What it does**: Creates a basic weather assistant using ToolCallAgent (default)
**Agent Type**: ToolCallAgent
**Dapr Required**: No

#### 2. Reasoning Agent
```bash
python 02_reasoning_agent.py
```
**What it does**: Creates a travel planner using ReActAgent with reasoning
**Agent Type**: ReActAgent
**Dapr Required**: No

#### 3. API Integration Agent
```bash
python 03_openapi_agent.py
```
**What it does**: Creates an agent that interacts with APIs using OpenAPI specifications
**Agent Type**: OpenAPIReActAgent
**Dapr Required**: No

### Running With Dapr (Durable Agents)

These examples require Dapr for stateful, durable agent workflows:

#### 4. Durable State Agent
```bash
# Start Dapr with the required components
dapr run --app-id durable-agent --app-port 8001 --dapr-http-port 3500 --resources-path components/ -- python 04_durable_agent.py
```
**What it does**: Creates a stateful agent that maintains conversation state
**Agent Type**: AssistantAgent
**Dapr Required**: Yes

Unlike simpler agents, this stateful agent exposes a REST API for workflow interactions:

##### Start a new workflow:
```bash
curl -i -X POST http://localhost:8001/start-workflow \
  -H "Content-Type: application/json" \
  -d '{"task": "Process this dataset: [1, 2, 3, 4, 5]"}'
```

You'll receive a workflow ID in response, which you can use to track progress.

##### Check workflow status:
```bash
# Replace WORKFLOW_ID with the ID from the previous response
curl -i -X GET http://localhost:3500/v1.0/workflows/durableTaskHub/WORKFLOW_ID
```

##### Check agent service status:
```bash
curl -i -X GET http://localhost:8001/status
```

### How the Durable Agent Works

The key components of this implementation are:

1. **Persistent Memory**: The agent stores conversation state in Dapr's state store, enabling it to remember context across sessions and system restarts.

2. **Workflow Orchestration**: Long-running tasks are managed through Dapr's workflow system, providing:
    - Durability - workflows survive process crashes
    - Observability - track status and progress
    - Recoverability - automatic retry on failures

3. **Stateful Processing**: The agent can:
    - Process data and maintain analysis results
    - Store results persistently across interactions
    - Retrieve previous analysis when requested
    - Maintain conversation context

#### 5. Advanced Configuration
```bash
# Start Dapr with the required components
dapr run --app-id advanced-agent --app-port 8002 --dapr-http-port 3501 --resources-path components/ -- python 05_advanced_config.py
```
**What it does**: Demonstrates advanced configuration patterns with shared configs
**Agent Type**: Varies based on configuration
**Dapr Required**: Yes

## Configuration Structure

### Master Configuration (`configs/agents_config.yaml`)

Contains shared configurations for:
# TODO(@Sicoyle): I need to look if we can just pick up on Dapr LLM providers component instead of this! and see that they have all fields we would want.
- **LLM providers**: OpenAI, Anthropic, Azure
- **Dapr**: Dapr specific fields for creating durable agents
- **Agent behaviors**: Simple, reasoning, durable, advanced patterns

### Individual Agent Configuration

Each agent references configurations from the master config:
- `llm_config`: Which LLM provider to use
- `dapr_config`: Dapr specific configurations
- `agent_config`: Which agent behavior pattern to use

## Examples

### 1. Simple Tool Agent
```python
agent = Agent(
    name="WeatherBot",
    role="Weather Assistant",
    goal="Provide weather information",
    instructions=["Get current weather", "Provide forecasts"],
    tools=[get_weather, get_forecast],
    config_file="configs/weather_agent.yaml"
)
```

### 2. Reasoning Agent
```python
agent = Agent(
    name="TravelPlanner",
    role="Travel Assistant",
    goal="Plan travel itineraries with reasoning",
    instructions=["Plan travel itineraries", "Search for flights and hotels"],
    tools=[search_flights, book_hotel, get_attractions],
    reasoning=True,  # Triggers ReActAgent
    config_file="configs/travel_agent.yaml"
)
```

### 3. API Integration Agent
```python
agent = Agent(
    name="APIBot",
    role="API Integration Assistant",
    goal="Help with API integrations",
    instructions=["Analyze OpenAPI specifications", "Execute API calls"],
    openapi_spec_path="./api_specs/petstore.yaml",  # Triggers OpenAPIReActAgent
    config_file="configs/api_agent.yaml"
)
```

### 4. Durable State Agent
```python
agent = Agent(
    name="DataProcessor",
    role="Data Analysis Assistant",
    goal="Process data and maintain state",
    instructions=["Process and analyze data", "Store results persistently"],
    tools=[process_data, store_results, get_previous_analysis],
    config_file="configs/data_agent.yaml"  # Contains state store configuration
)
```

## Configuration Overrides

You can override any configuration by passing parameters directly:

```python
agent = Agent(
    name="CustomBot",
    role="Custom Assistant",
    goal="Custom goal",
    instructions=["Custom instructions"],
    tools=[custom_tool],
    # Direct configuration overrides
    max_iterations=15,
    reasoning=True,
    state_store_name="customstatestore"
)
```

## Agent Type Selection Logic

The system automatically selects the agent type based on configuration:

1. **AssistantAgent**: If `state_store_name` is specified
2. **OpenAPIReActAgent**: If `openapi_spec_path` is specified
3. **ReActAgent**: If `reasoning` is True
4. **ToolCallAgent**: Default (simple tool execution)

This provides a seamless experience where users don't need to understand the underlying agent types - the system chooses the best one automatically.

## Comparison with Other Quickstarts

This unified agent approach simplifies the patterns shown in other quickstarts:

| Quickstart | Traditional Approach | Unified Approach |
|------------|---------------------|------------------|
| [01-hello-world](./01-hello-world) | Manual agent type selection | Automatic selection |
| [03-agent-tool-call](./03-agent-tool-call) | ToolCallAgent only | All agent types supported |
| [04-agentic-workflow](./04-agentic-workflow) | Complex workflow setup | Simple configuration |

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the package with `pip install -e .` from the project root
2. **API Key Issues**: Ensure your `OPENAI_API_KEY` environment variable is set correctly
3. **Dapr Not Found**: For durable agents, make sure Dapr CLI is installed and initialized
4. **Configuration File Not Found**: Ensure you're running from the correct directory (`quickstarts/poc-new-dx-agent-creation`)

### Debug Mode

To see detailed logs, set the log level:
```bash
export LOG_LEVEL=DEBUG
```

### Testing Individual Components

You can test the configuration loading separately:
```python
from dapr_agents.agent.utils.factory import AgentConfig

config = AgentConfig()
config.load_yaml_config("configs/weather_agent.yaml")
print(config)
```

### Dapr Component Setup

For durable agents, ensure you have the required Dapr components in a `components/` directory:

```yaml
# components/statestore.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: workflowstatestore
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: redisPassword
    value: ""
```

```yaml
# components/messagebus.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: messagepubsub
spec:
  type: pubsub.redis
  version: v1
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: redisPassword
    value: ""
``` 
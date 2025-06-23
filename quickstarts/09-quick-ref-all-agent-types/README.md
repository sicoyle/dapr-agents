# Quick Reference For Trying out Each Agent Type

This quickstart demonstrates the unified agent experience with automatic agent type selection and YAML configuration support.
It supports trying out the various agents and parameters supported.

## Overview

The unified agent interface provides a single entry point for all agent types with automatic selection based on configuration parameters:

- **ToolCallAgent** (default): Simple, stateless tool execution
- **ReActAgent**: Reasoning-action pattern with explicit thought processes
- **OpenAPIReActAgent**: API integration with vector store capabilities
- **DurableAgent**: Durable, stateful workflow with Dapr integration


## Agent Type Selection Logic

The system automatically selects the agent type based on configuration:

1. **DurableAgent**: If `state_store_name` is specified
2. **OpenAPIReActAgent**: If `openapi_spec_path` is specified
3. **ReActAgent**: If `reasoning` is True
4. **ToolCallAgent**: Default (simple tool execution)

This provides a seamless experience where you don't need to understand the underlying agent types - the system chooses the best one automatically.

## Prerequisites

### Basic Requirements
- **Python 3.10+** with pip
- **OpenAI API Key** for LLM interactions
- **Dapr Agents package** installed

### For Durable Agents (DurableAgent)
- **Dapr CLI** installed and initialized
- Optionally, **Docker** (for Dapr components)

## Setup

### 1. Install Dapr Agents
From the project root directory:
```bash
pip-compile pyproject.toml
pip install -e .
```

### 2. Set Environment Variables
Optionally,
```bash
export OPENAI_API_KEY="your-openai-key"
```

### 3. Install Dapr CLI (for durable agents only)

Follow the instructions to install the latest [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/). 

## Running the Quickstarts

Navigate to the quickstart directory:
```bash
cd quickstarts/09-quick-ref-all-agent-types
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

This is a simple agent that leverages specified `tools`, and is stateless.

#### 2. Reasoning Agent
```bash
python 02_reasoning_agent.py
```
**What it does**: Creates a travel planner using ReActAgent with reasoning
**Agent Type**: ReActAgent
**Dapr Required**: No

This is a simple agent leveraging the ReAct pattern, denoted by `reasoning=True` in the agent configuration.

#### 3. API Integration Agent
```bash
python 03_openapi_agent.py
```
**What it does**: Creates an agent that interacts with APIs using OpenAPI specifications
**Agent Type**: OpenAPIReActAgent
**Dapr Required**: No

This is a simple agent supporting OpenAPI integrations and vector store support, denoted by `openapi_spec_path` in the agent configuration.

### Running With Dapr (Durable Agents)

These examples require Dapr for stateful, durable agent workflows:

#### 4. Durable State Agent
```bash
# Start Dapr with the required components
dapr run --app-id durable-agent --app-port 8001 --dapr-http-port 3500 --resources-path components/ -- python 04_durable_agent.py
```
**What it does**: Creates a stateful agent that maintains conversation state
**Agent Type**: DurableAgent
**Dapr Required**: Yes

Unlike simpler agents, this stateful agent is instantiated through the `DurableAgent` class, and exposes a REST API for workflow interactions:

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
curl -i -X GET http://localhost:3500/v1.0/workflows/dapr/WORKFLOW_ID
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

### Global Configuration (`configs/agents_config.yaml`)

Contains shared configurations for:
- **LLM providers**: OpenAI, etc
- **Dapr**: Dapr specific fields for creating durable agents
- **Agent behaviors**: Simple, reasoning, and durable

### Individual Agent Configuration

Each agent references configurations from the global config:
- `llm_config`: Which LLM provider to use
- `dapr_config`: Dapr specific configurations
- `agent_config`: Which agent behavior pattern to use

## Reference Examples

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
# components/pubsub.yaml
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
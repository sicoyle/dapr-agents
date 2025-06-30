# Quick Reference For Trying out Agent VS DurableAgent

## Overview

The unified agent interface provides a single entry point for all agent types with automatic selection based on configuration parameters:

- **Agent** (default): Simple, stateless tool execution
- **DurableAgent**: Durable, stateful workflow leveraging Dapr

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

### 3. Install Dapr CLI (for DurableAgent only)

Follow the instructions to install the latest [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/). 

## Running the Quickstarts

Navigate to the quickstart directory:
```bash
cd quickstarts/09-quick-ref-agents
```

### Running Without Dapr (Stateless Agents)

These examples work without Dapr and demonstrate the unified agent interface:

#### 1. Simple Tool Agent
```bash
python 01_simple_agent.py
```
**What it does**: Creates a basic weather assistant
**Agent Type**: Agent
**Dapr Required**: No

This is a simple agent that leverages specified `tools`, and is stateless.

#### 2. Reasoning Agent
```bash
python 02_more_complex_agent.py
```
**What it does**: Creates a travel planner using Agent reasoning
**Agent Type**: Agent
**Dapr Required**: No

#### 3. VectorStore Integration with Agent

Make sure you have the required dependencies installed:
```
pip install sentence-transformers chromadb
```

```bash
python 03_agent_with_vectorstore.py
```
**What it does**: Creates an agent that interacts with a vectorstore
**Agent Type**: Agent
**Dapr Required**: No


This agent demonstrates integration with vector stores for document storage and retrieval capabilities.

### Running With Dapr (Durable Agents)

DurableAgent requires Dapr for stateful, durable agent workflows:

#### 4. Durable Agent
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
# Understanding DurableAgent Architecture

## Overview

The `DurableAgent` is fundamentally different from other agent types (`ToolCallAgent`, `ReActAgent`, `OpenAPIReActAgent`) because it's built on **Dapr Workflows** rather than direct function calls.

## 1. Why DurableAgent Needs a Service

### The Problem
- **Other agents**: Direct function calls → Immediate response → Done
- **DurableAgent**: Workflow-based → Complex state management → Long-running processes

### Why Workflows?
DurableAgent uses Dapr Workflows because it needs to:
- **Maintain state** across multiple interactions
- **Handle complex multi-step processes** (tool calls, reasoning loops)
- **Support actor-based communication** between agents
- **Provide durability** (survive restarts, handle failures)
- **Enable distributed coordination** in multi-agent systems

### The Service Requirement
The workflow runtime needs to be **continuously running** to:
- Listen for workflow events
- Manage workflow state
- Handle actor lifecycle
- Process pub/sub messages
- Maintain persistent connections

### What the Service Actually Does

The service is essentially a **bridge between your app and the Dapr sidecar**:

```
Your App ←→ Service ←→ Dapr Sidecar ←→ Redis/State Store
```

**The service handles:**
- **Workflow Runtime**: Manages Dapr workflow execution
- **State Management**: Coordinates with Dapr state store
- **Pub/Sub**: Handles message routing between agents
- **Actor Lifecycle**: Manages Dapr actor instances
- **HTTP Endpoints**: Exposes REST API for external communication

**Without the service running:**
- Dapr sidecar can't communicate back to your app
- Workflows can't execute
- State can't be persisted
- Pub/sub messages can't be processed

## 2. agent.run() vs agent.start()

### agent.run() - Direct Execution Pattern
```python
# Pattern: Request → Process → Response → Done
response = await agent.run("Process this data")
print(response)  # Get immediate result
# Script ends here
```

**What it does:**
- Takes input, processes it, returns result
- **Synchronous/blocking** - waits for completion
- **Single request** - one input, one output
- **Stateless** - no persistent service running

**Use cases:**
- Simple tool execution
- One-off analysis
- Script-based automation
- When you want immediate results

### agent.start() - Service Pattern
```python
# Pattern: Start → Listen → Handle Requests → Keep Running
await agent.start()  # Starts service, keeps running
# Service is now listening for HTTP requests
# Multiple clients can send requests
# Service stays alive until stopped
```

**What it does:**
- Starts a **long-running service**
- **Asynchronous/non-blocking** - service runs in background
- **Multiple requests** - handles many concurrent requests
- **Stateful** - maintains state across requests
- **HTTP endpoints** - exposes REST API

**Use cases:**
- Multi-agent systems
- Production deployments
- Real-time applications
- When you need persistent state

## 3. Concrete Examples: When to Use Each

### Use agent.run() When:

#### Example 1: Data Processing Script
```python
# You want to process some data and get results immediately
agent = Agent(config_file="configs/data_agent.yaml")

# Process one dataset and get result
result = await agent.run("Analyze sales data from Q1")
print(f"Analysis complete: {result}")

# Script ends, no persistent service needed
```

#### Example 2: One-off Analysis
```python
# You need a quick analysis, don't need state persistence
agent = Agent(config_file="configs/data_agent.yaml")

# Get weather forecast
forecast = await agent.run("What's the weather in San Francisco?")
print(forecast)

# Done - no need for long-running service
```

#### Example 3: Batch Processing
```python
# Process multiple items in a loop
agent = Agent(config_file="configs/data_agent.yaml")

datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
for dataset in datasets:
    result = await agent.run(f"Process {dataset}")
    print(f"Processed {dataset}: {result}")

# Each call is independent, no state needed between calls
```

### Use agent.start() When:

#### Example 1: Multi-Agent System
```python
# Multiple agents need to communicate with each other
wizard_agent = Agent(name="Gandalf", config_file="configs/wizard_agent.yaml")
hobbit_agent = Agent(name="Frodo", config_file="configs/hobbit_agent.yaml")

# Start both as services so they can communicate
await wizard_agent.start()  # Now listening for messages
await hobbit_agent.start()  # Now listening for messages

# Agents can now send messages to each other via pub/sub
# Service stays running to handle ongoing communication
```

#### Example 2: Production Web Service
```python
# You're building a web app that needs to handle multiple users
agent = Agent(config_file="configs/customer_service_agent.yaml")

# Start as service to handle HTTP requests
await agent.start()

# Now your web app can send requests to the agent:
# POST /start-workflow {"task": "Help customer with refund"}
# Multiple users can make requests simultaneously
# Agent maintains context for each user session
```

#### Example 3: Real-time Chat Application
```python
# Building a chat app where the agent needs to remember conversation history
agent = Agent(config_file="configs/chat_agent.yaml")

# Start service to handle ongoing conversations
await agent.start()

# Chat app sends messages:
# User 1: "Hello" → Agent responds with context
# User 1: "What did we talk about?" → Agent remembers previous conversation
# User 2: "Hello" → Agent starts new conversation thread
# Service maintains separate state for each user
```

#### Example 4: IoT Data Processing
```python
# Processing real-time sensor data that needs stateful analysis
agent = Agent(config_file="configs/iot_agent.yaml")

# Start service to handle continuous data streams
await agent.start()

# IoT devices send data continuously:
# Sensor 1: temperature reading → Agent analyzes trend
# Sensor 1: another reading → Agent compares to previous, detects anomaly
# Sensor 2: humidity reading → Agent correlates with temperature data
# Service maintains historical context for anomaly detection
```

## 4. Should We Wrap agent.start() in agent.run()?

### The Dilemma

**Option A: Wrap agent.start() in agent.run()**
```python
# What the wrapper does:
async def run(self, input_data):
    # Start service if not running
    if not self._service_started:
        await self.agent.start()  # Long-running service
        self._service_started = True
    
    # Send request to service
    # Wait for response
    # Return result
    # Service keeps running for next call
```

**Option B: Keep them separate**
```python
# User chooses:
# For immediate results:
response = await agent.run("Process this")

# For long-running service:
await agent.start()  # Service runs indefinitely
```

### Analysis

**Arguments for Option A (Wrapper):**
- ✅ **Unified interface** - all agents use same pattern
- ✅ **Simpler for users** - don't need to understand internals
- ✅ **Automatic lifecycle management** - service starts when needed
- ✅ **Stateful conversations** - agent remembers context between calls

**Arguments for Option B (Separate):**
- ✅ **Explicit control** - user decides when to start/stop service
- ✅ **Resource efficiency** - don't start service for one-off calls
- ✅ **Clear semantics** - run() = immediate, start() = persistent
- ✅ **Production flexibility** - can run service independently

### Recommendation

**Use Option A (Wrapper)** for the unified agent interface because:

1. **User Experience**: Users expect `agent.run()` to work the same way across all agent types
2. **Stateful Nature**: DurableAgent is designed for stateful conversations, which aligns with persistent service
3. **Simplicity**: Hides the complexity of workflow management from end users
4. **Consistency**: Matches the pattern of other agent types

### Implementation Strategy

```python
class AssistantAgentWrapper:
    def __init__(self, assistant_agent):
        self.agent = assistant_agent
        self._service_started = False
    
    async def run(self, input_data):
        # Start service if needed (lazy initialization)
        if not self._service_started:
            await self._start_service()
        
        # Execute workflow and return result
        return await self._execute_workflow(input_data)
    
    async def start(self):
        # Explicit service start (for advanced users)
        await self._start_service()
    
    async def _start_service(self):
        # Start the workflow runtime and service
        self.agent.start_runtime()
        # Could also start HTTP service if needed
        self._service_started = True
```

## 5. Real-World Usage Patterns

### Pattern 1: Simple Script (agent.run())
```python
agent = Agent(config_file="configs/data_agent.yaml")

# Process one dataset
result1 = await agent.run("Process dataset A")
print(result1)

# Process another dataset (agent remembers context)
result2 = await agent.run("What was the previous result?")
print(result2)
```

### Pattern 2: Production Service (agent.start())
```python
agent = Agent(config_file="configs/data_agent.yaml")

# Start as long-running service
await agent.start()
# Service now handles HTTP requests at /start-workflow
# Multiple clients can send requests
# Service maintains state across all requests
```

### Pattern 3: Hybrid (both patterns)
```python
agent = Agent(config_file="configs/data_agent.yaml")

# Use run() for immediate processing
result = await agent.run("Process this data")

# Later, start as service for ongoing work
await agent.start()
# Service continues running
```

## 6. Configuration Implications

The wrapper approach means:
- **agent.run()** automatically handles service lifecycle
- **agent.start()** is still available for explicit control
- **State persistence** works seamlessly across both patterns
- **Resource management** is handled automatically

This gives users the **best of both worlds**: simple interface for common use cases, with advanced control when needed. 
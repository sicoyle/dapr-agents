# Dapr Python SDK Tool Calling Implementation Guide

This document provides the correct implementation guide for adding tool calling support to the Dapr Python SDK. **Architecture**: Tools are defined in Dapr Agents applications using decorators, and the Python SDK provides the transport layer between Dapr Agents and Dapr Runtime.

## Correct Architecture Flow

```
Python App (defines tools with @tool decorators)
    ↓ uses dapr-agents library
Dapr Agents (tool registry, uses Python SDK for transport)
    ↓ calls Python SDK
Python SDK (transport layer, talks to Dapr Runtime)
    ↓ gRPC calls
Dapr Runtime (proxies requests to LLM)
    ↓ HTTP/API calls  
LLM Component (OpenAI, etc.)
```

## Python SDK Role

The Python SDK is **only responsible for**:
1. **Sending tool definitions** to Dapr Runtime (which forwards to LLM)
2. **Receiving tool call responses** from LLM (via Dapr Runtime)
3. **Sending tool execution results** back to continue conversation
4. **Transforming data formats** for easier consumption by Dapr Agents

**The Python SDK does NOT:**
- Define tools (done in Dapr Agents apps with decorators)
- Execute tools (done in Dapr Agents)
- Manage tool registry (done in Dapr Agents)

## Required Changes

### 1. Extended ConversationInput

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ConversationInput:
    content: str
    role: Optional[str] = None
    scrub_pii: Optional[bool] = None
    # NEW: Tool calling support
    tools: Optional[List[Dict[str, Any]]] = None      # Tool definitions to send to LLM
    tool_call_id: Optional[str] = None                # For tool result messages
    name: Optional[str] = None                        # Function name for tool results
```

### 2. Enhanced ConversationResponse

```python
@dataclass 
class ConversationResult:
    result: str
    parameters: Dict[str, Any]
    # NEW: Tool calling fields
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool calls from LLM
    finish_reason: Optional[str] = None                # Why generation stopped
```

### 3. Updated gRPC Protocol

The `ConversationRequest` protobuf needs:

```protobuf
message ConversationInput {
  string content = 1;
  optional string role = 2;
  optional bool scrubPII = 3;
  // NEW fields:
  repeated Tool tools = 4;           // Tool definitions
  optional string tool_call_id = 5;  // For tool result messages  
  optional string name = 6;          // Function name for tool results
}

message Tool {
  string type = 1;                   // Always "function"
  ToolFunction function = 2;
}

message ToolFunction {
  string name = 1;
  string description = 2; 
  string parameters = 3;             // JSON schema as string
}

message ConversationResult {
  string result = 1;
  map<string, string> parameters = 2;
  // NEW fields:
  repeated ToolCall tool_calls = 3;
  optional string finish_reason = 4;
}

message ToolCall {
  string id = 1;
  string type = 2;                   // Always "function"
  ToolCallFunction function = 3;
}

message ToolCallFunction {
  string name = 1;
  string arguments = 2;              // JSON string
}
```

## Usage Examples

### 1. Dapr Agents Application (Tool Definition)

```python
# weather_agent.py - Dapr Agents application
from dapr_agents import tool
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    location: str = Field(description="City and state, e.g. San Francisco, CA")

@tool(args_model=WeatherSchema)
def get_weather(location: str) -> str:
    """Get current weather conditions"""
    # Tool implementation
    return f"Weather in {location}: 72°F, sunny"

# The dapr-agents library handles:
# 1. Tool registry
# 2. Converting @tool to OpenAI format
# 3. Using Python SDK for transport
```

### 2. Python SDK Usage (Transport Layer)

```python
# Inside dapr-agents library code
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput

# Tool definition in OpenAI format (from @tool decorator)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions",
            "parameters": {
                "type": "object", 
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

# Send initial request with tools
inputs = [ConversationInput(
    content="What's the weather in San Francisco?",
    role="user",
    tools=tools  # NEW: Tools sent to LLM via Dapr
)]

with DaprClient() as client:
    response = client.converse_alpha1(
        name="openai",
        inputs=inputs
    )
    
    # Response contains tool calls
    result = response.outputs[0]
    if result.tool_calls:
        # Execute tools (in Dapr Agents)
        for tool_call in result.tool_calls:
            function_name = tool_call["function"]["name"] 
            arguments = json.loads(tool_call["function"]["arguments"])
            
            # Execute the tool (handled by Dapr Agents)
            result = execute_tool(function_name, arguments)
            
            # Send tool result back
            tool_result_input = ConversationInput(
                content=result,
                role="tool", 
                tool_call_id=tool_call["id"],
                name=function_name
            )
            
            # Continue conversation with tool result
            final_response = client.converse_alpha1(
                name="openai",
                inputs=[tool_result_input]
            )
```

### 3. Streaming with Tool Calls

```python
# Streaming tool calls work similarly
for chunk in client.converse_stream_alpha1(
    name="openai",
    inputs=inputs_with_tools
):
    if chunk.result and chunk.result.tool_calls:
        # Handle tool calls in stream
        pass
    elif chunk.result and chunk.result.result:
        # Handle content chunks
        pass
```

## Implementation Strategy

### Phase 1: Core Protocol Support
1. Update `ConversationInput` dataclass
2. Update `ConversationResult` dataclass  
3. Update gRPC protobuf definitions
4. Update client methods to handle new fields

### Phase 2: Enhanced Responses
1. Add tool call parsing from LLM responses
2. Add finish_reason support
3. Update streaming to handle tool calls

### Phase 3: Developer Experience
1. Add helper methods for tool result formatting
2. Add validation for tool call flows
3. Update documentation and examples

## Backward Compatibility

All changes are additive:
- Existing `ConversationInput` usage continues to work
- New fields default to `None`
- Existing responses work unchanged
- No breaking changes to method signatures

## Testing Strategy

1. **Unit Tests**: Test new dataclass fields and serialization
2. **Integration Tests**: Test with actual LLM components
3. **End-to-End Tests**: Test full tool calling flow with Dapr Agents
4. **Streaming Tests**: Test tool calls in streaming responses

## Key Points

1. **Python SDK = Transport Only**: Just moves data between Dapr Agents and Dapr Runtime
2. **Tools Live in Dapr Agents**: Defined with `@tool` decorators in application code
3. **Dapr Runtime = Proxy**: Forwards requests/responses to/from LLM components
4. **Similar to Streaming**: Follow the same pattern as streaming implementation
5. **JSON Transformation**: SDK can provide JSON formatting for easier consumption

This approach keeps the Python SDK focused on its core responsibility as a transport layer while allowing Dapr Agents to handle the higher-level tool calling logic. 
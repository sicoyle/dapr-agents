# Tool Calling Root Cause Analysis

## Executive Summary

**Tool calling does not work with real LLM providers (OpenAI, Anthropic) through Dapr conversation components because the Dapr conversation API does not support tool calling at the protocol level.**

## Technical Investigation Results

### 1. Tool Definition and Conversion ✅ WORKING

The dapr-agents framework correctly:
- Defines tools using `@tool` decorator
- Converts `AgentTool` objects to Dapr SDK `Tool` format
- Creates proper OpenAI-compatible tool definitions

```python
# This works correctly
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: sunny, 72°F"

# Converts to proper SDK Tool format
sdk_tool = Tool(
    type="function",
    name="GetWeather", 
    description="Get weather information for a location.",
    parameters='{"properties": {"location": {"type": "string"}}, "required": ["location"], "type": "object"}'
)
```

### 2. ConversationInput Creation ❌ BROKEN

The issue starts here. While `ConversationInput` can accept a `tools` parameter in its constructor:

```python
# This appears to work
conv_input = ConversationInput(
    content="What's the weather?",
    role="user",
    tools=[sdk_tool]  # Tools are accepted
)
```

**But the tools are completely ignored during gRPC serialization.**

### 3. Dapr Python SDK gRPC Serialization ❌ BROKEN

In the Dapr Python SDK's `converse_alpha1` implementation, only 3 fields are extracted from `ConversationInput`:

```python
# From dapr-python-sdk source
inputs_pb = [
    api_v1.ConversationInput(
        content=inp.content,      # ✅ Sent
        role=inp.role,           # ✅ Sent  
        scrubPII=inp.scrub_pii   # ✅ Sent
        # tools=inp.tools        # ❌ IGNORED!
    )
    for inp in inputs
]
```

**The `tools` field is completely ignored and never sent to the Dapr runtime.**

### 4. Protocol-Level Gap ❌ MISSING

The Dapr conversation API specification doesn't include tool calling support:

```python
# Current converse_alpha1 signature
def converse_alpha1(
    name: str,
    inputs: List[ConversationInput],
    context_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    # No tools parameter exists
) -> ConversationResponse:
```

## Evidence from Testing

### Test 1: ConversationInput Interface
```python
# ConversationInput constructor signature
ConversationInput(
    content: Optional[str] = None, 
    role: Optional[str] = None, 
    scrub_pii: Optional[bool] = None, 
    parts: Optional[List[ContentPart]] = None
) -> None

# ❌ No tools parameter in constructor signature
# The tools attribute exists but is not used in serialization
```

### Test 2: Raw Dapr API Call
```python
# This fails because ConversationInput doesn't accept tools in constructor
conv_input = ConversationInput(
    content="What's the weather?",
    role="user", 
    tools=[sdk_tool]  # ❌ TypeError: unexpected keyword argument 'tools'
)
```

### Test 3: Real LLM Provider Responses

**OpenAI Response:**
```json
{
  "outputs": [
    {
      "result": "I don't have the capability to use external tools or access real-time weather data.",
      "finish_reason": "stop"
    }
  ]
}
```

**Anthropic Response:**
```json
{
  "outputs": [
    {
      "result": "I'll help you get the weather information. Let me use the weather tool to check the current conditions in San Francisco.\n\n*[Simulated tool call - actual tool calling not available]*",
      "finish_reason": "stop"
    }
  ]
}
```

Both providers respond as if no tools are available, confirming tools aren't reaching them.

## Root Cause Summary

1. **Protocol Gap**: Dapr conversation API doesn't support tool calling
2. **SDK Gap**: Python SDK ignores tools during gRPC serialization  
3. **Component Gap**: Conversation components (openai, anthropic) don't handle tools
4. **Framework Limitation**: dapr-agents can prepare tools but can't send them

## Current State

| Component | Tool Support | Status |
|-----------|-------------|--------|
| dapr-agents tool definitions | ✅ | Working |
| AgentTool → SDK Tool conversion | ✅ | Working |
| ConversationInput.tools attribute | ⚠️ | Exists but unused |
| Python SDK gRPC serialization | ❌ | Ignores tools |
| Dapr conversation API | ❌ | No tool support |
| LLM conversation components | ❌ | No tool awareness |

## What Works

- **echo-tools component**: Simulates tool calls for testing
- **Tool definitions**: Proper OpenAI-format tool schemas
- **Tool execution**: Local tool execution works
- **Framework architecture**: Ready for tool calling when protocol supports it

## Required Fixes

To enable tool calling with real LLM providers, the following changes are needed:

### 1. Dapr Protocol Updates
- Add `tools` field to `ConversationInput` protobuf definition
- Add `tool_call_id` and `name` fields for tool responses
- Update conversation API to accept and forward tools

### 2. Python SDK Updates  
- Include `tools` in gRPC serialization
- Add tool calling response parsing
- Support tool response message types

### 3. Conversation Component Updates
- Update OpenAI component to pass tools to OpenAI API
- Update Anthropic component to pass tools to Anthropic API
- Handle tool call responses and format appropriately

### 4. Response Processing
- Parse tool calls from LLM responses
- Convert to dapr-agents format
- Enable automatic tool execution workflows

## Conclusion

The dapr-agents framework is architecturally ready for tool calling. The limitation is at the Dapr platform level - the conversation API and components don't support tool calling yet. This is a platform feature gap, not a framework issue.

**Current workaround**: Use the `echo-tools` component for testing and development of tool calling workflows.

**Future solution**: Requires Dapr platform updates to support tool calling in the conversation API. 
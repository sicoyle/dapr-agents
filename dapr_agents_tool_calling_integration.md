# Dapr Agents Tool Calling Integration Guide

This document explains how Dapr Agents will integrate with the enhanced Python SDK tool calling functionality after the SDK changes from `python_sdk_tool_calling_implementation.md` are implemented.

## Correct Architecture Overview

**Key Point**: Tools are defined in Dapr Agents applications using `@tool` decorators. The Python SDK is purely a transport layer between Dapr Agents and Dapr Runtime.

With the Python SDK enhanced to support tool calling, the flow is:
1. **Dapr Agents apps define tools** using `@tool` decorators  
2. **Dapr Agents converts tools** to OpenAI format and sends via Python SDK
3. **Python SDK transports** tool definitions to Dapr Runtime → LLM
4. **Python SDK receives** tool call responses from LLM via Dapr Runtime
5. **Dapr Agents executes tools** locally and sends results back via Python SDK

## Correct Architecture Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dapr Agents   │    │   Python SDK    │    │  Dapr Runtime   │    │  LLM Component  │
│   Application   │    │ (Transport Only)│    │    (Proxy)      │    │  (OpenAI, etc.) │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ 1. Define Tools │───▶│ 2. Send Request │───▶│ 3. Proxy to LLM │───▶│ 4. Process &    │
│    with @tool   │    │    with Tools   │    │    with Tools   │    │    Generate     │
│    decorators   │    │                 │    │                 │    │    Tool Calls   │
│                 │    │                 │    │                 │    │                 │
│ 8. Execute Tools│◀───│ 7. Return Tool  │◀───│ 6. Proxy Tool   │◀───│ 5. Return Tool  │
│    Locally      │    │    Call Response│    │    Call Response│    │    Calls        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Integration Points in Dapr Agents

### 1. Tool Definition and Registration

**File:** `dapr_agents/tool/base.py`

Tools are defined using the existing `@tool` decorator pattern:

```python
from dapr_agents import tool
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

@tool(args_model=WeatherSchema)  
def get_weather(location: str) -> str:
    """Get the current weather in a given location"""
    # Tool implementation
    return f"Weather in {location}: 72°F, sunny"

@tool
def calculate_tip(amount: float, percentage: float = 18.0) -> dict:
    """Calculate tip amount"""
    tip = amount * percentage / 100
    return {"tip": round(tip, 2), "total": round(amount + tip, 2)}
```

### 2. Tool Registry and Conversion

**File:** `dapr_agents/tool/base.py`

The tool registry converts `@tool` decorators to OpenAI format for the SDK:

```python
class ToolRegistry:
    """Manages tools defined with @tool decorators."""
    
    def __init__(self):
        self.registered_tools = {}
    
    def register_tool(self, tool_instance):
        """Register a tool defined with @tool decorator."""
        self.registered_tools[tool_instance.name] = tool_instance
    
    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Convert all registered tools to OpenAI format for SDK."""
        tool_definitions = []
        
        for tool in self.registered_tools.values():
            # Use existing tool.to_function_call() method
            tool_def = tool.to_function_call()
            tool_definitions.append(tool_def)
            
        return tool_definitions
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self.registered_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool = self.registered_tools[tool_name]
        return tool.execute(**arguments)
```

### 3. Enhanced Chat Client

**File:** `dapr_agents/llm/dapr/chat.py`

The chat client sends tools via the enhanced Python SDK:

```python
from dapr.clients.grpc._request import ConversationInput

class DaprChatClient:
    """Enhanced chat client with tool calling support."""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        # Tools are automatically registered via @tool decorators
        
    def generate(
        self,
        messages: List[Dict[str, Any]], 
        stream: bool = False,
        tools: Optional[List] = None,  # Local tool instances (from @tool)
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Generate response with tool calling support."""
        
        # Convert local @tool instances to OpenAI format
        if tools:
            # Register any additional tools passed in
            for tool in tools:
                self.tool_registry.register_tool(tool)
        
        # Get all tool definitions in OpenAI format
        tool_definitions = self.tool_registry.get_openai_tool_definitions()
        
        # Prepare conversation input with tools
        conversation_inputs = []
        
        # Add conversation messages
        for msg in messages:
            conv_input = ConversationInput(
                content=msg.get("content", ""),
                role=msg.get("role", "user")
            )
            conversation_inputs.append(conv_input)
        
        # Add tools to the first user message (SDK requirement)
        if tool_definitions and conversation_inputs:
            conversation_inputs[0].tools = tool_definitions
        
        # Send request via Python SDK
        if stream:
            return self._handle_streaming_response(conversation_inputs, **kwargs)
        else:
            return self._handle_single_response(conversation_inputs, **kwargs)
    
    def _handle_single_response(self, inputs, **kwargs):
        """Handle non-streaming response with tool execution."""
        with DaprClient() as client:
            response = client.converse_alpha1(
                name=self.llm_component,
                inputs=inputs,
                **kwargs
            )
            
            result = response.outputs[0]
            
            # Check for tool calls
            if hasattr(result, 'tool_calls') and result.tool_calls:
                return self._execute_tools_and_continue(result.tool_calls, inputs, **kwargs)
            else:
                # No tool calls, return response
                return {
                    "content": result.result,
                    "finish_reason": getattr(result, 'finish_reason', 'stop')
                }
    
    def _execute_tools_and_continue(self, tool_calls, original_inputs, **kwargs):
        """Execute tool calls and continue conversation."""
        
        # Execute each tool call
        tool_results = []
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
            
            try:
                import json
                arguments = json.loads(arguments_str)
                
                # Execute tool using registry
                result = self.tool_registry.execute_tool(function_name, arguments)
                
                # Create tool result input
                tool_result_input = ConversationInput(
                    content=json.dumps(result) if not isinstance(result, str) else result,
                    role="tool",
                    tool_call_id=tool_call.get("id"),
                    name=function_name
                )
                tool_results.append(tool_result_input)
                
            except Exception as e:
                # Handle tool execution error
                error_result = ConversationInput(
                    content=f"Error executing {function_name}: {str(e)}",
                    role="tool",
                    tool_call_id=tool_call.get("id"),
                    name=function_name
                )
                tool_results.append(error_result)
        
        # Continue conversation with tool results
        with DaprClient() as client:
            final_response = client.converse_alpha1(
                name=self.llm_component,
                inputs=tool_results,
                **kwargs
            )
            
            final_result = final_response.outputs[0]
            return {
                "content": final_result.result,
                "finish_reason": getattr(final_result, 'finish_reason', 'stop'),
                "tool_calls_executed": len(tool_calls)
            }
```

### 4. Streaming Support

**File:** `dapr_agents/llm/dapr/chat.py`

Streaming with tool calls:

```python
def _handle_streaming_response(self, inputs, **kwargs):
    """Handle streaming response with tool execution."""
    
    with DaprClient() as client:
        tool_calls_buffer = []
        
        for chunk in client.converse_stream_alpha1(
            name=self.llm_component,
            inputs=inputs,
            **kwargs
        ):
            # Handle streaming content
            if hasattr(chunk, 'result') and chunk.result and hasattr(chunk.result, 'result'):
                yield {
                    "type": "content", 
                    "data": chunk.result.result
                }
            
            # Handle tool calls in streaming response  
            elif hasattr(chunk, 'result') and chunk.result and hasattr(chunk.result, 'tool_calls'):
                tool_calls_buffer.extend(chunk.result.tool_calls)
                
            # Handle completion with tool calls
            elif hasattr(chunk, 'context_id') and tool_calls_buffer:
                # Execute tools and continue
                final_result = self._execute_tools_and_continue(tool_calls_buffer, inputs, **kwargs)
                yield {
                    "type": "tool_execution_complete",
                    "data": final_result
                }
```

## Usage Examples

### Example 1: Simple Tool Calling in Dapr Agents App

```python
# weather_agent.py - Dapr Agents application
from dapr_agents import tool
from dapr_agents.llm.dapr import DaprChatClient
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    location: str = Field(description="City and state, e.g. San Francisco, CA")

@tool(args_model=WeatherSchema)
def get_weather(location: str) -> str:
    """Get current weather conditions"""
    # Simulate API call
    return f"Weather in {location}: 72°F, sunny"

@tool
def calculate_tip(amount: float, percentage: float = 18.0) -> dict:
    """Calculate tip amount"""
    tip = amount * percentage / 100
    return {"tip": round(tip, 2), "total": round(amount + tip, 2)}

# Usage
def main():
    client = DaprChatClient()
    
    # Tools are automatically discovered from @tool decorators
    messages = [
        {"role": "user", "content": "What's the weather in San Francisco and calculate a 20% tip on $45.60?"}
    ]
    
    # The client automatically:
    # 1. Finds tools defined with @tool decorators
    # 2. Converts them to OpenAI format  
    # 3. Sends via Python SDK to Dapr Runtime → LLM
    # 4. Executes tools locally when LLM requests them
    # 5. Continues conversation with results
    response = client.generate(messages=messages)
    
    print(f"Response: {response['content']}")
    
if __name__ == "__main__":
    main()
```

### Example 2: Streaming with Tool Calls

```python
# streaming_agent.py
from dapr_agents import tool
from dapr_agents.llm.dapr import DaprChatClient

@tool
def search_database(query: str) -> dict:
    """Search company database"""
    # Simulate database search
    return {"results": [f"Result for: {query}"], "count": 1}

def main():
    client = DaprChatClient()
    
    messages = [
        {"role": "user", "content": "Search for customer John Doe and write a summary"}
    ]
    
    # Streaming with automatic tool execution
    for chunk in client.generate(messages=messages, stream=True):
        if chunk.get("type") == "content":
            print(chunk["data"], end="", flush=True)
        elif chunk.get("type") == "tool_execution_complete":
            print(f"\n[Tools executed: {chunk['data'].get('tool_calls_executed', 0)}]")
    
if __name__ == "__main__":
    main()
```

## Key Changes from Previous Architecture

### ✅ **Correct Approach**
1. **Tools defined in Dapr Agents** using `@tool` decorators
2. **Python SDK is transport only** - no tool definitions sent through it
3. **Dapr Agents manages tool registry** and execution
4. **SDK provides enhanced data structures** for tool call messages

### ❌ **Previous Incorrect Assumptions**
1. ~~Tools received from Python apps via SDK~~
2. ~~Mixed tool execution (SDK + Dapr Agents)~~  
3. ~~Tool definitions sent through ConversationInput.tools~~

## Implementation Strategy

### Phase 1: Core Integration
1. ✅ **Use existing `@tool` decorator pattern** - no changes needed
2. ✅ **Enhance tool registry** to convert to OpenAI format
3. ✅ **Update DaprChatClient** to use enhanced Python SDK
4. ✅ **Add tool execution logic** in conversation flow

### Phase 2: Enhanced Features  
1. **Streaming tool execution** support
2. **Error handling** for tool failures
3. **Tool execution monitoring** and logging
4. **Performance optimization** for tool calls

### Phase 3: Advanced Features
1. **Async tool execution** for I/O bound tools
2. **Tool execution timeouts** and circuit breakers
3. **Tool result caching** for expensive operations
4. **Tool execution metrics** and monitoring

## Configuration

No special configuration needed - tools are discovered automatically from `@tool` decorators in the application code.

```python
# No configuration required - tools auto-discovered
from dapr_agents import tool

@tool
def my_tool(param: str) -> str:
    return f"Processed: {param}"

# Tool is automatically available to LLM calls
```

## Testing Strategy

### Unit Tests
```python
def test_tool_registry():
    """Test tool registration and conversion."""
    registry = ToolRegistry()
    
    @tool  
    def test_tool(input: str) -> str:
        return f"test: {input}"
    
    # Tool should be auto-registered
    definitions = registry.get_openai_tool_definitions()
    assert len(definitions) == 1
    assert definitions[0]["function"]["name"] == "test_tool"

def test_tool_execution():
    """Test tool execution."""
    registry = ToolRegistry()
    
    result = registry.execute_tool("test_tool", {"input": "hello"})
    assert result == "test: hello"
```

### Integration Tests
```python
def test_end_to_end_tool_calling():
    """Test complete tool calling flow."""
    
    @tool
    def get_time() -> str:
        return "12:00 PM"
    
    client = DaprChatClient()
    response = client.generate(
        messages=[{"role": "user", "content": "What time is it?"}]
    )
    
    # Should have executed the tool and returned response
    assert "12:00 PM" in response["content"]
```

## Best Practices

1. **Keep tools simple and focused** - one responsibility per tool
2. **Use proper type hints** - helps with OpenAI schema generation  
3. **Handle errors gracefully** - tools should not crash the conversation
4. **Add good descriptions** - helps LLM choose appropriate tools
5. **Consider performance** - tools block conversation until complete

## Migration Path

1. **No migration needed** - existing `@tool` decorators continue to work
2. **Enhanced Python SDK** provides transport layer improvements
3. **Automatic tool discovery** - no configuration changes required
4. **Backward compatible** - existing Dapr Agents apps work unchanged

This approach maintains the existing developer experience while adding powerful tool calling capabilities through the enhanced Python SDK transport layer. 
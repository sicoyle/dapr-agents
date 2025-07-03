# 10 - Dapr Tool Calling

This quickstart demonstrates tool calling capabilities using Dapr agents with two different approaches:

1. **Direct Tool Calling** - Using `DaprChatClient` for immediate tool execution
2. **AssistantAgent** - Using workflow-based agents for advanced tool calling with memory and state management

## ✅ Current Status

- ✅ **Tool Definition**: `@tool` decorator and `AgentTool` classes work perfectly
- ✅ **Tool Registration**: Tools are properly registered in dapr-agents
- ✅ **Basic LLM Conversation**: Full support via Dapr components
- ✅ **Tool Calling Integration**: Working with improved Python SDK
- ✅ **Streaming + Tools**: Streaming tool calling supported
- ✅ **AssistantAgent**: Workflow-based agents with persistent memory

## What You'll Learn

- How to define tools using the `@tool` decorator and `AgentTool` classes
- Direct tool calling with `DaprChatClient`
- Workflow-based tool calling with `AssistantAgent`
- Streaming tool calling capabilities
- Memory and state management in agents
- Tool function signatures and documentation

## Prerequisites

- Dapr CLI installed
- Python 3.9+
- Redis (for AssistantAgent state storage)
- Optional: OpenAI/Anthropic API keys for real LLM providers

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Dapr with Components

```bash
# Option A: Use the development helper (recommended)
python ../../tools/run_dapr_dev.py --components ../../tests/components/local_dev

# Option B: Manual Dapr start (from quickstarts/10-dapr-tool-calling directory)
dapr run --app-id test-app --dapr-http-port 3500 --dapr-grpc-port 50001 --components-path ./components
```

### 3. Try the Examples

#### Direct Tool Calling (Simple)
```bash
# Basic tool calling with echo provider
python simple_example.py

# With OpenAI (requires API key)
python simple_example.py --provider openai

# With streaming
python simple_example.py --provider openai --streaming
```

#### AssistantAgent (Advanced)
```bash
# Workflow-based agent with memory
python assistant_agent_example.py

# With real LLM provider
python assistant_agent_example.py --provider openai

# Demo mode
python assistant_agent_example.py --demo
```

## Two Approaches to Tool Calling

### 🔧 Direct Tool Calling (`simple_example.py`)

**Best for**: Simple tool execution, quick prototyping, stateless operations

**Features**:
- Immediate tool execution with `DaprChatClient`
- Function-based (`@tool`) and class-based (`AgentTool`) tools
- Streaming support
- Minimal setup required

**Example**:
```python
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool

@tool
def get_weather(location: str) -> str:
    return f"Weather in {location}: 72°F, sunny"

client = DaprChatClient()
response = client.generate(
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=[get_weather],
    llm_component="openai"
)
```

### 🤖 AssistantAgent (`assistant_agent_example.py`)

**Best for**: Complex workflows, persistent memory, multi-agent systems

**Features**:
- Workflow-based execution with Dapr Workflows
- Persistent memory across conversations
- Automatic tool calling iteration (up to max_iterations)
- State management and recovery
- Multi-agent communication
- Service-based architecture

**Example**:
```python
from dapr_agents import AssistantAgent
from dapr_agents.memory import ConversationDaprStateMemory

assistant = AssistantAgent(
    name="ToolAssistant",
    tools=[get_weather, calculate],
    memory=ConversationDaprStateMemory(
        store_name="conversationstore",
        session_id="my-session"
    ),
    max_iterations=5
)

assistant.as_service(port=8002)
await assistant.start()
```

### When to Use Which?

| Feature | Direct Tool Calling | AssistantAgent |
|---------|-------------------|----------------|
| **Setup Complexity** | Low | Medium |
| **Memory** | None | Persistent |
| **State Management** | Manual | Automatic |
| **Multi-iteration** | Manual | Automatic |
| **Service Architecture** | Optional | Built-in |
| **Agent Communication** | No | Yes |
| **Workflow Support** | No | Yes |
| **Best For** | Quick tasks | Complex workflows |

## Tool Definition Examples

### 📊 Mathematical Tools

```python
from dapr_agents.tool import tool

@tool
def calculate_math(expression: str) -> str:
    """Safely calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        Result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"
```

### 🕐 Time and Date Tools

```python
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in the specified timezone.
    
    Args:
        timezone: The timezone to get the time for (e.g., "UTC", "EST", "PST")
    
    Returns:
        Current time as a formatted string
    """
    current_time = datetime.now()
    return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
```

### 📝 Text Analysis Tools

```python
@tool
def text_analysis(text: str) -> str:
    """Analyze text and provide statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Analysis results including character count, word count, etc.
    """
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    analysis = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": sentences,
        "average_word_length": sum(len(word.strip('.,!?;:')) for word in words) / len(words) if words else 0
    }
    
    return f"Text Analysis: {json.dumps(analysis, indent=2)}"
```

## ✅ Tool Calling in Action

### Intelligent Tool Selection
```python
# Working tool calling API
from dapr_agents.llm.dapr import DaprChatClient

client = DaprChatClient()

response = client.generate(
    messages=[{"role": "user", "content": "What time is it and calculate 25 * 4?"}],
    tools=[get_current_time, calculate_math],
    llm_component="openai"
)

# The LLM automatically chooses and executes the right tools
```

### Streaming with Tool Calling
```python
# Streaming + tool calling working now!
response_stream = client.generate(
    messages=[{"role": "user", "content": "Analyze this text and tell me the time"}],
    tools=[text_analysis, get_current_time],
    stream=True,
    llm_component="openai"
)

for chunk in response_stream:
    if hasattr(chunk, 'chunk') and chunk.chunk:
        print(chunk.chunk.content, end='', flush=True)
    elif hasattr(chunk, 'complete'):
        print(f"\n✅ Completed: {chunk.complete.usage}")
```

## Architecture Overview

### How Tools Work ✅
1. **Tool Definition**: `@tool` decorator and `AgentTool` classes register functions
2. **Tool Registry**: Tools are collected and formatted for LLMs
3. **OpenAI Format**: Tools convert to proper OpenAI tool calling format
4. **Execution**: Tools execute locally in dapr-agents applications
5. **Integration**: Works with improved Python SDK for Dapr

### Direct Tool Calling Flow
1. Define tools with `@tool` or `AgentTool`
2. Pass tools to `DaprChatClient.generate()`
3. LLM decides which tools to call
4. Tools execute and return results
5. LLM incorporates results into response

### AssistantAgent Workflow Flow
1. Define tools and create AssistantAgent
2. Agent runs as Dapr workflow
3. Automatic tool calling iteration
4. Persistent state and memory management
5. Multi-agent communication support

## Complete Feature Set ✅

### Core Features
- [x] `@tool` decorator implementation
- [x] `AgentTool` class-based tools
- [x] Tool registry and management
- [x] OpenAI format conversion
- [x] Local tool execution
- [x] Direct tool calling via `DaprChatClient`
- [x] Streaming + tool calling
- [x] AssistantAgent workflow-based execution
- [x] Persistent memory and state
- [x] Multi-iteration tool workflows
- [x] Error handling and recovery

### Advanced Features
- [x] Function-based and class-based tools
- [x] Tool parameter validation
- [x] Async tool execution support
- [x] Tool execution history
- [x] Memory integration
- [x] Multi-agent communication
- [x] Workflow state management

## Testing the Implementation

Both approaches are fully functional:

```python
# Test direct tool calling
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool

@tool
def example_tool(message: str) -> str:
    return f"Processed: {message}"

client = DaprChatClient()
response = client.generate(
    messages=[{"role": "user", "content": "Use the example tool"}],
    tools=[example_tool],
    llm_component="echo"
)

@tool
def my_tool(param: str) -> str:
    """My tool description."""
    return f"Result: {param}"

# Tools are properly registered and formatted
factory = AgentFactory()
tools = factory.get_tools()
print(f"Registered tools: {[t.name for t in tools]}")
```

## Error Handling

Current examples include proper error handling for the development state:

```python
try:
    # This will work for basic conversation
    response = llm.generate(
        messages=[{"role": "user", "content": "Hello"}],
        llm_component="openai"
    )
    print(response.choices[0].message.content)
    
    # This is not yet supported but won't crash
    response_with_tools = llm.generate(
        messages=[{"role": "user", "content": "What time is it?"}],
        tools=[get_current_time],  # Tools defined but not yet called by LLM
        llm_component="openai"
    )
    
except Exception as e:
    print(f"Error: {e}")
    # Handle gracefully
```

## Performance Considerations

### Current Performance ✅
- **Tool Definition**: Instant registration
- **Tool Formatting**: Fast OpenAI format conversion
- **Local Execution**: Direct function calls

### Future Performance 🚧
- **Tool Calling Latency**: Will depend on LLM provider
- **Streaming Tools**: Real-time tool execution
- **Caching**: Tool result caching for repeated calls

## Troubleshooting

### Common Issues

1. **"Tools not being called by LLM"**
   - This is expected - tool calling integration is in development
   - Tools are defined correctly but not yet sent to LLM

2. **"gRPC protocol doesn't support tools"**
   - Correct - this is the main development blocker
   - Protocol updates are in progress

3. **"Tool definitions work but execution doesn't"**
   - Tool definitions and local execution work perfectly
   - The missing piece is LLM integration

### Getting Help

- Check tool definitions with `@tool` decorator
- Verify tools are registered in the factory
- Test tool execution locally
- Monitor development progress for integration updates

## Next Steps

1. **Monitor Development**: Tool calling integration is actively being developed
2. **Prepare Tools**: Define and test your tools locally
3. **Plan Integration**: Design your tool calling workflows
4. **Stay Updated**: Follow progress on gRPC protocol updates

## Resources

- [Tool Definition Guide](../../docs/concepts/tools.md)
- [Agent Patterns](../../docs/concepts/agents.md)
- [Development Roadmap](../../docs/roadmap.md)

---

**Note**: This quickstart demonstrates the foundation for tool calling. The complete integration is in active development and will be available soon. All tool definitions and local execution work perfectly - we're just connecting the final pieces to enable LLM-driven tool selection and execution. 

# Dapr Tool Calling Examples

This directory contains comprehensive examples demonstrating tool calling functionality with various Dapr conversation providers.

## 🚀 Quick Start

### 1. Setup API Keys (Optional)

Create a `.env` file in the **root of the dapr-agents repository** with your API keys:

```bash
# Create .env file at repo root (not in this directory!)
cd ../../  # Go to repo root
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_AI_API_KEY=your-google-ai-api-key-here
EOF
```

**Note:** The `.env` file should be in the root directory of the repository, not in this quickstart directory.

### 2. Install Dependencies

```bash
# Install python-dotenv if not already installed
pip install python-dotenv
```

### 3. Start Dapr Runtime

```bash
# From the root of dapr-agents
python tools/run_dapr_dev.py --app-id tool-calling-demo --components ./components
```

### 4. Run Simple Example

```bash
# Basic example with echo provider (always works)
python simple_example.py

# With OpenAI (requires OPENAI_API_KEY in .env)
python simple_example.py --provider openai

# With streaming
python simple_example.py --provider openai --streaming

# With Anthropic
python simple_example.py --provider anthropic

# With Google AI (Gemini)
python simple_example.py --provider gemini
```

### 5. Run Comprehensive Examples

```bash
# Run all provider examples
python tool_calling_examples.py
```

## 📋 Prerequisites

### Required
- Dapr runtime with conversation components
- Python environment with dapr-agents installed
- `python-dotenv` package (`pip install python-dotenv`)

### Optional (for real LLM providers)
- API keys configured in `.env` file at repo root:
  - `OPENAI_API_KEY` for OpenAI provider
  - `ANTHROPIC_API_KEY` for Anthropic provider  
  - `GOOGLE_AI_API_KEY` for Google AI (Gemini) provider

## 🔧 Supported Providers

| Provider | Component Name | Description | API Key Required | .env Variable | Status |
|----------|----------------|-------------|------------------|---------------|--------|
| Echo | `echo-tools` | Testing/development provider | No | - | ✅ Available |
| OpenAI | `openai` | GPT models (GPT-4, GPT-3.5) | Yes | `OPENAI_API_KEY` | ✅ Available |
| Anthropic | `anthropic` | Claude models | Yes | `ANTHROPIC_API_KEY` | ✅ Available |
| Google AI | `gemini` | Gemini models | Yes | `GOOGLE_AI_API_KEY` | ⚠️ Not available in current Dapr build |

**Note**: The Google AI (Gemini) provider requires the `conversation.googleai` component type, which may not be available in all Dapr builds. If you encounter "failed finding conversation component gemini", this means your Dapr build doesn't include the Google AI conversation component.

## 🛠️ Tool Examples

The examples include these sample tools:

### Weather Tool
```python
@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location."""
    return f"Weather in {location}: 72°F, sunny, light breeze"
```

### Calculator Tool
```python
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Time Tool
```python
@tool
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
```

## 💡 Usage Patterns

### Basic Tool Calling
```python
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool

@tool
def my_tool(param: str) -> str:
    return f"Processed: {param}"

client = DaprChatClient()
response = client.generate(
    messages=[{"role": "user", "content": "Use my tool with 'hello'"}],
    llm_component="echo-tools",  # or "openai", "anthropic", etc.
    tools=[my_tool],
    stream=False
)
```

### Streaming with Tools
```python
for chunk in client.generate(
    messages=[{"role": "user", "content": "Get weather for NYC"}],
    llm_component="openai",
    tools=[get_weather],
    stream=True
):
    if chunk.get("choices") and chunk["choices"]:
        choice = chunk["choices"][0]
        if choice.get("delta", {}).get("content"):
            print(choice["delta"]["content"], end="")
        elif choice.get("delta", {}).get("tool_calls"):
            print(f"Tool called: {choice['delta']['tool_calls']}")
```

### Multiple Tools
```python
tools = [get_weather, calculate, get_time]

response = client.generate(
    messages=[{"role": "user", "content": "Get weather for London, calculate 25*4, and tell me the time"}],
    llm_component="anthropic",
    tools=tools,
    stream=False
)
```

## 🔧 Component Configuration

### Echo (Testing)
```yaml
# components/echo-tools.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: echo-tools
spec:
  type: conversation.echo
  version: v1
```

### OpenAI
```yaml
# components/openai.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  version: v1
  metadata:
  - name: apiKey
    value: "${OPENAI_API_KEY}"
  - name: model
    value: "gpt-4"
```

### Anthropic
```yaml
# components/anthropic.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: anthropic
spec:
  type: conversation.anthropic
  version: v1
  metadata:
  - name: apiKey
    value: "${ANTHROPIC_API_KEY}"
  - name: model
    value: "claude-3-sonnet-20240229"
```

### Google AI (Gemini)
```yaml
# components/gemini.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: gemini
spec:
  type: conversation.googleai
  version: v1
  metadata:
  - name: apiKey
    value: "${GOOGLE_AI_API_KEY}"
  - name: model
    value: "gemini-1.5-pro"
```

## 🧪 Testing Different Scenarios

### 1. Single Tool Call
```bash
python simple_example.py --provider echo-tools
```

### 2. Multiple Tool Calls
```bash
python tool_calling_examples.py
```

### 3. Streaming Responses
```bash
python simple_example.py --provider openai --streaming
```

### 4. Error Handling
The examples include error handling for:
- Invalid calculations
- Missing API keys
- Network errors
- Tool execution failures

## 📊 Expected Output

### Echo Provider (Development)
```json
{
  "outputs": [
    {
      "result": "I need to call some tools to help you with that.",
      "tool_calls": [
        {
          "id": "call_echo_1234567890",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\"}"
          }
        }
      ],
      "finish_reason": "tool_calls"
    }
  ]
}
```

### Real LLM Providers
Real providers (OpenAI, Anthropic, etc.) will:
1. Understand the user's request
2. Decide which tools to call
3. Generate appropriate tool calls
4. Process tool results
5. Provide a natural language response

## 🔍 Troubleshooting

### Common Issues

1. **"Unsupported format type: dapr"**
   - This was fixed in the latest version
   - Make sure you're using the updated dapr-agents code

2. **Tool names not matching**
   - Echo component expects snake_case names (`get_weather`)
   - This is handled automatically by the Dapr client

3. **API key errors**
   - Make sure your `.env` file is in the **repo root**, not in this directory
   - Check that your API keys are valid and properly formatted in the `.env` file
   - Ensure `python-dotenv` is installed: `pip install python-dotenv`

4. **Component not found**
   - Ensure Dapr is running with the correct components path
   - Check component YAML files are properly configured

5. **".env file not found"**
   - The `.env` file should be in the root of the dapr-agents repository
   - Path should be: `/path/to/dapr-agents/.env` (not `/path/to/dapr-agents/quickstarts/10-dapr-tool-calling/.env`)

### Debug Mode
Run with debug logging:
```bash
export DAPR_LOG_LEVEL=debug
python simple_example.py --provider your-provider
```

### Verify .env File Location
```bash
# From the quickstart directory, check if .env exists at repo root
ls -la ../../.env

# Should show the .env file with your API keys
```

## 📚 Additional Resources

- [Dapr Conversation API Documentation](https://docs.dapr.io/reference/components-reference/supported-conversation/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [Google AI Gemini API](https://ai.google.dev/docs)
- [dapr-agents Documentation](../../docs/)

## 🤝 Contributing

To add support for new providers:

1. Create component configuration in `components/`
2. Add provider test in `tool_calling_examples.py`
3. Update this README
4. Test with both streaming and non-streaming modes
5. Add any required API key variables to the `.env` example 
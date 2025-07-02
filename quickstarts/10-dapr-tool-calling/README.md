# 10 - Dapr Tool Calling (In Development)

This quickstart demonstrates tool calling capabilities using Dapr agents. **Tool calling functionality is currently in active development** and will be available in future releases.

## 🚧 Current Status

- ✅ **Tool Definition**: `@tool` decorator works perfectly
- ✅ **Tool Registration**: Tools are properly registered in dapr-agents
- ✅ **Basic LLM Conversation**: Full support via Dapr components
- 🚧 **Tool Calling Integration**: gRPC protocol support in development
- 🚧 **Streaming + Tools**: Coming with streaming support

## What You'll Learn

- How to define tools using the `@tool` decorator
- Tool function signatures and documentation
- **Preview**: How tool calling will work when available
- Current architecture and development progress
- Future capabilities and roadmap

## Prerequisites

- Dapr CLI installed
- Python 3.9+
- OpenAI API key (for when tool calling becomes available)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Dapr Sidecar

```bash
# From the quickstarts/10-dapr-tool-calling directory
dapr run --app-id test-app --dapr-http-port 3500 --dapr-grpc-port 50001 --components-path ./components
```

### 3. Explore Tool Definitions

```bash
python tool_calling_openai.py
```

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

## 🚀 Coming Soon: Full Tool Calling

When tool calling becomes available, you'll be able to:

### Intelligent Tool Selection
```python
# Future tool calling API (in development)
from dapr_agents.llm import DaprChatClient

llm = DaprChatClient()

response = llm.generate(
    messages=[{"role": "user", "content": "What time is it and calculate 25 * 4?"}],
    tools=[get_current_time, calculate_math],  # This will work soon!
    llm_component="openai"
)

# The LLM will automatically choose and execute the right tools
```

### Streaming with Tool Calling
```python
# Future streaming + tool calling (in development)
response_stream = llm.generate(
    messages=[{"role": "user", "content": "Analyze this text and tell me the time"}],
    tools=[text_analysis, get_current_time],
    stream=True,  # Both streaming AND tool calling!
    llm_component="openai"
)

for chunk in response_stream:
    if chunk.get("type") == "content":
        print(chunk["data"], end='', flush=True)
    elif chunk.get("type") == "tool_call":
        print(f"\n[Using tool: {chunk['data']['name']}]")
    elif chunk.get("type") == "tool_result":
        print(f"[Tool completed: {chunk['data']}]")
```

## Current Architecture

### How Tools Work Today ✅
1. **Tool Definition**: `@tool` decorator registers functions
2. **Tool Registry**: Tools are collected and formatted
3. **OpenAI Format**: Tools convert to proper OpenAI tool format
4. **Local Execution**: Tools execute in dapr-agents applications

### What's Missing 🚧
1. **gRPC Protocol**: Tool transport from Python SDK to Dapr Runtime
2. **LLM Integration**: Tools need to reach OpenAI/other LLM components
3. **Response Handling**: Tool results need to flow back through the system
4. **Streaming Support**: Real-time tool calling with streaming responses

## Development Progress

### Phase 1: Foundation ✅
- [x] `@tool` decorator implementation
- [x] Tool registry and management
- [x] OpenAI format conversion
- [x] Local tool execution

### Phase 2: Transport Layer 🚧
- [ ] gRPC protocol update for tools
- [ ] Python SDK tool serialization
- [ ] Dapr Runtime tool forwarding
- [ ] LLM component tool integration

### Phase 3: Advanced Features 📋
- [ ] Streaming + tool calling
- [ ] Multi-step tool workflows
- [ ] Tool calling with context
- [ ] Error handling and retries

## Testing Current Implementation

You can test the tool definition and formatting:

```python
# This works today - tool definition and conversion
from dapr_agents.tool import tool
from dapr_agents.agent.utils.factory import AgentFactory

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
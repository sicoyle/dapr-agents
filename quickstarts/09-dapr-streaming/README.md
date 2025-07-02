# 09 - Dapr LLM Conversation (Streaming Coming Soon!)

This quickstart demonstrates LLM conversation capabilities using Dapr components. **Streaming functionality is currently in development** and will be available in future releases.

## 🚧 Current Status

- ✅ **Basic LLM Conversation**: Full support via `converse_alpha1`
- ✅ **Multiple LLM Providers**: Echo, OpenAI components working
- ✅ **Simulated Streaming Display**: Character-by-character output for better UX
- 🚧 **True Streaming Responses**: In development (`converse_stream_alpha1`)
- 🚧 **Tool Calling**: In development (Phase 3)

## What You'll Learn

- How to use Dapr LLM components for conversation
- Working with different LLM providers (Echo, OpenAI)
- Managing conversation context and history
- Performance monitoring and error handling
- **Preview**: What streaming will enable when available

## Prerequisites

- Dapr CLI installed
- Python 3.9+
- For OpenAI examples: OpenAI API key

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Dapr Sidecar

```bash
# From the quickstarts/09-dapr-streaming directory
dapr run --app-id test-app --dapr-http-port 3500 --dapr-grpc-port 50001 --components-path ./components
```

### 3. Run Examples

#### Echo Component (No API Key Required)
```bash
python streaming_echo.py
```

#### OpenAI Component (Requires API Key)
```bash
export OPENAI_API_KEY=your_api_key_here
python streaming_openai.py
```

## Examples

### 📝 Basic Conversation

```python
from dapr_agents.llm import DaprChatClient

# Initialize client
llm = DaprChatClient()

# Generate response
response = llm.generate(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    llm_component="echo"
)

print(response.choices[0].message.content)
```

### 🔄 Multi-turn Conversation

```python
conversation = [
    {"role": "user", "content": "What's machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give me an example?"}
]

response = llm.generate(
    messages=conversation,
    llm_component="openai"
)
```

## 🚀 Coming Soon: Streaming Support

When streaming becomes available, you'll be able to:

### Real-time Streaming
```python
# Future streaming API (in development)
response_stream = llm.generate(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,  # This will work soon!
    llm_component="openai"
)

for chunk in response_stream:
    if chunk.get("type") == "content":
        print(chunk["data"], end='', flush=True)
```

### Advanced Streaming Features
- **Token-by-token responses**: See text appear as it's generated
- **Usage tracking**: Monitor token consumption in real-time  
- **Context continuity**: Maintain conversation state across streams
- **Error handling**: Robust streaming error recovery
- **Performance metrics**: Latency and throughput monitoring

## Components

### Echo Component (`components/echo-conversation.yaml`)
- **Purpose**: Testing without API keys
- **Behavior**: Echoes input with conversational formatting
- **Use case**: Development and testing

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: echo
spec:
  type: conversation.echo
  version: v1
```

### OpenAI Component (`components/openai-conversation.yaml`)
- **Purpose**: Real AI conversations with GPT models
- **Requirements**: OpenAI API key
- **Models**: GPT-4, GPT-3.5-turbo, etc.

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  version: v1
  metadata:
  - name: apiKey
    secretKeyRef:
      name: openai-secret
      key: api-key
```

## Error Handling

The examples include comprehensive error handling:

```python
try:
    response = llm.generate(
        messages=[{"role": "user", "content": prompt}],
        llm_component="openai"
    )
    # Process response
except Exception as e:
    print(f"Error: {e}")
    # Handle specific error types
```

## Performance Tips

1. **Component Selection**: Use `echo` for testing, real LLMs for production
2. **Context Management**: Keep conversation history manageable
3. **Error Recovery**: Implement retry logic for production use
4. **Resource Monitoring**: Monitor token usage and costs

## Development Roadmap

### Phase 1: Basic Conversation ✅
- [x] Non-streaming conversation API
- [x] Multiple LLM provider support
- [x] Context management
- [x] Error handling

### Phase 2: Streaming Support 🚧 
- [ ] `converse_stream_alpha1` API
- [ ] Real-time token streaming
- [ ] Stream error handling
- [ ] Performance optimization

### Phase 3: Advanced Features 📋
- [ ] Tool calling with streaming
- [ ] Multi-modal support
- [ ] Advanced context management
- [ ] Production monitoring

## Troubleshooting

### Common Issues

1. **"DaprClient object has no attribute 'converse_stream_alpha1'"**
   - This is expected - streaming is in development
   - Use non-streaming examples for now

2. **"Component not found"**
   - Ensure Dapr sidecar is running
   - Check component configuration files

3. **OpenAI API errors**
   - Verify API key is set correctly
   - Check API key permissions and quotas

### Getting Help

- Check the [Dapr documentation](https://docs.dapr.io/)
- Review component configuration
- Ensure all prerequisites are met
- Look at error messages for specific guidance

## Next Steps

1. **Try the Tool Calling Quickstart**: `../10-dapr-tool-calling/`
2. **Explore Multi-Agent Workflows**: `../05-multi-agent-workflow-actors/`
3. **Build Production Apps**: Use these patterns in your applications

---

**Note**: This quickstart is actively being developed. Streaming functionality will be added as soon as the underlying Dapr Python SDK support is available. The current examples show the foundation that streaming will build upon. 
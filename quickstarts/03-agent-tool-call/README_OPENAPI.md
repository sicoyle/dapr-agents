# OpenAPIReActAgent Example

This example demonstrates how to create an `OpenAPIReActAgent` that can dynamically interact with APIs using OpenAPI specifications. The agent can discover and execute API endpoints based on natural language queries.

## What is OpenAPIReActAgent?

The `OpenAPIReActAgent` is a specialized agent that:

1. **Loads OpenAPI specifications** - Parses API documentation to understand available endpoints
2. **Uses vector search** - Finds relevant API endpoints based on natural language queries
3. **Executes API calls** - Makes actual HTTP requests to the API with proper parameters
4. **Follows ReAct pattern** - Uses reasoning and action cycles for complex API interactions

## Prerequisites

- Python 3.10+
- OpenAI API key
- Optional: `sentence-transformers` for better vector search (install with `pip install sentence-transformers`)

## Setup

1. **Install dependencies**:
   ```bash
   pip install dapr-agents python-dotenv
   # Optional: pip install sentence-transformers
   ```

2. **Set environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Example

```bash
python openapi_agent_example.py
```

## How It Works

### 1. OpenAPI Specification Creation
```python
def create_simple_openapi_spec():
    """Create a simple OpenAPI 3.0 specification for demonstration."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Simple Pet Store API", "version": "1.0.0"},
        "paths": {
            "/pet/findByStatus": {
                "get": {
                    "summary": "Find pets by status",
                    "parameters": [...],
                    "responses": {...}
                }
            }
        }
    }
    return spec

# Create and load the spec
spec_dict = create_simple_openapi_spec()
spec_parser = OpenAPISpecParser.from_file("temp_petstore_openapi.json")
```

**Note**: This example creates a local OpenAPI 3.0 specification to avoid compatibility issues with OpenAPI 2.0 (Swagger) specifications that use deprecated parameter types like `formData` and `body`.

### 2. Vector Store Setup
```python
from dapr_agents.document.embedder import SentenceTransformerEmbedder
from dapr_agents.storage import ChromaVectorStore

# Create embedding function for semantic search
embedding_function = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

# Initialize vector store for API endpoint search
api_vector_store = ChromaVectorStore(
    name="petstore_api_toolbox",
    embedding_function=embedding_function,
)
```

### 3. Agent Creation
```python
from dapr_agents import OpenAPIReActAgent

agent = OpenAPIReActAgent(
    role="Pet Store API Assistant",
    goal="Help users interact with the Pet Store API",
    instructions=[
        "Find appropriate API endpoints for user requests",
        "Execute API calls with proper parameters",
        "Provide clear responses about API operations"
    ],
    spec_parser=spec_parser,
    api_vector_store=api_vector_store,
)
```

### 4. Using the Agent
```python
# Natural language queries that get converted to API calls
response = await agent.run("Get all pets with status 'available'")
response = await agent.run("Find pet by ID 1")
response = await agent.run("Add a new pet with name 'Fluffy'")
```

## Key Features

### Automatic Tool Generation
The agent automatically creates two tools from the OpenAPI specification:

1. **GetOpenapiDefinition** - Searches for relevant API endpoints
2. **OpenApiCallExecutor** - Executes the selected API calls

### Vector Search
The agent uses semantic search to find the most relevant API endpoints for your query, even if you don't know the exact endpoint names.

### Authentication Support
You can add authentication headers for APIs that require them:

```python
agent = OpenAPIReActAgent(
    # ... other parameters ...
    auth_header={"Authorization": "Bearer your-token-here"}
)
```

## OpenAPI Version Compatibility

### OpenAPI 3.0 (Recommended)
- ✅ Fully supported by `OpenAPISpecParser`
- ✅ Uses modern parameter types (`query`, `header`, `path`, `cookie`)
- ✅ Supports `requestBody` for POST/PUT operations
- ✅ Better schema definitions

### OpenAPI 2.0 (Swagger)
- ❌ Not fully compatible with current parser
- ❌ Uses deprecated parameter types (`formData`, `body`)
- ❌ Different schema structure

### Solution for OpenAPI 2.0 APIs
If you need to use an OpenAPI 2.0 API:

1. **Convert the spec** to OpenAPI 3.0 using tools like:
   ```bash
   npm install -g swagger-codegen
   swagger-codegen generate -i swagger.json -l openapi-yaml -o output
   ```

2. **Use a different API** that provides OpenAPI 3.0 documentation

3. **Create a custom OpenAPI 3.0 wrapper** for the API

## Example Output

When you run the example, you'll see the agent:

1. **Creating the OpenAPI spec** and parsing endpoints
2. **Setting up vector search** for endpoint discovery
3. **Creating the agent** with the necessary tools
4. **Executing queries** and showing the reasoning process:
   ```
   Thought: To get pets with status 'available', I need to find the appropriate API endpoint...
   Action: {"name": "GetOpenapiDefinition", "arguments": {...}}
   Observation: Found endpoint: GET /pet/findByStatus
   Action: {"name": "OpenApiCallExecutor", "arguments": {...}}
   Observation: [{"id": 1, "name": "Fluffy", "status": "available"}, ...]
   ```

## Use Cases

- **API Integration**: Connect to any API with OpenAPI 3.0 documentation
- **Dynamic API Discovery**: Let the agent find the right endpoints for your needs
- **Natural Language API Access**: Use plain English to interact with APIs
- **Complex API Workflows**: Chain multiple API calls with reasoning

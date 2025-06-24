#!/usr/bin/env python3
"""
OpenAPIReActAgent Example

This example demonstrates how to create an OpenAPIReActAgent that can interact with APIs
using OpenAPI specifications. The agent can dynamically discover and execute API endpoints
based on natural language queries.

Prerequisites:
- OpenAI API key
- Optional: sentence-transformers for vector search (pip install sentence-transformers)
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from dapr_agents import OpenAPIReActAgent
from dapr_agents.tool.utils import OpenAPISpecParser
from dapr_agents.document.embedder import SentenceTransformerEmbedder
from dapr_agents.storage import ChromaVectorStore

# Load environment variables
load_dotenv()


def create_simple_openapi_spec():
    """
    Create a simple OpenAPI 3.0 specification for demonstration.
    This avoids compatibility issues with OpenAPI 2.0 specs.
    """
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Simple Pet Store API",
            "version": "1.0.0",
            "description": "A simple API for managing pets",
        },
        "servers": [
            {
                "url": "https://petstore.swagger.io/v2",
                "description": "Pet Store API server",
            }
        ],
        "paths": {
            "/pet/findByStatus": {
                "get": {
                    "summary": "Find pets by status",
                    "description": "Multiple status values can be provided with comma separated strings",
                    "operationId": "findPetsByStatus",
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "description": "Status values that need to be considered for filter",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["available", "pending", "sold"],
                                "default": "available",
                            },
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "successful operation",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Pet"},
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/pet": {
                "post": {
                    "summary": "Add a new pet to the store",
                    "description": "Add a new pet to the store",
                    "operationId": "addPet",
                    "requestBody": {
                        "description": "Pet object that needs to be added to the store",
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Pet"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "successful operation",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Pet"}
                                }
                            },
                        }
                    },
                }
            },
            "/pet/{petId}": {
                "get": {
                    "summary": "Find pet by ID",
                    "description": "Returns a single pet",
                    "operationId": "getPetById",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "description": "ID of pet to return",
                            "required": True,
                            "schema": {"type": "integer", "format": "int64"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "successful operation",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Pet"}
                                }
                            },
                        }
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "Pet": {
                    "type": "object",
                    "required": ["name", "photoUrls"],
                    "properties": {
                        "id": {"type": "integer", "format": "int64"},
                        "name": {"type": "string", "example": "doggie"},
                        "status": {
                            "type": "string",
                            "enum": ["available", "pending", "sold"],
                            "default": "available",
                        },
                        "photoUrls": {"type": "array", "items": {"type": "string"}},
                    },
                }
            }
        },
    }
    return spec


async def main():
    """
    Create and run an OpenAPIReActAgent that can interact with a Pet Store API.
    """

    print("Setting up OpenAPIReActAgent...")

    # 1. Create OpenAPI specification (OpenAPI 3.0 compatible)
    print("Creating OpenAPI 3.0 specification...")

    # Create a simple OpenAPI 3.0 spec to avoid compatibility issues
    spec_dict = create_simple_openapi_spec()

    # Save to a temporary file for demonstration
    spec_file = "temp_petstore_openapi.json"
    with open(spec_file, "w") as f:
        json.dump(spec_dict, f, indent=2)

    print(f"Created OpenAPI spec file: {spec_file}")

    # Load the spec from the file
    spec_parser = OpenAPISpecParser.from_file(spec_file)
    print(f"Loaded {len(spec_parser.endpoints)} API endpoints")

    # Clean up the temporary file
    os.remove(spec_file)

    # 2. Set up vector store for API endpoint search
    print("Setting up vector store for API endpoint search...")

    # Initialize embedding function
    embedding_function = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    # Initialize vector store
    api_vector_store = ChromaVectorStore(
        name="petstore_api_toolbox",
        embedding_function=embedding_function,
    )

    # 3. Create OpenAPIReActAgent
    print("Creating OpenAPIReActAgent...")

    agent = OpenAPIReActAgent(
        role="Pet Store API Assistant",
        goal="Help users interact with the Pet Store API",
        instructions=[
            "Find appropriate API endpoints for user requests",
            "Execute API calls with proper parameters",
            "Provide clear responses about API operations",
        ],
        spec_parser=spec_parser,
        api_vector_store=api_vector_store,
    )

    print("OpenAPIReActAgent created successfully!")
    print(f"Available tools: {[tool.name for tool in agent.tool_executor.tools]}")

    # 4. Test the agent with different queries
    test_queries = [
        "Get all pets with status 'available'",
        "Find pet by ID 1",
        "Add a new pet with name 'Fluffy' and status 'available'",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print(f"{'='*60}")

        try:
            response = await agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

    print("\nOpenAPIReActAgent example completed!")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        exit(1)

    # Run the example
    asyncio.run(main())

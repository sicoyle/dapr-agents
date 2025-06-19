# Set your OpenAI API key before running:
# export OPENAI_API_KEY="your-api-key-here"
#
# For OpenAPI agents, you may also need:
# pip install sentence-transformers chromadb

from dapr_agents import Agent
from dapr_agents.tool.utils.openapi import OpenAPISpecParser
# from dapr_agents.storage import VectorStore

async def main():
    # Agent with OpenAPI integration - automatically uses OpenAPIReActAgent
    agent = Agent(
        name="APIBot",
        role="API Integration Assistant",
        goal="Help with API integrations and OpenAPI specifications",
        instructions=[
            "Analyze OpenAPI specifications",
            "Execute API calls",
            "Handle authentication"
        ],
        openapi_spec_path="./api_specs/petstore.yaml",  # Triggers OpenAPIReActAgent
        config_file="configs/api_agent.yaml"
    )
    
    response = await agent.run("Find pets with status 'available'")
    print(response)
    
    response = await agent.run("Add a new pet with name 'Fluffy', status 'available', and photoUrls ['https://example.com/fluffy.jpg']")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 
from dapr_agents import DurableAgent
from dapr_agents.tool import tool

@tool
def process_data(data: str) -> dict:
    """Process and analyze data"""
    return {"analysis": "Complete", "score": 0.95, "processed_data": data}

@tool
def store_results(results: dict) -> str:
    """Store analysis results"""
    return f"Results stored successfully: {results}"

@tool
def get_previous_analysis() -> str:
    """Retrieve previous analysis results"""
    return "Previous analysis: Data processed with 95% accuracy"

async def main():
    # Create the durable agent using config file
    print("Creating agent...")
    agent = DurableAgent(
        name="DataProcessor",
        role="Data Analysis Assistant",
        goal="Process data and maintain state across interactions",
        instructions=[
            "Process and analyze data",
            "Store results persistently",
            "Maintain context across conversations"
        ],
        tools=[process_data, store_results, get_previous_analysis],
        config_file="configs/data_agent.yaml"
    )
    
    agent = agent.as_service()
    
    print(f"Agent created successfully. Type: {type(agent).__name__}")
    print("Starting durable agent as a service...")
    print("The agent is now running and ready to receive requests.")
    print("You can interact with it via HTTP endpoints or other Dapr services.")
    print("Press Ctrl+C to stop the service.")
    
    # Start the agent service
    await agent.start()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 
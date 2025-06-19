from dapr_agents import Agent
from dapr_agents.tool import tool

@tool
def complex_analysis(data: str, parameters: dict) -> dict:
    """Perform complex data analysis"""
    return {
        "analysis_type": "complex",
        "input_data": data,
        "parameters": parameters,
        "result": "Analysis completed successfully"
    }

async def main():
    # Advanced agent with shared configuration
    agent = Agent(
        name="AdvancedBot",
        role="Advanced Analysis Assistant",
        goal="Perform complex analyses with advanced configuration",
        instructions=[
            "Perform complex data analysis",
            "Use advanced reasoning",
            "Maintain persistent state"
        ],
        tools=[complex_analysis],
        reasoning=True,
        config_file="configs/advanced_agent.yaml"
    )
    
    response = await agent.run("Analyze this data with custom parameters")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 
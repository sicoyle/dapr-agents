from dapr_agents import Agent
from dapr_agents.tool import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: Sunny, 75°F"


@tool
def get_forecast(city: str, days: int = 5) -> str:
    """Get weather forecast for a city"""
    return f"Forecast for {city} next {days} days: Mostly sunny"


async def main():
    # Simple agent - automatically uses ToolCallAgent
    agent = Agent(
        name="WeatherBot",
        role="Weather Assistant",
        goal="Provide weather information",
        instructions=["Get current weather", "Provide forecasts"],
        tools=[get_weather, get_forecast],
        config_file="configs/weather_agent.yaml",
    )

    response = await agent.run("What's the weather in New York?")
    print(response)

    response = await agent.run("What's the 3-day forecast for London?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

import asyncio
from dapr_agents import tool, Agent, DurableAgent
from dapr_agents import OpenAIChatClient
from dapr_agents.agents.durableagent.storage import Storage


@tool
def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"


async def main():
    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[my_weather_func],
        message_bus_name="messagepubsub",
        storage=Storage(
            name="workflowstatestore",

            # Optional
            session_id="weather_agent_session",
            local_directory="./temporary-state",
        ),
        llm=OpenAIChatClient(model="gpt-3.5-turbo"),
    )
    try:
        # Can override session_id per run
        response = await weather_agent.run(
            "What's the weather?",
            # session_id="new_session"  # Optional: override default session
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
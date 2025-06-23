from dapr_agents import Agent
from dapr_agents.tool import tool


@tool
def search_flights(from_city: str, to_city: str, date: str) -> str:
    """Search for available flights"""
    return (
        f"Found flights from {from_city} to {to_city} on {date}: Flight 123, Flight 456"
    )


@tool
def book_hotel(city: str, check_in: str, check_out: str) -> str:
    """Book a hotel in the specified city"""
    return f"Booked hotel in {city} from {check_in} to {check_out}"


@tool
def get_attractions(city: str) -> str:
    """Get tourist attractions in a city"""
    return f"Top attractions in {city}: Museum, Park, Historic District"


async def main():
    # Agent that needs reasoning - automatically uses ReActAgent
    agent = Agent(
        name="TravelPlanner",
        role="Travel Assistant",
        goal="Plan travel itineraries with reasoning",
        instructions=[
            "Plan travel itineraries",
            "Search for flights and hotels",
            "Provide reasoning for recommendations",
        ],
        tools=[search_flights, book_hotel, get_attractions],
        reasoning=True,  # Triggers ReActAgent
        config_file="configs/travel_agent.yaml",
    )

    response = await agent.run("Plan a 3-day trip to Paris")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

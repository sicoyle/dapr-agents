import asyncio
import logging

from agent_tools import tools
from dotenv import load_dotenv
import time

from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents import DurableAgent
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.agents.configs import AgentMemoryConfig, AgentStateConfig


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger(__name__)

    memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="agentstatestore",
            session_id=f"sam-session",
        )
    )
    state = AgentStateConfig(
        store=StateStoreService(store_name="agentstatestore", key_prefix="sam")
    )

    # Instantiate your agent
    weather_agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data.",
        ],
        memory=memory,
        state=state,
        tools=tools,
    )
    # Start the agent (registers workflows with the runtime)
    weather_agent.start()

    # Create an AgentRunner to execute the workflow
    runner = AgentRunner()

    try:
        # prompt = "Send me the weather in Austin, Texas, then call the tool to jump 5 feet, then calculate 2+3"

        # # Run the workflow and wait for completion
        # result = await runner.run(
        #     weather_agent,
        #     payload={"task": prompt},
        # )

        # print(f"\n✅ Final Result:\n{result}\n", flush=True)
        time.sleep(30)

    except Exception as e:
        logger.error(f"Error running workflow: {e}", exc_info=True)
        raise
    finally:
        # Stop agent first (tears down durabletask runtime)
        weather_agent.stop()
        # Then shut down runner (unwire/close clients)
        runner.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

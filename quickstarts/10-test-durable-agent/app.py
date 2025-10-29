from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.prompting import AgentProfileConfig
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.workflow.utils.core import wait_for_shutdown

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("durable-agent-app")


async def main() -> None:
    """
    Minimal durable-agent quickstart.

    Notes:
    - If you do not specify a custom state/message model, AgentStateConfig defaults to
      AgentWorkflowState / AgentWorkflowMessage under the hood.
    """
    agent_name = "blog-agent"

    # --- Pub/Sub & State/Registry wiring -------------------------------------
    pubsub_config = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="blog.requests",
        broadcast_topic="agents.broadcast",
    )

    state_config = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore", key_prefix="blog:")
        # No default_state/state_model_cls/message_model_cls → uses defaults.
    )

    registry_config = AgentRegistryConfig(
        store=StateStoreService(store_name="agentregistrystore"),
        team_name="bloggers",
    )

    # --- Profile / Prompting ---------------------------------------------------
    profile = AgentProfileConfig(
        name="Blog Agent",
        role="AI Blogger",
        goal="Write engaging blog updates and summarize content for readers.",
        instructions=[
            "Summarize new information clearly in 2-3 sentences.",
            "Recommend one follow-up topic the audience might enjoy.",
        ],
        style_guidelines=[
            "Use friendly, professional tone.",
            "Avoid technical jargon unless asked.",
        ],
    )

    # --- Memory (Dapr-backed conversation history) -----------------------------
    memory_config = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="memorystore",
            session_id=f"{agent_name}-session",
        )
    )

    # --- LLM Client ------------------------------------------------------------
    llm = OpenAIChatClient()

    # --- Assemble durable agent ------------------------------------------------
    agent = DurableAgent(
        profile_config=profile,
        pubsub_config=pubsub_config,
        state_config=state_config,
        registry_config=registry_config,
        memory_config=memory_config,
        llm=llm,
    )
    agent.start()

    # --- HTTP runner for workflow endpoints -----------------------------------
    runner = AgentRunner()
    try:
        runner.register_routes(agent)
        await wait_for_shutdown()
    finally:
        runner.shutdown()
        agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
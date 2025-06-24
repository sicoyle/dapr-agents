#!/usr/bin/env python3
"""
Example demonstrating how AgenticWorkflow can load configuration from YAML
and override it with instantiation parameters.
"""

import asyncio
from dapr_agents.workflow.agentic import AgenticWorkflow


async def main():
    # Example 1: Using YAML config only
    print("=== Example 1: YAML config only ===")
    workflow1 = AgenticWorkflow.from_config("workflow_config_example.yaml")
    print(f"Message bus: {workflow1.message_bus_name}")
    print(f"State store: {workflow1.state_store_name}")
    print(f"Max iterations: {workflow1.max_iterations}")
    print()

    # Example 2: Overriding YAML config with instantiation parameters
    print("=== Example 2: YAML config + overrides ===")
    workflow2 = AgenticWorkflow.from_config(
        "workflow_config_example.yaml",
        message_bus_name="custom_message_bus",  # Override YAML value
        max_iterations=50,  # Override YAML value
        name="CustomWorkflow",  # Override YAML value
    )
    print(f"Message bus: {workflow2.message_bus_name} (overridden)")
    print(f"State store: {workflow2.state_store_name} (from YAML)")
    print(f"Max iterations: {workflow2.max_iterations} (overridden)")
    print(f"Name: {workflow2.name} (overridden)")
    print()

    # Example 3: Direct instantiation with config_file
    print("=== Example 3: Direct instantiation with config_file ===")
    workflow3 = AgenticWorkflow(
        name="DirectWorkflow",
        message_bus_name="direct_message_bus",
        state_store_name="direct_state_store",
        config_file="workflow_config_example.yaml",  # YAML config loaded but not applied
    )
    print(f"Message bus: {workflow3.message_bus_name} (from instantiation)")
    print(f"State store: {workflow3.state_store_name} (from instantiation)")
    print(f"Max iterations: {workflow3.max_iterations} (default)")
    print(f"Name: {workflow3.name} (from instantiation)")
    print()

    print("✅ Configuration override examples completed!")


if __name__ == "__main__":
    asyncio.run(main())

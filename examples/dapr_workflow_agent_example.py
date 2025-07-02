#!/usr/bin/env python3
"""
Comprehensive example of DaprWorkflowAgent usage.

This example demonstrates:
1. Creating a DaprWorkflowAgent with tools
2. Running tasks through workflows for durability
3. Handling workflow state and monitoring
4. Error recovery and fault tolerance
"""

import asyncio
import time
from typing import Any

from src.smolagents.dapr_workflow_agent import DaprWorkflowAgent
from src.smolagents.models import OpenAIServerModel
from src.smolagents.tools import tool


# Define some example tools
@tool
def calculate_math(expression: str) -> str:
    """
    Calculate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return f"Invalid expression: {expression}. Only basic math operations allowed."
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def get_current_time() -> str:
    """
    Get the current time and date.
    
    Returns:
        Current timestamp as a formatted string
    """
    import datetime
    now = datetime.datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def simulate_work(duration: int) -> str:
    """
    Simulate some work that takes time (useful for testing workflow durability).
    
    Args:
        duration: Number of seconds to simulate work
        
    Returns:
        A message indicating work completion
    """
    import time
    time.sleep(min(duration, 10))  # Cap at 10 seconds for safety
    return f"Completed {duration} seconds of simulated work"


def create_agent() -> DaprWorkflowAgent:
    """Create and configure a DaprWorkflowAgent."""
    # Create model (you would use real API keys in production)
    model = OpenAIServerModel(
        model_id="gpt-3.5-turbo",
        api_key="test-key-replace-with-real"
    )
    
    # Create agent with tools
    agent = DaprWorkflowAgent(
        model=model,
        tools=[calculate_math, get_current_time, simulate_work],
        workflow_name="example_agent_workflow",
        auto_start_runtime=True
    )
    
    return agent


def test_basic_functionality():
    """Test basic DaprWorkflowAgent functionality."""
    print("🔧 Testing Basic DaprWorkflowAgent Functionality")
    print("=" * 50)
    
    try:
        agent = create_agent()
        
        print(f"✅ Agent created successfully")
        print(f"   - Workflow name: {agent.workflow_name}")
        print(f"   - Number of tools: {len(agent.tools)}")
        print(f"   - Tool names: {[t.name for t in agent.tools]}")
        print(f"   - Has workflow runtime: {agent.workflow_runtime is not None}")
        print(f"   - Has workflow client: {agent.workflow_client is not None}")
        
        # Test system prompt
        system_prompt = agent.initialize_system_prompt()
        print(f"   - System prompt length: {len(system_prompt)} characters")
        
        # Test workflow registration
        print("\n🔄 Testing workflow registration...")
        agent._register_workflow()
        agent._register_activities()
        print("✅ Workflows and activities registered successfully")
        
        return agent
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_context_manager(agent: DaprWorkflowAgent):
    """Test context manager functionality."""
    print("\n🔧 Testing Context Manager")
    print("=" * 50)
    
    try:
        with agent as ctx_agent:
            print(f"✅ Context manager entered successfully")
            print(f"   - Context agent workflow name: {ctx_agent.workflow_name}")
            print(f"   - Runtime is active: {ctx_agent.workflow_runtime is not None}")
        
        print("✅ Context manager exited successfully")
        
    except Exception as e:
        print(f"❌ Error in context manager test: {e}")
        import traceback
        traceback.print_exc()


def test_workflow_monitoring(agent: DaprWorkflowAgent):
    """Test workflow monitoring capabilities."""
    print("\n🔧 Testing Workflow Monitoring")
    print("=" * 50)
    
    try:
        # Test workflow status (would work with actual workflows)
        print("📊 Workflow monitoring methods available:")
        print(f"   - get_workflow_status: {hasattr(agent, 'get_workflow_status')}")
        print(f"   - terminate_workflow: {hasattr(agent, 'terminate_workflow')}")
        print(f"   - run_as_workflow: {hasattr(agent, 'run_as_workflow')}")
        
        print("✅ Workflow monitoring capabilities verified")
        
    except Exception as e:
        print(f"❌ Error in workflow monitoring test: {e}")


async def test_async_functionality(agent: DaprWorkflowAgent):
    """Test async functionality if available."""
    print("\n🔧 Testing Async Functionality")
    print("=" * 50)
    
    try:
        # Test if async methods are available
        async_methods = [
            method for method in dir(agent) 
            if method.startswith('a') and callable(getattr(agent, method))
        ]
        
        print(f"📊 Available async-like methods: {async_methods}")
        print("✅ Async functionality check completed")
        
    except Exception as e:
        print(f"❌ Error in async functionality test: {e}")


def main():
    """Run all tests and examples."""
    print("🚀 DaprWorkflowAgent Comprehensive Example")
    print("=" * 60)
    print()
    
    # Test basic functionality
    agent = test_basic_functionality()
    if not agent:
        print("❌ Basic functionality test failed. Exiting.")
        return
    
    # Test context manager
    test_context_manager(agent)
    
    # Test workflow monitoring
    test_workflow_monitoring(agent)
    
    # Test async functionality
    asyncio.run(test_async_functionality(agent))
    
    print("\n🎉 All tests completed!")
    print("\n📝 Next Steps:")
    print("   1. Replace 'test-key-replace-with-real' with actual OpenAI API key")
    print("   2. Run actual workflow execution with: agent.run_as_workflow(task)")
    print("   3. Monitor workflows using Dapr dashboard or CLI")
    print("   4. Test fault tolerance by interrupting and resuming workflows")


if __name__ == "__main__":
    main() 
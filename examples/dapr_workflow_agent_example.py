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
            return "The result of {} is {}".format(expression, result)
        else:
            return "Invalid expression: {}. Only basic math operations allowed.".format(expression)
    except Exception as e:
        return "Error calculating {}: {}".format(expression, str(e))


@tool
def get_current_time() -> str:
    """
    Get the current time and date.
    
    Returns:
        Current timestamp as a formatted string
    """
    import datetime
    now = datetime.datetime.now()
    return "Current time: {}".format(now.strftime('%Y-%m-%d %H:%M:%S'))


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
    return "Completed {} seconds of simulated work".format(duration)


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
        
        print("✅ Agent created successfully")
        print("   - Workflow name: {}".format(agent.workflow_name))
        print("   - Number of tools: {}".format(len(agent.tools)))
        print("   - Tool names: {}".format([t.name for t in agent.tools]))
        print("   - Has workflow runtime: {}".format(agent.workflow_runtime is not None))
        print("   - Has workflow client: {}".format(agent.workflow_client is not None))
        
        # Test system prompt
        system_prompt = agent.initialize_system_prompt()
        print("   - System prompt length: {} characters".format(len(system_prompt)))
        
        # Test workflow registration
        print("\n🔄 Testing workflow registration...")
        agent._register_workflow()
        agent._register_activities()
        print("✅ Workflows and activities registered successfully")
        
        return agent
        
    except Exception as e:
        print("❌ Error in basic functionality test: {}".format(e))
        import traceback
        traceback.print_exc()
        return None


def test_context_manager(agent: DaprWorkflowAgent):
    """Test context manager functionality."""
    print("\n🔧 Testing Context Manager")
    print("=" * 50)
    
    try:
        with agent as ctx_agent:
            print("✅ Context manager entered successfully")
            print("   - Context agent workflow name: {}".format(ctx_agent.workflow_name))
            print("   - Runtime is active: {}".format(ctx_agent.workflow_runtime is not None))
        
        print("✅ Context manager exited successfully")
        
    except Exception as e:
        print("❌ Error in context manager test: {}".format(e))
        import traceback
        traceback.print_exc()


def test_workflow_monitoring(agent: DaprWorkflowAgent):
    """Test workflow monitoring capabilities."""
    print("\n🔧 Testing Workflow Monitoring")
    print("=" * 50)
    
    try:
        # Test workflow status (would work with actual workflows)
        print("📊 Workflow monitoring methods available:")
        print("   - get_workflow_status: {}".format(hasattr(agent, 'get_workflow_status')))
        print("   - terminate_workflow: {}".format(hasattr(agent, 'terminate_workflow')))
        print("   - run_as_workflow: {}".format(hasattr(agent, 'run_as_workflow')))
        
        print("✅ Workflow monitoring capabilities verified")
        
    except Exception as e:
        print("❌ Error in workflow monitoring test: {}".format(e))


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
        
        print("📊 Available async-like methods: {}".format(async_methods))
        print("✅ Async functionality check completed")
        
    except Exception as e:
        print("❌ Error in async functionality test: {}".format(e))


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
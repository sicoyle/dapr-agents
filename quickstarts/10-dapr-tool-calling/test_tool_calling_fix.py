#!/usr/bin/env python3

"""
Test script to verify tool calling fix is working
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dapr_agents.tool import tool
from dapr_agents.llm.dapr import DaprChatClient

def load_env_file():
    """Load environment variables from .env file"""
    env_file = project_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"✅ Loaded environment from {env_file}")
        except ImportError:
            print("⚠️  python-dotenv not available, skipping .env file")
    else:
        print("⚠️  No .env file found")

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: 72°F, sunny"

def test_tool_calling_fix():
    """Test that tools are now properly passed to the Dapr conversation API"""
    print("🔧 Testing Tool Calling Fix")
    print("=" * 50)
    
    # Load environment
    load_env_file()
    
    # Create client
    client = DaprChatClient()
    
    # Test with different providers
    providers = ["echo-tools", "openai", "anthropic"]
    
    for provider in providers:
        print(f"\n🧪 Testing with provider: {provider}")
        
        try:
            response = client.generate(
                messages=[
                    {"role": "user", "content": "What's the weather in San Francisco? Use the weather tool."}
                ],
                tools=[get_weather],
                llm_component=provider,
                stream=False,
                temperature=0.7
            )
            
            print(f"✅ Response received from {provider}")
            print(f"📝 Response: {json.dumps(response, indent=2)}")
            
            # Check if response contains tool calls
            if response and "outputs" in response:
                output = response["outputs"][0]
                if "tool_calls" in output:
                    print(f"🎉 Tool calls detected! {len(output['tool_calls'])} tool call(s)")
                    for i, tool_call in enumerate(output["tool_calls"]):
                        print(f"   Tool call {i+1}: {tool_call}")
                else:
                    print(f"ℹ️  No tool calls in response (provider may not support tools)")
            
        except Exception as e:
            print(f"❌ Error with {provider}: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_tool_calling_fix() 
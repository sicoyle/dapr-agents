#!/usr/bin/env python3

"""
Comprehensive debug script to trace tool calling request pipeline
"""

import json
import os
from pathlib import Path
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput, Tool
import logging

# Set up logging to see internal operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_file():
    """Load environment variables from .env file"""
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"✅ Loaded environment from {env_file}")
        except ImportError:
            print("❌ python-dotenv not available")
    else:
        print(f"❌ No .env file found at {env_file}")

@tool
def test_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {location} is sunny and 72°F"

def debug_tool_conversion():
    """Debug how tools are converted to SDK format"""
    print("\n🔧 TOOL CONVERSION DEBUG")
    print("=" * 50)
    
    agent_tool = test_weather
    print(f"📋 Original tool: {agent_tool.name}")
    print(f"📋 Tool description: {agent_tool.description}")
    print(f"📋 Tool args_schema: {json.dumps(agent_tool.args_schema, indent=2)}")
    
    # Show the tool's OpenAI function definition
    openai_def = agent_tool.to_function_call("openai")
    print(f"📋 OpenAI function definition: {json.dumps(openai_def, indent=2)}")
    
    # Test the conversion process
    client = DaprChatClient()
    sdk_tools = client._convert_tools_to_sdk_format([agent_tool])
    
    print(f"\n🔄 Converted to {len(sdk_tools)} SDK tools:")
    for i, sdk_tool in enumerate(sdk_tools):
        print(f"  Tool {i+1}:")
        print(f"    Type: {sdk_tool.type}")
        print(f"    Name: {sdk_tool.name}")
        print(f"    Description: {sdk_tool.description}")
        print(f"    Parameters: {sdk_tool.parameters}")
    
    return sdk_tools

def debug_conversation_input():
    """Debug how ConversationInput is created with tools"""
    print("\n📝 CONVERSATION INPUT DEBUG")
    print("=" * 50)
    
    # Get converted tools
    client = DaprChatClient()
    sdk_tools = client._convert_tools_to_sdk_format([test_weather])
    
    # Create conversation inputs
    inputs = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    conversation_inputs = client.convert_to_conversation_inputs(inputs, sdk_tools)
    
    print(f"📋 Created {len(conversation_inputs)} conversation inputs:")
    for i, conv_input in enumerate(conversation_inputs):
        print(f"  Input {i+1}:")
        print(f"    Content: {conv_input.content}")
        print(f"    Role: {conv_input.role}")
        print(f"    Tools: {len(conv_input.tools) if conv_input.tools else 0}")
        if conv_input.tools:
            for j, tool in enumerate(conv_input.tools):
                print(f"      Tool {j+1}: {tool.name}")
    
    return conversation_inputs

def debug_raw_dapr_call():
    """Debug raw Dapr conversation API call"""
    print("\n🌐 RAW DAPR API DEBUG")
    print("=" * 50)
    
    # Create tools manually
    sdk_tool = Tool(
        type="function",
        name="test_weather",
        description="Get weather information for a location.",
        parameters=json.dumps({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        })
    )
    
    # Create conversation input with tools
    conv_input = ConversationInput(
        content="What's the weather in San Francisco? Use the weather tool.",
        role="user",
        tools=[sdk_tool]
    )
    
    print(f"📋 ConversationInput created:")
    print(f"  Content: {conv_input.content}")
    print(f"  Role: {conv_input.role}")
    print(f"  Tools: {len(conv_input.tools)} tool(s)")
    
    # Test with different providers
    providers = ["openai", "anthropic", "echo-tools"]
    
    for provider in providers:
        print(f"\n🧪 Testing with {provider}:")
        try:
            with DaprClient() as client:
                response = client.converse_alpha1(
                    name=provider,
                    inputs=[conv_input]
                )
                
                print(f"  ✅ Response received")
                print(f"  📤 Response outputs: {len(response.outputs)}")
                
                for i, output in enumerate(response.outputs):
                    print(f"    Output {i+1}:")
                    print(f"      Result: {output.result[:100]}...")
                    
                    # Check for tool calls
                    if hasattr(output, 'tool_calls') and output.tool_calls:
                        print(f"      ✅ Tool calls: {len(output.tool_calls)}")
                        for j, tool_call in enumerate(output.tool_calls):
                            print(f"        Tool call {j+1}:")
                            print(f"          ID: {tool_call.id}")
                            if hasattr(tool_call, 'function'):
                                print(f"          Function: {tool_call.function.name}")
                                print(f"          Arguments: {tool_call.function.arguments}")
                            else:
                                print(f"          Raw tool call: {tool_call}")
                    else:
                        print(f"      ❌ No tool calls found")
                        
        except Exception as e:
            print(f"  ❌ Error: {e}")

def debug_dapr_chat_client_full_flow():
    """Debug the full DaprChatClient flow"""
    print("\n🤖 DAPR CHAT CLIENT FULL FLOW DEBUG")
    print("=" * 50)
    
    load_env_file()
    
    client = DaprChatClient()
    
    # Test with different providers
    providers = ["openai", "anthropic", "echo-tools"]
    
    for provider in providers:
        print(f"\n🧪 Testing DaprChatClient with {provider}:")
        
        try:
            response = client.generate(
                messages=[{
                    "role": "user", 
                    "content": "What's the weather in San Francisco? Please use the test_weather tool."
                }],
                llm_component=provider,
                tools=[test_weather],
                stream=False,
            )
            
            print(f"  ✅ Response received")
            print(f"  📤 Response type: {type(response)}")
            print(f"  📤 Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            
            # Print full response for debugging
            print(f"  📤 Full response:")
            print(json.dumps(response, indent=4, default=str))
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run all debugging functions"""
    print("🔍 COMPREHENSIVE TOOL CALLING DEBUG")
    print("=" * 60)
    
    # Step 1: Debug tool conversion
    debug_tool_conversion()
    
    # Step 2: Debug conversation input creation
    debug_conversation_input()
    
    # Step 3: Debug raw Dapr API call
    debug_raw_dapr_call()
    
    # Step 4: Debug full DaprChatClient flow
    debug_dapr_chat_client_full_flow()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUG COMPLETE")
    print("=" * 60)
    print("Check the output above to identify where tools are lost in the pipeline")

if __name__ == "__main__":
    main() 
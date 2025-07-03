#!/usr/bin/env python3

"""
Debug script to understand why tools aren't being called
"""

import json
from pathlib import Path
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool


# Load environment variables
def load_env_file():
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except ImportError:
            pass


@tool
def simple_tool(message: str) -> str:
    """A simple test tool that just returns a formatted message."""
    return f"Tool called with: {message}"


def test_tool_calling():
    """Test tool calling with debug output."""
    print("🔍 Debugging Tool Calling")
    print("=" * 40)

    load_env_file()

    client = DaprChatClient()
    tools = [simple_tool]

    print(
        f"📋 Tools to pass: {[getattr(tool, 'name', getattr(tool, '__name__', str(tool))) for tool in tools]}"
    )

    # Test message that should trigger tool use
    message = "Please use the simple_tool with the message 'Hello World'"

    print(f"📝 User message: {message}")
    print()

    # Test with OpenAI
    print("🧪 Testing with OpenAI...")
    try:
        response = client.generate(
            messages=[{"role": "user", "content": message}],
            llm_component="openai",
            tools=tools,
            stream=False,
        )

        print("📤 Full OpenAI response:")
        print(json.dumps(response, indent=2, default=str))
        print()

        # Check if tools were called
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                print("✅ Tool calls found in OpenAI response!")
                for tool_call in choice.message.tool_calls:
                    print(f"   Tool: {tool_call.function.name}")
                    print(f"   Args: {tool_call.function.arguments}")
            else:
                print("❌ No tool calls found in OpenAI response")

    except Exception as e:
        print(f"❌ OpenAI error: {e}")

    print()

    # Test with echo-tools
    print("🧪 Testing with echo-tools...")
    try:
        response = client.generate(
            messages=[{"role": "user", "content": message}],
            llm_component="echo-tools",
            tools=tools,
            stream=False,
        )

        print("📤 Full echo-tools response:")
        print(json.dumps(response, indent=2, default=str))
        print()

        # Check if tools were called
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                print("✅ Tool calls found in echo-tools response!")
                for tool_call in choice.message.tool_calls:
                    print(f"   Tool: {tool_call.function.name}")
                    print(f"   Args: {tool_call.function.arguments}")
            else:
                print("❌ No tool calls found in echo-tools response")

    except Exception as e:
        print(f"❌ echo-tools error: {e}")

    print()

    # Test with Anthropic
    print("🧪 Testing with Anthropic...")
    try:
        response = client.generate(
            messages=[{"role": "user", "content": message}],
            llm_component="anthropic",
            tools=tools,
            stream=False,
        )

        print("📤 Full Anthropic response:")
        print(json.dumps(response, indent=2, default=str))
        print()

        # Check if tools were called
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                print("✅ Tool calls found in Anthropic response!")
                for tool_call in choice.message.tool_calls:
                    print(f"   Tool: {tool_call.function.name}")
                    print(f"   Args: {tool_call.function.arguments}")
            else:
                print("❌ No tool calls found in Anthropic response")

    except Exception as e:
        print(f"❌ Anthropic error: {e}")


if __name__ == "__main__":
    test_tool_calling()

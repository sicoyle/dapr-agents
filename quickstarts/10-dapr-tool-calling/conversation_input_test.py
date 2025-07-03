#!/usr/bin/env python3

"""
Simple test to inspect ConversationInput interface
"""

from dapr.clients.grpc._request import ConversationInput, Tool
import inspect


def test_conversation_input():
    """Test what ConversationInput accepts"""
    print("🔍 CONVERSATION INPUT INTERFACE TEST")
    print("=" * 50)

    # Check ConversationInput signature
    sig = inspect.signature(ConversationInput.__init__)
    print(f"📋 ConversationInput.__init__ signature:")
    print(f"   {sig}")

    # Check ConversationInput attributes
    print(f"\n📋 ConversationInput attributes:")
    for attr in dir(ConversationInput):
        if not attr.startswith("_"):
            print(f"   {attr}")

    # Try to create a basic ConversationInput
    print(f"\n🧪 Creating basic ConversationInput:")
    try:
        conv_input = ConversationInput(content="Hello", role="user")
        print(f"   ✅ Success: {conv_input}")
        print(f"   Content: {conv_input.content}")
        print(f"   Role: {conv_input.role}")

        # Check if it has tools attribute
        if hasattr(conv_input, "tools"):
            print(f"   Tools attribute: {conv_input.tools}")
        else:
            print(f"   ❌ No tools attribute")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Check Tool interface
    print(f"\n📋 Tool interface:")
    tool_sig = inspect.signature(Tool.__init__)
    print(f"   Tool.__init__ signature: {tool_sig}")

    # Try to create a Tool
    print(f"\n🧪 Creating Tool:")
    try:
        tool = Tool(
            type="function",
            name="test_tool",
            description="A test tool",
            parameters='{"type": "object"}',
        )
        print(f"   ✅ Success: {tool}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_conversation_input()

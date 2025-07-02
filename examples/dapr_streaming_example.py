#!/usr/bin/env python3

"""
Example demonstrating Dapr streaming chat completion using dapr-agents.

This example shows how to use the streaming capability with Dapr LLM components.

Prerequisites:
- Dapr sidecar running with LLM components
- Set DAPR_LLM_COMPONENT_DEFAULT environment variable
"""

import asyncio
import os
from dapr_agents.llm.dapr import DaprChatClient


def streaming_example():
    """Example of streaming chat completion with Dapr."""
    print("🚀 Dapr Streaming Chat Completion Example")
    
    # Initialize the Dapr chat client
    client = DaprChatClient()
    
    # Example messages
    messages = [
        {"role": "user", "content": "Tell me a story about a robot learning to paint"}
    ]
    
    print("\n📡 Starting streaming conversation...")
    print("🤖 Assistant: ", end="", flush=True)
    
    try:
        # Generate streaming response
        response_stream = client.generate(
            messages=messages,
            stream=True,
            temperature=0.7
        )
        
        # Process streaming chunks
        full_content = ""
        for chunk in response_stream:
            if chunk.get("type") == "content":
                print(chunk["data"], end="", flush=True)
                full_content += chunk["data"]
            elif chunk.get("type") == "final_content":
                print(f"\n\n✅ Complete response: {len(chunk['data'])} characters")
            elif chunk.get("type") == "final_usage":
                usage = chunk["data"]
                print(f"💰 Usage: {usage.get('total_tokens', 'N/A')} tokens")
        
        # Also show the total content received
        if full_content:
            print(f"\n📝 Total content received: {len(full_content)} characters")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. Dapr sidecar is running")
        print("   2. DAPR_LLM_COMPONENT_DEFAULT environment variable is set")
        print("   3. LLM component is properly configured")


def non_streaming_example():
    """Example of non-streaming chat completion with Dapr for comparison."""
    print("\n🔄 Non-streaming Chat Completion Example")
    
    # Initialize the Dapr chat client
    client = DaprChatClient()
    
    # Example messages
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    print("\n📞 Making non-streaming request...")
    
    try:
        # Generate non-streaming response
        response = client.generate(
            messages=messages,
            stream=False,
            temperature=0.3
        )
        
        print(f"🤖 Assistant: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function to run examples."""
    print("=" * 60)
    print("  Dapr Streaming Chat Completion Examples")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("DAPR_LLM_COMPONENT_DEFAULT"):
        print("⚠️  Warning: DAPR_LLM_COMPONENT_DEFAULT not set")
        print("   Setting default to 'echo' for demo purposes")
        os.environ["DAPR_LLM_COMPONENT_DEFAULT"] = "echo"
    
    # Run streaming example
    streaming_example()
    
    # Run non-streaming example for comparison
    non_streaming_example()
    
    print("\n" + "=" * 60)
    print("  Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 
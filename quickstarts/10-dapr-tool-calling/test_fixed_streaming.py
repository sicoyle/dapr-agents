#!/usr/bin/env python3

"""
Test script for the fixed streaming transformation
"""

import os
import logging
from dotenv import load_dotenv
from dapr_agents.llm import DaprChatClient
from dapr_agents.tool import tool

# Set up logging to see debug output
logging.basicConfig(level=logging.DEBUG)

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def main():
    load_dotenv()
    
    print("🧪 Testing Fixed Streaming Transformation")
    print("=" * 50)
    
    # Initialize client
    llm = DaprChatClient()
    
    # Simple tool calling test
    prompt = "What time is it?"
    tools = [get_current_time]
    
    print(f"📝 Prompt: {prompt}")
    print(f"🔧 Tools: {[tool.name for tool in tools]}")
    print("\n🔄 Streaming Response:")
    print("-" * 30)
    
    try:
        # Test streaming with tools
        response_stream = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            stream=True,
            llm_component="openai",
            temperature=0.1,
        )
        
        chunk_count = 0
        for chunk in response_stream:
            chunk_count += 1
            print(f"\n📦 Chunk {chunk_count}:")
            print(f"   Type: {type(chunk)}")
            print(f"   Content: {chunk}")
            
            # Check for choices
            if isinstance(chunk, dict) and "choices" in chunk:
                choices = chunk["choices"]
                if choices:
                    choice = choices[0]
                    if "delta" in choice and choice["delta"]:
                        delta = choice["delta"]
                        if "content" in delta:
                            print(f"   📝 Content: '{delta['content']}'")
                        if "tool_calls" in delta:
                            print(f"   🔧 Tool calls: {len(delta['tool_calls'])}")
                            for tc in delta["tool_calls"]:
                                print(f"      • {tc['function']['name']}")
                    if choice.get("finish_reason"):
                        print(f"   🏁 Finish reason: {choice['finish_reason']}")
            
            # Check for usage
            if isinstance(chunk, dict) and chunk.get("usage"):
                print(f"   📊 Usage: {chunk['usage']}")
        
        print(f"\n✅ Test completed! Processed {chunk_count} chunks")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

"""
Test script to verify streaming tool calling fix is working
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
    return f"Weather in {location}: sunny, 72°F"

def test_streaming_tool_calling():
    """Test streaming tool calling with real LLM providers"""
    print("🧪 STREAMING TOOL CALLING TEST")
    print("=" * 50)
    
    load_env_file()
    
    # Test with OpenAI
    print("\n🔥 Testing OpenAI Streaming...")
    client = DaprChatClient()
    
    try:
        # Create a streaming request with tools using the correct API
        stream = client.generate(
            messages="What's the weather like in San Francisco?",
            llm_component="openai",
            tools=[get_weather],
            stream=True  # Enable streaming
        )
        
        print("📡 Streaming response chunks:")
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            print(f"  Chunk {chunk_count}: {json.dumps(chunk, indent=2)}")
            
            # Stop after a few chunks to avoid too much output
            if chunk_count >= 5:
                print("  ... (stopping after 5 chunks)")
                break
                
        print(f"✅ OpenAI streaming completed ({chunk_count} chunks received)")
        
    except Exception as e:
        print(f"❌ OpenAI streaming failed: {e}")
    
    # Test with Anthropic
    print("\n🔥 Testing Anthropic Streaming...")
    
    try:
        # Create a streaming request with tools using the correct API
        stream = client.generate(
            messages="What's the weather like in New York?",
            llm_component="anthropic",
            tools=[get_weather],
            stream=True  # Enable streaming
        )
        
        print("📡 Streaming response chunks:")
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            print(f"  Chunk {chunk_count}: {json.dumps(chunk, indent=2)}")
            
            # Stop after a few chunks to avoid too much output
            if chunk_count >= 5:
                print("  ... (stopping after 5 chunks)")
                break
                
        print(f"✅ Anthropic streaming completed ({chunk_count} chunks received)")
        
    except Exception as e:
        print(f"❌ Anthropic streaming failed: {e}")

if __name__ == "__main__":
    test_streaming_tool_calling() 
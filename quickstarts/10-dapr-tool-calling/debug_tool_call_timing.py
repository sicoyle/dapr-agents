#!/usr/bin/env python3

"""
Debug Tool Call Timing in Streaming Responses

This test specifically examines when and where tool calls appear in streaming responses
for both OpenAI and Anthropic providers.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
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
            print("⚠️  python-dotenv not available")

@tool
def get_weather(location: str) -> str:
    """Get current weather information for a location."""
    return f"Weather in {location}: sunny, 72°F"

def debug_streaming_response(provider: str):
    """Debug the exact structure of streaming responses with tool calls."""
    print(f"\n🔍 DEBUGGING {provider.upper()} STREAMING TOOL CALLS")
    print("=" * 60)
    
    client = DaprChatClient()
    
    # Use a clear tool-calling prompt
    message = "What's the weather like in San Francisco? Please use the weather tool."
    
    print(f"📤 Sending: {message}")
    print(f"🔧 Tools available: get_weather")
    print(f"📡 Streaming response analysis:")
    
    try:
        stream = client.generate(
            messages=[{"role": "user", "content": message}],
            llm_component=provider,
            tools=[get_weather],
            stream=True,
            temperature=0.1  # Low temperature for consistent behavior
        )
        
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            print(f"\n--- Chunk {chunk_count} ---")
            
            # Print the complete chunk structure
            print(f"Raw chunk type: {type(chunk)}")
            print(f"Raw chunk keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'Not a dict'}")
            
            # Focus on outputs since choices is empty
            if isinstance(chunk, dict):
                if 'outputs' in chunk and chunk['outputs']:
                    print(f"🎯 OUTPUTS FOUND: {chunk['outputs']}")
                    for i, output in enumerate(chunk['outputs']):
                        print(f"  Output {i}: {output}")
                        if hasattr(output, 'tool_calls') and output.tool_calls:
                            print(f"  🔧 TOOL CALLS: {output.tool_calls}")
                        if hasattr(output, 'result'):
                            print(f"  📝 RESULT: {output.result}")
                        if hasattr(output, 'finish_reason'):
                            print(f"  🏁 FINISH REASON: {output.finish_reason}")
                
                if 'choices' in chunk:
                    print(f"📋 Choices: {chunk['choices']} (length: {len(chunk['choices'])})")
                
                if 'usage' in chunk and chunk['usage']:
                    print(f"📊 Usage: {chunk['usage']}")
                
                if 'context_id' in chunk and chunk['context_id']:
                    print(f"🆔 Context ID: {chunk['context_id']}")
            
            print(f"Raw chunk: {chunk}")
            
            # Check for content
            if hasattr(chunk, 'chunk') and chunk.chunk:
                if hasattr(chunk.chunk, 'content') and chunk.chunk.content:
                    print(f"✅ Content chunk: '{chunk.chunk.content}'")
                else:
                    print(f"🔍 Non-content chunk: {chunk.chunk}")
            
            # Check for complete message
            elif hasattr(chunk, 'complete') and chunk.complete:
                print(f"🏁 Complete message detected!")
                print(f"Complete structure: {chunk.complete}")
                
                # Deep dive into complete message structure
                if hasattr(chunk.complete, 'outputs'):
                    print(f"📋 Outputs found: {len(chunk.complete.outputs)}")
                    for i, output in enumerate(chunk.complete.outputs):
                        print(f"  Output {i}: {output}")
                        if hasattr(output, 'tool_calls'):
                            print(f"  🔧 Tool calls in output {i}: {output.tool_calls}")
                        if hasattr(output, 'result'):
                            print(f"  📝 Result in output {i}: {output.result}")
                        if hasattr(output, 'finish_reason'):
                            print(f"  🏁 Finish reason: {output.finish_reason}")
                
                if hasattr(chunk.complete, 'usage'):
                    print(f"📊 Usage: {chunk.complete.usage}")
            
            # Check for direct outputs
            elif hasattr(chunk, 'outputs'):
                print(f"📋 Direct outputs detected: {len(chunk.outputs)}")
                for i, output in enumerate(chunk.outputs):
                    print(f"  Output {i}: {output}")
                    if hasattr(output, 'tool_calls'):
                        print(f"  🔧 Tool calls: {output.tool_calls}")
            
            # Check for any other structure
            else:
                print(f"❓ Unknown chunk structure")
                for attr in dir(chunk):
                    if not attr.startswith('_'):
                        value = getattr(chunk, attr)
                        if value is not None:
                            print(f"  {attr}: {value}")
            
            # Stop after reasonable number of chunks
            if chunk_count >= 10:
                print(f"\n... (stopping after {chunk_count} chunks)")
                break
        
        print(f"\n✅ {provider.upper()} streaming analysis complete ({chunk_count} total chunks)")
        
    except Exception as e:
        print(f"❌ Error with {provider}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debugging function."""
    print("🔍 TOOL CALL TIMING DEBUG")
    print("=" * 60)
    print("This test examines exactly when tool calls appear in streaming responses")
    
    load_env_file()
    
    # Test both providers if available
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    
    if not providers:
        print("❌ No API keys found!")
        return
    
    for provider in providers:
        debug_streaming_response(provider)
    
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    print("Check the output above to see:")
    print("1. When tool calls appear (during streaming vs at the end)")
    print("2. What the exact response structure looks like")
    print("3. Where tool calls are located in the response")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

"""
Comprehensive Tool Calling Examples with Different Dapr Providers

This script demonstrates how to use tool calling with various Dapr conversation components:
- Echo (for testing/development)
- OpenAI (GPT models)
- Anthropic (Claude models)
- Any other provider that supports tool calling

Prerequisites:
1. Dapr runtime running with conversation components configured
2. API keys configured in .env file at repo root (for real LLM providers)
3. Components configured in ./components/ directory
"""

import os
import json
from pathlib import Path
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()


# Load environment variables from .env file at repo root
def load_env_file():
    """Load environment variables from .env file at the root of the repo."""
    # Get the repo root (two levels up from this file)
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"

    if env_file.exists():
        print(f"📄 Loading environment variables from: {env_file}")
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            print("✅ Environment variables loaded successfully")
        except ImportError:
            print(
                "⚠️  python-dotenv not installed. Install with: pip install python-dotenv"
            )
            print("⚠️  Falling back to system environment variables")
        except Exception as e:
            print(f"⚠️  Error loading .env file: {e}")
            print("⚠️  Falling back to system environment variables")
    else:
        print(f"📄 No .env file found at: {env_file}")
        print("⚠️  Using system environment variables")


def check_api_key(key_name: str, provider_name: str) -> bool:
    """Check if an API key is available and provide helpful feedback."""
    api_key = os.getenv(key_name)
    if not api_key:
        print(
            f"{Fore.YELLOW}⚠️  {key_name} not found in .env file or environment variables.{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}⚠️  Add '{key_name}=your-api-key' to .env file to test {provider_name}.{Style.RESET_ALL}"
        )
        return False
    else:
        print(
            f"{Fore.GREEN}✅ {key_name} found for {provider_name} provider{Style.RESET_ALL}"
        )
        return True


# =============================================================================
# Tool Definitions
# =============================================================================


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather information
    """
    # Simulate weather API call
    weather_data = {
        "location": location,
        "temperature": "72°F",
        "condition": "sunny",
        "humidity": "65%",
        "wind": "5 mph",
    }
    return f"Weather in {location}: {weather_data['temperature']}, {weather_data['condition']}, humidity {weather_data['humidity']}, wind {weather_data['wind']}"


@tool
def get_time() -> str:
    """Get the current time.

    Returns:
        Current time as a string
    """
    from datetime import datetime

    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        Result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic mathematical operations are allowed"

        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for information (simulated).

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Simulate web search
    return f"Search results for '{query}': Found 3 relevant articles about {query}. Top result: Latest information about {query} from reliable sources."


# =============================================================================
# Provider Examples
# =============================================================================


def print_section(title: str):
    """Print a colored section header."""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def print_subsection(title: str):
    """Print a colored subsection header."""
    print(f"\n{Fore.YELLOW}{title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-'*40}{Style.RESET_ALL}")


def test_echo_provider():
    """Test tool calling with Echo provider (for development/testing)."""
    print_section("🔄 Echo Provider - Tool Calling Test")

    client = DaprChatClient()
    tools = [get_weather, get_time, calculate]

    test_cases = [
        "What's the weather like in San Francisco?",
        "What time is it?",
        "Calculate 15 * 8 + 12",
        "Can you get the weather for New York and also tell me the time?",
    ]

    for i, message in enumerate(test_cases, 1):
        print_subsection(f"Test {i}: {message}")

        try:
            # Non-streaming call
            response = client.generate(
                messages=[{"role": "user", "content": message}],
                llm_component="echo-tools",
                tools=tools,
                stream=False,
            )

            print(f"{Fore.GREEN}✅ Response:{Style.RESET_ALL}")
            print(json.dumps(response, indent=2))

        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def test_openai_provider():
    """Test tool calling with OpenAI provider."""
    print_section("🤖 OpenAI Provider - Tool Calling Test")

    # Check if OpenAI API key is available
    if not check_api_key("OPENAI_API_KEY", "OpenAI"):
        print(f"{Fore.YELLOW}⚠️  Skipping OpenAI tests.{Style.RESET_ALL}")
        return

    client = DaprChatClient()
    tools = [get_weather, get_time, calculate, search_web]

    test_cases = [
        "What's the weather like in Tokyo? Also, what time is it?",
        "Calculate the result of 25 * 4 + 10, then search for information about that number",
        "I need the current weather in London and the time",
    ]

    for i, message in enumerate(test_cases, 1):
        print_subsection(f"OpenAI Test {i}: {message}")

        try:
            # Test with streaming
            print(f"{Fore.BLUE}📡 Streaming response:{Style.RESET_ALL}")
            for chunk in client.generate(
                messages=[{"role": "user", "content": message}],
                llm_component="openai",
                tools=tools,
                temperature=0.7,
                stream=True,
            ):
                if chunk.get("choices") and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if choice.get("delta", {}).get("content"):
                        print(choice["delta"]["content"], end="", flush=True)
                    elif choice.get("delta", {}).get("tool_calls"):
                        print(
                            f"\n{Fore.MAGENTA}🔧 Tool calls: {json.dumps(choice['delta']['tool_calls'], indent=2)}{Style.RESET_ALL}"
                        )
                    elif choice.get("finish_reason"):
                        print(
                            f"\n{Fore.GREEN}✅ Finished: {choice['finish_reason']}{Style.RESET_ALL}"
                        )

            print()  # New line after streaming

        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def test_anthropic_provider():
    """Test tool calling with Anthropic provider."""
    print_section("🧠 Anthropic Provider - Tool Calling Test")

    # Check if Anthropic API key is available
    if not check_api_key("ANTHROPIC_API_KEY", "Anthropic"):
        print(f"{Fore.YELLOW}⚠️  Skipping Anthropic tests.{Style.RESET_ALL}")
        return

    client = DaprChatClient()
    tools = [get_weather, get_time, calculate, search_web]

    test_cases = [
        "Please get the weather for Paris and calculate 100 / 4",
        "What's the current time and can you search for information about Python programming?",
    ]

    for i, message in enumerate(test_cases, 1):
        print_subsection(f"Anthropic Test {i}: {message}")

        try:
            # Non-streaming call
            response = client.generate(
                messages=[{"role": "user", "content": message}],
                llm_component="anthropic",
                tools=tools,
                temperature=0.7,
                stream=False,
            )

            print(f"{Fore.GREEN}✅ Response:{Style.RESET_ALL}")
            print(json.dumps(response, indent=2))

        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def test_gemini_provider():
    """Test tool calling with Google AI (Gemini) provider."""
    print_section("🤖 Google AI (Gemini) Provider - Tool Calling Test")

    # Check if Google AI API key is available
    if not check_api_key("GOOGLE_AI_API_KEY", "Google AI (Gemini)"):
        print(f"{Fore.YELLOW}⚠️  Skipping Google AI tests.{Style.RESET_ALL}")
        return

    client = DaprChatClient()
    tools = [get_weather, get_time, calculate, search_web]

    test_cases = [
        "What's the weather in Mountain View? Also calculate 42 * 7",
        "Get the current time and search for information about AI",
    ]

    for i, message in enumerate(test_cases, 1):
        print_subsection(f"Gemini Test {i}: {message}")

        try:
            # Non-streaming call - using 'gemini' as the component name
            response = client.generate(
                messages=[{"role": "user", "content": message}],
                llm_component="gemini",  # This should match the component name in gemini-conversation.yaml
                tools=tools,
                temperature=0.7,
                stream=False,
            )

            print(f"{Fore.GREEN}✅ Response:{Style.RESET_ALL}")
            print(json.dumps(response, indent=2))

        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
            # Add some debugging info
            if "failed finding conversation component" in str(e):
                print(
                    f"{Fore.YELLOW}💡 Debug: This error suggests the 'gemini' component is not loaded by Dapr.{Style.RESET_ALL}"
                )
                print(f"{Fore.YELLOW}💡 Possible causes:{Style.RESET_ALL}")
                print(
                    f"{Fore.YELLOW}   - The conversation.googleai component type is not supported in this Dapr build{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}   - The component file has a syntax error{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}   - The component is not in the components directory being loaded{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}💡 Check: Verify that components/gemini-conversation.yaml exists and is valid{Style.RESET_ALL}"
                )


def test_multi_turn_conversation():
    """Test multi-turn conversation with tool calling."""
    print_section("💬 Multi-Turn Conversation with Tool Calling")

    client = DaprChatClient()
    tools = [get_weather, get_time, calculate]

    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "What's the weather in Seattle?"},
        # Response would include tool call and result
        {
            "role": "assistant",
            "content": "I'll check the weather in Seattle for you.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Seattle"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Weather in Seattle: 65°F, cloudy, humidity 80%, wind 10 mph",
        },
        {
            "role": "assistant",
            "content": "The weather in Seattle is currently 65°F and cloudy, with 80% humidity and 10 mph winds.",
        },
        {"role": "user", "content": "Thanks! Now what time is it?"},
    ]

    try:
        # Use echo for this example since it's always available
        response = client.generate(
            messages=conversation, llm_component="echo-tools", tools=tools, stream=False
        )

        print(f"{Fore.GREEN}✅ Multi-turn conversation response:{Style.RESET_ALL}")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def test_error_handling():
    """Test error handling in tool calling."""
    print_section("🚨 Error Handling Tests")

    client = DaprChatClient()

    # Test with invalid tool call
    print_subsection("Test: Invalid calculation")
    try:
        response = client.generate(
            messages=[{"role": "user", "content": "Calculate 1/0"}],
            llm_component="echo-tools",
            tools=[calculate],
            stream=False,
        )
        print(f"{Fore.GREEN}✅ Handled division by zero:{Style.RESET_ALL}")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def print_usage_examples():
    """Print usage examples and setup instructions."""
    print_section("📚 Usage Examples & Setup")

    print(
        f"""
{Fore.GREEN}🔧 Setup Instructions:{Style.RESET_ALL}

1. {Fore.YELLOW}Create .env file at repo root:{Style.RESET_ALL}
   Create a file named '.env' in the root directory with your API keys:

   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   GOOGLE_AI_API_KEY=your-google-ai-api-key-here
   ```

2. {Fore.YELLOW}Install python-dotenv (if not already installed):{Style.RESET_ALL}
   pip install python-dotenv

3. {Fore.YELLOW}Start Dapr with components:{Style.RESET_ALL}
   python tools/run_dapr_dev.py --app-id tool-calling-demo --components ./components

4. {Fore.YELLOW}Component Configuration Examples:{Style.RESET_ALL}

   Echo (components/echo-tools.yaml):
   ```yaml
   apiVersion: dapr.io/v1alpha1
   kind: Component
   metadata:
     name: echo-tools
   spec:
     type: conversation.echo
     version: v1
   ```

   OpenAI (components/openai.yaml):
   ```yaml
   apiVersion: dapr.io/v1alpha1
   kind: Component
   metadata:
     name: openai
   spec:
     type: conversation.openai
     version: v1
     metadata:
     - name: apiKey
       value: "${{OPENAI_API_KEY}}"
     - name: model
       value: "gpt-4"
   ```

{Fore.GREEN}💡 Code Examples:{Style.RESET_ALL}

{Fore.YELLOW}Basic Tool Calling:{Style.RESET_ALL}
```python
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool

@tool
def my_tool(param: str) -> str:
    return f"Processed: {{param}}"

client = DaprChatClient()
response = client.generate(
    messages=[{{"role": "user", "content": "Use my tool with 'hello'"}}],
    llm_component="echo-tools",  # or "openai", "anthropic", etc.
    tools=[my_tool],
    stream=False
)
```

{Fore.YELLOW}Streaming with Tools:{Style.RESET_ALL}
```python
for chunk in client.generate(
    messages=[{{"role": "user", "content": "Get weather for NYC"}}],
    llm_component="openai",
    tools=[get_weather],
    stream=True
):
    if chunk.get("choices") and chunk["choices"]:
        choice = chunk["choices"][0]
        if choice.get("delta", {{}}).get("content"):
            print(choice["delta"]["content"], end="")
        elif choice.get("delta", {{}}).get("tool_calls"):
            print(f"Tool called: {{choice['delta']['tool_calls']}}")
```
"""
    )


def main():
    """Run all tool calling examples."""
    print_section("🚀 Dapr Agents - Tool Calling Examples")

    print(
        f"""
{Fore.GREEN}This script demonstrates tool calling with various Dapr conversation providers.{Style.RESET_ALL}

{Fore.YELLOW}Available Providers:{Style.RESET_ALL}
• Echo - For testing and development (always available)
• OpenAI - GPT models (requires OPENAI_API_KEY in .env)
• Anthropic - Claude models (requires ANTHROPIC_API_KEY in .env)
• Google AI - Gemini models (requires GOOGLE_AI_API_KEY in .env) ⚠️ May not be available in all Dapr builds

{Fore.YELLOW}Tools Available:{Style.RESET_ALL}
• get_weather(location) - Get weather for a location
• get_time() - Get current time
• calculate(expression) - Perform mathematical calculations
• search_web(query) - Search the web (simulated)

{Fore.CYAN}Note:{Style.RESET_ALL} If you see "failed finding conversation component" errors, it means that
conversation component type is not supported in your current Dapr build.
"""
    )

    # Load environment variables from .env file
    load_env_file()

    # Run examples
    test_echo_provider()
    test_openai_provider()
    test_anthropic_provider()
    test_gemini_provider()
    test_multi_turn_conversation()
    test_error_handling()

    # Print usage examples
    print_usage_examples()

    print_section("✅ All Examples Completed")
    print(
        f"{Fore.GREEN}Check the output above for results from each provider.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Note: Some tests may be skipped if API keys are not configured in .env file.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Note: Some providers may not be available if the Dapr build doesn't include them.{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()

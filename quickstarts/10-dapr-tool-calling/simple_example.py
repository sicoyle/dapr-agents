#!/usr/bin/env python3

"""
Simple Tool Calling Example

A minimal example showing how to use tool calling with Dapr conversation components.
This example works with any provider (echo, openai, anthropic, etc.).

Usage:
    python simple_example.py
    python simple_example.py --provider openai
    python simple_example.py --provider anthropic --streaming
"""

import argparse
import json
import os
from pathlib import Path
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool
from dapr_agents.tool.base import AgentTool


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


# Define function-based tools using @tool decorator
@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather information
    """
    return f"Weather in {location}: 72°F, sunny, light breeze"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Define class-based tools inheriting from AgentTool
class GetTimeZone(AgentTool):
    """Get the current time zone information for a location."""

    def __init__(self):
        super().__init__(
            name="get_time_zone",
            description="Get the current time zone information for a given location",
        )

    def _run(self, location: str) -> str:
        """Execute the time zone lookup.

        Args:
            location: The city and country, e.g. 'Tokyo, Japan'
        """
        # Simple mock time zone data
        time_zones = {
            "tokyo": "JST (UTC+9)",
            "london": "GMT (UTC+0)",
            "new york": "EST (UTC-5)",
            "san francisco": "PST (UTC-8)",
            "sydney": "AEDT (UTC+11)",
            "paris": "CET (UTC+1)",
        }

        location_key = location.lower().split(",")[0].strip()
        timezone = time_zones.get(location_key, "Unknown timezone")
        return f"Time zone for {location}: {timezone}"


class ConvertCurrency(AgentTool):
    """Convert currency from one type to another."""

    def __init__(self):
        super().__init__(
            name="convert_currency",
            description="Convert an amount from one currency to another",
        )

    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        """Execute the currency conversion.

        Args:
            amount: The amount to convert
            from_currency: The source currency code (e.g., 'USD')
            to_currency: The target currency code (e.g., 'EUR')
        """
        # Simple mock exchange rates
        rates = {
            ("USD", "EUR"): 0.85,
            ("EUR", "USD"): 1.18,
            ("USD", "GBP"): 0.73,
            ("GBP", "USD"): 1.37,
            ("USD", "JPY"): 110.0,
            ("JPY", "USD"): 0.009,
        }

        rate = rates.get((from_currency.upper(), to_currency.upper()))
        if rate:
            converted_amount = amount * rate
            return f"{amount} {from_currency.upper()} = {converted_amount:.2f} {to_currency.upper()}"
        else:
            return f"Exchange rate not available for {from_currency} to {to_currency}"


class GeneratePassword(AgentTool):
    """Generate a secure random password."""

    def __init__(self):
        super().__init__(
            name="generate_password",
            description="Generate a secure random password with specified length",
        )

    def _run(self, length: int = 12, include_symbols: bool = True) -> str:
        """Execute the password generation.

        Args:
            length: Length of the password (default: 12)
            include_symbols: Whether to include special symbols (default: true)
        """
        import random
        import string

        characters = string.ascii_letters + string.digits
        if include_symbols:
            characters += "!@#$%^&*"

        password = "".join(random.choice(characters) for _ in range(length))
        return f"Generated password ({length} chars): {password}"


def check_provider_requirements(provider: str):
    """Check if the required API keys are available for the provider."""
    requirements = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_AI_API_KEY",
        "googleai": "GOOGLE_AI_API_KEY",
    }

    if provider in requirements:
        api_key = os.getenv(requirements[provider])
        if not api_key:
            print(
                f"⚠️  Warning: {requirements[provider]} not found in environment variables or .env file"
            )
            print(f"⚠️  Provider '{provider}' may not work without the API key")
            return False
        else:
            print(f"✅ API key found for provider '{provider}'")
            return True
    else:
        print(f"✅ Provider '{provider}' doesn't require an API key")
        return True


def run_simple_example(
    provider: str = "echo-tools", streaming: bool = False, use_class_tools: bool = False
):
    """Run a simple tool calling example."""

    print(f"🚀 Running tool calling example with provider: {provider}")
    print(f"📡 Streaming: {'Yes' if streaming else 'No'}")
    print(f"🔧 Tool type: {'Class-based' if use_class_tools else 'Function-based'}")
    print("=" * 50)

    # Check provider requirements
    check_provider_requirements(provider)

    # Initialize client
    client = DaprChatClient()

    # Choose tools based on the flag
    if use_class_tools:
        # Use class-based tools
        tools = [GetTimeZone(), ConvertCurrency(), GeneratePassword()]
        message = "What's the time zone in Tokyo? Also convert 100 USD to EUR and generate a 16-character password."
    else:
        # Use function-based tools
        tools = [get_weather, calculate]
        message = "What's the weather like in San Francisco? Also calculate 15 * 8"

    print(f"User: {message}")
    print("Assistant: ", end="", flush=True)

    try:
        if streaming:
            # Streaming response
            for chunk in client.generate(
                messages=[{"role": "user", "content": message}],
                llm_component=provider,
                tools=tools,
                stream=True,
            ):
                if hasattr(chunk, "chunk") and chunk.chunk and hasattr(chunk.chunk, "content"):
                    print(chunk.chunk.content, end="", flush=True)
                elif hasattr(chunk, "complete") and chunk.complete:
                    print(f"\n✅ Completed with usage: {chunk.complete.usage}")

        else:
            # Non-streaming response
            response = client.generate(
                messages=[{"role": "user", "content": message}],
                llm_component=provider,
                tools=tools,
                stream=False,
            )

            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
                print(content)
            elif isinstance(response, dict) and "outputs" in response:
                # Handle the raw response format
                output = response["outputs"][0]
                content = output.get("result", "No content found")
                print(content)
            else:
                print(json.dumps(response, indent=2, default=str))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Simple Tool Calling Example")
    parser.add_argument(
        "--provider",
        default="echo-tools",
        help="Dapr conversation provider (echo-tools, openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming responses"
    )
    parser.add_argument(
        "--class-tools",
        action="store_true",
        help="Use class-based tools instead of function-based tools",
    )

    args = parser.parse_args()

    tool_type = "Class-based" if args.class_tools else "Function-based"

    if args.class_tools:
        tools_desc = """
• GetTimeZone - Get time zone information for a location
• ConvertCurrency - Convert amounts between currencies
• GeneratePassword - Generate secure random passwords"""
    else:
        tools_desc = """
• get_weather(location) - Get weather for a location
• calculate(expression) - Perform mathematical calculations"""

    print(
        f"""
🔧 Simple Tool Calling Example

This example demonstrates both function-based and class-based tool calling
with Dapr conversation components.

Provider: {args.provider}
Streaming: {'Yes' if args.streaming else 'No'}
Tool Type: {tool_type}

Available tools:{tools_desc}

Make sure Dapr is running with your components configured!
"""
    )

    # Load environment variables from .env file
    load_env_file()

    run_simple_example(args.provider, args.streaming, args.class_tools)


if __name__ == "__main__":
    main()

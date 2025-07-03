#!/usr/bin/env python3

"""
Dapr Tool Calling with Streaming Example
This example demonstrates tool calling combined with streaming responses.
"""

import os
import json
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from colorama import init, Fore, Style
from dapr_agents.llm import DaprChatClient
from dapr_agents.tool import tool

# Initialize colorama for colored output
init()


# Tool definitions
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in the specified timezone.

    Args:
        timezone: The timezone to get the time for (e.g., "UTC", "EST", "PST")

    Returns:
        Current time as a formatted string
    """
    current_time = datetime.now()
    return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculate_math(expression: str) -> str:
    """Safely calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        Result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def get_random_fact() -> str:
    """Get a random interesting fact.

    Returns:
        A random fact string
    """
    facts = [
        "Octopuses have three hearts and blue blood.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
        "The shortest war in history lasted only 38-45 minutes.",
        "Bananas are berries, but strawberries aren't.",
        "A day on Venus is longer than its year.",
        "The human brain uses about 20% of the body's total energy.",
    ]
    return random.choice(facts)


@tool
def text_analysis(text: str) -> str:
    """Analyze text and provide statistics.

    Args:
        text: Text to analyze

    Returns:
        Analysis results including character count, word count, etc.
    """
    words = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")

    analysis = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": sentences,
        "average_word_length": sum(len(word.strip(".,!?;:")) for word in words)
        / len(words)
        if words
        else 0,
    }

    return f"Text Analysis: {json.dumps(analysis, indent=2)}"


def main():
    """Main streaming tool calling example."""
    load_dotenv()

    print(f"{Fore.CYAN}🚀 Testing Dapr Tool Calling with Streaming{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}This example demonstrates intelligent tool selection with streaming responses{Style.RESET_ALL}\n"
    )

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(
            f"{Fore.RED}❌ Please set OPENAI_API_KEY environment variable{Style.RESET_ALL}"
        )
        return

    # Initialize the Dapr chat client
    llm = DaprChatClient()

    # Test scenarios that should trigger tool calls
    test_scenarios = [
        {
            "name": "Time Query",
            "prompt": "What time is it right now?",
            "expected_tool": "get_current_time",
        },
        {
            "name": "Math Calculation",
            "prompt": "Can you calculate 15 * 7 + 23 for me?",
            "expected_tool": "calculate_math",
        },
        {
            "name": "Text Analysis",
            "prompt": "Please analyze this text: 'The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet!'",
            "expected_tool": "text_analysis",
        },
        {
            "name": "Random Fact",
            "prompt": "Tell me an interesting random fact.",
            "expected_tool": "get_random_fact",
        },
    ]

    # Collect available tools
    available_tools = [get_current_time, calculate_math, get_random_fact, text_analysis]

    for i, scenario in enumerate(test_scenarios, 1):
        print(
            f"{Fore.CYAN}📋 Test {i}/{len(test_scenarios)}: {scenario['name']}{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}❓ Query:{Style.RESET_ALL} {scenario['prompt']}")
        print(
            f"{Fore.BLUE}🔧 Expected Tool:{Style.RESET_ALL} {scenario['expected_tool']}"
        )
        print(f"{Fore.MAGENTA}🤖 Streaming Response:{Style.RESET_ALL}")

        try:
            start_time = time.time()

            # Generate streaming response with tools
            response_stream = llm.generate(
                messages=[{"role": "user", "content": scenario["prompt"]}],
                tools=available_tools,
                stream=True,
                llm_component="openai",
                temperature=0.1,  # Low temperature for consistent tool calling
            )

            tool_calls_made = []
            content_parts = []

            for chunk in response_stream:
                # Handle OpenAI-compatible streaming format
                if isinstance(chunk, dict):
                    # Handle content chunks
                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        if "delta" in choice and choice["delta"]:
                            delta = choice["delta"]

                            # Handle text content
                            if "content" in delta and delta["content"]:
                                content = delta["content"]
                                print(content, end="", flush=True)
                                content_parts.append(content)

                            # Handle tool calls
                            if "tool_calls" in delta:
                                for tool_call in delta["tool_calls"]:
                                    tool_name = tool_call["function"]["name"]
                                    tool_args = tool_call["function"]["arguments"]
                                    tool_calls_made.append(f"{tool_name}({tool_args})")
                                    print(
                                        f"\n{Fore.YELLOW}🔧 Tool Call: {tool_name}{Style.RESET_ALL}",
                                        end="",
                                        flush=True,
                                    )

                        # Handle finish reason
                        if choice.get("finish_reason") == "tool_calls":
                            print(
                                f"\n{Fore.GREEN}✅ Tool calls completed{Style.RESET_ALL}"
                            )

                    # Handle usage information
                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]
                        print(
                            f"\n{Fore.CYAN}📊 Usage: {usage.get('total_tokens', 'N/A')} tokens{Style.RESET_ALL}"
                        )

            end_time = time.time()

            print(f"\n{Fore.GREEN}✅ Scenario {i} completed!{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 Performance:{Style.RESET_ALL}")
            print(f"   • Response time: {end_time - start_time:.2f} seconds")
            print(f"   • Content length: {len(''.join(content_parts))} characters")

            # Brief pause between tests
            time.sleep(1)
            print("-" * 60)

        except Exception as e:
            print(f"\n{Fore.RED}❌ Error in scenario {i}: {e}{Style.RESET_ALL}")
            import traceback

            traceback.print_exc()

    print(
        f"\n{Fore.GREEN}🎉 All streaming tool calling tests completed!{Style.RESET_ALL}"
    )
    print(f"{Fore.MAGENTA}🚀 Key Features Demonstrated:{Style.RESET_ALL}")
    print("   • Intelligent tool selection based on user queries")
    print("   • Real-time streaming responses")
    print("   • Multiple tool types (time, math, text analysis, facts)")
    print("   • Performance monitoring and usage tracking")


if __name__ == "__main__":
    main()

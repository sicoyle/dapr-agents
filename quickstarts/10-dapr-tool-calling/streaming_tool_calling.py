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
                chunk_type = chunk.get("type")

                if chunk_type == "content":
                    content = chunk["data"]
                    print(content, end="", flush=True)
                    content_parts.append(content)

                elif chunk_type == "tool_call":
                    tool_call = chunk["data"]
                    tool_calls_made.append(tool_call)
                    print(
                        f"\n{Fore.YELLOW}🔧 Tool Call: {tool_call.get('name')} with args {tool_call.get('arguments')}{Style.RESET_ALL}"
                    )

                elif chunk_type == "tool_result":
                    tool_result = chunk["data"]
                    print(f"{Fore.GREEN}✅ Tool Result: {tool_result}{Style.RESET_ALL}")

                elif chunk_type == "final_usage":
                    usage = chunk["data"]
                    print(
                        f"\n{Fore.CYAN}📊 Usage: {usage.get('total_tokens', 'N/A')} tokens{Style.RESET_ALL}"
                    )

            end_time = time.time()

            print(
                f"\n{Fore.MAGENTA}✅ Scenario completed in {end_time - start_time:.2f}s{Style.RESET_ALL}"
            )
            if tool_calls_made:
                print(
                    f"{Fore.GREEN}🎯 Tools used: {[tc.get('name') for tc in tool_calls_made]}{Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.YELLOW}⚠️  No tools were called{Style.RESET_ALL}")

            print("-" * 60)

        except Exception as e:
            print(
                f"\n{Fore.RED}❌ Error in scenario {scenario['name']}: {e}{Style.RESET_ALL}"
            )
            print("-" * 60)

    # Interactive mode
    print(f"{Fore.CYAN}💬 Interactive Tool Calling Mode{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Try these examples:{Style.RESET_ALL}")
    print("   • 'What time is it?'")
    print("   • 'Calculate 25 * 4 + 10'")
    print("   • 'Analyze this text: Hello world!'")
    print("   • 'Give me a random fact'")
    print(f"\n{Fore.CYAN}💬 Interactive Tool Calling Mode{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}Try asking questions that might use the available tools:{Style.RESET_ALL}"
    )
    print("   • 'What time is it?'")
    print("   • 'Calculate 25 * 4 + 10'")
    print("   • 'Analyze this text: Hello world!'")
    print("   • 'Give me a random fact'")
    print("   • Type 'quit' to exit\n")

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            print(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}", end="", flush=True)

            response_stream = llm.generate(
                messages=[{"role": "user", "content": user_input}],
                tools=available_tools,
                stream=True,
                llm_component="openai",
            )

            for chunk in response_stream:
                if chunk.get("type") == "content":
                    content = chunk["data"]
                    print(content, end="", flush=True)
                elif chunk.get("type") == "tool_call":
                    tool_call = chunk["data"]
                    print(
                        f"\n{Fore.YELLOW}[Using tool: {tool_call.get('name')}]{Style.RESET_ALL}"
                    )
                elif chunk.get("type") == "tool_result":
                    print(f"{Fore.GREEN}[Tool completed]{Style.RESET_ALL}")

            print("\n")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

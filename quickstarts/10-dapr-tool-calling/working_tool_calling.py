#!/usr/bin/env python3

"""
Working Dapr Tool Calling Example
This example demonstrates tool calling using the current available API (non-streaming).
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
    """Main tool calling example."""
    load_dotenv()

    print(f"{Fore.CYAN}🚀 Testing Dapr Tool Calling (Non-Streaming){Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}This example demonstrates intelligent tool selection using available APIs{Style.RESET_ALL}\n"
    )

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(
            f"{Fore.RED}❌ Please set OPENAI_API_KEY environment variable{Style.RESET_ALL}"
        )
        return

    # Set the required environment variable
    os.environ["DAPR_LLM_COMPONENT_DEFAULT"] = "openai"

    # Initialize the Dapr chat client
    try:
        llm = DaprChatClient()
        print(
            f"{Fore.GREEN}✅ DaprChatClient initialized successfully{Style.RESET_ALL}"
        )
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to initialize DaprChatClient: {e}{Style.RESET_ALL}")
        return

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
        print(f"{Fore.MAGENTA}🤖 Response:{Style.RESET_ALL}")

        try:
            start_time = time.time()

            # Generate response with tools (non-streaming)
            response = llm.generate(
                messages=[{"role": "user", "content": scenario["prompt"]}],
                tools=available_tools,
                stream=False,  # Use non-streaming for now
                llm_component="openai",
                temperature=0.1,  # Low temperature for consistent tool calling
            )

            end_time = time.time()

            # Process response
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
                print(f"{Fore.YELLOW}{content}{Style.RESET_ALL}")

                # Check for tool calls (this depends on the response format)
                tool_calls_detected = (
                    "function" in content.lower() or "tool" in content.lower()
                )

                print(
                    f"\n{Fore.MAGENTA}✅ Scenario completed in {end_time - start_time:.2f}s{Style.RESET_ALL}"
                )
                if tool_calls_detected:
                    print(
                        f"{Fore.GREEN}🎯 Tool calling detected in response{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"{Fore.YELLOW}⚠️  No obvious tool calling detected{Style.RESET_ALL}"
                    )
            else:
                print(
                    f"{Fore.RED}❌ Unexpected response format: {response}{Style.RESET_ALL}"
                )

            print("-" * 60)

        except Exception as e:
            print(
                f"\n{Fore.RED}❌ Error in scenario {scenario['name']}: {e}{Style.RESET_ALL}"
            )
            print("-" * 60)

    # Test simple conversation without tools
    print(f"\n{Fore.CYAN}💬 Testing Simple Conversation (No Tools){Style.RESET_ALL}")
    try:
        simple_response = llm.generate(
            messages=[{"role": "user", "content": "Hello! How are you today?"}],
            stream=False,
            llm_component="openai",
            temperature=0.7,
        )

        if hasattr(simple_response, "choices") and simple_response.choices:
            content = simple_response.choices[0].message.content
            print(f"{Fore.YELLOW}{content}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Simple conversation working!{Style.RESET_ALL}")
        else:
            print(
                f"{Fore.RED}❌ Unexpected response format: {simple_response}{Style.RESET_ALL}"
            )

    except Exception as e:
        print(f"{Fore.RED}❌ Error in simple conversation: {e}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}🎉 Tool Calling Test Complete!{Style.RESET_ALL}")
    print(
        f"{Fore.MAGENTA}📝 Note: Full streaming + tool calling will be available when converse_stream_alpha1 is implemented{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()

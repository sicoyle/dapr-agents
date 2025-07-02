#!/usr/bin/env python3

"""
Dapr Tool Calling Example with OpenAI
This example demonstrates real tool calling with OpenAI GPT models via Dapr.
"""

import os
import json
import time
import math
from datetime import datetime
from dotenv import load_dotenv
from colorama import init, Fore, Style
from dapr_agents.llm import DaprChatClient
from dapr_agents.tool import tool

# Initialize colorama for colored output
init()

# Define tools that the LLM can call
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_circle_area(radius: float) -> dict:
    """
    Calculate the area of a circle.
    
    Args:
        radius: The radius of the circle
        
    Returns:
        Dictionary with calculation results
    """
    if radius < 0:
        return {"error": "Radius cannot be negative"}
    
    area = math.pi * radius ** 2
    return {
        "radius": radius,
        "area": round(area, 2),
        "circumference": round(2 * math.pi * radius, 2)
    }

@tool
def get_weather_info(location: str) -> dict:
    """
    Get weather information for a location (simulated).
    
    Args:
        location: City name or location
        
    Returns:
        Weather information dictionary
    """
    # Simulate weather data
    import random
    
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "snowy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    
    return {
        "location": location,
        "temperature": f"{temperature}°C",
        "condition": condition,
        "humidity": f"{random.randint(30, 90)}%",
        "wind_speed": f"{random.randint(5, 25)} km/h"
    }

@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    compound_frequency: int = 12
) -> dict:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount invested
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Investment period in years
        compound_frequency: How many times interest compounds per year
    
    Returns:
        Dictionary with calculation results
    """
    if principal <= 0 or rate < 0 or time <= 0:
        return {"error": "Invalid input parameters"}
    
    amount = principal * (1 + rate/compound_frequency) ** (compound_frequency * time)
    interest = amount - principal
    
    return {
        "principal": principal,
        "final_amount": round(amount, 2),
        "interest_earned": round(interest, 2),
        "annual_rate": f"{rate*100}%",
        "years": time,
        "compound_frequency": f"{compound_frequency}x per year"
    }

def main():
    """Main tool calling example with OpenAI."""
    load_dotenv()
    
    print(f"{Fore.CYAN}🔧 Testing Dapr Tool Calling with OpenAI{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}This example demonstrates intelligent tool selection and execution{Style.RESET_ALL}\n")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.RED}❌ Please set OPENAI_API_KEY environment variable{Style.RESET_ALL}")
        return
    
    # Initialize the Dapr chat client with tools
    llm = DaprChatClient()
    
    # Register all our tools
    tools = [
        get_current_time,
        calculate_circle_area,
        get_weather_info,
        calculate_compound_interest
    ]
    
    print(f"{Fore.GREEN}🛠️  Tools available:{Style.RESET_ALL}")
    for tool_func in tools:
        print(f"   • {tool_func.__name__}: {tool_func.__doc__.split('.')[0]}")
    print()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Time Query",
            "prompt": "What time is it right now?",
            "expected_tool": "get_current_time"
        },
        {
            "name": "Math Calculation",
            "prompt": "Calculate the area of a circle with radius 5.5 meters",
            "expected_tool": "calculate_circle_area"
        },
        {
            "name": "Weather Query",
            "prompt": "What's the weather like in Tokyo?",
            "expected_tool": "get_weather_info"
        },
        {
            "name": "Financial Calculation",
            "prompt": "If I invest $10,000 at 7% annual interest for 5 years with monthly compounding, how much will I have?",
            "expected_tool": "calculate_compound_interest"
        },
        {
            "name": "Multi-step Query",
            "prompt": "What time is it, and what's the weather like in Paris? Also calculate the area of a circle with radius 3.",
            "expected_tool": "multiple"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{Fore.MAGENTA}📋 Test {i}: {scenario['name']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}❓ Query:{Style.RESET_ALL} {scenario['prompt']}")
        
        try:
            start_time = time.time()
            
            # Send the query with tools available
            response = llm.generate(
                prompt=scenario['prompt'],
                tools=tools
            )
            
            end_time = time.time()
            
            print(f"{Fore.BLUE}🤖 Response:{Style.RESET_ALL} {response.get_content()}")
            
            # Check if tools were called
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"{Fore.CYAN}🔧 Tools called:{Style.RESET_ALL}")
                for tool_call in response.tool_calls:
                    print(f"   • {tool_call.function.name}({tool_call.function.arguments})")
            
            print(f"{Fore.YELLOW}⏱️  Response time: {end_time - start_time:.2f}s{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        
        print("-" * 60)
    
    # Interactive mode
    print(f"\n{Fore.CYAN}💬 Interactive Tool Calling Mode{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type your questions, and the AI will use tools as needed. Type 'quit' to exit.{Style.RESET_ALL}\n")
    
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input.strip():
                continue
            
            print(f"{Fore.BLUE}AI: {Style.RESET_ALL}", end="")
            
            # Stream response with tools
            full_response = ""
            for chunk in llm.stream(prompt=user_input, tools=tools):
                content = chunk.get_content()
                if content:
                    print(content, end='', flush=True)
                    full_response += content
            
            print()  # New line after streaming
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 
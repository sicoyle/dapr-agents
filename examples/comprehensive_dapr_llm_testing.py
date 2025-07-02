#!/usr/bin/env python3

"""
Comprehensive Dapr LLM Testing Suite

This comprehensive example demonstrates production-ready testing of Dapr LLM components
across multiple providers with automatic component creation and detailed diagnostics.

Unlike the quickstart examples (which focus on learning), this is a diagnostic tool for:
- Multi-provider testing (Echo, OpenAI, Anthropic)
- Component auto-creation and validation
- Production readiness checking
- API key validation and troubleshooting

Prerequisites:
1. Start Dapr sidecar: python tools/run_dapr_dev.py --build
2. Set environment variables for API keys (optional)
3. Components will be auto-created in ./components/ directory

Usage:
    python examples/comprehensive_dapr_llm_testing.py [--component echo|openai|anthropic]
    python examples/comprehensive_dapr_llm_testing.py --show-config
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput

# Load environment variables
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"📄 Loaded environment from: {env_file}")


def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")

    # Check if Dapr is running
    try:
        import requests

        response = requests.get("http://localhost:3500/v1.0/healthz", timeout=2)
        if response.status_code == 204:
            print("✅ Dapr sidecar is running")
        else:
            print("❌ Dapr sidecar not responding correctly")
            return False
    except Exception:
        print("❌ Dapr sidecar not running")
        print("💡 Start with: python tools/run_dapr_dev.py --build")
        return False

    return True


def test_echo_component():
    """Test the echo conversation component."""
    print("\n🔊 Testing Echo Component")
    print("=" * 40)
    print("Echo component simply returns the input - useful for testing")

    try:
        with DaprClient() as client:
            inputs = [
                ConversationInput(
                    content="Hello from the echo component test!", role="user"
                )
            ]

            print("📤 Sending: Hello from the echo component test!")

            # Non-streaming test
            response = client.converse_alpha1(
                name="echo", inputs=inputs, context_id="echo-test-123"
            )

            print(f"📥 Received: {response.outputs[0].result}")
            print(f"🆔 Context ID: {response.context_id}")

            # Streaming test
            print("\n📡 Testing streaming...")
            inputs[0].content = "This is a streaming test with echo!"
            print("📤 Streaming: This is a streaming test with echo!")
            print("📥 Streamed response: ", end="", flush=True)

            for chunk in client.converse_stream_alpha1(
                name="echo", inputs=inputs, context_id="echo-stream-456"
            ):
                if chunk.result and chunk.result.result:
                    print(chunk.result.result, end="", flush=True)

            print("\n✅ Echo component test completed successfully")
            return True

    except Exception as e:
        print(f"❌ Echo component test failed: {e}")
        return False


def test_openai_component():
    """Test the OpenAI conversation component."""
    print("\n🤖 Testing OpenAI Component")
    print("=" * 40)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        print("💡 Set with: export OPENAI_API_KEY=your_key_here")
        return False

    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"✅ OpenAI API Key: {masked_key}")

    try:
        with DaprClient() as client:
            inputs = [
                ConversationInput(
                    content="Write a haiku about programming", role="user"
                )
            ]

            print("📤 Sending: Write a haiku about programming")

            # Non-streaming test
            print("\n📡 Testing non-streaming...")
            response = client.converse_alpha1(
                name="openai",
                inputs=inputs,
                temperature=0.7,
                context_id="openai-test-123",
            )

            print(f"📥 Response: {response.outputs[0].result}")
            if response.usage:
                print(f"📊 Usage: {response.usage.total_tokens} tokens")

            # Streaming test
            print("\n📡 Testing streaming...")
            inputs[0].content = "Tell me a short joke about AI"
            print("📤 Streaming: Tell me a short joke about AI")
            print("📥 Streamed response: ", end="", flush=True)

            total_tokens = 0
            for chunk in client.converse_stream_alpha1(
                name="openai",
                inputs=inputs,
                temperature=0.8,
                context_id="openai-stream-456",
            ):
                if chunk.result and chunk.result.result:
                    print(chunk.result.result, end="", flush=True)
                if chunk.usage:
                    total_tokens = chunk.usage.total_tokens

            if total_tokens:
                print(f"\n📊 Total tokens used: {total_tokens}")

            print("\n✅ OpenAI component test completed successfully")
            return True

    except Exception as e:
        print(f"❌ OpenAI component test failed: {e}")
        print("💡 Check your OpenAI API key and component configuration")
        return False


def test_anthropic_component():
    """Test the Anthropic conversation component."""
    print("\n🧠 Testing Anthropic Component")
    print("=" * 40)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        print("💡 Set with: export ANTHROPIC_API_KEY=your_key_here")
        print("💡 Create component: components/anthropic-conversation.yaml")
        return False

    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"✅ Anthropic API Key: {masked_key}")

    try:
        with DaprClient() as client:
            inputs = [
                ConversationInput(
                    content="Explain quantum computing in simple terms", role="user"
                )
            ]

            print("📤 Sending: Explain quantum computing in simple terms")

            # Non-streaming test
            print("\n📡 Testing non-streaming...")
            response = client.converse_alpha1(
                name="anthropic",
                inputs=inputs,
                temperature=0.6,
                context_id="anthropic-test-123",
            )

            print(f"📥 Response: {response.outputs[0].result[:200]}...")
            if response.usage:
                print(f"📊 Usage: {response.usage.total_tokens} tokens")

            # Streaming test
            print("\n📡 Testing streaming...")
            inputs[0].content = "What's the difference between AI and machine learning?"
            print(
                "📤 Streaming: What's the difference between AI and machine learning?"
            )
            print("📥 Streamed response: ", end="", flush=True)

            total_tokens = 0
            for chunk in client.converse_stream_alpha1(
                name="anthropic",
                inputs=inputs,
                temperature=0.7,
                context_id="anthropic-stream-456",
            ):
                if chunk.result and chunk.result.result:
                    print(chunk.result.result, end="", flush=True)
                if chunk.usage:
                    total_tokens = chunk.usage.total_tokens

            if total_tokens:
                print(f"\n📊 Total tokens used: {total_tokens}")

            print("\n✅ Anthropic component test completed successfully")
            return True

    except Exception as e:
        print(f"❌ Anthropic component test failed: {e}")
        print("💡 Check your Anthropic API key and component configuration")
        return False


def create_anthropic_component():
    """Create Anthropic component configuration if it doesn't exist."""
    components_dir = Path("components")
    anthropic_file = components_dir / "anthropic-conversation.yaml"

    if not anthropic_file.exists():
        print("📝 Creating Anthropic component configuration...")
        components_dir.mkdir(exist_ok=True)

        config = """apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: anthropic
spec:
  type: conversation.anthropic
  metadata:
    - name: key
      value: "${ANTHROPIC_API_KEY}"
    - name: model
      value: "claude-3-haiku-20240307"
    - name: cacheTTL
      value: "10m"
    - name: temperature
      value: "0.7"
    - name: maxTokens
      value: "1000"
"""
        anthropic_file.write_text(config)
        print(f"✅ Created: {anthropic_file}")
    else:
        print(f"✅ Anthropic component already exists: {anthropic_file}")


def show_component_configurations():
    """Show the current component configurations."""
    print("\n📋 Component Configurations")
    print("=" * 40)

    components_dir = Path("components")
    if not components_dir.exists():
        print("❌ Components directory not found")
        return

    for yaml_file in components_dir.glob("*-conversation.yaml"):
        print(f"\n📄 {yaml_file.name}:")
        try:
            content = yaml_file.read_text()
            # Extract component name and type
            lines = content.split("\n")
            name = next(
                (line.split(":")[1].strip() for line in lines if "name:" in line),
                "unknown",
            )
            comp_type = next(
                (line.split(":")[1].strip() for line in lines if "type:" in line),
                "unknown",
            )
            print(f"   Name: {name}")
            print(f"   Type: {comp_type}")
        except Exception as e:
            print(f"   Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Dapr LLM components")
    parser.add_argument(
        "--component",
        choices=["echo", "openai", "anthropic", "all"],
        default="all",
        help="Which component to test (default: all)",
    )
    parser.add_argument(
        "--show-config", action="store_true", help="Show component configurations"
    )

    args = parser.parse_args()

    print("🚀 Dapr LLM Components Test Suite")
    print("=" * 50)

    if args.show_config:
        show_component_configurations()
        return

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Create Anthropic component if needed
    create_anthropic_component()

    # Show available components
    show_component_configurations()

    # Run tests based on selection
    success = True

    if args.component == "all" or args.component == "echo":
        success &= test_echo_component()

    if args.component == "all" or args.component == "openai":
        success &= test_openai_component()

    if args.component == "all" or args.component == "anthropic":
        success &= test_anthropic_component()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("\n💡 Next steps:")
        print("   - Try the dapr-agents examples in quickstarts/")
        print("   - Build your own agents with tool calling")
        print("   - Explore streaming capabilities")
    else:
        print("❌ Some tests failed")
        print("\n🔧 Troubleshooting:")
        print("   - Check API keys are set correctly")
        print("   - Verify component configurations")
        print("   - Ensure Dapr sidecar is running")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

"""Pytest configuration and shared fixtures for Dapr Agents testing."""

import pytest
import logging
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Generator
from unittest.mock import MagicMock

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from tests.utils import (
    # ScenarioManager,
    # DevelopmentScenario,
    # EnvManager,
    # ComponentManager,
    # DaprManager,
    # VersionManager,
# )

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# This file is used by pytest to configure the test environment
# and provide shared fixtures across all tests.


# @pytest.fixture(scope="session")
# def scenario_manager():
#     """Scenario manager for detecting development environment."""
#     return ScenarioManager()


# @pytest.fixture(scope="session")
# def development_scenario(scenario_manager):
#     """Detect and configure development scenario."""
#     scenario = scenario_manager.detect_scenario()
#     logger.info(f"Detected development scenario: {scenario.value}")
#     logger.info(f"Description: {scenario_manager.get_scenario_description(scenario)}")
#     return scenario


# @pytest.fixture(scope="session")
# def env_manager():
#     """Environment manager for test setup."""
#     return EnvManager()


# @pytest.fixture(scope="session")
# def component_manager():
#     """Component manager for test configurations."""
#     return ComponentManager()


# @pytest.fixture(scope="session")
# def version_manager():
#     """Version manager for compatibility checking."""
#     return VersionManager()


# @pytest.fixture(scope="session")
# def test_environment(env_manager, development_scenario):
#     """Setup test environment based on development scenario."""
#     setup_result = env_manager.setup_for_scenario(development_scenario)

#     if not setup_result["requirements_met"]:
#         pytest.skip(
#             f"Test environment requirements not met for scenario {development_scenario.value}"
#         )

#     if setup_result["warnings"]:
#         for warning in setup_result["warnings"]:
#             logger.warning(f"Test environment warning: {warning}")

#     return setup_result


# @pytest.fixture(scope="session")
# def dapr_runtime(
#     development_scenario, component_manager, test_environment
# ) -> Generator[DaprManager, None, None]:
#     """Session-scoped Dapr runtime based on development scenario."""
#     # Use CLI for production-like scenarios, local binary for development
#     use_cli = development_scenario in [
#         DevelopmentScenario.AGENT_ONLY,
#         DevelopmentScenario.PRODUCTION,
#     ]
#     manager = DaprManager(use_cli=use_cli)

#     # Setup CLI if needed
#     if use_cli and not manager._is_dapr_cli_installed():
#         logger.info("Setting up Dapr CLI for production-like testing...")
#         if not manager.setup_dapr_cli(interactive=False):
#             pytest.skip("Could not setup Dapr CLI for production testing")

#     # Get components directory for scenario
#     components_dir = component_manager.get_components_for_scenario(development_scenario)

#     # Ensure components exist for scenario
#     component_manager.create_scenario_components(development_scenario)

#     # Start Dapr runtime
#     app_id = f"test-{development_scenario.value.replace('_', '-')}"
#     success = manager.start_dapr(components_dir, app_id, development_scenario)

#     if not success:
#         pytest.skip(
#             f"Could not start Dapr runtime for scenario {development_scenario.value}"
#         )

#     # Validate that required components are loaded
#     components = manager.list_components()
#     logger.info(f"Dapr started with {len(components)} components: {components}")

#     yield manager

#     # Cleanup
#     manager.stop_dapr()


@pytest.fixture(scope="session")
def api_keys(env_manager):
    """Available API keys for testing."""
    return env_manager.check_api_keys()


@pytest.fixture
def skip_if_no_api_key(api_keys):
    """Decorator to skip tests if required API key is missing."""

    def _skip_if_no_key(provider: str):
        if not api_keys.get(provider, False):
            pytest.skip(f"API key for {provider} not available")

    return _skip_if_no_key


@pytest.fixture
def dapr_endpoints(dapr_runtime):
    """Dapr endpoints for testing."""
    return dapr_runtime.get_endpoints()


@pytest.fixture
def sample_tools():
    """Common tool definitions for testing."""
    from dapr_agents.tool.base import tool

    @tool
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"Weather in {location}: Sunny, 72°F"

    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            # Simple safe evaluation for testing
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def get_time() -> str:
        """Get current time."""
        from datetime import datetime

        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return [get_weather, calculate, get_time]


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
        {"role": "user", "content": "Calculate 2 + 2"},
        {"role": "user", "content": "What time is it?"},
        {
            "role": "user",
            "content": "Tell me about the weather in New York and what time it is",
        },
    ]


@pytest.fixture
def dapr_chat_client(dapr_endpoints):
    """DaprChatClient instance for testing."""
    from dapr_agents.llm import DaprChatClient

    return DaprChatClient()


@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Mark tests in integration/ directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark tests in unit/ directory as unit tests
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def test_logging(request):
    """Configure logging for tests."""
    import logging

    # Set up logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# @pytest.fixture(scope="session", autouse=True)
# def session_setup(development_scenario, version_manager):
#     """Session-level setup and teardown."""
#     import logging

#     logger = logging.getLogger(__name__)

#     logger.info("=== Test Session Starting ===")
#     logger.info(f"Development Scenario: {development_scenario.value}")

#     # Check version compatibility
#     versions = version_manager.get_all_versions()
#     compatibility = version_manager.check_compatibility(versions)

#     # Check if all compatibility checks pass
#     all_compatible = all(compatibility.values())
    
#     if not all_compatible:
#         logger.warning("Environment compatibility issues detected:")
#         for check_name, is_compatible in compatibility.items():
#             if not is_compatible:
#                 logger.warning(f"  - {check_name}: Not compatible")

#     yield

#     logger.info("=== Test Session Complete ===")


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data."""
    from pathlib import Path

    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_components_dir(tmp_path_factory):
    """Temporary components directory for tests."""
    return tmp_path_factory.mktemp("components")


def get_available_providers(api_keys: Dict[str, bool]) -> list:
    """Get list of available providers based on API keys."""
    providers = ["echo"]  # Echo is always available

    for provider, available in api_keys.items():
        if available:
            providers.append(provider)

    return providers


# def require_scenario(scenario: DevelopmentScenario):
#     """Decorator to require specific development scenario."""

#     def decorator(func):
#         return pytest.mark.skipif(
#             ScenarioManager().detect_scenario() != scenario,
#             reason=f"Test requires {scenario.value} development scenario",
#         )(func)

#     return decorator


@pytest.fixture(autouse=True)
def patch_openai_client(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", MagicMock())


@pytest.fixture(autouse=True)
def set_llm_component_default_env(monkeypatch):
    """Ensure DAPR_LLM_COMPONENT_DEFAULT is set for all tests."""
    monkeypatch.setenv("DAPR_LLM_COMPONENT_DEFAULT", "openai")


@pytest.fixture(autouse=True)
def set_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


# Export commonly used items
__all__ = [
    # "development_scenario",
    "dapr_runtime",
    "dapr_chat_client",
    "sample_tools",
    "sample_messages",
    "api_keys",
    "skip_if_no_api_key",
    "get_available_providers",
    # "require_scenario",
]

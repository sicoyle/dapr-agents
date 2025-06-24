# Dapr Agents Test Suite

This directory contains the unit tests for the dapr-agents library.

## Test Structure

The test suite is organized as follows:

- `conftest.py` - Pytest configuration and common fixtures
- `test_agent_base.py` - Tests for the base AgentBase class
- `test_durable_agent.py` - Tests for the DurableAgent class
- `test_agent_factory.py` - Tests for the AgentFactory and Agent classes
- `test_agent_patterns.py` - Tests for agent patterns (ReActAgent, ToolCallAgent, OpenAPIReActAgent)
- `test_types.py` - Tests for core types used in the library
- `run_tests.py` - Simple test runner script

## Running Tests

### Using pytest directly
```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=dapr_agents --cov-report=term-missing

# Run specific test file
pytest tests/test_agent_base.py -v

# Run specific test class
pytest tests/test_agent_base.py::TestAgentBase -v

# Run specific test method
pytest tests/test_agent_base.py::TestAgentBase::test_agent_initialization -v
```

### Using Makefile
```bash
# Run basic tests
make test

# Run tests with coverage
make test-cov

# Install test dependencies and run tests
make test-all
```

### Using tox
```bash
# Run tests in isolated environment
tox -e pytest
```

### Using the test runner script
```bash
# Run the test runner
python tests/run_tests.py
```

## Test Dependencies

Install test dependencies:
```bash
pip install -r test-requirements.txt
```

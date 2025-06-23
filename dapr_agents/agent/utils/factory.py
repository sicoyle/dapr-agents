import os
import yaml
import warnings
import asyncio
import signal
from typing import Optional, List, Union, Type, TypeVar, Dict, Any
from pathlib import Path

from dapr_agents.agent.patterns import ReActAgent, ToolCallAgent, OpenAPIReActAgent
from dapr_agents.workflow.agents import DurableAgent
from dapr_agents.tool.utils.openapi import OpenAPISpecParser
from dapr_agents.memory import ConversationListMemory
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.llm import LLMClientBase
from dapr_agents.memory import MemoryBase
from dapr_agents.tool import AgentTool
from dapr_agents.storage import ChromaVectorStore
from dapr_agents.config import Config

T = TypeVar("T", ToolCallAgent, ReActAgent, OpenAPIReActAgent, DurableAgent)

class AgentFactory:
    """
    Returns agent classes based on the provided pattern.
    """

    AGENT_PATTERNS = {
        "react": ReActAgent,
        "toolcalling": ToolCallAgent,
        "openapireact": OpenAPIReActAgent,
    }

    @staticmethod
    def create_agent_class(pattern: str) -> Type[T]:
        """
        Selects the agent class based on the pattern.

        Args:
            pattern (str): Pattern type ('react', 'toolcalling', 'openapireact').

        Returns:
            Type: Corresponding agent class.

        Raises:
            ValueError: If the pattern is unsupported.
        """
        pattern = pattern.lower()
        agent_class = AgentFactory.AGENT_PATTERNS.get(pattern)
        if not agent_class:
            raise ValueError(f"Unsupported agent pattern: {pattern}")
        return agent_class


class Agent:
    """
    Agent class with common interface and external configuration support.
    This is a wrapper that automatically selects the appropriate agent type.
    Agents created can be a ToolCallAgent, ReActAgent, or OpenAPIReActAgent based on the parameters passed in.
    """

    def __init__(
        self,
        role: str,
        name: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        tools: Optional[List[AgentTool]] = None,
        llm: Optional[LLMClientBase] = None,
        memory: Optional[MemoryBase] = None,
        pattern: Optional[str] = None,  # Deprecating but supported - can I just rm?
        reasoning: bool = False,
        openapi_spec_path: Optional[str] = None,
        config_file: Optional[str] = None,
        # TODO(@Sicoyle): add api_vector_store and am i missing anything else?
        **kwargs
    ):
        """
        Initialize the unified agent with automatic type selection.

        Args:
            role: Agent role
            name: Agent name
            goal: Agent goal
            instructions: List of instructions
            tools: List of tools
            llm: LLM client (supported for backward compatibility)
            memory: Memory instance (supported for backward compatibility)
            pattern: Agent pattern (deprecated, use reasoning/openapi_spec_path instead)
            reasoning: Whether to use reasoning (triggers ReActAgent)
            openapi_spec_path: Path to OpenAPI spec (triggers OpenAPIReActAgent)
            config_file: Path to YAML configuration file
            **kwargs: Additional parameters
        """
        # Handle deprecated pattern  - can I just rm?
        if pattern is not None:
            warnings.warn(
                "The 'pattern' parameter is deprecated and will be removed in a future version. "
                "Use 'reasoning=True' for ReActAgent, and 'openapi_spec_path' for OpenAPIReActAgent. "
                "Default is ToolCallAgent.",
                DeprecationWarning,
                stacklevel=2
            )
            # Map pattern to new parameters for backward compatibility
            if pattern.lower() == "react":
                reasoning = True
            elif pattern.lower() == "openapireact":
                if not openapi_spec_path:
                    warnings.warn(
                        "OpenAPIReAct pattern requires 'openapi_spec_path' parameter. "
                        "Please provide the path to your OpenAPI specification.",
                        UserWarning,
                        stacklevel=2
                    )
        
        config = self._load_configuration(config_file, kwargs)
        
        # Lazy initialization for LLM and memory with error handling
        try:
            llm = llm or OpenAIChatClient()
        except Exception as e:
            if "api_key" in str(e).lower() or "openai_api_key" in str(e).lower():
                raise ValueError(
                    "OpenAI API key is required. Please set the OPENAI_API_KEY environment variable:\n"
                    "export OPENAI_API_KEY='your-api-key-here'\n"
                    "Or pass it directly to the Agent constructor."
                ) from e
            else:
                raise ValueError(f"Failed to initialize LLM client: {e}") from e
        
        try:
            memory = memory or ConversationListMemory()
        except Exception as e:
            raise ValueError(f"Failed to initialize memory: {e}") from e
        
        agent_type = self._determine_agent_type(config, pattern, reasoning, openapi_spec_path)
        self._agent_type = agent_type
        
        # Handle OpenAPI-specific kwargs
        if self._agent_type == "openapireact":
            # Only create spec_parser if we have an openapi_spec_path
            if openapi_spec_path and not kwargs.get("spec_parser"):
                try:
                    spec_parser = OpenAPISpecParser.from_file(openapi_spec_path)
                    kwargs["spec_parser"] = spec_parser
                except Exception as e:
                    warnings.warn(f"Failed to load OpenAPI spec from {openapi_spec_path}: {e}")
            
            # Add required vector store for OpenAPI agents?
            # TODO(@Sicoyle): should this be supported for all agent types or just this one??
            if not kwargs.get("api_vector_store"):
                try:
                    kwargs["api_vector_store"] = ChromaVectorStore()
                except ImportError as e:
                    raise ImportError(
                        f"OpenAPIReActAgent requires additional dependencies. "
                        f"Install them with: pip install sentence-transformers chromadb\n"
                        f"Original error: {e}"
                    ) from e
            
            kwargs.update({
                "auth_header": kwargs.get("auth_header", {}),
            })

         # Store config file path for as_service method
        self._config_file = config_file
        self.agent = self._create_agent(config, kwargs, role, name, goal, instructions, tools, llm, memory)
        self._validate_environment()
        
        # Set up graceful shutdown
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (OSError, ValueError):
            # TODO: test this bc signal handlers may not work in all environments (e.g., Windows)
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self._shutdown_event.set()
    
    def _validate_environment(self):
        """Validate that the environment is properly set up"""
        # Check for OpenAI API key only if using OpenAI client
        if hasattr(self, 'agent') and hasattr(self.agent, 'llm') and self.agent.llm:
            llm_class_name = self.agent.llm.__class__.__name__
            if 'OpenAI' in llm_class_name and not os.getenv("OPENAI_API_KEY"):
                print("⚠️  Warning: OPENAI_API_KEY environment variable is not set.")
                print("   This is required for OpenAI LLM interactions.")
                print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
                print()
    
    def _load_configuration(self, config_file: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file or params"""
        config_loader = Config()
        config = config_loader.load_defaults()
        
        if config_file:
            # Resolve relative path to config file
            if not os.path.isabs(config_file):
                config_file = os.path.abspath(config_file)
            config.update(config_loader.load_config_with_global(config_file))
        
        # Params override file config (instantiation parameters take precedence)
        config.update(params)
        return config
    
    def _determine_agent_type(self, config: Dict[str, Any], pattern: Optional[str] = None, 
                            reasoning: bool = False, openapi_spec_path: Optional[str] = None) -> str:
        """Automatic agent type selection based on configuration"""
        # Priority order for agent selection
        if config.get('agent', {}).get('openapi_spec_path') or openapi_spec_path:
            return 'openapireact'  # OpenAPIReActAgent
        elif config.get('agent', {}).get('reasoning') or reasoning:
            return 'react'  # ReActAgent
        elif pattern:
            # Fallback to pattern for backward compatibility
            pattern_lower = pattern.lower()
            if pattern_lower in ['react', 'toolcalling', 'openapireact']:
                return pattern_lower
        else:
            return 'toolcall'  # ToolCallAgent (default)
    
    def _create_agent(self, config: Dict[str, Any], kwargs: Dict[str, Any], 
                     role: str, name: Optional[str], goal: Optional[str], 
                     instructions: Optional[List[str]], tools: Optional[List[AgentTool]], 
                     llm: LLMClientBase, memory: MemoryBase):
        """Create the appropriate agent type with configuration"""
        agent_params = {
            'role': role,
            'name': name,
            'goal': goal,
            'instructions': instructions,
            'tools': tools or [],
            'llm': llm,
            'memory': memory,
        }
        
        # Add agent-specific configuration
        if 'agent' in config:
            agent_params.update(config['agent'])

        # Add any additional kwargs (excluding factory-specific ones)
        agent_params.update(kwargs)
        
        if self._agent_type == 'openapireact':
            return OpenAPIReActAgent(**agent_params)
        elif self._agent_type == 'react':
            return ReActAgent(**agent_params)
        else:
            return ToolCallAgent(**agent_params)
    
    async def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """Run the agent with the given input"""
        try:
            # Check for shutdown before running
            if self._shutdown_event.is_set():
                print("Shutdown requested. Skipping agent execution.")
                return None
            
            # Create a task that can be cancelled
            task = asyncio.create_task(self.agent.run(input_data))
            
            # Wait for either completion or shutdown
            done, pending = await asyncio.wait(
                [task, asyncio.create_task(self._shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for p in pending:
                p.cancel()
            
            if self._shutdown_event.is_set():
                print("Shutdown requested during execution. Cancelling agent.")
                task.cancel()
                return None
            
            if task in done:
                return await task
            
        except asyncio.CancelledError:
            print("Agent execution was cancelled.")
            return None
        except Exception as e:
            print(f"Error during agent execution: {e}")
            raise

    def __getattr__(self, name):
        """Delegate attribute access to the underlying agent"""
        return getattr(self.agent, name)
    
    # TODO(@Sicoyle): add as_service method in future PR so these agents have the same start options as durable agent
    # TODO(@Sicoyle): add start method in future PR so these agents have the same start options as durable agent

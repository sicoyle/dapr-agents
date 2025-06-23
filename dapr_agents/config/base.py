import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """
    Unified configuration management for Dapr Agents.
    Supports loading from YAML files with global config references and merging.
    """

    def __init__(self):
        self.config = {}

    def load_defaults(self) -> Dict[str, Any]:
        """Load sensible defaults for all agent types"""
        return {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 4000
            },
            'dapr': {
                'message_bus_name': 'messagepubsub',
                'state_store_name': 'workflowstatestore',
                'state_key': 'workflow_state',
                'agents_registry_store_name': 'workflowstatestore',
                'agents_registry_key': 'agents_registry',
                'service_port': 8001,
                'grpc_port': 50001
            },
            'agent': {
                'max_iterations': 10,
                'tool_choice': 'auto',
                'reasoning': False,
            },
            'workflow': {
                'max_iterations': 20,
                'save_state_locally': True,
                'local_state_path': None
            }
        }

    def load_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            Dict containing the loaded configuration
        """
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config or {}

    def load_config_with_global(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file with support for global config references.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            Dict containing the merged configuration
        """
        # Resolve config file path
        config_file = os.path.abspath(config_file)
        if not os.path.exists(config_file):
            raise ValueError(f"Config file not found: {config_file}")
        
        # Load base config
        config = self.load_yaml_config(config_file)
        
        # Load global config if specified
        if 'global_config' in config:
            global_config_path = os.path.join(os.path.dirname(config_file), config['global_config'])
            if not os.path.exists(global_config_path):
                raise ValueError(f"Global config file not found: {global_config_path}")
            
            global_config = self.load_yaml_config(global_config_path)
            
            # If agent_config is specified, get that section from global config
            if 'agent_config' in config and config['agent_config'] in global_config.get('agent', {}):
                agent_config = global_config['agent'][config['agent_config']]
                config.update(agent_config)
            
            # Merge global dapr config with local overrides
            if 'dapr' in global_config:
                global_dapr = global_config['dapr'].copy()
                global_dapr.update(config.get('dapr', {}))
                config['dapr'] = global_dapr
        
        return config

    def load_global_config(self, config_path: str) -> Dict[str, Any]:
        """Load global configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def merge_configs(self, global_config: Dict[str, Any], local_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge global config with local config, applying references and overrides.
        
        Args:
            global_config: The global configuration dictionary
            local_config: The local configuration dictionary
            
        Returns:
            Dict containing the merged configuration
        """
        merged = {}
        
        # Merge LLM config
        if 'llm_config' in local_config:
            llm_key = local_config['llm_config']
            if llm_key in global_config.get('llm', {}):
                merged['llm'] = global_config['llm'][llm_key]
        
        # Merge Dapr config
        if 'dapr_config' in local_config:
            dapr_key = local_config['dapr_config']
            if dapr_key in global_config.get('dapr', {}):
                merged['dapr'] = global_config['dapr'][dapr_key]
        else:
            # If no dapr_config specified, use the flat dapr config
            merged['dapr'] = global_config.get('dapr', {})
        
        # Merge agent config
        if 'agent_config' in local_config:
            agent_key = local_config['agent_config']
            if agent_key in global_config.get('agent', {}):
                merged['agent'] = global_config['agent'][agent_key]
        
        # Merge workflow config
        if 'workflow_config' in local_config:
            workflow_key = local_config['workflow_config']
            if workflow_key in global_config.get('workflow', {}):
                merged['workflow'] = global_config['workflow'][workflow_key]
        
        # Apply local overrides
        for section in ['llm', 'dapr', 'agent', 'workflow']:
            if section in local_config:
                if section not in merged:
                    merged[section] = {}
                merged[section].update(local_config[section])
        
        return merged

    def get_dapr_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Dapr configuration from merged config"""
        return config.get('dapr', {})

    def get_llm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract LLM configuration from merged config"""
        return config.get('llm', {})

    def get_agent_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent configuration from merged config"""
        return config.get('agent', {})

    def get_workflow_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract workflow configuration from merged config"""
        return config.get('workflow', {})

    def apply_to_object(self, obj: Any, config: Dict[str, Any], section: str):
        """
        Apply configuration to an object's attributes.
        
        Args:
            obj: The object to apply configuration to
            config: The configuration dictionary
            section: The configuration section to apply (e.g., 'dapr', 'workflow')
        """
        section_config = config.get(section, {})
        
        for key, value in section_config.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
            else:
                # Log warning for unknown attributes
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Unknown configuration key '{key}' for section '{section}'")

    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """
        Create a Config instance from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            Config instance with loaded configuration
        """
        config_instance = cls()
        config_instance.config = config_instance.load_config_with_global(file_path)
        return config_instance 
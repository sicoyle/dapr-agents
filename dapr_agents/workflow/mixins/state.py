import json
import logging
import os
import tempfile
import threading
from typing import Optional, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

state_lock = threading.Lock()


class StateManagementMixin:
    """
    Mixin providing workflow state initialization, validation, and persistence.
    """

    # TODO: Delete this once we rm orchestrators in favor of agents as tools.
    @property
    def state(self) -> dict:
        """
        Get the current workflow state.
        
        Returns:
            dict: The current workflow state.
        """
        return self.storage._current_state if hasattr(self, 'storage') else {}

    # TODO: Delete this once we rm orchestrators in favor of agents as tools.
    @state.setter
    def state(self, value: dict) -> None:
        """
        Set the current workflow state.
        
        Args:
            value (dict): The new workflow state.
        """
        if hasattr(self, 'storage'):
            self.storage._current_state = value

    def initialize_state(self) -> None:
        """
        Initialize workflow state from provided value or storage.

        Raises:
            RuntimeError: If state initialization or loading from storage fails.
        """
        try:
            if self.storage._current_state is None:
                logger.debug("No user-provided state. Attempting to load from storage.")
                self.storage._current_state = self.load_state()

            if isinstance(self.storage._current_state, BaseModel):
                logger.debug(
                    "User provided a state as a Pydantic model. Converting to dict."
                )
                self.storage._current_state = self.storage._current_state.model_dump()

            if not isinstance(self.storage._current_state, dict):
                raise TypeError(
                    f"Invalid state type: {type(self.storage._current_state)}. Expected dict."
                )

            logger.debug(
                f"Workflow state initialized with {len(self.storage._current_state)} key(s)."
            )
            self.save_state()
        except Exception as e:
            raise RuntimeError(f"Error initializing workflow state: {e}") from e

    def load_state(self) -> dict:
        """
        Load the workflow state from the configured Dapr state store.

        Returns:
            dict: The loaded and optionally validated state.

        Raises:
            RuntimeError: If the state store is not properly configured.
            TypeError: If the retrieved state is not a dictionary.
            ValidationError: If state schema validation fails.
        """
        try:
            if (
                not self._state_store_client
                or not self.storage.name
                or not self.storage._key
            ):
                logger.error("State store is not configured. Cannot load state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'storage.name'."
                )

            # For durable agents, always load from database to ensure it's the source of truth
            has_state, state_data = self._state_store_client.try_get_state(
                self.storage._key
            )
            if has_state and state_data:
                logger.debug(
                    f"Existing state found for key '{self.storage._key}'. Validating it."
                )
                # Deserialize JSON string if necessary
                if isinstance(state_data, str):
                    import json
                    state_data = json.loads(state_data)
                
                if not isinstance(state_data, dict):
                    raise TypeError(
                        f"Invalid state type retrieved: {type(state_data)}. Expected dict."
                    )
                self.storage._current_state = state_data
            else:
                self.storage._current_state = {}

            # Load workflow instances from the current session
            session_id = self.storage._get_session_id()
            session_key = self.storage._get_session_key(session_id)
            has_session, session_data = self._state_store_client.try_get_state(session_key)
            
            # Always ensure "instances" key exists
            self.storage._current_state.setdefault("instances", {})
            
            if has_session and session_data:
                # Handle JSON string if necessary
                if isinstance(session_data, str):
                    import json

                    session_data = json.loads(session_data)

                instance_ids = session_data.get("workflow_instances", [])

                # Load each instance
                for instance_id in instance_ids:
                    instance_key = self.storage._get_instance_key(instance_id)
                    (
                        has_instance,
                        instance_data,
                    ) = self._state_store_client.try_get_state(instance_key)
                    if has_instance and instance_data:
                        # Deserialize if it's a JSON string
                        if isinstance(instance_data, str):
                            instance_data = json.loads(instance_data)
                        self.storage._current_state["instances"][
                            instance_id
                        ] = instance_data
                        logger.debug(
                            f"Loaded workflow instance {instance_id} from key '{instance_key}'"
                        )

            logger.debug(
                f"Set self.storage._current_state to loaded data: {self.storage._current_state}"
            )
            return self.storage._current_state

            logger.debug(
                f"No existing state found for key '{self.storage._key}'. Initializing empty state."
            )
            return {}
        except Exception as e:
            logger.error(f"Failed to load state for key '{self.storage._key}': {e}")
            raise RuntimeError(f"Error loading workflow state: {e}") from e

    def get_local_state_file_path(self) -> str:
        """
        Return the file path for saving the local state.

        Returns:
            str: The absolute path to the local state file.
        """
        if not self.storage.local_directory:
            return os.path.join(os.getcwd(), f"{self.name}_state.json")
        os.makedirs(self.storage.local_directory, exist_ok=True)

        # If relative path, make it absolute from workspace root
        if not os.path.isabs(self.storage.local_directory):
            abs_path = os.path.join(os.getcwd(), self.storage.local_directory)
        else:
            abs_path = self.storage.local_directory

        return os.path.join(abs_path, f"{self.name}_state.json")

    def save_state_to_disk(
        self, state_data: str, filename: Optional[str] = None
    ) -> None:
        """
        Safely save the workflow state to a local JSON file.

        Args:
            state_data: The state data to save (as JSON string or dict).
            filename: Optional filename for the state file.

        Raises:
            RuntimeError: If saving to disk fails.
        """
        try:
            file_path = filename or self.get_local_state_file_path()
            save_directory = os.path.dirname(file_path)
            os.makedirs(save_directory, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                "w", dir=save_directory, delete=False
            ) as tmp_file:
                tmp_file.write(state_data)
                temp_path = tmp_file.name

            with state_lock:
                existing_state = {}
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        try:
                            existing_state = json.load(file)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Existing state file is corrupt or empty. Overwriting."
                            )

                new_state = (
                    json.loads(state_data)
                    if isinstance(state_data, str)
                    else state_data
                )
                merged_state = {**existing_state, **new_state}

                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(merged_state, file, indent=4)

                os.replace(temp_path, file_path)

            logger.debug(f"Workflow state saved locally at '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to save workflow state to disk: {e}")
            raise RuntimeError(f"Error saving workflow state to disk: {e}")

    def save_state(
        self,
        state: Optional[Union[dict, BaseModel, str]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Save the current workflow state to Dapr and optionally to disk.

        Args:
            state: The new state to save. If not provided, saves the existing state.
            force_reload: If True, reloads the state from the store after saving.

        Raises:
            RuntimeError: If the state store is not configured.
            TypeError: If the provided state is not a supported type.
            ValueError: If the provided state is a string but not valid JSON.
        """
        try:
            if (
                not self._state_store_client
                or not self.storage.name
                or not self.storage._key
            ):
                logger.error("State store is not configured. Cannot save state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'storage.name'."
                )

            self.storage._current_state = state or self.storage._current_state
            if not self.storage._current_state:
                logger.warning("Skipping state save: Empty state.")
                return

            if isinstance(self.storage._current_state, BaseModel):
                state_to_save = self.storage._current_state.model_dump_json()
            elif isinstance(self.storage._current_state, dict):
                state_to_save = json.dumps(self.storage._current_state)
            elif isinstance(self.storage._current_state, str):
                try:
                    json.loads(self.storage._current_state)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string provided as state: {e}")
                state_to_save = self.storage._current_state
            else:
                raise TypeError(
                    f"Invalid state type: {type(self.storage._current_state)}. Expected dict, BaseModel, or JSON string."
                )

            # Save each workflow instance separately
            if "instances" in self.storage._current_state:
                for instance_id, instance_data in self.storage._current_state[
                    "instances"
                ].items():
                    instance_key = self.storage._get_instance_key(instance_id)
                    # Handle both dict and already-serialized string
                    if isinstance(instance_data, dict):
                        instance_json = json.dumps(instance_data)
                    elif isinstance(instance_data, str):
                        instance_json = instance_data
                    else:
                        instance_json = json.dumps(instance_data)
                    self._state_store_client.save_state(instance_key, instance_json)
                    logger.debug(
                        f"Saved workflow instance {instance_id} to key '{instance_key}'"
                    )

            # Save other state data (like chat_history) to main key
            other_state = {
                k: v for k, v in self.storage._current_state.items() if k != "instances"
            }
            if other_state:
                other_state_json = json.dumps(other_state)
                self._state_store_client.save_state(self.storage._key, other_state_json)
                logger.debug(f"Saved non-instance state to key '{self.storage._key}'")

            if self.storage.local_directory is not None:
                self.save_state_to_disk(state_data=state_to_save)

            if force_reload:
                self.storage._current_state = self.load_state()
                logger.debug(
                    f"State reloaded after saving for key '{self.storage._key}'."
                )
        except Exception as e:
            logger.error(f"Failed to save state for key '{self.storage._key}': {e}")
            raise

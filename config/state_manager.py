import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class RenderState:
    """
    Represents the complete state of a rendering task.
    """
    system_type: str
    system_parameters: Dict[str, float]
    numerical_settings: Dict[str, Any]
    camera_settings: Dict[str, Any]
    pipeline_settings: Dict[str, Any]


class RenderConfigManager:
    """
    Manages the serialization and deserialization of the rendering state.
    """
    @staticmethod
    def save_config(state: RenderState, filepath: str) -> None:
        """
        Saves the render state to a JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, indent=4)

    @staticmethod
    def load_config(filepath: str) -> RenderState:
        """
        Loads the render state from a JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return RenderState(**data)

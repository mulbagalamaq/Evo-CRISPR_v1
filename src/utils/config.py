"""
Configuration Utilities for EvoBeevos+ Variant Predictor

This module provides functions to load, save, and manage configuration settings
including API keys, default parameters, and user preferences.
"""

import os
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Constants
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "api_keys": {
        "groq": "",
        "evo": ""
    },
    "prediction": {
        "default_window_size": 2048,
        "default_reference_genome": "GRCh38",
        "confidence_threshold": 0.7,
        "use_mock_predictions": False
    },
    "interface": {
        "theme": "light",
        "show_advanced_options": False,
        "default_input_method": "position",
        "save_results": True,
        "results_folder": "results"
    },
    "ensembl": {
        "use_cache": True,
        "cache_ttl_days": 30
    },
    "clinvar": {
        "use_cache": True,
        "cache_ttl_days": 30
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or create default if not exists.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    if config_path is None:
        # Use the default location
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), CONFIG_FILE)
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update with any new default settings
            merged_config = DEFAULT_CONFIG.copy()
            _update_nested_dict(merged_config, config)
            return merged_config
        else:
            # Create default config
            save_config(DEFAULT_CONFIG, config_path)
            return DEFAULT_CONFIG
            
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to the configuration file
        
    Returns:
        Boolean indicating success
    """
    if config_path is None:
        # Use the default location
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), CONFIG_FILE)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
        
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False

def get_api_key(key_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get an API key from environment variables or config.
    
    Args:
        key_name: Name of the API key
        config: Configuration dictionary (optional)
        
    Returns:
        API key as string or None if not found
    """
    # First try environment variable
    env_var_name = f"{key_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    if api_key:
        return api_key
    
    # Then try Streamlit secrets
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    # Finally try the config file
    if config is None:
        config = load_config()
    
    return config.get("api_keys", {}).get(key_name.lower())

def save_api_key(key_name: str, api_key: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save an API key to the configuration.
    
    Args:
        key_name: Name of the API key
        api_key: API key value to save
        config: Configuration dictionary (optional)
        
    Returns:
        Boolean indicating success
    """
    if config is None:
        config = load_config()
    
    # Make sure the api_keys section exists
    if "api_keys" not in config:
        config["api_keys"] = {}
    
    # Update the key
    config["api_keys"][key_name.lower()] = api_key
    
    # Save the updated config
    return save_config(config)

def get_setting(setting_path: str, default_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a setting from the configuration using a dot-notation path.
    
    Args:
        setting_path: Path to the setting (e.g., "prediction.default_window_size")
        default_value: Default value to return if setting not found
        config: Configuration dictionary (optional)
        
    Returns:
        Setting value or default if not found
    """
    if config is None:
        config = load_config()
    
    # Split the path and navigate through the config
    parts = setting_path.split('.')
    current = config
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default_value
    
    return current

def save_setting(setting_path: str, value: Any, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save a setting to the configuration using a dot-notation path.
    
    Args:
        setting_path: Path to the setting (e.g., "prediction.default_window_size")
        value: Value to save
        config: Configuration dictionary (optional)
        
    Returns:
        Boolean indicating success
    """
    if config is None:
        config = load_config()
    
    # Split the path and navigate through the config
    parts = setting_path.split('.')
    current = config
    
    # Navigate to the parent object that will contain our setting
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Set the value
    current[parts[-1]] = value
    
    # Save the updated config
    return save_config(config)

def _update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a nested dictionary with another dictionary.
    
    Args:
        d: Target dictionary
        u: Source dictionary
        
    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d

def init_results_directory(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Initialize the results directory where variant analysis results are saved.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        Path to the results directory
    """
    if config is None:
        config = load_config()
    
    # Get the results folder from config
    results_folder = config.get("interface", {}).get("results_folder", "results")
    
    # Use absolute path if provided, otherwise create relative to the app directory
    if os.path.isabs(results_folder):
        results_path = results_folder
    else:
        results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), results_folder)
    
    # Create the directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    return results_path

def get_cache_directory() -> str:
    """
    Get the path to the cache directory.
    
    Returns:
        Path to the cache directory
    """
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".cache")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path 
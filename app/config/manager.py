"""
Configuration Manager

Centralized configuration management with validation, encryption,
and dynamic updates.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime
import logging
from cryptography.fernet import Fernet
import threading
from contextlib import contextmanager

from .settings import Settings
from .validators import ConfigValidator, ValidationError
from .loader import ConfigLoader, FileConfigLoader, EnvironmentConfigLoader

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration manager

    Features:
    - Load configuration from multiple sources
    - Validate configuration
    - Encrypt sensitive values
    - Support dynamic updates
    - Provide configuration change notifications
    - Environment-specific overrides
    """

    def __init__(self, config_file: Optional[str] = None,
                 environment: Optional[str] = None,
                 encryption_key: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file
            environment: Environment name (development, production, etc.)
            encryption_key: Key for encrypting sensitive configuration values
        """
        self.config_file = config_file
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.encryption_key = encryption_key
        self._config: Optional[Settings] = None
        self._validators: List[Callable] = []
        self._change_callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._last_updated: Optional[datetime] = None

        # Initialize encryption
        self._cipher: Optional[Fernet] = None
        if encryption_key:
            self._cipher = Fernet(encryption_key.encode())

        # Load initial configuration
        self.reload()

    @property
    def config(self) -> Settings:
        """Get current configuration"""
        if self._config is None:
            self.reload()
        return self._config

    def reload(self) -> Settings:
        """Reload configuration from all sources"""
        with self._lock:
            # Load base configuration
            config_data = self._load_base_config()

            # Apply environment overrides
            config_data = self._apply_environment_overrides(config_data)

            # Apply environment variable overrides
            config_data = self._apply_env_overrides(config_data)

            # Decrypt sensitive values
            config_data = self._decrypt_sensitive_values(config_data)

            # Create settings object
            self._config = Settings(**config_data)

            # Validate configuration
            self._validate_config()

            # Update timestamp
            self._last_updated = datetime.utcnow()

            # Notify change callbacks
            self._notify_change_callbacks()

            logger.info(f"Configuration reloaded for environment: {self.environment}")
            return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key

        Args:
            key: Configuration key (e.g., "database.host")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.config
        keys = key.split('.')

        try:
            for k in keys:
                config = getattr(config, k)
            return config
        except (AttributeError, KeyError):
            return default

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set configuration value

        Args:
            key: Configuration key (dot-separated)
            value: Value to set
            persist: Whether to persist the change to file
        """
        with self._lock:
            config_dict = self.config.dict()
            self._set_nested_value(config_dict, key, value)

            # Update config object
            self._config = Settings(**config_dict)

            # Validate after update
            self._validate_config()

            # Persist if requested
            if persist and self.config_file:
                self._persist_config()

            # Notify callbacks
            self._notify_key_change_callbacks(key, value)

    def encrypt_value(self, value: Any) -> str:
        """
        Encrypt a configuration value

        Args:
            value: Value to encrypt

        Returns:
            Encrypted value string
        """
        if not self._cipher:
            raise RuntimeError("Encryption key not configured")

        try:
            # Convert value to JSON string
            value_str = json.dumps(value)
            # Encrypt
            encrypted = self._cipher.encrypt(value_str.encode())
            # Return base64 encoded string
            return f"enc:{encrypted.decode()}"
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            raise

    def decrypt_value(self, encrypted_value: str) -> Any:
        """
        Decrypt a configuration value

        Args:
            encrypted_value: Encrypted value string

        Returns:
            Decrypted value
        """
        if not self._cipher:
            raise RuntimeError("Encryption key not configured")

        try:
            # Remove prefix
            if not encrypted_value.startswith("enc:"):
                return encrypted_value

            encrypted = encrypted_value[4:]
            # Decrypt
            decrypted = self._cipher.decrypt(encrypted.encode())
            # Parse JSON
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise

    def add_validator(self, validator: Callable[[Settings], None]) -> None:
        """Add configuration validator"""
        self._validators.append(validator)

    def add_change_callback(self, key: str, callback: Callable[[Any], None]) -> None:
        """Add callback for configuration changes"""
        if key not in self._change_callbacks:
            self._change_callbacks[key] = []
        self._change_callbacks[key].append(callback)

    def add_global_change_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for any configuration changes"""
        self.add_change_callback("*", callback)

    def validate(self) -> List[str]:
        """
        Validate current configuration

        Returns:
            List of validation errors
        """
        errors = []

        # Run built-in validators
        try:
            ConfigValidator.validate(self.config)
        except ValidationError as e:
            errors.extend(e.errors)

        # Run custom validators
        for validator in self._validators:
            try:
                validator(self.config)
            except Exception as e:
                errors.append(str(e))

        return errors

    def export_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Export configuration to dictionary

        Args:
            include_sensitive: Whether to include sensitive values

        Returns:
            Configuration dictionary
        """
        config_dict = self.config.dict()

        if not include_sensitive:
            config_dict = self._mask_sensitive_values(config_dict)

        return config_dict

    def import_config(self, config_data: Dict[str, Any],
                     merge: bool = False) -> None:
        """
        Import configuration from dictionary

        Args:
            config_data: Configuration data to import
            merge: Whether to merge with existing configuration
        """
        with self._lock:
            if merge:
                current = self.export_config()
                current.update(config_data)
                config_data = current

            # Decrypt values
            config_data = self._decrypt_sensitive_values(config_data)

            # Update configuration
            self._config = Settings(**config_data)

            # Validate
            self._validate_config()

            # Persist
            if self.config_file:
                self._persist_config()

            # Notify callbacks
            self._notify_change_callbacks()

    @contextmanager
    def transaction(self):
        """Context manager for configuration transactions"""
        old_config = self._config
        try:
            yield self
        except Exception:
            # Rollback on error
            self._config = old_config
            raise

    # Private methods

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from file"""
        if not self.config_file:
            return {}

        loader = FileConfigLoader(self.config_file)
        return loader.load()

    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific overrides"""
        env_file = f"config.{self.environment}.yaml"
        env_path = Path(self.config_file).parent / env_file

        if env_path.exists():
            env_loader = FileConfigLoader(str(env_path))
            env_config = env_loader.load()
            config_data.update(env_config)

        return config_data

    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # This would parse environment variables like DATABASE__HOST, etc.
        # For now, return unchanged
        return config_data

    def _decrypt_sensitive_values(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration values"""
        if not self._cipher:
            return config_data

        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                return {k: decrypt_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("enc:"):
                try:
                    return self.decrypt_value(obj)
                except Exception:
                    return obj
            else:
                return obj

        return decrypt_recursive(config_data)

    def _mask_sensitive_values(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive values for export"""
        sensitive_keys = [
            "password", "secret", "key", "token", "auth",
            "jwt_secret_key", "minio_secret_key", "openai_api_key"
        ]

        def mask_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: "*****" if any(sensitive in k.lower() for sensitive in sensitive_keys)
                    else mask_recursive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [mask_recursive(item) for item in obj]
            else:
                return obj

        return mask_recursive(config_data)

    def _validate_config(self) -> None:
        """Validate current configuration"""
        errors = self.validate()
        if errors:
            raise ValidationError(f"Configuration validation failed: {', '.join(errors)}")

    def _set_nested_value(self, config_dict: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value in dictionary"""
        keys = key.split('.')
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _persist_config(self) -> None:
        """Persist configuration to file"""
        if not self.config_file:
            return

        try:
            config_dict = self.export_config(include_sensitive=True)

            # Determine format based on file extension
            if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                with open(self.config_file, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                with open(self.config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration persisted to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to persist configuration: {e}")
            raise

    def _notify_change_callbacks(self) -> None:
        """Notify all change callbacks"""
        for key, callbacks in self._change_callbacks.items():
            if key == "*":
                for callback in callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in change callback: {e}")

    def _notify_key_change_callbacks(self, key: str, value: Any) -> None:
        """Notify callbacks for specific key"""
        # Check for exact match
        if key in self._change_callbacks:
            for callback in self._change_callbacks[key]:
                try:
                    callback(value)
                except Exception as e:
                    logger.error(f"Error in change callback for key {key}: {e}")

        # Check for wildcard matches
        key_parts = key.split('.')
        for i in range(1, len(key_parts)):
            wildcard_key = '.'.join(key_parts[:i]) + '.*'
            if wildcard_key in self._change_callbacks:
                for callback in self._change_callbacks[wildcard_key]:
                    try:
                        callback(value)
                    except Exception as e:
                        logger.error(f"Error in change callback for key {wildcard_key}: {e}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> Optional[ConfigManager]:
    """Get global configuration manager instance"""
    return _config_manager


def initialize_config(config_file: Optional[str] = None,
                     environment: Optional[str] = None,
                     encryption_key: Optional[str] = None) -> ConfigManager:
    """
    Initialize global configuration manager

    Args:
        config_file: Path to configuration file
        environment: Environment name
        encryption_key: Encryption key for sensitive values

    Returns:
        Configuration manager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_file, environment, encryption_key)
    return _config_manager


def get_config() -> Settings:
    """Get current application settings"""
    manager = get_config_manager()
    if manager is None:
        raise RuntimeError("Configuration manager not initialized")
    return manager.config


def reload_config() -> Settings:
    """Reload configuration"""
    manager = get_config_manager()
    if manager is None:
        raise RuntimeError("Configuration manager not initialized")
    return manager.reload()
"""
Security Module

Provides encrypted storage for API keys and sensitive configuration.
Uses keyring for system-level secure storage with fallback to encrypted file storage.
"""

import os
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


def _get_machine_id() -> str:
    """Get a machine-specific identifier for key derivation."""
    identifiers = []
    
    if os.path.exists("/etc/machine-id"):
        with open("/etc/machine-id", "r") as f:
            identifiers.append(f.read().strip())
    
    if os.path.exists("/var/lib/dbus/machine-id"):
        with open("/var/lib/dbus/machine-id", "r") as f:
            identifiers.append(f.read().strip())
    
    identifiers.append(os.environ.get("USER", "default"))
    identifiers.append(os.environ.get("HOME", "/tmp"))
    
    combined = ":".join(identifiers)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def _derive_key(master_key: bytes, salt: bytes) -> bytes:
    """Derive an encryption key from master key and salt."""
    return hashlib.pbkdf2_hmac("sha256", master_key, salt, 100000, dklen=32)


def _simple_encrypt(data: str, key: bytes) -> bytes:
    """Simple XOR-based encryption (for basic obfuscation)."""
    data_bytes = data.encode("utf-8")
    key_extended = (key * ((len(data_bytes) // len(key)) + 1))[:len(data_bytes)]
    encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_extended))
    return encrypted


def _simple_decrypt(encrypted: bytes, key: bytes) -> str:
    """Simple XOR-based decryption."""
    key_extended = (key * ((len(encrypted) // len(key)) + 1))[:len(encrypted)]
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key_extended))
    return decrypted.decode("utf-8")


@dataclass
class SecureConfig:
    """
    Secure configuration storage for API keys and sensitive data.
    
    Uses system keyring when available, falls back to encrypted file storage.
    
    Attributes
    ----------
    config_dir : str
        Directory for storing encrypted configuration files
    app_name : str
        Application name for keyring service identification
        
    Examples
    --------
    >>> config = SecureConfig()
    >>> config.store_api_key("openai", "sk-...")
    >>> key = config.get_api_key("openai")
    """
    
    config_dir: str = field(default_factory=lambda: os.path.expanduser("~/.organ_generator"))
    app_name: str = "organ_generator"
    _keyring_available: bool = field(default=False, init=False)
    _master_key: Optional[bytes] = field(default=None, init=False)
    
    def __post_init__(self):
        os.makedirs(self.config_dir, exist_ok=True)
        
        try:
            import keyring
            keyring.get_keyring()
            self._keyring_available = True
        except (ImportError, Exception):
            self._keyring_available = False
        
        self._master_key = _get_machine_id().encode("utf-8")
    
    def _get_credentials_file(self) -> Path:
        """Get path to encrypted credentials file."""
        return Path(self.config_dir) / "credentials.enc"
    
    def _load_credentials(self) -> Dict[str, str]:
        """Load credentials from encrypted file."""
        creds_file = self._get_credentials_file()
        
        if not creds_file.exists():
            return {}
        
        try:
            with open(creds_file, "rb") as f:
                data = f.read()
            
            salt = data[:16]
            encrypted = data[16:]
            
            key = _derive_key(self._master_key, salt)
            decrypted = _simple_decrypt(encrypted, key)
            
            return json.loads(decrypted)
        except Exception:
            return {}
    
    def _save_credentials(self, credentials: Dict[str, str]) -> None:
        """Save credentials to encrypted file."""
        creds_file = self._get_credentials_file()
        
        salt = os.urandom(16)
        key = _derive_key(self._master_key, salt)
        
        json_data = json.dumps(credentials)
        encrypted = _simple_encrypt(json_data, key)
        
        with open(creds_file, "wb") as f:
            f.write(salt + encrypted)
        
        os.chmod(creds_file, 0o600)
    
    def store_api_key(self, provider: str, key: str) -> bool:
        """
        Store an API key securely.
        
        Parameters
        ----------
        provider : str
            Provider name (e.g., "openai", "anthropic")
        key : str
            API key to store
            
        Returns
        -------
        bool
            True if storage was successful
        """
        if self._keyring_available:
            try:
                import keyring
                keyring.set_password(self.app_name, provider, key)
                return True
            except Exception:
                pass
        
        try:
            credentials = self._load_credentials()
            credentials[provider] = key
            self._save_credentials(credentials)
            return True
        except Exception:
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Retrieve an API key.
        
        Parameters
        ----------
        provider : str
            Provider name (e.g., "openai", "anthropic")
            
        Returns
        -------
        str or None
            API key if found, None otherwise
        """
        if self._keyring_available:
            try:
                import keyring
                key = keyring.get_password(self.app_name, provider)
                if key:
                    return key
            except Exception:
                pass
        
        try:
            credentials = self._load_credentials()
            return credentials.get(provider)
        except Exception:
            return None
    
    def delete_api_key(self, provider: str) -> bool:
        """
        Delete a stored API key.
        
        Parameters
        ----------
        provider : str
            Provider name
            
        Returns
        -------
        bool
            True if deletion was successful
        """
        if self._keyring_available:
            try:
                import keyring
                keyring.delete_password(self.app_name, provider)
            except Exception:
                pass
        
        try:
            credentials = self._load_credentials()
            if provider in credentials:
                del credentials[provider]
                self._save_credentials(credentials)
            return True
        except Exception:
            return False
    
    def list_providers(self) -> list:
        """
        List all providers with stored API keys.
        
        Returns
        -------
        list
            List of provider names
        """
        providers = set()
        
        try:
            credentials = self._load_credentials()
            providers.update(credentials.keys())
        except Exception:
            pass
        
        return list(providers)
    
    def store_config(self, key: str, value: Any) -> bool:
        """
        Store a general configuration value.
        
        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Value to store (must be JSON-serializable)
            
        Returns
        -------
        bool
            True if storage was successful
        """
        config_file = Path(self.config_dir) / "config.json"
        
        try:
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
            else:
                config = {}
            
            config[key] = value
            
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.
        
        Parameters
        ----------
        key : str
            Configuration key
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value or default
        """
        config_file = Path(self.config_dir) / "config.json"
        
        try:
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                return config.get(key, default)
        except Exception:
            pass
        
        return default
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Retrieve all configuration values.
        
        Returns
        -------
        Dict[str, Any]
            All configuration values
        """
        config_file = Path(self.config_dir) / "config.json"
        
        try:
            if config_file.exists():
                with open(config_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {}

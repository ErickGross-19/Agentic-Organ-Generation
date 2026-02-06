"""Encryption service for storing API keys securely."""
import os
import sys
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Encryption key from environment or generate deterministic one from secret
ENV = os.getenv("ENV", "development")

if ENV == "development":
    ENCRYPTION_SECRET = os.getenv("ENCRYPTION_SECRET", "dev-only-secret-not-for-production")
else:
    ENCRYPTION_SECRET = os.getenv("ENCRYPTION_SECRET")
    if not ENCRYPTION_SECRET:
        print("FATAL: ENCRYPTION_SECRET environment variable is required in production", file=sys.stderr)
        sys.exit(1)

def _get_fernet(key_id: str = "default") -> Fernet:
    """
    Get a Fernet instance for encryption/decryption.
    Uses a deterministic salt derived from the encryption secret and key_id,
    so encrypted values can be decrypted even after server restart.
    """
    # Derive a deterministic salt from the secret and key_id
    # This ensures the same key_id always produces the same Fernet key
    salt_input = f"{ENCRYPTION_SECRET}:{key_id}".encode()
    salt = hashlib.sha256(salt_input).digest()[:16]

    # Derive key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(ENCRYPTION_SECRET.encode()))
    return Fernet(key)

def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage."""
    fernet = _get_fernet()
    encrypted = fernet.encrypt(api_key.encode())
    return encrypted.decode()

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from storage."""
    fernet = _get_fernet()
    decrypted = fernet.decrypt(encrypted_key.encode())
    return decrypted.decode()

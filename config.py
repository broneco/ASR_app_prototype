"""
Configuration module for Azure Product Recognition App.
Loads settings from environment variables with fallback to defaults.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT: str = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "YOUR_AZURE_OPENAI_ENDPOINT"  # https://your-resource.openai.azure.com/
)
AZURE_OPENAI_API_KEY: str = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "YOUR_AZURE_OPENAI_API_KEY"
)
AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "gpt-4o"
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "text-embedding-3-small"
)
AZURE_OPENAI_API_VERSION: str = os.getenv(
    "AZURE_OPENAI_API_VERSION",
    "2024-08-01-preview"
)

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT: str = os.getenv(
    "AZURE_SEARCH_ENDPOINT",
    "YOUR_AZURE_SEARCH_ENDPOINT"  # https://your-service.search.windows.net
)
AZURE_SEARCH_API_KEY: str = os.getenv(
    "AZURE_SEARCH_API_KEY",
    "YOUR_AZURE_SEARCH_API_KEY"
)
AZURE_SEARCH_INDEX_NAME: str = os.getenv(
    "AZURE_SEARCH_INDEX_NAME",
    "products-index"
)

# Application Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
VECTOR_SEARCH_TOP_K: int = int(os.getenv("VECTOR_SEARCH_TOP_K", "3"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Vector dimensions for text-embedding-3-small
EMBEDDING_DIMENSIONS: int = 1536


def validate_config() -> tuple[bool, Optional[str]]:
    """
    Validates that all required configuration values are set.

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_configs = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
        "AZURE_SEARCH_API_KEY": AZURE_SEARCH_API_KEY,
    }

    for key, value in required_configs.items():
        if value.startswith("YOUR_") or not value:
            return False, f"Configuration error: {key} is not set. Please check your .env file."

    return True, None


if __name__ == "__main__":
    # Quick validation check
    is_valid, error = validate_config()
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print(f"✗ {error}")

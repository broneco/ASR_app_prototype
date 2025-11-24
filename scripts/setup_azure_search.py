#!/usr/bin/env python3
"""
Setup script for Azure AI Search.
Loads products from JSON, generates embeddings, creates index, and uploads documents.
"""
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.azure_client import AzureSearchClient
from src.embeddings import EmbeddingGenerator
from src.models import Product

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_products(file_path: str) -> list[Product]:
    """
    Loads products from JSON file.

    Args:
        file_path: Path to products JSON file

    Returns:
        List of Product objects
    """
    logger.info(f"Loading products from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Products file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    products = [Product(**item) for item in data]
    logger.info(f"✓ Loaded {len(products)} products")
    return products


def generate_embeddings(products: list[Product], generator: EmbeddingGenerator) -> list[Product]:
    """
    Generates embeddings for all products.

    Args:
        products: List of Product objects
        generator: EmbeddingGenerator instance

    Returns:
        List of Product objects with embeddings
    """
    logger.info("Generating embeddings for products...")

    # Extract product names
    product_names = [p.name for p in products]

    # Generate embeddings in batch
    embeddings = generator.generate_embeddings_batch(product_names)

    # Assign embeddings to products
    for product, embedding in zip(products, embeddings):
        product.embedding = embedding

    logger.info(f"✓ Generated embeddings for {len(products)} products")
    return products


def create_index(search_client: AzureSearchClient) -> None:
    """
    Creates the search index.

    Args:
        search_client: AzureSearchClient instance
    """
    logger.info("Creating search index...")

    try:
        created = search_client.create_index()
        if created:
            logger.info("✓ Index created successfully")
        else:
            logger.info("✓ Index already exists")
    except Exception as e:
        logger.error(f"✗ Failed to create index: {e}")
        raise


def upload_products(products: list[Product], search_client: AzureSearchClient) -> None:
    """
    Uploads products to the search index.

    Args:
        products: List of Product objects with embeddings
        search_client: AzureSearchClient instance
    """
    logger.info("Uploading products to search index...")

    # Convert products to dictionaries
    documents = [p.model_dump() for p in products]

    try:
        uploaded = search_client.upload_documents(documents)
        logger.info(f"✓ Uploaded {uploaded} products successfully")
    except Exception as e:
        logger.error(f"✗ Failed to upload products: {e}")
        raise


def verify_search(search_client: AzureSearchClient, generator: EmbeddingGenerator) -> None:
    """
    Verifies search functionality with a test query.

    Args:
        search_client: AzureSearchClient instance
        generator: EmbeddingGenerator instance
    """
    logger.info("Verifying search functionality...")

    test_query = "Castrol Magnatec 5 litrů"
    logger.info(f"Test query: '{test_query}'")

    try:
        # Generate embedding for test query
        query_embedding = generator.generate_embedding(test_query)

        # Perform search
        results = search_client.vector_search(query_embedding, top_k=3)

        if results:
            logger.info(f"✓ Search returned {len(results)} results:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result.product_name} (score: {result.score:.4f})")
        else:
            logger.warning("Search returned no results")

    except Exception as e:
        logger.error(f"✗ Search verification failed: {e}")
        raise


def main():
    """Main setup function."""
    print("\n" + "=" * 80)
    print("AZURE AI SEARCH SETUP")
    print("=" * 80 + "\n")

    # Validate configuration
    print("Step 1: Validating configuration...")
    is_valid, error = config.validate_config()
    if not is_valid:
        print(f"✗ {error}")
        print("\nPlease configure your .env file with Azure credentials.")
        print("See .env.example for the required format.")
        sys.exit(1)
    print("✓ Configuration is valid\n")

    # Initialize clients
    print("Step 2: Initializing Azure clients...")
    try:
        search_client = AzureSearchClient()
        generator = EmbeddingGenerator()
        print("✓ Clients initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize clients: {e}")
        sys.exit(1)

    # Test connections
    print("Step 3: Testing Azure connections...")
    try:
        if not generator.test_connection():
            print("✗ Azure OpenAI connection failed")
            sys.exit(1)
        if not search_client.test_connection():
            print("✗ Azure AI Search connection failed")
            sys.exit(1)
        print("✓ All connections successful\n")
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        sys.exit(1)

    # Load products
    print("Step 4: Loading products...")
    try:
        products_file = os.path.join(
            Path(__file__).parent.parent,
            "data",
            "sample_products.json"
        )
        products = load_products(products_file)
        print(f"✓ Loaded {len(products)} products\n")
    except Exception as e:
        print(f"✗ Failed to load products: {e}")
        sys.exit(1)

    # Generate embeddings
    print("Step 5: Generating embeddings...")
    try:
        products = generate_embeddings(products, generator)
        print("✓ Embeddings generated\n")
    except Exception as e:
        print(f"✗ Failed to generate embeddings: {e}")
        sys.exit(1)

    # Create index
    print("Step 6: Creating search index...")
    try:
        create_index(search_client)
        print("✓ Index ready\n")
    except Exception as e:
        print(f"✗ Failed to create index: {e}")
        sys.exit(1)

    # Upload products
    print("Step 7: Uploading products to index...")
    try:
        upload_products(products, search_client)
        print("✓ Products uploaded\n")
    except Exception as e:
        print(f"✗ Failed to upload products: {e}")
        sys.exit(1)

    # Verify search
    print("Step 8: Verifying search functionality...")
    try:
        verify_search(search_client, generator)
        print("✓ Search verification successful\n")
    except Exception as e:
        print(f"✗ Search verification failed: {e}")
        sys.exit(1)

    print("=" * 80)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nYou can now use the application to process transcripts.")
    print("Try running: python scripts/test_recognition.py")
    print()


if __name__ == "__main__":
    main()

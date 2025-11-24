"""
Embeddings generation module using Azure OpenAI.
Handles creating vector embeddings for text using text-embedding-3-small model.
"""
import logging
from typing import List, Union
from openai import AzureOpenAI, OpenAIError
import config

# Setup logging
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using Azure OpenAI text-embedding-3-small model.
    """

    def __init__(
        self,
        api_key: str = config.AZURE_OPENAI_API_KEY,
        endpoint: str = config.AZURE_OPENAI_ENDPOINT,
        deployment_name: str = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version: str = config.AZURE_OPENAI_API_VERSION
    ):
        """
        Initializes the embedding generator.

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of the embedding model deployment
            api_version: Azure OpenAI API version
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        logger.info(f"Initialized EmbeddingGenerator with deployment: {deployment_name}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If text is empty
            OpenAIError: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            logger.debug(f"Generating embedding for text: {text[:50]}...")
            response = self.client.embeddings.create(
                input=text.strip(),
                model=self.deployment_name
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding

        except OpenAIError as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[List[float]]:
        """
        Generates embeddings for multiple texts in batches.

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch (default: 16)

        Returns:
            List of embedding vectors, one per input text

        Raises:
            ValueError: If texts list is empty
            OpenAIError: If API call fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Generating embeddings for {len(texts)} texts in {total_batches} batches")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                logger.debug(f"Processing batch {batch_num}/{total_batches}")
                # Clean and validate batch
                cleaned_batch = [text.strip() for text in batch if text and text.strip()]

                if not cleaned_batch:
                    logger.warning(f"Batch {batch_num} is empty after cleaning, skipping")
                    continue

                response = self.client.embeddings.create(
                    input=cleaned_batch,
                    model=self.deployment_name
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {batch_num} completed: {len(batch_embeddings)} embeddings")

            except OpenAIError as e:
                logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
                raise

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def test_connection(self) -> bool:
        """
        Tests the connection to Azure OpenAI by generating a test embedding.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing Azure OpenAI connection...")
            test_embedding = self.generate_embedding("test")
            if len(test_embedding) == config.EMBEDDING_DIMENSIONS:
                logger.info("✓ Azure OpenAI connection test successful")
                return True
            else:
                logger.warning(
                    f"Unexpected embedding dimensions: {len(test_embedding)} "
                    f"(expected {config.EMBEDDING_DIMENSIONS})"
                )
                return False
        except Exception as e:
            logger.error(f"✗ Azure OpenAI connection test failed: {e}")
            return False


def create_embedding_generator() -> EmbeddingGenerator:
    """
    Factory function to create an EmbeddingGenerator instance with default configuration.

    Returns:
        Configured EmbeddingGenerator instance
    """
    return EmbeddingGenerator()


if __name__ == "__main__":
    # Test the embedding generator
    logging.basicConfig(level=logging.INFO)

    generator = create_embedding_generator()

    # Test single embedding
    print("\nTest 1: Single embedding")
    try:
        embedding = generator.generate_embedding("CASTROL MAGNATEC 5W-30 A5 5 lt")
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test batch embeddings
    print("\nTest 2: Batch embeddings")
    try:
        texts = [
            "CASTROL MAGNATEC 5W-30 A5 5 lt",
            "SHERON Celoroční ostřikovač eMotion -5 °C 4 lt",
            "EUROL Sportbike 5W-40 1 lt"
        ]
        embeddings = generator.generate_embeddings_batch(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test connection
    print("\nTest 3: Connection test")
    success = generator.test_connection()
    print(f"Connection test: {'✓ PASSED' if success else '✗ FAILED'}")

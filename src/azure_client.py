"""
Azure clients for OpenAI and AI Search services.
"""
import logging
import json
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAIError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchField,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

import config
from src.models import SearchResult

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    Wrapper for Azure OpenAI API with function calling support.
    """

    def __init__(
        self,
        api_key: str = config.AZURE_OPENAI_API_KEY,
        endpoint: str = config.AZURE_OPENAI_ENDPOINT,
        deployment_name: str = config.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version: str = config.AZURE_OPENAI_API_VERSION
    ):
        """
        Initializes the Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of the GPT-4o deployment
            api_version: Azure OpenAI API version
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        logger.info(f"Initialized AzureOpenAIClient with deployment: {deployment_name}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request to Azure OpenAI.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            functions: Optional list of function definitions for function calling
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary containing the response

        Raises:
            OpenAIError: If API call fails
        """
        try:
            logger.debug(f"Sending chat completion request with {len(messages)} messages")

            kwargs: Dict[str, Any] = {
                "model": self.deployment_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if functions:
                kwargs["tools"] = [
                    {"type": "function", "function": func} for func in functions
                ]
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            logger.debug("Chat completion successful")

            return response

        except OpenAIError as e:
            logger.error(f"Chat completion failed: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Tests the connection to Azure OpenAI.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing Azure OpenAI connection...")
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            if response and response.choices:
                logger.info("✓ Azure OpenAI connection test successful")
                return True
            return False
        except Exception as e:
            logger.error(f"✗ Azure OpenAI connection test failed: {e}")
            return False


class AzureSearchClient:
    """
    Wrapper for Azure AI Search operations including vector search.
    """

    def __init__(
        self,
        endpoint: str = config.AZURE_SEARCH_ENDPOINT,
        api_key: str = config.AZURE_SEARCH_API_KEY,
        index_name: str = config.AZURE_SEARCH_INDEX_NAME
    ):
        """
        Initializes the Azure AI Search client.

        Args:
            endpoint: Azure Search service endpoint
            api_key: Azure Search admin key
            index_name: Name of the search index
        """
        self.endpoint = endpoint
        self.index_name = index_name
        self.credential = AzureKeyCredential(api_key)

        self.index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=self.credential
        )

        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self.credential
        )

        logger.info(f"Initialized AzureSearchClient for index: {index_name}")

    def create_index(self, vector_dimensions: int = config.EMBEDDING_DIMENSIONS) -> bool:
        """
        Creates a search index with vector search capabilities.

        Args:
            vector_dimensions: Dimensions of the embedding vectors

        Returns:
            True if index was created successfully, False if it already exists

        Raises:
            HttpResponseError: If index creation fails
        """
        try:
            # Check if index already exists
            try:
                self.index_client.get_index(self.index_name)
                logger.info(f"Index '{self.index_name}' already exists")
                return False
            except ResourceNotFoundError:
                pass

            # Define the index schema
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True
                ),
                SearchableField(
                    name="name",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    filterable=True
                ),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimensions,
                    vector_search_profile_name="default-vector-profile"
                )
            ]

            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw-config"
                    )
                ]
            )

            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )

            result = self.index_client.create_index(index)
            logger.info(f"✓ Created index '{self.index_name}' successfully")
            return True

        except HttpResponseError as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def upload_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Uploads documents to the search index.

        Args:
            documents: List of document dictionaries with 'id', 'name', and 'embedding' fields

        Returns:
            Number of successfully uploaded documents

        Raises:
            HttpResponseError: If upload fails
        """
        if not documents:
            logger.warning("No documents to upload")
            return 0

        try:
            logger.info(f"Uploading {len(documents)} documents to index '{self.index_name}'")
            result = self.search_client.upload_documents(documents=documents)

            succeeded = sum(1 for r in result if r.succeeded)
            failed = len(result) - succeeded

            if failed > 0:
                logger.warning(f"Upload completed with {failed} failures")
            else:
                logger.info(f"✓ Successfully uploaded {succeeded} documents")

            return succeeded

        except HttpResponseError as e:
            logger.error(f"Failed to upload documents: {e}")
            raise

    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = config.VECTOR_SEARCH_TOP_K
    ) -> List[SearchResult]:
        """
        Performs a vector similarity search.

        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects

        Raises:
            HttpResponseError: If search fails
        """
        try:
            logger.debug(f"Performing vector search (top_k={top_k})")

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="embedding"
            )

            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["id", "name"]
            )

            search_results = []
            for result in results:
                search_result = SearchResult(
                    product_id=result["id"],
                    product_name=result["name"],
                    score=result["@search.score"]
                )
                search_results.append(search_result)
                logger.debug(
                    f"Found: {search_result.product_name} "
                    f"(score: {search_result.score:.4f})"
                )

            logger.info(f"Vector search returned {len(search_results)} results")
            return search_results

        except HttpResponseError as e:
            logger.error(f"Vector search failed: {e}")
            raise

    def delete_index(self) -> bool:
        """
        Deletes the search index.

        Returns:
            True if index was deleted, False if it didn't exist

        Raises:
            HttpResponseError: If deletion fails
        """
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"✓ Deleted index '{self.index_name}'")
            return True
        except ResourceNotFoundError:
            logger.info(f"Index '{self.index_name}' does not exist")
            return False
        except HttpResponseError as e:
            logger.error(f"Failed to delete index: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Tests the connection to Azure AI Search.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing Azure AI Search connection...")
            # Try to list indices
            indices = list(self.index_client.list_indexes())
            logger.info(f"✓ Azure AI Search connection test successful (found {len(indices)} indices)")
            return True
        except Exception as e:
            logger.error(f"✗ Azure AI Search connection test failed: {e}")
            return False


def create_azure_clients() -> tuple[AzureOpenAIClient, AzureSearchClient]:
    """
    Factory function to create both Azure clients with default configuration.

    Returns:
        Tuple of (AzureOpenAIClient, AzureSearchClient)
    """
    openai_client = AzureOpenAIClient()
    search_client = AzureSearchClient()
    return openai_client, search_client


if __name__ == "__main__":
    # Test the clients
    logging.basicConfig(level=logging.INFO)

    print("Testing Azure Clients")
    print("=" * 80)

    # Test OpenAI client
    print("\n1. Testing Azure OpenAI Client...")
    openai_client = AzureOpenAIClient()
    if openai_client.test_connection():
        print("   ✓ Azure OpenAI is ready")
    else:
        print("   ✗ Azure OpenAI connection failed")

    # Test Search client
    print("\n2. Testing Azure AI Search Client...")
    search_client = AzureSearchClient()
    if search_client.test_connection():
        print("   ✓ Azure AI Search is ready")
    else:
        print("   ✗ Azure AI Search connection failed")

    print("\n" + "=" * 80)

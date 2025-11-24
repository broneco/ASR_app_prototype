"""
Unit tests for product matcher module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.product_matcher import ProductMatcher, SYSTEM_PROMPT
from src.models import ProcessingResult, MatchedProduct, SearchResult
from src.azure_client import AzureOpenAIClient, AzureSearchClient
from src.embeddings import EmbeddingGenerator


class TestProductMatcher:
    """Tests for ProductMatcher class."""

    @pytest.fixture
    def mock_clients(self):
        """Creates mock Azure clients."""
        openai_client = Mock(spec=AzureOpenAIClient)
        search_client = Mock(spec=AzureSearchClient)
        embedding_generator = Mock(spec=EmbeddingGenerator)
        return openai_client, search_client, embedding_generator

    @pytest.fixture
    def matcher(self, mock_clients):
        """Creates ProductMatcher instance with mock clients."""
        openai_client, search_client, embedding_generator = mock_clients
        return ProductMatcher(
            openai_client=openai_client,
            search_client=search_client,
            embedding_generator=embedding_generator
        )

    def test_initialization(self, matcher):
        """Tests that ProductMatcher initializes correctly."""
        assert matcher.openai_client is not None
        assert matcher.search_client is not None
        assert matcher.embedding_generator is not None
        assert matcher.search_function is not None
        assert matcher.search_function["name"] == "search_products"

    def test_process_transcript_empty_text(self, matcher):
        """Tests that processing empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            matcher.process_transcript("")

        with pytest.raises(ValueError, match="cannot be empty"):
            matcher.process_transcript("   ")

    def test_search_products_function(self, matcher, mock_clients):
        """Tests the internal search function."""
        _, search_client, embedding_generator = mock_clients

        # Setup mocks
        test_embedding = [0.1] * 1536
        embedding_generator.generate_embedding.return_value = test_embedding

        search_results = [
            SearchResult(
                product_id="TEST_1",
                product_name="Test Product 1",
                score=0.85
            ),
            SearchResult(
                product_id="TEST_2",
                product_name="Test Product 2",
                score=0.75
            )
        ]
        search_client.vector_search.return_value = search_results

        # Call function
        results = matcher._search_products_function("test query")

        # Verify
        embedding_generator.generate_embedding.assert_called_once_with("test query")
        search_client.vector_search.assert_called_once()
        assert len(results) == 2
        assert results[0]["product_name"] == "Test Product 1"
        assert results[0]["confidence"] == 0.85

    def test_search_products_function_low_confidence(self, matcher, mock_clients):
        """Tests that low confidence results are filtered out."""
        _, search_client, embedding_generator = mock_clients

        # Setup mocks
        test_embedding = [0.1] * 1536
        embedding_generator.generate_embedding.return_value = test_embedding

        search_results = [
            SearchResult(
                product_id="TEST_1",
                product_name="Test Product 1",
                score=0.85
            ),
            SearchResult(
                product_id="TEST_2",
                product_name="Test Product 2",
                score=0.50  # Below threshold
            )
        ]
        search_client.vector_search.return_value = search_results

        # Call function
        results = matcher._search_products_function("test query")

        # Verify - only high confidence result should be returned
        assert len(results) == 1
        assert results[0]["product_name"] == "Test Product 1"

    def test_process_transcript_with_products(self, matcher, mock_clients):
        """Tests processing a transcript with product mentions."""
        openai_client, search_client, embedding_generator = mock_clients

        # Setup embedding mock
        embedding_generator.generate_embedding.return_value = [0.1] * 1536

        # Setup search mock
        search_client.vector_search.return_value = [
            SearchResult(
                product_id="CASTROL_1",
                product_name="CASTROL MAGNATEC 5W-30 A5 5 lt",
                score=0.92
            )
        ]

        # Mock OpenAI response - simulate function calling flow
        # First response: GPT wants to call function
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search_products"
        mock_tool_call.function.arguments = json.dumps({"query": "Castrol Magnatec 5W-30 A5 5 litrů"})
        mock_tool_call.model_dump.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "search_products",
                "arguments": json.dumps({"query": "Castrol Magnatec 5W-30 A5 5 litrů"})
            }
        }

        first_message = MagicMock()
        first_message.tool_calls = [mock_tool_call]
        first_message.content = None

        first_response = MagicMock()
        first_response.choices = [MagicMock(message=first_message)]

        # Second response: GPT provides final answer
        final_result = {
            "matched_products": [
                {
                    "product_name": "CASTROL MAGNATEC 5W-30 A5 5 lt",
                    "confidence": 0.92,
                    "context": "Castrol Magnatec 5W-30 A5 5 litrů"
                }
            ],
            "competitor_advantage_mentioned": False,
            "bad_placement_mentioned": False
        }

        second_message = MagicMock()
        second_message.tool_calls = None
        second_message.content = json.dumps(final_result, ensure_ascii=False)

        second_response = MagicMock()
        second_response.choices = [MagicMock(message=second_message)]

        # Configure chat_completion to return responses in sequence
        openai_client.chat_completion.side_effect = [first_response, second_response]

        # Process transcript
        text = "Na polici je Castrol Magnatec pětilitrový A5."
        result = matcher.process_transcript(text)

        # Verify
        assert isinstance(result, ProcessingResult)
        assert len(result.matched_products) == 1
        assert result.matched_products[0].product_name == "CASTROL MAGNATEC 5W-30 A5 5 lt"
        assert result.competitor_advantage_mentioned is False
        assert result.bad_placement_mentioned is False
        assert result.raw_text == text

    def test_process_transcript_with_competitor_mention(self, matcher, mock_clients):
        """Tests detection of competitor advantage mentions."""
        openai_client, _, embedding_generator = mock_clients

        # Setup mocks - no products, just competitor detection
        final_result = {
            "matched_products": [],
            "competitor_advantage_mentioned": True,
            "bad_placement_mentioned": False
        }

        message = MagicMock()
        message.tool_calls = None
        message.content = json.dumps(final_result, ensure_ascii=False)

        response = MagicMock()
        response.choices = [MagicMock(message=message)]

        openai_client.chat_completion.return_value = response

        # Process transcript
        text = "Konkurence má výraznější obal."
        result = matcher.process_transcript(text)

        # Verify
        assert result.competitor_advantage_mentioned is True
        assert result.bad_placement_mentioned is False

    def test_process_transcript_with_placement_issue(self, matcher, mock_clients):
        """Tests detection of placement issues."""
        openai_client, _, embedding_generator = mock_clients

        # Setup mocks
        final_result = {
            "matched_products": [],
            "competitor_advantage_mentioned": False,
            "bad_placement_mentioned": True
        }

        message = MagicMock()
        message.tool_calls = None
        message.content = json.dumps(final_result, ensure_ascii=False)

        response = MagicMock()
        response.choices = [MagicMock(message=message)]

        openai_client.chat_completion.return_value = response

        # Process transcript
        text = "Produkt je schovaný a není dobře vidět."
        result = matcher.process_transcript(text)

        # Verify
        assert result.competitor_advantage_mentioned is False
        assert result.bad_placement_mentioned is True

    def test_system_prompt_content(self):
        """Tests that system prompt contains required instructions."""
        assert "vyhledávání produktů" in SYSTEM_PROMPT.lower() or "search_products" in SYSTEM_PROMPT.lower()
        assert "konkurence" in SYSTEM_PROMPT.lower()
        assert "umístění" in SYSTEM_PROMPT.lower() or "schovaný" in SYSTEM_PROMPT.lower()
        assert "json" in SYSTEM_PROMPT.lower()

    def test_function_definition(self, matcher):
        """Tests that function definition is correctly structured."""
        func = matcher.search_function
        assert func["name"] == "search_products"
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"
        assert "query" in func["parameters"]["properties"]
        assert "query" in func["parameters"]["required"]


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_processing_result_creation(self):
        """Tests creating a ProcessingResult."""
        result = ProcessingResult(
            matched_products=[
                MatchedProduct(
                    product_name="Test Product",
                    confidence=0.85,
                    context="test context"
                )
            ],
            competitor_advantage_mentioned=True,
            bad_placement_mentioned=False,
            raw_text="test text"
        )

        assert len(result.matched_products) == 1
        assert result.competitor_advantage_mentioned is True
        assert result.bad_placement_mentioned is False
        assert result.raw_text == "test text"

    def test_processing_result_to_summary(self):
        """Tests summary generation."""
        result = ProcessingResult(
            matched_products=[
                MatchedProduct(
                    product_name="Test Product",
                    confidence=0.85,
                    context="test context"
                )
            ],
            competitor_advantage_mentioned=True,
            bad_placement_mentioned=True,
            raw_text="test text"
        )

        summary = result.to_summary()
        assert "Test Product" in summary
        assert "0.85" in summary
        assert "ANO" in summary  # Both flags should show ANO


class TestMatchedProduct:
    """Tests for MatchedProduct model."""

    def test_matched_product_creation(self):
        """Tests creating a MatchedProduct."""
        product = MatchedProduct(
            product_name="Test Product",
            confidence=0.8567,
            context="test context"
        )

        assert product.product_name == "Test Product"
        assert product.confidence == 0.86  # Should be rounded to 2 decimals
        assert product.context == "test context"

    def test_confidence_validation(self):
        """Tests confidence score validation."""
        # Valid confidence
        product = MatchedProduct(
            product_name="Test",
            confidence=0.5,
            context="ctx"
        )
        assert product.confidence == 0.5

        # Invalid confidence - should raise error
        with pytest.raises(Exception):  # Pydantic will raise validation error
            MatchedProduct(
                product_name="Test",
                confidence=1.5,  # > 1.0
                context="ctx"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

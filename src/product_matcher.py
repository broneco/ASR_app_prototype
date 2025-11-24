"""
Product matcher using Azure OpenAI with function calling and Azure AI Search.
"""
import logging
import json
from typing import List, Dict, Any, Optional

import config
from src.azure_client import AzureOpenAIClient, AzureSearchClient
from src.embeddings import EmbeddingGenerator
from src.models import ProcessingResult, MatchedProduct

logger = logging.getLogger(__name__)


# System prompt for GPT-4o
SYSTEM_PROMPT = """Jsi asistent pro identifikaci produktů z českých transkriptů obchodních návštěv.

Tvým úkolem je:
1. Identifikovat zmínky o produktech v textu a pro každou zmínku zavolat funkci search_products
2. Detekovat zmínky o výhodách konkurence
3. Detekovat zmínky o špatném umístění produktů

PRAVIDLA PRO VYHLEDÁVÁNÍ PRODUKTŮ:
- Pro každou zmínku produktu v textu zavolej funkci search_products s vhodným dotazem
- Dotaz by měl obsahovat: značku, typ produktu, parametry (objem, specifikace)
- Příklady dobrých dotazů:
  * "Castrol Magnatec 5W-30 A5 5 litrů"
  * "Sheron ostřikovač eMotion 4 litry"
  * "Eurol Syntence 0W-20 5 litrů"

PRAVIDLA PRO DETEKCI VÝHOD KONKURENCE:
Hledej zmínky typu:
- "konkurence má akci"
- "konkurence má výraznější obal"
- "konkurence má lepší cenu"
- "konkurenční produkt je lépe vidět"
- slovo "konkurence" nebo "konkurenční" v kontextu výhody

PRAVIDLA PRO DETEKCI ŠPATNÉHO UMÍSTĚNÍ:
Hledej zmínky typu:
- "je schovaný" / "je schovaná"
- "není dobře vidět"
- "není vidět"
- "špatně čitelné cenovky"
- "špatně umístěný"
- "částečně schovaný"
- "není úplně vidět"
- "chybí na polici"
- "mimo správné místo"

FORMÁT VÝSTUPU:
Po dokončení všech vyhledávání vrať strukturovaný JSON s:
{
  "matched_products": [
    {
      "product_name": "název nalezeného produktu",
      "confidence": 0.85,
      "context": "originální zmínka z textu"
    }
  ],
  "competitor_advantage_mentioned": true/false,
  "bad_placement_mentioned": true/false
}

Buď pečlivý a systematický. Nezapomeň na žádnou zmínku produktu v textu.
"""


class ProductMatcher:
    """
    Main class for processing transcripts and matching products.
    Uses Azure OpenAI with function calling and Azure AI Search.
    """

    def __init__(
        self,
        openai_client: Optional[AzureOpenAIClient] = None,
        search_client: Optional[AzureSearchClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initializes the ProductMatcher.

        Args:
            openai_client: Azure OpenAI client (creates new if None)
            search_client: Azure AI Search client (creates new if None)
            embedding_generator: Embedding generator (creates new if None)
        """
        self.openai_client = openai_client or AzureOpenAIClient()
        self.search_client = search_client or AzureSearchClient()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        # Function definition for OpenAI function calling
        self.search_function = {
            "name": "search_products",
            "description": "Vyhledá produkty v katalogu na základě textového dotazu. "
                          "Použij tuto funkci pro každou zmínku produktu v transkriptu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Vyhledávací dotaz obsahující značku, typ produktu a parametry "
                                      "(např. 'Castrol Magnatec 5W-30 A5 5 litrů')"
                    }
                },
                "required": ["query"]
            }
        }

        logger.info("ProductMatcher initialized")

    def _search_products_function(self, query: str) -> List[Dict[str, Any]]:
        """
        Internal function called by GPT-4o to search for products.

        Args:
            query: Search query string

        Returns:
            List of matching products with scores
        """
        try:
            logger.info(f"Searching for products with query: '{query}'")

            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query)

            # Perform vector search
            search_results = self.search_client.vector_search(
                query_vector=query_embedding,
                top_k=config.VECTOR_SEARCH_TOP_K
            )

            # Convert to function result format
            results = []
            for result in search_results:
                if result.score >= config.CONFIDENCE_THRESHOLD:
                    results.append({
                        "product_name": result.product_name,
                        "confidence": round(result.score, 2)
                    })

            logger.debug(f"Found {len(results)} products above threshold")
            return results

        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return []

    def process_transcript(self, text: str) -> ProcessingResult:
        """
        Processes a transcript and identifies products and issues.

        Args:
            text: Transcript text to process

        Returns:
            ProcessingResult with matched products and detected issues

        Raises:
            ValueError: If text is empty
            Exception: If processing fails
        """
        if not text or not text.strip():
            raise ValueError("Transcript text cannot be empty")

        logger.info("=" * 80)
        logger.info(f"Processing transcript: {text[:100]}...")

        try:
            # Prepare messages for GPT-4o
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyzuj tento transkript:\n\n{text}"}
            ]

            # Storage for all matched products
            all_matched_products: List[Dict[str, Any]] = []
            current_context = ""

            # Start conversation with function calling
            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                logger.debug(f"Function calling iteration {iteration}")

                # Call GPT-4o
                response = self.openai_client.chat_completion(
                    messages=messages,
                    functions=[self.search_function],
                    temperature=0.3
                )

                message = response.choices[0].message

                # Check if GPT wants to call a function
                if message.tool_calls:
                    # Process each function call
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        logger.info(f"GPT-4o calling function: {function_name}")
                        logger.debug(f"Function arguments: {function_args}")

                        if function_name == "search_products":
                            # Execute the search
                            query = function_args.get("query", "")
                            current_context = query  # Store context
                            search_results = self._search_products_function(query)

                            # Add results to collection
                            for result in search_results:
                                result["context"] = query
                                all_matched_products.append(result)

                            # Add function response to conversation
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call.model_dump()]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(search_results, ensure_ascii=False)
                            })
                else:
                    # No more function calls - GPT is ready to provide final answer
                    final_content = message.content
                    logger.debug(f"GPT-4o final response: {final_content}")

                    # Parse the final JSON response
                    try:
                        # Extract JSON from response
                        json_start = final_content.find("{")
                        json_end = final_content.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = final_content[json_start:json_end]
                            result_data = json.loads(json_str)
                        else:
                            # Fallback: try to parse entire content as JSON
                            result_data = json.loads(final_content)

                        # Create ProcessingResult
                        # Use products from function calls if available
                        matched_products = []
                        if all_matched_products:
                            matched_products = [
                                MatchedProduct(
                                    product_name=p["product_name"],
                                    confidence=p["confidence"],
                                    context=p["context"]
                                )
                                for p in all_matched_products
                            ]
                        elif "matched_products" in result_data:
                            # Use products from JSON if no function calls were made
                            matched_products = [
                                MatchedProduct(**p)
                                for p in result_data.get("matched_products", [])
                            ]

                        result = ProcessingResult(
                            matched_products=matched_products,
                            competitor_advantage_mentioned=result_data.get(
                                "competitor_advantage_mentioned", False
                            ),
                            bad_placement_mentioned=result_data.get(
                                "bad_placement_mentioned", False
                            ),
                            raw_text=text
                        )

                        logger.info(f"✓ Processing completed: {len(result.matched_products)} products found")
                        logger.info("=" * 80)
                        return result

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {e}")
                        # Return results based on function calls only
                        matched_products = [
                            MatchedProduct(
                                product_name=p["product_name"],
                                confidence=p["confidence"],
                                context=p["context"]
                            )
                            for p in all_matched_products
                        ]

                        result = ProcessingResult(
                            matched_products=matched_products,
                            competitor_advantage_mentioned=False,
                            bad_placement_mentioned=False,
                            raw_text=text
                        )
                        logger.info("=" * 80)
                        return result

            # Max iterations reached
            logger.warning("Max iterations reached without final response")
            matched_products = [
                MatchedProduct(
                    product_name=p["product_name"],
                    confidence=p["confidence"],
                    context=p["context"]
                )
                for p in all_matched_products
            ]

            result = ProcessingResult(
                matched_products=matched_products,
                competitor_advantage_mentioned=False,
                bad_placement_mentioned=False,
                raw_text=text
            )
            logger.info("=" * 80)
            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            logger.info("=" * 80)
            raise


def create_product_matcher() -> ProductMatcher:
    """
    Factory function to create a ProductMatcher with default configuration.

    Returns:
        Configured ProductMatcher instance
    """
    return ProductMatcher()


if __name__ == "__main__":
    # Test the product matcher
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("PRODUCT MATCHER TEST")
    print("=" * 80)

    matcher = create_product_matcher()

    # Test transcript
    test_text = "Na polici je Castrol Magnatec pětilitrový A5 a vedle něj je Sheron ostřikovač čtyřlitrový."

    try:
        result = matcher.process_transcript(test_text)
        print(result.to_summary())
    except Exception as e:
        print(f"\n✗ Test failed: {e}")

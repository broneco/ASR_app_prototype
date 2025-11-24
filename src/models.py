"""
Data models for the Product Recognition application.
Uses Pydantic for data validation and serialization.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class Product(BaseModel):
    """
    Represents a product in the catalog.

    Attributes:
        id: Unique identifier for the product
        name: Full name of the product
        embedding: Vector embedding of the product name (optional, added during indexing)
    """
    id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

    @validator('name')
    def name_not_empty(cls, v: str) -> str:
        """Validates that product name is not empty."""
        if not v or not v.strip():
            raise ValueError("Product name cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "id": "CASTROL_MAGNATEC_5W30_A5_5L",
                "name": "CASTROL MAGNATEC 5W-30 A5 5 lt"
            }
        }


class MatchedProduct(BaseModel):
    """
    Represents a product that was matched from the transcript.

    Attributes:
        product_name: Name of the matched product
        confidence: Confidence score (0.0 to 1.0)
        context: Original mention from the transcript text
    """
    product_name: str = Field(..., description="Name of the matched product")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    context: str = Field(..., description="Original text context")

    @validator('confidence')
    def round_confidence(cls, v: float) -> float:
        """Rounds confidence to 2 decimal places."""
        return round(v, 2)

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "CASTROL MAGNATEC 5W-30 A5 5 lt",
                "confidence": 0.92,
                "context": "Na polici je Castrol Magnatec pětilitrový A5"
            }
        }


class ProcessingResult(BaseModel):
    """
    Result of processing a transcript.

    Attributes:
        matched_products: List of products identified in the transcript
        competitor_advantage_mentioned: Whether competitive advantage was mentioned
        bad_placement_mentioned: Whether poor product placement was mentioned
        raw_text: Original transcript text
    """
    matched_products: List[MatchedProduct] = Field(
        default_factory=list,
        description="List of matched products"
    )
    competitor_advantage_mentioned: bool = Field(
        False,
        description="Flag indicating mention of competitor advantage"
    )
    bad_placement_mentioned: bool = Field(
        False,
        description="Flag indicating mention of poor placement"
    )
    raw_text: str = Field(..., description="Original transcript text")

    @validator('raw_text')
    def text_not_empty(cls, v: str) -> str:
        """Validates that raw text is not empty."""
        if not v or not v.strip():
            raise ValueError("Raw text cannot be empty")
        return v

    def to_summary(self) -> str:
        """
        Returns a human-readable summary of the processing result.

        Returns:
            Formatted string summary
        """
        summary_lines = [
            "=" * 80,
            "VÝSLEDEK ZPRACOVÁNÍ TRANSKRIPTU",
            "=" * 80,
            "",
            f"Originální text: {self.raw_text}",
            "",
            f"Nalezeno produktů: {len(self.matched_products)}",
        ]

        if self.matched_products:
            summary_lines.append("")
            summary_lines.append("Identifikované produkty:")
            for i, product in enumerate(self.matched_products, 1):
                summary_lines.append(
                    f"  {i}. {product.product_name} "
                    f"(confidence: {product.confidence:.2f})"
                )
                summary_lines.append(f"     Kontext: \"{product.context}\"")

        summary_lines.append("")
        summary_lines.append("Detekované problémy:")
        summary_lines.append(
            f"  • Výhoda konkurence: "
            f"{'ANO ⚠️' if self.competitor_advantage_mentioned else 'NE'}"
        )
        summary_lines.append(
            f"  • Špatné umístění: "
            f"{'ANO ⚠️' if self.bad_placement_mentioned else 'NE'}"
        )
        summary_lines.append("=" * 80)

        return "\n".join(summary_lines)

    class Config:
        json_schema_extra = {
            "example": {
                "matched_products": [
                    {
                        "product_name": "CASTROL MAGNATEC 5W-30 A5 5 lt",
                        "confidence": 0.92,
                        "context": "Na polici je Castrol Magnatec pětilitrový A5"
                    }
                ],
                "competitor_advantage_mentioned": False,
                "bad_placement_mentioned": False,
                "raw_text": "Na polici je Castrol Magnatec pětilitrový A5."
            }
        }


class SearchResult(BaseModel):
    """
    Result from Azure AI Search vector search.

    Attributes:
        product_id: ID of the found product
        product_name: Name of the found product
        score: Similarity score from vector search
    """
    product_id: str = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    score: float = Field(..., description="Search similarity score")

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "CASTROL_MAGNATEC_5W30_A5_5L",
                "product_name": "CASTROL MAGNATEC 5W-30 A5 5 lt",
                "score": 0.89
            }
        }


class TranscriptSample(BaseModel):
    """
    Sample transcript for testing purposes.

    Attributes:
        id: Unique identifier for the transcript
        text: Transcript text
        expected_products: List of product names that should be found
        has_competitor_mention: Whether transcript mentions competitor advantage
        has_placement_issue: Whether transcript mentions placement issues
    """
    id: str = Field(..., description="Transcript ID")
    text: str = Field(..., description="Transcript text")
    expected_products: List[str] = Field(
        default_factory=list,
        description="Expected product names"
    )
    has_competitor_mention: bool = Field(
        False,
        description="Has competitor advantage mention"
    )
    has_placement_issue: bool = Field(
        False,
        description="Has placement issue mention"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "transcript_001",
                "text": "Na polici je Castrol Magnatec pětilitrový A5.",
                "expected_products": ["CASTROL MAGNATEC 5W-30 A5 5 lt"],
                "has_competitor_mention": False,
                "has_placement_issue": False
            }
        }

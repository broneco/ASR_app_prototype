#!/usr/bin/env python3
"""
Test script for product recognition.
Loads sample transcripts and allows interactive testing.
"""
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.product_matcher import ProductMatcher
from src.models import TranscriptSample

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_transcripts(file_path: str) -> list[TranscriptSample]:
    """
    Loads sample transcripts from JSON file.

    Args:
        file_path: Path to transcripts JSON file

    Returns:
        List of TranscriptSample objects
    """
    logger.info(f"Loading sample transcripts from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcripts file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcripts = [TranscriptSample(**item) for item in data]
    logger.info(f"✓ Loaded {len(transcripts)} sample transcripts")
    return transcripts


def test_sample_transcripts(matcher: ProductMatcher, transcripts: list[TranscriptSample]) -> None:
    """
    Tests all sample transcripts and displays results.

    Args:
        matcher: ProductMatcher instance
        transcripts: List of sample transcripts
    """
    print("\n" + "=" * 80)
    print("TESTING SAMPLE TRANSCRIPTS")
    print("=" * 80)

    total = len(transcripts)
    successful = 0

    for i, sample in enumerate(transcripts, 1):
        print(f"\n[{i}/{total}] Testing transcript: {sample.id}")
        print("-" * 80)

        try:
            result = matcher.process_transcript(sample.text)

            # Display results
            print(result.to_summary())

            # Validate against expected results
            print("\nValidace výsledků:")

            # Check products
            found_names = {p.product_name for p in result.matched_products}
            expected_names = set(sample.expected_products)

            if found_names == expected_names:
                print(f"  ✓ Produkty: OK (nalezeno {len(found_names)}/{len(expected_names)})")
            else:
                missing = expected_names - found_names
                extra = found_names - expected_names
                if missing:
                    print(f"  ⚠ Produkty: Chybí {len(missing)}: {missing}")
                if extra:
                    print(f"  ⚠ Produkty: Navíc {len(extra)}: {extra}")

            # Check competitor mention
            if result.competitor_advantage_mentioned == sample.has_competitor_mention:
                print(f"  ✓ Konkurence: OK")
            else:
                print(f"  ✗ Konkurence: Očekáváno {sample.has_competitor_mention}, "
                      f"detekováno {result.competitor_advantage_mentioned}")

            # Check placement issue
            if result.bad_placement_mentioned == sample.has_placement_issue:
                print(f"  ✓ Umístění: OK")
            else:
                print(f"  ✗ Umístění: Očekáváno {sample.has_placement_issue}, "
                      f"detekováno {result.bad_placement_mentioned}")

            successful += 1

        except Exception as e:
            print(f"✗ Zpracování selhalo: {e}")
            logger.exception("Error processing transcript")

        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print(f"SOUHRN: {successful}/{total} transkriptů úspěšně zpracováno")
    print("=" * 80 + "\n")


def interactive_mode(matcher: ProductMatcher) -> None:
    """
    Interactive mode for testing custom transcripts.

    Args:
        matcher: ProductMatcher instance
    """
    print("\n" + "=" * 80)
    print("INTERAKTIVNÍ REŽIM")
    print("=" * 80)
    print("\nZadejte text transkriptu (nebo 'quit' pro ukončení):")
    print("Tip: Můžete zadat více řádků. Ukončete zadávání prázdným řádkem.\n")

    while True:
        print("-" * 80)
        lines = []
        print("Transkript:")

        while True:
            try:
                line = input()
                if not line:
                    break
                if line.lower() == 'quit':
                    print("\nKonec interaktivního režimu.")
                    return
                lines.append(line)
            except EOFError:
                print("\nKonec interaktivního režimu.")
                return

        if not lines:
            continue

        text = " ".join(lines)

        try:
            result = matcher.process_transcript(text)
            print(result.to_summary())
        except Exception as e:
            print(f"\n✗ Zpracování selhalo: {e}")
            logger.exception("Error processing transcript")

        print("\nZadejte další text (nebo 'quit' pro ukončení):")


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("PRODUCT RECOGNITION TEST")
    print("=" * 80 + "\n")

    # Validate configuration
    print("Krok 1: Ověření konfigurace...")
    is_valid, error = config.validate_config()
    if not is_valid:
        print(f"✗ {error}")
        print("\nProsím nakonfigurujte váš .env soubor s Azure credentials.")
        print("Viz .env.example pro požadovaný formát.")
        sys.exit(1)
    print("✓ Konfigurace je validní\n")

    # Initialize matcher
    print("Krok 2: Inicializace ProductMatcher...")
    try:
        matcher = ProductMatcher()
        print("✓ ProductMatcher inicializován\n")
    except Exception as e:
        print(f"✗ Inicializace selhala: {e}")
        sys.exit(1)

    # Load sample transcripts
    print("Krok 3: Načítání ukázkových transkriptů...")
    try:
        transcripts_file = os.path.join(
            Path(__file__).parent.parent,
            "data",
            "sample_transcripts.json"
        )
        transcripts = load_sample_transcripts(transcripts_file)
        print(f"✓ Načteno {len(transcripts)} transkriptů\n")
    except Exception as e:
        print(f"✗ Načítání transkriptů selhalo: {e}")
        sys.exit(1)

    # Ask user what to do
    print("Vyberte režim:")
    print("  1. Testovat ukázkové transkripty")
    print("  2. Interaktivní režim (vlastní text)")
    print("  3. Obojí")

    try:
        choice = input("\nVaše volba (1-3): ").strip()
    except EOFError:
        choice = "3"

    if choice == "1":
        test_sample_transcripts(matcher, transcripts)
    elif choice == "2":
        interactive_mode(matcher)
    elif choice == "3":
        test_sample_transcripts(matcher, transcripts)
        interactive_mode(matcher)
    else:
        print("Neplatná volba. Spouštím oba režimy.")
        test_sample_transcripts(matcher, transcripts)
        interactive_mode(matcher)


if __name__ == "__main__":
    main()

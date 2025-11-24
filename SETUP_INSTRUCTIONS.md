# Návod k nastavení Azure Product Recognition App

Tento dokument obsahuje detailní pokyny pro nastavení všech potřebných Azure služeb a spuštění aplikace.

## Přehled

Aplikace vyžaduje následující Azure služby:
1. **Azure OpenAI Service** - pro GPT-4o a text-embedding-3-small
2. **Azure AI Search** - pro vector database s embeddingy produktů

## Předpoklady

- Aktivní Azure subscription
- Python 3.9 nebo novější
- Přístup k Azure Portal (portal.azure.com)

---

## Část 1: Vytvoření Azure OpenAI Resource

### Krok 1.1: Vytvoření Azure OpenAI Service

1. Přihlaste se do [Azure Portal](https://portal.azure.com)

2. Klikněte na **"Create a resource"** (+ ikona v levém menu)

3. Vyhledejte **"Azure OpenAI"**

4. Klikněte na **"Create"**

5. Vyplňte formulář:
   - **Subscription**: Vyberte vaši subscription
   - **Resource group**: Vytvořte novou nebo vyberte existující (např. "product-recognition-rg")
   - **Region**: Vyberte region, kde je Azure OpenAI k dispozici (např. "East US", "Sweden Central")
   - **Name**: Zadejte unikátní jméno (např. "product-recognition-openai")
   - **Pricing tier**: Vyberte "Standard S0"

6. Klikněte na **"Review + create"** a pak **"Create"**

7. Počkejte na dokončení deployment (1-2 minuty)

### Krok 1.2: Deploy GPT-4o Model

1. Po vytvoření resource přejděte do služby

2. V levém menu klikněte na **"Model deployments"** nebo **"Deployments"**

3. Klikněte na **"Create new deployment"** nebo **"+ Create"**

4. Vyplňte:
   - **Select a model**: Zvolte **"gpt-4o"** (nebo nejnovější dostupnou verzi GPT-4o)
   - **Deployment name**: Zadejte **"gpt-4o"** (přesně toto jméno!)
   - **Model version**: Vyberte nejnovější
   - **Deployment type**: Standard
   - **Tokens per Minute Rate Limit**: Doporučeno minimálně 30K

5. Klikněte na **"Create"**

### Krok 1.3: Deploy text-embedding-3-small Model

1. Opět klikněte na **"Create new deployment"**

2. Vyplňte:
   - **Select a model**: Zvolte **"text-embedding-3-small"**
   - **Deployment name**: Zadejte **"text-embedding-3-small"** (přesně toto jméno!)
   - **Model version**: Vyberte nejnovější
   - **Deployment type**: Standard
   - **Tokens per Minute Rate Limit**: Doporučeno minimálně 50K

3. Klikněte na **"Create"**

### Krok 1.4: Získání API Key a Endpoint

1. V Azure OpenAI resource přejděte na **"Keys and Endpoint"** v levém menu

2. Uložte si následující hodnoty:
   - **KEY 1** nebo **KEY 2** (jedna z nich)
   - **Endpoint** (formát: `https://your-resource.openai.azure.com/`)

3. Tyto hodnoty budete potřebovat v `.env` souboru

---

## Část 2: Vytvoření Azure AI Search Resource

### Krok 2.1: Vytvoření Azure AI Search Service

1. V Azure Portal klikněte na **"Create a resource"**

2. Vyhledejte **"Azure AI Search"** (dříve "Azure Cognitive Search")

3. Klikněte na **"Create"**

4. Vyplňte formulář:
   - **Subscription**: Vyberte vaši subscription
   - **Resource group**: Použijte stejnou jako pro OpenAI (např. "product-recognition-rg")
   - **Service name**: Zadejte unikátní jméno (např. "product-recognition-search")
   - **Location**: Ideálně stejný region jako OpenAI (např. "East US")
   - **Pricing tier**: Vyberte minimálně **"Basic"** (Free tier nepodporuje vector search!)

5. Klikněte na **"Review + create"** a pak **"Create"**

6. Počkejte na dokončení deployment (2-3 minuty)

### Krok 2.2: Získání Search Service Credentials

1. Po vytvoření přejděte do Azure AI Search resource

2. V levém menu klikněte na **"Keys"**

3. Uložte si:
   - **Primary admin key** nebo **Secondary admin key**
   - **URL** (formát: `https://your-service.search.windows.net`)

### Krok 2.3: Poznámka o indexu

Index bude vytvořen automaticky setup scriptem v další části. Není potřeba ho vytvářet ručně v portálu.

---

## Část 3: Konfigurace Aplikace

### Krok 3.1: Instalace Python Dependencies

```bash
# Přejděte do složky projektu
cd product-recognition-app

# Vytvořte virtuální prostředí (doporučeno)
python -m venv venv

# Aktivujte virtuální prostředí
# Na Windows:
venv\Scripts\activate
# Na Linux/Mac:
source venv/bin/activate

# Nainstalujte dependencies
pip install -r requirements.txt
```

### Krok 3.2: Vytvoření .env souboru

1. Zkopírujte `.env.example` na `.env`:

```bash
# Na Windows:
copy .env.example .env

# Na Linux/Mac:
cp .env.example .env
```

2. Otevřete `.env` soubor v textovém editoru

3. Vyplňte hodnoty, které jste získali z Azure Portal:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_API_KEY=your-actual-search-key-here
AZURE_SEARCH_INDEX_NAME=products-index

# Application Configuration
LOG_LEVEL=INFO
VECTOR_SEARCH_TOP_K=3
CONFIDENCE_THRESHOLD=0.7
```

**Důležité:**
- Nahraďte `your-resource` a `your-service` skutečnými názvy vašich služeb
- API klíče jsou dlouhé alfanumerické řetězce (32+ znaků)
- Endpoint URL musí končit lomítkem `/`
- Deployment names musí přesně odpovídat názvům, které jste použili při vytváření deploymentů

### Krok 3.3: Ověření konfigurace

```bash
# Spusťte config validaci
python config.py
```

Měli byste vidět:
```
✓ Configuration is valid
```

Pokud vidíte chybu, zkontrolujte:
- Že všechny placeholders jsou nahrazeny
- Že API klíče jsou správné
- Že endpoints mají správný formát

---

## Část 4: Setup Azure AI Search

### Krok 4.1: Spuštění Setup Scriptu

Tento script:
- Načte produkty z `data/sample_products.json`
- Vygeneruje embeddingy pomocí Azure OpenAI
- Vytvoří AI Search index
- Nahraje produkty s embeddingy do indexu
- Ověří funkčnost

```bash
python scripts/setup_azure_search.py
```

**Očekávaný výstup:**

```
================================================================================
AZURE AI SEARCH SETUP
================================================================================

Step 1: Validating configuration...
✓ Configuration is valid

Step 2: Initializing Azure clients...
✓ Clients initialized

Step 3: Testing Azure connections...
✓ Azure OpenAI is ready
✓ Azure AI Search is ready
✓ All connections successful

Step 4: Loading products...
✓ Loaded 20 products

Step 5: Generating embeddings...
✓ Embeddings generated

Step 6: Creating search index...
✓ Index ready

Step 7: Uploading products to index...
✓ Products uploaded

Step 8: Verifying search functionality...
Test query: 'Castrol Magnatec 5 litrů'
✓ Search returned 3 results:
  1. CASTROL MAGNATEC 5W-30 A5 5 lt (score: 0.8923)
  2. CASTROL MAGNATEC 5W-40 C3 60 lt (score: 0.8456)
  3. CASTROL GTX 5W-30 A5 1 lt (score: 0.7891)
✓ Search verification successful

================================================================================
SETUP COMPLETED SUCCESSFULLY!
================================================================================
```

### Krok 4.2: Možné problémy a řešení

**Problém: "Configuration error: AZURE_OPENAI_ENDPOINT is not set"**
- Řešení: Zkontrolujte `.env` soubor, že všechny hodnoty jsou vyplněné

**Problém: "Authentication failed"**
- Řešení: Ověřte, že API klíče jsou správné
- Zkontrolujte, že jste zkopírovali celý klíč včetně všech znaků
- Vyzkoušejte druhý klíč (Key 2)

**Problém: "Model deployment not found"**
- Řešení: Ověřte, že deployment names v `.env` přesně odpovídají názvům v Azure Portal
- Zkontrolujte v Azure Portal, že deploymenty jsou skutečně vytvořené a aktivní

**Problém: "Vector search not supported"**
- Řešení: Azure AI Search musí být minimálně na Basic tier
- Free tier nepodporuje vector search!

**Problém: "Rate limit exceeded"**
- Řešení: Počkejte chvíli a zkuste znovu
- Případně zvyšte token rate limit v Azure Portal

---

## Část 5: Testování Aplikace

### Krok 5.1: Spuštění Test Scriptu

```bash
python scripts/test_recognition.py
```

### Krok 5.2: Výběr režimu

Script nabídne 3 možnosti:
1. **Testovat ukázkové transkripty** - zpracuje všech 10 předpřipravených transkriptů
2. **Interaktivní režim** - můžete zadávat vlastní text
3. **Obojí** - nejprve ukázkové transkripty, pak interaktivní režim

### Krok 5.3: Očekávaný výsledek

Pro ukázkový transkript `"Na polici je Castrol Magnatec pětilitrový A5."` by měl být výstup podobný:

```
================================================================================
VÝSLEDEK ZPRACOVÁNÍ TRANSKRIPTU
================================================================================

Originální text: Na polici je Castrol Magnatec pětilitrový A5.

Nalezeno produktů: 1

Identifikované produkty:
  1. CASTROL MAGNATEC 5W-30 A5 5 lt (confidence: 0.92)
     Kontext: "Castrol Magnatec pětilitrový A5"

Detekované problémy:
  • Výhoda konkurence: NE
  • Špatné umístění: NE
================================================================================
```

---

## Část 6: Použití v Kódu

### Základní použití

```python
from src.product_matcher import ProductMatcher

# Vytvořte matcher
matcher = ProductMatcher()

# Zpracujte text
text = "Na polici je Castrol Magnatec pětilitrový A5."
result = matcher.process_transcript(text)

# Výsledky
print(f"Nalezeno produktů: {len(result.matched_products)}")
for product in result.matched_products:
    print(f"  - {product.product_name} (confidence: {product.confidence})")

print(f"Konkurence: {result.competitor_advantage_mentioned}")
print(f"Špatné umístění: {result.bad_placement_mentioned}")
```

### Přidání vlastních produktů

1. Upravte `data/sample_products.json` - přidejte nové produkty

2. Spusťte znovu setup script:
```bash
python scripts/setup_azure_search.py
```

---

## Část 7: Troubleshooting

### Obecné problémy

**Aplikace je pomalá**
- Azure OpenAI a embeddings vyžadují několik API callů
- Typický čas zpracování: 3-10 sekund na transkript
- Zvažte zvýšení token rate limits v Azure Portal

**Produkty nejsou nalezeny**
- Zkontrolujte, že setup script úspěšně dokončil
- Ověřte, že produkty jsou v indexu (Azure Portal → AI Search → Indexes)
- Zkuste nižší `CONFIDENCE_THRESHOLD` v `.env`

**Chyby při detekci konkurence/umístění**
- GPT-4o model se učí z promptu v `src/product_matcher.py`
- Můžete upravit `SYSTEM_PROMPT` pro lepší detekci
- Přidejte více příkladů klíčových slov

### Logování

Pro debug účely zvyšte log level:

```env
LOG_LEVEL=DEBUG
```

Logy uvidíte v konzoli při běhu aplikace.

### Azure Portal Diagnostika

1. **Azure OpenAI**: Metrics → Request Count, Token Usage
2. **Azure AI Search**: Monitoring → Metrics → Search queries

---

## Další kroky

Teď můžete:

1. ✓ Zpracovávat transkripty pomocí `ProductMatcher`
2. ✓ Přidat vlastní produkty do katalogu
3. ✓ Upravit detection pravidla v system promptu
4. ✓ Integrovat do vlastní aplikace

Pro více informací viz `README.md`.

---

## Časté otázky (FAQ)

**Q: Musím platit za Azure služby?**
A: Ano, Azure OpenAI a Azure AI Search jsou placené služby. Doporučujeme sledovat usage v Azure Portal a nastavit budget alerts.

**Q: Mohu použít Free tier pro AI Search?**
A: Ne, Free tier nepodporuje vector search. Minimálně Basic tier je potřeba.

**Q: Jak často mohu spouštět setup script?**
A: Kdykoliv potřebujete aktualizovat produkty v indexu. Script vytvoří index pokud neexistuje, nebo použije existující.

**Q: Mohu změnit embedding model?**
A: Ano, ale musíte změnit `EMBEDDING_DIMENSIONS` v `config.py` a znovu vytvořit index.

**Q: Podporuje aplikace jiné jazyky než češtinu?**
A: Ano, GPT-4o je multilingual. Stačí upravit system prompt a sample data.

---

Pokud narazíte na problém, který není zde řešený, zkontrolujte logy nebo vytvořte issue v repository.

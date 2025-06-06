# JWST Preprint DOI Analyzer

Automatically analyzes astronomy preprints from arXiv to identify papers containing James Webb Space Telescope (JWST) science results and checks for proper JWST data DOI citations from MAST. Supports both batch processing (by month) and single paper analysis.

## Usage

After installation, the `jwst-preprint-analyzer` command will be available.

### Batch Mode (by Month)
Analyzes all astronomy preprints for a given month and year.
```bash
jwst-preprint-analyzer --year-month YYYY-MM
```
Example:
```bash
jwst-preprint-analyzer --year-month 2024-01 --output-dir ./analysis_results
```

### Single Paper Mode (by arXiv ID)
Analyzes a single preprint specified by its arXiv ID.
```bash
jwst-preprint-analyzer --arxiv-id XXXXX.YYYYY
```
Example:
```bash
jwst-preprint-analyzer --arxiv-id 2309.16028 --gpt-model gpt-4o
```

### Command-Line Options
```bash
jwst-preprint-analyzer [-h] (--year-month YEAR_MONTH | --arxiv-id ARXIV_ID)
                      [--output-dir OUTPUT_DIR] [--prompts-dir PROMPTS_DIR]
                      [--science-threshold SCIENCE_THRESHOLD]
                      [--doi-threshold DOI_THRESHOLD]
                      [--reranker-threshold RERANKER_THRESHOLD]
                      [--reprocess] [--top-k-snippets TOP_K_SNIPPETS]
                      [--context-sentences CONTEXT_SENTENCES]
                      [--reranker-model RERANKER_MODEL]
                      [--gpt-model GPT_MODEL] [--validate-llm]
                      [--ads-key ADS_KEY] [--openai-key OPENAI_KEY]
                      [--cohere-key COHERE_KEY]
```

**Mode Selection (choose one):**
-   `--year-month YEAR_MONTH`: Month to analyze in `YYYY-MM` format (e.g., `2024-01`) for batch processing.
-   `--arxiv-id ARXIV_ID`: Specific arXiv ID (e.g., `2301.12345`) to process for single paper analysis.

**General Options:**
-   `-h, --help`: Show this help message and exit.
-   `--output-dir OUTPUT_DIR, -o OUTPUT_DIR`: Directory to store all outputs (papers, texts, results). Default: current directory (`./`).
-   `--prompts-dir PROMPTS_DIR, -p PROMPTS_DIR`: Directory containing LLM prompt template files (e.g., `science_system.txt`). Default: `./prompts`.
-   `--science-threshold SCIENCE_THRESHOLD`: Threshold for classifying papers as JWST science (0-1). Default: `0.5`.
-   `--doi-threshold DOI_THRESHOLD`: Threshold for considering DOIs as properly cited (0-1). Default: `0.8`.
-   `--reranker-threshold RERANKER_THRESHOLD`: Minimum reranker score for the top snippet to proceed with LLM analysis. Scores below this will skip LLM. Default: `0.05`.
-   `--reprocess`: Force reprocessing of downloaded/analyzed papers, ignoring caches.
-   `--top-k-snippets TOP_K_SNIPPETS`: Number of top reranked snippets to send to the LLM. Default: `5`.
-   `--context-sentences CONTEXT_SENTENCES`: Number of sentences before and after a keyword sentence to include in a snippet. Default: `3`.
-   `--reranker-model RERANKER_MODEL`: Cohere reranker model name. Default: `rerank-v3.5`.
-   `--gpt-model GPT_MODEL`: OpenAI GPT model for science and DOI analysis. Default: `gpt-4o-mini-2024-07-18`.
-   `--validate-llm`: Perform a second LLM call to validate the first analysis (increases cost/time).

**API Key Options:**
-   `--ads-key ADS_KEY`: ADS API key (optional if `ADS_API_KEY` environment variable is set).
-   `--openai-key OPENAI_KEY`: OpenAI API key (optional if `OPENAI_API_KEY` environment variable is set).
-   `--cohere-key COHERE_KEY`: Cohere API key (optional if `COHERE_API_KEY` environment variable is set; reranking skipped if missing).

## Installation

### Prerequisites  
- Python 3.10 or higher
- Virtual environment (recommended)

### Install from Source
```bash
# Clone the repository
git clone <repository-url>
cd jwst-preprint-doi-automator

# Create virtual environment and install
uv venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync  # or: pip install -e .
```

### Environment Variables
Create a `.env` file in the project root:
```bash
ADS_API_KEY=your_ads_key_here        # Required for batch mode
OPENAI_API_KEY=your_openai_key_here  # Required
COHERE_API_KEY=your_cohere_key_here  # Optional (enables snippet reranking)
```

## Testing

```bash
# Install test dependencies and run tests
uv sync --dev
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=jwst_preprint_analyzer --cov-report=term
```

## Output

### Batch Mode
Generates a summary report at `results/YYYY-MM_report.json` with:
- Paper counts and analysis results
- Details on JWST science papers found
- DOI citation compliance status

### Single Paper Mode  
Outputs JSON to stdout with science/DOI analysis results for the specified arXiv paper.

## Development

### Code Formatting
```bash
# Format code  
uv run black jwst_preprint_analyzer/

# Check linting
uv run ruff check jwst_preprint_analyzer/
```